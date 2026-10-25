"""Model servers. `VLLMBackend` runs `vllm serve` on a GPU slot; `FakeBackend` simulates one for tests and dry runs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import resource
import secrets
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .graph import LocalModel
from .planner import Slot


@dataclass
class Reply:
    text: str
    n_in: int
    n_out: int


class Server(Protocol):
    model: LocalModel

    async def complete(self, messages: list[dict[str, str]]) -> Reply: ...

    async def stop(self) -> None: ...


class Backend(Protocol):
    async def start(self, model: LocalModel, slot: Slot) -> Server: ...


def free_port(start: int) -> int:
    """The first port from `start` up that nothing holds now. Another process can still take it before the server
    binds it; `VLLMBackend` gives each server its own key, so a server that is not ours is never used."""
    port = start
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1


def raise_open_files(want: int = 65536) -> None:
    """Raise this process's soft limit on open files toward `want`: every request in flight holds a socket."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = want if hard == resource.RLIM_INFINITY else min(want, hard)
    if soft != resource.RLIM_INFINITY and soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))


class VLLMServer:
    def __init__(self, model: LocalModel, proc: subprocess.Popen, port: int, client: Any, log: Path, key: str):
        self.model = model
        self.proc = proc
        self.port = port
        self.client = client
        self.log = log
        self.headers = {"Authorization": f"Bearer {key}"}

    async def complete(self, messages: list[dict[str, str]]) -> Reply:
        body = {"model": self.model.model, "messages": messages, **self.model.request_options()}
        response = await self.client.post(f"http://127.0.0.1:{self.port}/v1/chat/completions", json=body,
                                          headers=self.headers)
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}
        return Reply(text=data["choices"][0]["message"].get("content") or "",
                     n_in=int(usage.get("prompt_tokens", 0)), n_out=int(usage.get("completion_tokens", 0)))

    async def stop(self) -> None:
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            for _ in range(60):
                if self.proc.poll() is not None:
                    break
                await asyncio.sleep(0.5)
            if self.proc.poll() is None:
                os.killpg(self.proc.pid, signal.SIGKILL)
        await asyncio.sleep(3)  # let the driver release GPU memory before the next server starts


class VLLMBackend:
    """Starts one OpenAI-compatible `vllm serve` process per resident model, pinned to the slot's GPUs."""

    def __init__(self, log_dir: str | Path, port_base: int | None = None, vllm: str | list[str] | None = None,
                 extra_env: dict[str, str] | None = None, ready_timeout: float = 3600,
                 local_snapshots: str | Path | None = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        default_base = os.environ.get("QUERYPLAN_PORT_BASE") or (
            20000 + int(os.environ.get("SLURM_JOB_ID", os.getpid())) % 2000 * 10)
        self.port_base = port_base or int(default_base)
        if isinstance(vllm, str):
            vllm = [vllm]
        self.vllm = vllm or ([os.environ["QUERYPLAN_VLLM"]] if os.environ.get("QUERYPLAN_VLLM") else python_vllm())
        self.extra_env = extra_env or {}
        self.ready_timeout = ready_timeout
        # serve each model from its folder in this Hugging Face hub cache, so no hub lookup happens at all (useful on a
        # shared cache that is incomplete or not writable)
        self.local_snapshots = Path(local_snapshots).expanduser() if local_snapshots else None
        import httpx  # imported here so the package imports without it (tests use FakeBackend)

        self._httpx = httpx
        raise_open_files()
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(3600.0, connect=30.0),
                                        limits=httpx.Limits(max_connections=2048, max_keepalive_connections=512))

    async def start(self, model: LocalModel, slot: Slot) -> VLLMServer:
        port = free_port(self.port_base + 10 * int(slot.gpus[0]))
        source = snapshot_dir(self.local_snapshots, model.model) if self.local_snapshots else None
        key = secrets.token_hex(16)  # only this server accepts it, so a server from another run is never used
        args = [*self.vllm, "serve", str(source or model.model), "--host", "127.0.0.1", "--port", str(port),
                "--api-key", key, "--tensor-parallel-size", str(slot.size), *model.serve_args]
        if source is not None:
            args += ["--served-model-name", model.model]
        if "--gpu-memory-utilization" not in model.serve_args:
            args += ["--gpu-memory-utilization", f"{gpu_memory_share(slot.gpus):.2f}"]
        env = {**os.environ, **self.extra_env, "CUDA_VISIBLE_DEVICES": ",".join(map(str, slot.gpus)),
               "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        log = self.log_dir / f"vllm_{slot.id}_{model.name}_{int(time.time())}.log"
        handle = open(log, "w")
        proc = subprocess.Popen(args, stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        server = VLLMServer(model, proc, port, self.client, log, key)
        try:
            await self._wait_ready(server, proc, log)
        except BaseException:  # failed or cancelled: never leave a half-started server holding the GPU
            await server.stop()
            raise
        return server

    async def _wait_ready(self, server: VLLMServer, proc: subprocess.Popen, log: Path) -> None:
        model, port = server.model, server.port
        deadline = time.monotonic() + self.ready_timeout
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"vllm serve {model.model} exited with code {proc.returncode}; see {log}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"vllm serve {model.model} not ready after {self.ready_timeout}s; see {log}")
            try:
                # ours accepts our key and refuses requests without it; a server from another run fails one of these
                url = f"http://127.0.0.1:{port}/v1/models"
                r = await self.client.get(url, headers=server.headers, timeout=5)
                if r.status_code == 200 and (await self.client.get(url, timeout=5)).status_code == 401:
                    break
            except self._httpx.HTTPError:
                pass
            await asyncio.sleep(2)
        # "ready" is not "serving": one real completion before any work is sent
        await server.complete([{"role": "user", "content": "Say OK."}])

    async def close(self) -> None:
        await self.client.aclose()


class FakeServer:
    def __init__(self, model: LocalModel, backend: "FakeBackend"):
        self.model = model
        self.backend = backend
        self.inflight = 0

    async def complete(self, messages: list[dict[str, str]]) -> Reply:
        text_in = " ".join(m["content"] for m in messages)
        n_in = max(1, len(text_in) // 4)
        seed = int(hashlib.sha1((self.model.model + text_in).encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        mean = self.backend.out_tokens.get(self.model.model, 200)
        n_out = max(1, int(rng.lognormvariate(0, 0.4) * mean))
        self.inflight += 1
        try:
            theta = self.backend.theta.get(self.model.model, 2000.0)
            # each request decodes at theta / (inflight + half) tokens/s, matching the cost model's form
            rate = theta / (self.inflight + self.backend.half)
            await asyncio.sleep(n_out / rate / self.backend.speedup)
        finally:
            self.inflight -= 1
        reply = self.backend.reply(self.model, text_in, rng)
        return Reply(text=reply, n_in=n_in, n_out=n_out)

    async def stop(self) -> None:
        if self in self.backend.running:
            self.backend.running.remove(self)
        await asyncio.sleep(0)


class FakeBackend:
    """Simulated servers: load delays and token throughput follow the given parameters, `speedup` times faster."""

    def __init__(self, load_s: dict[str, float] | None = None, theta: dict[str, float] | None = None,
                 out_tokens: dict[str, float] | None = None, half: float = 8.0, speedup: float = 1.0,
                 reply: Any = None):
        self.load_s = load_s or {}
        self.theta = theta or {}
        self.out_tokens = out_tokens or {}
        self.half = half
        self.speedup = speedup
        self.reply = reply or (lambda model, text, rng: f"{model.name} reply {rng.randint(1, 9)}")
        self.starts: list[tuple[float, str, str]] = []
        self.running: list[FakeServer] = []  # started and not yet stopped

    async def start(self, model: LocalModel, slot: Slot) -> FakeServer:
        self.starts.append((time.monotonic(), slot.id, model.model))
        await asyncio.sleep(self.load_s.get(model.model, 30.0) / self.speedup)
        server = FakeServer(model, self)
        self.running.append(server)
        return server

    async def close(self) -> None:
        pass


def snapshot_dir(hub: Path, model: str) -> Path | None:
    """The newest complete snapshot of `model` in a Hugging Face hub cache directory, or None."""
    root = hub / f"models--{model.replace('/', '--')}" / "snapshots"
    candidates = [d for d in root.glob("*") if (d / "config.json").exists() and _weights_complete(d)]
    return max(candidates, key=lambda d: d.stat().st_mtime) if candidates else None


def _weights_complete(snapshot: Path) -> bool:
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        return all((snapshot / shard).exists() for shard in shards)
    return any(snapshot.glob("*.safetensors"))


def gpu_memory_share(gpus: tuple[int, ...], ceiling: float = 0.90, floor: float = 0.50) -> float:
    """The share of GPU memory a new server may claim: 0.90, or less if something already holds memory there."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ceiling
    shares = []
    for line in out.strip().splitlines():
        idx, free, total = (x.strip() for x in line.split(","))
        if int(idx) in gpus:
            shares.append(float(free) / float(total) - 0.03)
    return max(floor, min([ceiling, *shares]))


def python_vllm() -> list[str]:
    """How to run vLLM's command line: through the running Python, so the server uses the same environment (an
    installed `vllm` script can name an interpreter that does not exist on this host)."""
    return [sys.executable, "-m", "vllm.entrypoints.cli.main"]
