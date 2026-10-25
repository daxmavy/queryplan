import asyncio
import os
import resource
import signal
import subprocess
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import queryplan as qp
from queryplan import backends
from queryplan.executor import Runner

FAKE_VLLM = [sys.executable, str(Path(__file__).with_name("fake_vllm.py"))]
SLOT = qp.gpus([0], hw="sim", mem_gb=80)[0]


class Anything(BaseHTTPRequestHandler):
    """Another run's server: answers every request with 200."""

    def log_message(self, *a):
        pass

    def _ok(self):
        if self.command == "POST":
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = (b'{"data": [{"id": "m"}], "choices": [{"message": {"content": "not yours"}}], '
                b'"usage": {"prompt_tokens": 1, "completion_tokens": 1}}')
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = _ok


def test_a_started_server_answers_with_its_own_model(tmp_path):
    async def go():
        backend = qp.VLLMBackend(tmp_path, vllm=FAKE_VLLM, port_base=21000)
        server = await backend.start(qp.local("org/m"), SLOT)
        try:
            reply = await server.complete([{"role": "user", "content": "hi"}])
        finally:
            await server.stop()
            await backend.close()
        return reply

    assert asyncio.run(go()).text == "org/m OK"


def test_a_server_on_the_same_port_from_another_run_is_never_used(tmp_path, monkeypatch):
    foreign = HTTPServer(("127.0.0.1", 0), Anything)
    port = foreign.server_address[1]
    threading.Thread(target=foreign.serve_forever, daemon=True).start()
    # the race: both runs saw the port free, and the other run's server bound it first
    monkeypatch.setattr(backends, "free_port", lambda start: port)
    monkeypatch.setenv("FAKE_VLLM_DELAY", "1")

    async def go():
        backend = qp.VLLMBackend(tmp_path, vllm=FAKE_VLLM, ready_timeout=20)
        try:
            await backend.start(qp.local("org/m"), SLOT)
        finally:
            await backend.close()

    try:
        with pytest.raises(RuntimeError):
            asyncio.run(go())
    finally:
        foreign.shutdown()


def test_backend_raises_the_open_file_limit(tmp_path):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard))
    try:
        qp.VLLMBackend(tmp_path, vllm=FAKE_VLLM)
        now, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        assert now >= min(hard, 8192)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_cancelling_a_run_stops_its_servers(tmp_path):
    from test_executor import backend, make_experiments

    fake = backend(100)

    async def go():
        runner = Runner(qp.trace(make_experiments()), [["e1", "e2", "e3", "e4"]], qp.gpus([0, 1], hw="sim", mem_gb=80),
                           fake, qp.CostModel(), qp.Store(tmp_path / "s"), log=lambda _: None)
        task = asyncio.ensure_future(runner.run())
        while not fake.running:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    assert fake.running == []


def test_sigterm_stops_servers_and_exits(tmp_path):
    script = textwrap.dedent(f"""
        import os, signal, sys, threading
        sys.path.insert(0, {str(Path(__file__).parent)!r})
        import queryplan as qp
        from test_executor import backend, make_experiments
        fake = backend(5)
        started = fake.start
        async def start(model, slot):
            server = await started(model, slot)
            threading.Timer(0.2, os.kill, (os.getpid(), signal.SIGTERM)).start()
            return server
        fake.start = start
        try:
            qp.run(make_experiments(), slots=qp.gpus([0], hw="sim", mem_gb=80), store={str(tmp_path / 's')!r},
                   backend=fake, log=lambda _: None)
        except SystemExit as exc:
            print("exit", exc.code, "running", len(fake.running), flush=True)
            raise
    """)
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert "running 0" in done.stdout, done.stdout + done.stderr
    assert done.returncode == 128 + signal.SIGTERM
