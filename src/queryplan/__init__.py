"""queryplan: run a fixed set of LLM queries on local GPUs in the order and placement that finishes soonest.

Write experiments as ordinary Python generators that call `llm()`; queryplan traces them into a graph, learns how long
each kind of query and each model load takes, and decides which model each GPU holds and when.
"""

from __future__ import annotations

import asyncio
import json
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .backends import FakeBackend, VLLMBackend
from .costmodel import CostModel
from .executor import RunReport, Runner
from .graph import Experiment, Graph, LocalModel, experiment, llm, local, step, trace
from .planner import Plan, Problem, Slot, naive_plan, optimise
from .refs import OutputReadError, Ref
from .resources import gpus, visible_gpus
from .split import SitePlan, split
from .store import Store

__all__ = [
    "CostModel", "Experiment", "FakeBackend", "Graph", "LocalModel", "OutputReadError", "Plan", "Problem", "Ref",
    "Result", "RunReport", "Slot", "Store", "VLLMBackend", "experiment", "gpus", "llm", "local", "naive_plan",
    "optimise", "run", "SitePlan", "split", "step", "trace", "visible_gpus",
]

__version__ = "0.1.0"


@dataclass
class Result:
    values: dict[str, list[Any]]  # experiment name -> the values its generator yielded, with outputs filled in
    report: RunReport
    cost_model: CostModel
    graph: Graph


def run(tiers: Sequence[Experiment | Sequence[Experiment]], *, slots: list[Slot], store: str | Path,
        cost_model: CostModel | str | Path | None = None, backend: Any = "vllm", policy: str = "plan",
        save_cost_model: str | Path | None = None, site_plan: "SitePlan | str | Path | None" = None,
        site: str | None = None, imports: Sequence[str | Path] = (), vllm_options: dict[str, Any] | None = None,
        **options: Any) -> Result:
    """Trace the experiments and run them.

    `tiers` is a list whose entries are experiments or groups of experiments, highest priority first. Each tier is
    finished as early as possible before the next is considered: `[e1, e2, e3]` is strict order, `[e1, [e2, e3]]`
    puts e1 first and then finishes e2 and e3 together as early as possible.

    `policy="plan"` optimises; `policy="naive"` runs each experiment start to finish on one GPU at a time, the way a
    plain script would, for comparison.

    With `site_plan` (from `qp.split`) and `site`, only that site's share of the graph runs; `imports` are the
    `results.jsonl` files of earlier sites, which supply the values crossing into this one.

    `vllm_options` go to `VLLMBackend` (for example `local_snapshots=` a Hugging Face hub cache to serve from).
    """
    groups = [[t] if isinstance(t, Experiment) else list(t) for t in tiers]
    experiments = [e for g in groups for e in g]
    graph = trace(experiments)
    if isinstance(cost_model, (str, Path)):
        cost = CostModel.load(cost_model) if Path(cost_model).exists() else CostModel()
    else:
        cost = cost_model or CostModel()
    store_obj = Store(store)
    if backend == "vllm":
        backend = VLLMBackend(store_obj.path / "logs", **(vllm_options or {}))
    only = imported = None
    if site_plan is not None:
        plan = site_plan if isinstance(site_plan, SitePlan) else SitePlan.load(site_plan)
        if site not in plan.nodes:
            raise ValueError(f"site must be one of {list(plan.nodes)}")
        only = set(plan.nodes[site])
        imported = {}
        for path in imports:
            for line in Path(path).read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    imported[record["id"]] = record["value"]
    runner = Runner(graph, [[e.name for e in g] for g in groups], slots, backend, cost, store_obj, policy=policy,
                    only=only, imported=imported, **options)

    received: list[int] = []

    async def go() -> RunReport:
        # model servers run in their own sessions, so a signal that ends this process does not reach them: turn
        # SIGTERM and SIGHUP into a cancellation, which stops every server on the way out (Ctrl-C already cancels)
        loop, main = asyncio.get_running_loop(), asyncio.current_task()
        for signum in (signal.SIGTERM, signal.SIGHUP):
            try:
                loop.add_signal_handler(signum, lambda s=signum: (received.append(s), main.cancel()))
            except (ValueError, RuntimeError, NotImplementedError):  # not the main thread, or not supported here
                pass
        try:
            return await runner.run()
        finally:
            close = getattr(backend, "close", None)
            if close is not None:
                await close()

    try:
        report = asyncio.run(go())
    except asyncio.CancelledError:
        if received:
            raise SystemExit(128 + received[0]) from None
        raise
    cost.fit()
    if save_cost_model:
        cost.save(save_cost_model)
    return Result(values=runner.results(), report=report, cost_model=cost, graph=graph)
