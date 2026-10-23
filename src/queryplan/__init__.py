"""queryplan: run a fixed set of LLM queries on local GPUs in the order and placement that finishes soonest.

Write experiments as ordinary Python generators that call `llm()`; queryplan traces them into a graph, learns how long
each kind of query and each model load takes, and decides which model each GPU holds and when.
"""

from __future__ import annotations

import asyncio
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
from .store import Store

__all__ = [
    "CostModel", "Experiment", "FakeBackend", "Graph", "LocalModel", "OutputReadError", "Plan", "Problem", "Ref",
    "Result", "RunReport", "Slot", "Store", "VLLMBackend", "experiment", "gpus", "llm", "local", "naive_plan",
    "optimise", "run", "step", "trace", "visible_gpus",
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
        save_cost_model: str | Path | None = None, **options: Any) -> Result:
    """Trace the experiments and run them.

    `tiers` is a list whose entries are experiments or groups of experiments, highest priority first. Each tier is
    finished as early as possible before the next is considered: `[e1, e2, e3]` is strict order, `[e1, [e2, e3]]`
    puts e1 first and then finishes e2 and e3 together as early as possible.

    `policy="plan"` optimises; `policy="naive"` runs each experiment start to finish on one GPU at a time, the way a
    plain script would, for comparison.
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
        backend = VLLMBackend(store_obj.path / "logs")
    runner = Runner(graph, [[e.name for e in g] for g in groups], slots, backend, cost, store_obj, policy=policy,
                    **options)

    async def go() -> RunReport:
        try:
            return await runner.run()
        finally:
            close = getattr(backend, "close", None)
            if close is not None:
                await close()

    report = asyncio.run(go())
    cost.fit()
    if save_cost_model:
        cost.save(save_cost_model)
    return Result(values=runner.results(), report=report, cost_model=cost, graph=graph)
