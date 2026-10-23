"""Declaring work: model handles, `llm()`, `@step`, experiments, and tracing them into a graph."""

from __future__ import annotations

import contextvars
import functools
import hashlib
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

from .refs import Ref, find_keys, guard_regex, node_of


@dataclass(frozen=True)
class LocalModel:
    """A locally served model. Everything that changes outputs is fixed here and never changed by the planner."""

    model: str
    sampling: tuple[tuple[str, Any], ...] = ()
    chat_template_kwargs: tuple[tuple[str, Any], ...] = ()
    serve_args: tuple[str, ...] = ()
    memory_gb: float | None = None
    label: str | None = None

    @property
    def key(self) -> str:
        settings = json.dumps([self.sampling, self.chat_template_kwargs, self.serve_args], sort_keys=True, default=str)
        return f"{self.model}@{hashlib.sha1(settings.encode()).hexdigest()[:8]}"

    @property
    def name(self) -> str:
        return self.label or self.model.split("/")[-1]

    def request_options(self) -> dict[str, Any]:
        options = dict(self.sampling)
        if self.chat_template_kwargs:
            options["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        return options


def local(model: str, *, serve_args: Iterable[str] = (), memory_gb: float | None = None,
          chat_template_kwargs: dict[str, Any] | None = None, label: str | None = None,
          **sampling: Any) -> LocalModel:
    """Declare a locally served model with fixed sampling settings (temperature, top_p, max_tokens, seed, ...).

    `serve_args` are extra `vllm serve` arguments (e.g. a reasoning parser or max model length). `memory_gb` is the GPU
    memory the model needs; models without it are assumed to fit on any GPU.
    """
    return LocalModel(model=model, sampling=tuple(sorted(sampling.items())),
                      chat_template_kwargs=tuple(sorted((chat_template_kwargs or {}).items())),
                      serve_args=tuple(serve_args), memory_gb=memory_gb, label=label)


@dataclass
class Node:
    id: str
    kind: str  # "query" or "step"
    experiment: str
    unit: int
    deps: tuple[str, ...]  # ids of nodes whose outputs this node reads
    depth: int  # number of queries on the longest chain ending at this node (queries count themselves)
    # queries
    model: LocalModel | None = None
    prompt: Any = None
    system: Any = None
    validate: Callable[[str], bool] | None = None
    attempts: int = 1
    callsite: str = ""
    # steps
    fn: Callable[..., Any] | None = None
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    outputs: int | None = None

    @property
    def query_kind(self) -> str:
        """What the cost model learns output lengths for: one call site with one model."""
        assert self.model is not None
        return f"{self.callsite}|{self.model.key}"

    def fingerprint(self) -> str:
        """Changes when the declaration of this node changes, so a resumed run can tell its old results are stale."""
        if self.kind == "query":
            body = [self.model.key if self.model else None, repr(self.prompt), repr(self.system), self.attempts]
        else:
            body = [getattr(self.fn, "__qualname__", repr(self.fn)), repr(self.args), repr(sorted(self.kwargs.items()))]
        return hashlib.sha1(json.dumps(body, default=str).encode()).hexdigest()[:16]


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.children: dict[str, list[str]] = {}
        self.experiments: list[str] = []
        self.unit_outputs: dict[str, list[Any]] = {}

    def add(self, node: Node) -> None:
        self.nodes[node.id] = node
        self.children.setdefault(node.id, [])
        for dep in node.deps:
            self.children[dep].append(node.id)

    def queries(self) -> Iterator[Node]:
        return (n for n in self.nodes.values() if n.kind == "query")

    def nodes_of(self, experiment: str) -> list[str]:
        return [n.id for n in self.nodes.values() if n.experiment == experiment]

    def summary(self) -> dict[str, Any]:
        per_model: dict[str, int] = {}
        for q in self.queries():
            per_model[q.model.name] = per_model.get(q.model.name, 0) + 1
        return {"experiments": len(self.experiments), "queries": sum(per_model.values()),
                "steps": sum(1 for n in self.nodes.values() if n.kind == "step"), "queries_per_model": per_model}


@dataclass
class _Trace:
    graph: Graph
    experiment: str
    unit: int = 0
    counter: int = 0

    def new_id(self) -> str:
        self.counter += 1
        return f"{self.experiment}/{self.counter}"


_TRACE: contextvars.ContextVar[_Trace | None] = contextvars.ContextVar("queryplan_trace", default=None)


def _depth_of(graph: Graph, deps: Iterable[str]) -> int:
    return max((graph.nodes[d].depth for d in deps), default=0)


def _callsite(skip: int) -> str:
    frame = sys._getframe(skip)
    return f"{frame.f_code.co_filename.rsplit('/', 1)[-1]}:{frame.f_lineno}"


def llm(model: LocalModel, prompt: Any, *, system: Any = None, validate: Callable[[str], bool] | None = None,
        attempts: int = 1) -> Ref:
    """Declare one query to `model`. Returns a placeholder for its reply text.

    `validate`, if given, is called on the reply when the graph runs; a reply that fails it is requested again, up to
    `attempts` times in total, and the last reply is kept.
    """
    trace = _TRACE.get()
    if trace is None:
        raise RuntimeError("llm() can only be called inside an experiment (a function decorated with @experiment).")
    if not isinstance(model, LocalModel):
        raise TypeError("llm() needs a model handle made by queryplan.local(...)")
    deps = tuple(sorted({node_of(k) for k in find_keys((prompt, system))}))
    node = Node(id=trace.new_id(), kind="query", experiment=trace.experiment, unit=trace.unit, deps=deps,
                depth=_depth_of(trace.graph, deps) + 1, model=model, prompt=prompt, system=system,
                validate=validate, attempts=max(1, attempts), callsite=_callsite(2))
    trace.graph.add(node)
    return Ref(node.id)


def step(fn: Callable[..., Any] | None = None, *, outputs: int | None = None) -> Any:
    """Mark a function that reads LLM outputs. Called with placeholders, it is deferred until its inputs exist.

    With `outputs=n` the function must return a sequence of n values, and the deferred call returns n placeholders,
    so callers can unpack or iterate it while the graph is being built.
    """
    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            trace = _TRACE.get()
            keys = find_keys((args, kwargs))
            if trace is None or not keys:
                return func(*args, **kwargs)
            deps = tuple(sorted({node_of(k) for k in keys}))
            node = Node(id=trace.new_id(), kind="step", experiment=trace.experiment, unit=trace.unit, deps=deps,
                        depth=_depth_of(trace.graph, deps), fn=func, args=args, kwargs=kwargs, outputs=outputs)
            trace.graph.add(node)
            if outputs is None:
                return Ref(node.id)
            return tuple(Ref(node.id, i) for i in range(outputs))

        wrapper.__queryplan_step__ = True  # type: ignore[attr-defined]
        return wrapper

    return decorate(fn) if fn is not None else decorate


class Experiment:
    """A named generator function that yields one unit of work at a time (e.g. one market, one prompt)."""

    def __init__(self, name: str, fn: Callable[[], Iterable[Any]]):
        self.name = name
        self.fn = fn

    def __repr__(self) -> str:
        return f"<Experiment {self.name}>"


def experiment(name: str) -> Callable[[Callable[[], Iterable[Any]]], Experiment]:
    def decorate(fn: Callable[[], Iterable[Any]]) -> Experiment:
        return Experiment(name, fn)

    return decorate


def trace(experiments: Iterable[Experiment]) -> Graph:
    """Run each experiment's generator with placeholders, recording every query and step it declares."""
    graph = Graph()
    for exp in experiments:
        if exp.name in graph.unit_outputs:
            raise ValueError(f"two experiments are called {exp.name!r}")
        graph.experiments.append(exp.name)
        state = _Trace(graph=graph, experiment=exp.name)
        token = _TRACE.set(state)
        outputs: list[Any] = []
        try:
            with guard_regex():
                for value in exp.fn():
                    outputs.append(value)
                    state.unit += 1
        finally:
            _TRACE.reset(token)
        graph.unit_outputs[exp.name] = outputs
    return graph
