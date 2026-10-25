"""The driver: runs a traced graph on GPU slots by following a plan, recording observations and re-planning."""

from __future__ import annotations

import asyncio
import dataclasses
import heapq
import itertools
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .costmodel import CostModel
from .graph import Graph, LocalModel, Node
from .planner import ClassState, Phase, Plan, Problem, Slot, naive_plan, optimise
from .refs import node_of, resolve
from .store import Store

PILOT_TIER = -1


@dataclass
class SlotRun:
    slot: Slot
    server: Any = None
    model: str | None = None  # key of the resident (or loading) model
    loading: asyncio.Task | None = None
    load_started: float = 0.0
    inflight: int = 0
    phase_idx: int = 0
    entered: int = -1
    stopping: bool = False


@dataclass
class RunReport:
    policy: str
    wall_seconds: float
    tier_seconds: dict[int, float]
    experiment_seconds: dict[str, float]
    loads: int
    load_seconds: float
    queries: int
    output_tokens: int
    plans: list[dict[str, Any]] = field(default_factory=list)
    slots: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class Runner:
    def __init__(self, graph: Graph, tiers: list[list[str]], slots: list[Slot], backend: Any, cost: CostModel,
                 store: Store, policy: str = "plan", replan_seconds: float = 120.0, min_gain: float = 0.03,
                 pilot_units: int = 2, max_inflight: int = 256, log: Callable[[str], None] | None = None, seed: int = 0,
                 only: set[str] | None = None, imported: dict[str, Any] | None = None):
        if policy not in ("plan", "naive"):
            raise ValueError("policy must be 'plan' or 'naive'")
        self.graph, self.slots, self.backend, self.cost, self.store = graph, slots, backend, cost, store
        self.policy, self.replan_seconds, self.min_gain = policy, replan_seconds, min_gain
        self.max_inflight, self.pilot_units = max_inflight, pilot_units
        self.log = log or (lambda line: print(line, flush=True))  # flushed, so a log file shows progress as it happens
        self.rng = random.Random(seed)
        self.scope = set(only) if only is not None else set(graph.nodes)  # the nodes this run executes
        self.imported = imported or {}  # values produced elsewhere (another site)
        self.tier_of_exp = {exp: i for i, group in enumerate(tiers) for exp in group}
        missing = [e for e in graph.experiments if e not in self.tier_of_exp]
        if missing:
            raise ValueError(f"experiments not placed in any tier: {missing}")
        self.experiment_order = [e for group in tiers for e in group]
        self.models: dict[str, LocalModel] = {q.model.key: q.model for q in graph.queries()}
        self.values: dict[str, Any] = {}
        self.done: set[str] = set()
        self.waiting: dict[str, int] = {}
        self.node_tier: dict[str, int] = {}
        self.node_class: dict[str, str] = {}
        self.classes: dict[str, ClassState] = {}
        self.ready: dict[str, dict[str, list[tuple]]] = {}  # model key -> experiment -> heap of (tier, -depth, seq, id)
        self.counter = itertools.count()
        self.wake = asyncio.Event()
        self.slot_runs = {s.id: SlotRun(s) for s in slots}
        self.plan: Plan | None = None
        self.pilot_pending: set[str] = set()
        self.loads = 0
        self.load_time = 0.0
        self.output_tokens = 0
        self.t0 = 0.0
        self.tier_done: dict[int, float] = {}
        self.exp_done: dict[str, float] = {}
        self.plans_log: list[dict[str, Any]] = []
        self.class_rep: dict[str, Node] = {}
        self.exp_left: dict[str, int] = {}
        self.error: BaseException | None = None
        self.tasks: set[asyncio.Task] = set()

    # ---- setup ---------------------------------------------------------------------------------------------------
    def _choose_pilot(self) -> set[str]:
        """Units to run first so the cost model sees every call site and model before the main plan is made."""
        unknown_kinds = set()
        for q in self.graph.queries():
            _, _, known = self.cost.out_tokens(q.query_kind, q.model.key, q.model.request_options().get("max_tokens"))
            speed_known = all(self.cost.speed(q.model.key, s.hw, s.size).known for s in self.slots[:1])
            if not known or not speed_known:
                unknown_kinds.add((q.experiment, q.query_kind))
        if not unknown_kinds or self.pilot_units == 0:
            return set()
        chosen: set[str] = set()
        for exp in self.graph.experiments:
            if not any(e == exp for e, _ in unknown_kinds):
                continue
            units = sorted({n.unit for n in self.graph.nodes.values() if n.experiment == exp})
            for unit in self.rng.sample(units, min(self.pilot_units, len(units))):
                chosen |= {n.id for n in self.graph.nodes.values() if n.experiment == exp and n.unit == unit}
        return chosen

    def _setup(self) -> None:
        prior = {r["id"]: r for r in self.store.read("results")}
        resumed = 0
        for node in self.graph.nodes.values():
            if node.id not in self.scope:
                if node.id in self.imported:
                    self.values[node.id] = self.imported[node.id]
                    self.done.add(node.id)
                continue
            missing = [d for d in node.deps if d not in self.scope and d not in self.imported]
            if missing:
                raise ValueError(f"{node.id} needs {missing}, which this site neither runs nor was given")
            old = prior.get(node.id)
            if old is not None and old.get("fp") == node.fingerprint():
                self.values[node.id] = old["value"]
                self.done.add(node.id)
                self.exp_left[node.experiment] -= 1
                resumed += 1
        if resumed:
            self.log(f"resuming: {resumed} of {len(self.graph.nodes)} nodes already done")
        self.pilot_pending = self._choose_pilot() - self.done if self.policy == "plan" else set()
        if self.pilot_pending:
            self.log(f"pilot: {sum(1 for n in self.pilot_pending if self.graph.nodes[n].kind == 'query')} queries "
                     "run first to measure lengths and speeds")
        for node in self.graph.nodes.values():
            if node.id not in self.scope:
                continue
            self.node_tier[node.id] = PILOT_TIER if node.id in self.pilot_pending else self.tier_of_exp[node.experiment]
            if node.kind == "query":
                cid = f"{node.experiment}|{node.query_kind}|{node.depth}|{self.node_tier[node.id]}"
                self.node_class[node.id] = cid
                if cid not in self.classes:
                    self.classes[cid] = ClassState(id=cid, experiment=node.experiment, tier=self.node_tier[node.id],
                                                   model=node.model.key, depth=node.depth, total=0,
                                                   memory_gb=node.model.memory_gb,
                                                   max_tokens=node.model.request_options().get("max_tokens"))
                self.classes[cid].total += 1
                self.class_rep.setdefault(cid, node)
        parents: dict[str, set[str]] = {cid: set() for cid in self.classes}
        for node in self.graph.queries():
            if node.id not in self.scope:
                continue
            for anc in self._query_ancestors(node):
                parents[self.node_class[node.id]].add(self.node_class[anc])
        for cid, ps in parents.items():
            self.classes[cid].parents = tuple(sorted(ps - {cid}))
        for node in self.graph.nodes.values():
            if node.id not in self.scope:
                continue
            if node.id in self.done:
                if node.kind == "query":
                    c = self.classes[self.node_class[node.id]]
                    c.done += 1
                    c.taken += 1
                    c.released += 1
                continue
            self.waiting[node.id] = sum(1 for d in node.deps if d not in self.done)
        for node_id, count in list(self.waiting.items()):
            if count == 0:
                self._became_ready(node_id)

    def _query_ancestors(self, node: Node) -> set[str]:
        found: set[str] = set()
        stack = list(node.deps)
        seen: set[str] = set()
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            dep = self.graph.nodes[nid]
            if nid not in self.scope:
                continue
            if dep.kind == "query":
                found.add(nid)
            else:
                stack.extend(dep.deps)
        return found

    # ---- readiness -----------------------------------------------------------------------------------------------
    def _lookup(self, key: str) -> Any:
        node_id = node_of(key)
        value = self.values[node_id]
        if "#" in key:
            return value[int(key.split("#", 1)[1])]
        return value

    def _became_ready(self, node_id: str) -> None:
        node = self.graph.nodes[node_id]
        if node.kind == "step":
            try:
                args = resolve(node.args, self._lookup)
                kwargs = resolve(node.kwargs, self._lookup)
                value = node.fn(*args, **kwargs)
                if node.outputs is not None:
                    value = list(value)
                    if len(value) != node.outputs:
                        raise ValueError(f"step {node.fn.__name__} declared outputs={node.outputs} "
                                         f"but returned {len(value)} values")
            except BaseException as exc:  # a broken step stops the run: its dependents can never run
                self.error = exc
                self.wake.set()
                return
            self._finish(node, value)
            return
        cls = self.classes[self.node_class[node_id]]
        cls.released += 1
        heapq.heappush(self.ready.setdefault(node.model.key, {}).setdefault(node.experiment, []),
                       (self.node_tier[node_id], -node.depth, next(self.counter), node_id))

    def _finish(self, node: Node, value: Any, **extra: Any) -> None:
        self.values[node.id] = value
        self.done.add(node.id)
        self.store.result(node.id, node.fingerprint(), value, **extra)
        if node.kind == "query":
            self.classes[self.node_class[node.id]].done += 1
        self.pilot_pending.discard(node.id)
        self.exp_left[node.experiment] -= 1
        if self.exp_left[node.experiment] == 0:
            self._experiment_finished(node.experiment)
        for child in self.graph.children[node.id]:
            if child in self.done or child not in self.scope:
                continue
            self.waiting[child] -= 1
            if self.waiting[child] == 0:
                self._became_ready(child)
        self.wake.set()

    def _experiment_finished(self, exp: str) -> None:
        now = time.monotonic() - self.t0
        self.exp_done[exp] = now
        self.store.event("experiment_done", experiment=exp, seconds=now)
        self.log(f"{now:8.0f}s  experiment {exp} done")
        tier = self.tier_of_exp[exp]
        if all(e in self.exp_done for e, t in self.tier_of_exp.items() if t == tier):
            self.tier_done.setdefault(tier, now)

    # ---- planning ------------------------------------------------------------------------------------------------
    def _problem(self) -> Problem:
        for c in self.classes.values():
            q = self.class_rep[c.id]
            c.out_tokens, _, _ = self.cost.out_tokens(q.query_kind, c.model, c.max_tokens)
        self._estimate_inputs()
        resident = {sid: r.model for sid, r in self.slot_runs.items()}
        loading_left = {}
        for sid, r in self.slot_runs.items():
            if r.loading is not None and not r.loading.done() and r.model:
                expected = self.cost.load_seconds(r.model, r.slot.hw, r.slot.size)[0]
                loading_left[sid] = max(expected - (time.monotonic() - r.load_started), 1.0)
            else:
                loading_left[sid] = 0.0
        snapshot = {cid: dataclasses.replace(c) for cid, c in self.classes.items()}
        return Problem(snapshot, self.slots, self.cost, resident, loading_left)

    def _estimate_inputs(self) -> None:
        """Input tokens per class: rendered prompt length where we have it, else template plus parents' outputs."""
        sums: dict[str, list[float]] = {}
        for c in self.classes.values():
            sums[c.id] = []
        for node in self.graph.queries():
            cid = self.node_class.get(node.id)
            if cid is None or len(sums[cid]) >= 20:
                continue
            text = str(node.prompt) + str(node.system or "")
            parents = self._query_ancestors(node)
            est = len(text) / 4 + sum(self.classes[self.node_class[p]].out_tokens for p in parents)
            est += sum(len(str(self.imported.get(d, ""))) / 4 for d in node.deps if d not in self.scope)
            sums[cid].append(est)
        for cid, values in sums.items():
            if values:
                self.classes[cid].in_tokens = sum(values) / len(values)

    async def _make_plan(self, reason: str) -> None:
        problem = self._problem()
        started = time.monotonic()
        if self.policy == "naive":
            plan = naive_plan(problem, self.experiment_order)
        else:
            current = self._remaining_plan() if self.plan is not None else None  # as it stood at the snapshot
            # in a thread, and run() does not await this, so dispatching carries on while the planner works
            plan = await asyncio.to_thread(optimise, problem)
            if current is not None:
                current_result = await asyncio.to_thread(problem.simulate, current)
                current_key = current_result.tier_finish
                new_key = plan.predicted_tiers
                if not _better(new_key, current_key, self.min_gain):
                    self.plans_log.append({"t": time.monotonic() - self.t0, "reason": reason, "adopted": False,
                                           "current": list(current_key), "candidate": list(new_key)})
                    return
        self.plan = plan
        for r in self.slot_runs.values():
            r.phase_idx = 0
            r.entered = 0  # a new plan starts from whatever each slot holds
        record = {"t": time.monotonic() - self.t0, "reason": reason, "adopted": True,
                  "plan": plan.describe({k: m.name for k, m in self.models.items()}),
                  "predicted_tiers": list(plan.predicted_tiers), "predicted_loads": plan.predicted_loads,
                  "note": plan.note, "solve_seconds": time.monotonic() - started}
        self.plans_log.append(record)
        self.store.event("plan", **record)
        self.log(f"{record['t']:8.0f}s  plan ({reason}, {plan.note}): {record['plan']}  "
                 f"predicted tiers {[x if math.isinf(x) else round(x) for x in plan.predicted_tiers]}s")

    def _remaining_plan(self) -> Plan:
        assert self.plan is not None
        return Plan(phases={sid: self.plan.phases.get(sid, [])[r.phase_idx:] for sid, r in self.slot_runs.items()})

    def _refit(self) -> None:
        self.cost.fit()

    # ---- following the plan --------------------------------------------------------------------------------------
    def _phase(self, run: SlotRun) -> Phase | None:
        assert self.plan is not None
        phases = self.plan.phases.get(run.slot.id, [])
        while run.phase_idx < len(phases):
            phase = phases[run.phase_idx]
            if any(c.done < c.total and c.model in phase.models and
                   (phase.experiment is None or c.experiment == phase.experiment) for c in self.classes.values()):
                return phase
            run.phase_idx += 1
        return None

    def _has_ready(self, model: str, experiment: str | None) -> bool:
        heaps = self.ready.get(model, {})
        if experiment is None:
            return any(heaps.values())
        return bool(heaps.get(experiment))

    def _pop_ready(self, model: str, experiment: str | None) -> str | None:
        heaps = self.ready.get(model, {})
        if experiment is not None:
            heap = heaps.get(experiment)
            return heapq.heappop(heap)[3] if heap else None
        best = None
        for heap in heaps.values():
            if heap and (best is None or heap[0] < best[0]):
                best = heap
        return heapq.heappop(best)[3] if best else None

    def _remaining(self, model: str, experiment: str | None) -> bool:
        return any(c.done < c.total and c.model == model and (experiment is None or c.experiment == experiment)
                   for c in self.classes.values())

    def _next_model(self, run: SlotRun, phase: Phase) -> str | None:
        cycle = list(phase.models)
        start = (cycle.index(run.model) + 1) if run.model in cycle else 0
        ordered = cycle[start:] + cycle[:start]
        for m in ordered:
            if self._has_ready(m, phase.experiment):
                return m
        if run.model in cycle and self._remaining(run.model, phase.experiment):
            return run.model
        for m in ordered:
            if self._remaining(m, phase.experiment):
                return m
        return None

    def _steer(self) -> None:
        for run in self.slot_runs.values():
            if run.loading is not None and not run.loading.done():
                continue
            if run.stopping:
                continue
            phase = self._phase(run)
            if phase is None:
                if run.server is not None and run.inflight == 0:
                    self._switch(run, None)
                continue
            if run.entered != run.phase_idx:
                if phase.fresh and run.server is not None:
                    if run.inflight == 0:
                        self._switch(run, None)
                    continue
                run.entered = run.phase_idx
            if run.model in phase.models and self._has_ready(run.model, phase.experiment) and run.server is not None:
                self._dispatch(run, phase)
                continue
            target = self._next_model(run, phase)
            if target == run.model and run.server is not None:
                self._dispatch(run, phase)
                continue
            if target is not None and run.inflight == 0:
                self._switch(run, target)

    def _switch(self, run: SlotRun, target: str | None) -> None:
        previous = run.model  # run.model names the target from here on

        async def go() -> None:
            if run.server is not None:
                await run.server.stop()
                self.store.event("unload", slot=run.slot.id, model=self.models[previous].name if previous else None)
                run.server = None
            run.model = target
            if target is None:
                return
            model = self.models[target]
            run.load_started = time.monotonic()
            self.store.event("load_start", slot=run.slot.id, model=model.name)
            server = await self.backend.start(model, run.slot)
            seconds = time.monotonic() - run.load_started
            self.loads += 1
            self.load_time += seconds
            self.cost.add({"type": "load", "model": target, "hw": run.slot.hw, "gpus": run.slot.size,
                           "seconds": seconds})
            self.store.observation({"type": "load", "model": target, "hw": run.slot.hw, "gpus": run.slot.size,
                                    "seconds": seconds, "name": model.name})
            self.store.event("load_done", slot=run.slot.id, model=model.name, seconds=seconds)
            self.log(f"{time.monotonic() - self.t0:8.0f}s  {run.slot.id}: loaded {model.name} in {seconds:.0f}s")
            run.server = server

        run.model = target
        run.loading = asyncio.ensure_future(self._guard(go()))
        run.loading.add_done_callback(lambda _: self.wake.set())

    async def _guard(self, coro: Any) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self.error = exc
            self.wake.set()

    def _dispatch(self, run: SlotRun, phase: Phase) -> None:
        while run.inflight < self.max_inflight:
            node_id = self._pop_ready(run.model, phase.experiment)
            if node_id is None:
                break
            node = self.graph.nodes[node_id]
            run.inflight += 1
            self.classes[self.node_class[node.id]].taken += 1
            task = asyncio.ensure_future(self._guard(self._query(run, node)))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

    async def _query(self, run: SlotRun, node: Node) -> None:
        server = run.server
        messages: list[dict[str, str]] = []
        if node.system is not None:
            messages.append({"role": "system", "content": str(resolve(node.system, self._lookup))})
        prompt = resolve(node.prompt, self._lookup)
        if isinstance(prompt, list):
            messages.extend(prompt)
        else:
            messages.append({"role": "user", "content": str(prompt)})
        reply = None
        attempts = 0
        try:
            for attempts in range(1, node.attempts + 1):
                conc_start = run.inflight
                started = time.monotonic()
                for retry in range(4):
                    try:
                        reply = await server.complete(messages)
                        break
                    except Exception:
                        if retry == 3:
                            raise
                        await asyncio.sleep(2 ** retry)
                seconds = time.monotonic() - started
                record = {"type": "request", "model": node.model.key, "name": node.model.name, "hw": run.slot.hw,
                          "gpus": run.slot.size, "kind": node.query_kind, "n_in": reply.n_in, "n_out": reply.n_out,
                          "seconds": seconds, "concurrency": (conc_start + run.inflight) / 2, "t": time.time(),
                          "slot": run.slot.id}
                self.cost.add(record)
                self.store.observation(record)
                self.output_tokens += reply.n_out
                if node.validate is None or node.validate(reply.text):
                    break
        finally:
            run.inflight -= 1
        self._finish(node, reply.text, attempts=attempts)

    # ---- main loop -----------------------------------------------------------------------------------------------
    async def run(self) -> RunReport:
        self.t0 = time.monotonic()
        for exp in self.graph.experiments:
            self.exp_left[exp] = sum(1 for n in self.graph.nodes_of(exp) if n in self.scope)
        self._setup()
        for exp, left in self.exp_left.items():
            if left == 0 and exp not in self.exp_done:
                self._experiment_finished(exp)
        if self.error:
            raise self.error
        await self._make_plan("start")
        last_plan = time.monotonic()
        pilot_was_pending = bool(self.pilot_pending)
        planning: asyncio.Task | None = None
        while any(left > 0 for left in self.exp_left.values()):
            if self.error:
                break
            self._steer()
            self.wake.clear()
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            if planning is not None and planning.done():
                if planning.exception() is not None:
                    self.error = planning.exception()
                planning = None
                last_plan = time.monotonic()
            if self.error:
                break
            pilot_finished = pilot_was_pending and not any(
                self.graph.nodes[n].kind == "query" for n in self.pilot_pending)
            if self.policy == "plan" and planning is None and (
                    pilot_finished or time.monotonic() - last_plan > self.replan_seconds):
                if pilot_finished:
                    pilot_was_pending = False
                self._refit()
                planning = asyncio.ensure_future(self._make_plan("pilot finished" if pilot_finished else "periodic"))
                planning.add_done_callback(lambda _: self.wake.set())
        if planning is not None:
            planning.cancel()  # the work ran out first; the solver thread finishes on its own
        for run in self.slot_runs.values():
            if run.loading is not None and not run.loading.done():
                run.loading.cancel()  # a model still loading when the work ran out is not needed
                try:
                    await run.loading
                except BaseException:
                    pass
            if run.server is not None:
                await run.server.stop()
                self.store.event("unload", slot=run.slot.id, model=self.models[run.model].name if run.model else None)
                run.server = None
        if self.error:
            for task in list(self.tasks):
                task.cancel()
            self.store.close()
            raise self.error
        wall = time.monotonic() - self.t0
        self.store.event("run_done", seconds=wall)
        self.store.close()
        tiers = {t: self.tier_done.get(t, wall) for t in sorted(set(self.tier_of_exp.values()))}
        return RunReport(policy=self.policy, wall_seconds=wall, tier_seconds=tiers, experiment_seconds=self.exp_done,
                         loads=self.loads, load_seconds=self.load_time,
                         queries=sum(1 for _ in self.graph.queries()), output_tokens=self.output_tokens,
                         plans=self.plans_log, slots=[f"{s.id}:{s.hw}x{s.size}" for s in self.slots])

    def results(self) -> dict[str, list[Any]]:
        def lookup(key: str) -> Any:  # values another site produces are left as None
            return self._lookup(key) if node_of(key) in self.values else None

        return {exp: [resolve(v, lookup) for v in outs] for exp, outs in self.graph.unit_outputs.items()}


def _better(new: tuple[float, ...], current: tuple[float, ...], min_gain: float) -> bool:
    """New beats current if, at the first tier where they differ meaningfully, it finishes earlier by min_gain."""
    for n, c in zip(new, current):
        if c == float("inf") and n < c:
            return True
        if n < c * (1 - min_gain) and c - n > 30:
            return True
        if n > c * (1 + min_gain):
            return False
    return False
