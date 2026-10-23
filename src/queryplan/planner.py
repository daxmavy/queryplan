"""Choosing which models each GPU holds, and in what order, by simulating every candidate against the cost model.

The planner works on query classes (experiment x call site x model x depth), not single queries. A class is treated as
a fluid: its queries become ready as the classes it depends on complete, and a GPU serving its model drains it at the
throughput the cost model predicts for the current concurrency.

A plan gives each GPU slot a list of phases. A phase is a cycle of models (and optionally one experiment it is limited
to). The slot serves its resident model while that model has ready work, switches to the next model in the cycle that
has ready work when it runs dry, and moves to its next phase when the phase has no work left. The optimiser searches
plans with one unrestricted phase per slot; the naive baseline gives each slot whole experiments one after another.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from .costmodel import PREFILL_RATIO, CostModel

MAX_CONCURRENCY = 256
EPS = 1e-4  # fluid amounts below this many queries count as zero
CANDIDATE_LIMIT = 6000


@dataclass
class Slot:
    """A group of GPUs that serves one model at a time."""

    id: str
    gpus: tuple[int, ...]
    hw: str
    mem_gb: float
    host: str = "localhost"

    @property
    def size(self) -> int:
        return len(self.gpus)


@dataclass
class ClassState:
    id: str
    experiment: str
    tier: int
    model: str  # model key
    depth: int
    total: int
    done: float = 0.0
    released: float = 0.0  # queries whose inputs exist (done + in flight + waiting)
    taken: float = 0.0  # queries started (done + in flight)
    parents: tuple[str, ...] = ()
    out_tokens: float = 256.0
    in_tokens: float = 256.0
    memory_gb: float | None = None
    max_tokens: int | None = None

    @property
    def work_tokens(self) -> float:
        """Output-token-equivalents per query."""
        return self.out_tokens + self.in_tokens / PREFILL_RATIO


@dataclass(frozen=True)
class Phase:
    models: tuple[str, ...]
    experiment: str | None = None
    fresh: bool = False  # start the phase with nothing loaded (a separate script or job would)


@dataclass
class Plan:
    phases: dict[str, list[Phase]]  # slot id -> phases
    predicted_tiers: tuple[float, ...] = ()
    predicted_makespan: float = math.inf
    predicted_loads: int = 0
    note: str = ""

    def describe(self, names: dict[str, str] | None = None) -> dict[str, Any]:
        names = names or {}
        return {slot: [{"models": [names.get(m, m) for m in p.models], "experiment": p.experiment} for p in phases]
                for slot, phases in self.phases.items()}


@dataclass
class SimResult:
    tier_finish: tuple[float, ...]
    makespan: float
    loads: int
    timeline: list[tuple[float, str, str]] = field(default_factory=list)  # (time, slot, model loaded)


class Problem:
    """Everything the simulator needs: classes with their current progress, slots with what they hold now."""

    def __init__(self, classes: dict[str, ClassState], slots: list[Slot], cost: CostModel,
                 resident: dict[str, str | None] | None = None, loading_left: dict[str, float] | None = None):
        self.classes = classes
        self.slots = slots
        self.cost = cost
        self.resident = resident or {s.id: None for s in slots}
        self.loading_left = loading_left or {s.id: 0.0 for s in slots}
        self.tiers = sorted({c.tier for c in classes.values()})
        self.models = sorted({c.model for c in classes.values() if c.done < c.total - EPS})
        self.mem = {c.model: c.memory_gb for c in classes.values()}

    def fits(self, model: str, slot: Slot) -> bool:
        need = self.mem.get(model)
        return need is None or need <= slot.mem_gb * slot.size

    def simulate(self, plan: Plan, dt: float | None = None, horizon: float = 7 * 86400,
                 record: bool = False) -> SimResult:
        cls = {k: _Fluid(c) for k, c in self.classes.items()}
        order = sorted(cls.values(), key=lambda f: (f.c.tier, -f.c.depth))
        pools: dict[tuple[str, str | None], list[_Fluid]] = {}
        for f in order:
            pools.setdefault((f.c.model, None), []).append(f)
            pools.setdefault((f.c.model, f.c.experiment), []).append(f)
        with_parents = [f for f in order if f.c.parents]
        for f in order:
            if not f.c.parents:
                f.released = f.c.total
        slots = self.slots
        state = {s.id: _SlotSim(s, self.resident.get(s.id), self.loading_left.get(s.id, 0.0), plan.phases.get(s.id, []))
                 for s in slots}
        speeds = {(m, s.hw, s.size): self.cost.speed(m, s.hw, s.size) for m in self.models for s in slots}
        loads_s = {(m, s.hw, s.size): self.cost.load_seconds(m, s.hw, s.size)[0] for m in self.models for s in slots}
        if dt is None:
            dt = self._default_dt()
        tier_finish = {t: (0.0 if all(f.left <= EPS for f in cls.values() if f.c.tier == t) else None)
                       for t in self.tiers}
        loads = 0
        timeline: list[tuple[float, str, str]] = []
        tier_left = {tier: [f for f in order if f.c.tier == tier] for tier in self.tiers}
        t = 0.0
        idle_ticks = 0
        while t < horizon:
            if all(f.left <= EPS for f in order):
                break
            progress_before = sum(f.done for f in order)
            for f in with_parents:
                f.release(cls)
            # decide what each slot does this tick
            for sid, st in state.items():
                if st.loading > 0:
                    continue
                phase = st.current_phase(cls)
                if phase is None:
                    st.model = None
                    continue
                avail = {m: sum(f.available for f in pools.get((m, phase.experiment), ())) for m in phase.models}
                if st.model in phase.models and avail[st.model] > EPS:
                    continue
                remaining = {m: sum(f.left for f in pools.get((m, phase.experiment), ())) for m in phase.models}
                nxt = st.next_model(phase, avail, remaining)
                if nxt is not None and nxt != st.model:
                    st.model = nxt
                    st.loading = loads_s[(nxt, st.slot.hw, st.slot.size)]
                    loads += 1
                    if record:
                        timeline.append((t, sid, nxt))
            # process work: slots serving the same pool share it
            serving: dict[tuple[str, str | None], list[_SlotSim]] = {}
            for st in state.values():
                if st.loading > 0 or st.model is None:
                    continue
                phase = st.phases[st.phase_idx] if st.phase_idx < len(st.phases) else None
                serving.setdefault((st.model, phase.experiment if phase else None), []).append(st)
            for (model, exp), group in serving.items():
                pool = pools.get((model, exp), [])
                avail_q = sum(f.available for f in pool)
                if avail_q <= EPS:
                    continue
                for st in group:
                    share = avail_q / len(group)
                    conc = min(share, MAX_CONCURRENCY)
                    tokens = speeds[(model, st.slot.hw, st.slot.size)].rate(conc) * dt
                    for f in pool:
                        if tokens <= 0:
                            break
                        can = min(f.available, share)
                        if can <= 0:
                            continue
                        q = min(can, tokens / f.c.work_tokens)
                        f.taken += q
                        f.done += q
                        if f.c.total - f.done <= EPS:
                            f.done = f.taken = float(f.c.total)
                        tokens -= q * f.c.work_tokens
                        share -= q
            loading_any = any(st.loading > 0 for st in state.values())
            for st in state.values():
                if st.loading > 0:
                    st.loading = max(0.0, st.loading - dt)
            t += dt
            if not loading_any and sum(f.done for f in order) <= progress_before + EPS:
                idle_ticks += 1
                if idle_ticks > 3:  # nothing can make progress: this plan never finishes
                    break
            else:
                idle_ticks = 0
            for tier in self.tiers:
                if tier_finish[tier] is None and all(f.left <= EPS for f in tier_left[tier]):
                    tier_finish[tier] = t
        finish = tuple(tier_finish[t] if tier_finish[t] is not None else math.inf for t in self.tiers)
        return SimResult(tier_finish=finish, makespan=max(finish, default=0.0), loads=loads, timeline=timeline)

    def _default_dt(self) -> float:
        work = 0.0
        for c in self.classes.values():
            if c.total - c.done <= 0:
                continue
            slot = self.slots[0]
            work += (c.total - c.done) * c.work_tokens / self.cost.speed(c.model, slot.hw, slot.size).rate(MAX_CONCURRENCY)
        estimate = work / max(len(self.slots), 1) + 2 * max(
            (self.cost.load_seconds(m, self.slots[0].hw, self.slots[0].size)[0] for m in self.models), default=0)
        return max(estimate / 400, 0.25)


@dataclass
class _Fluid:
    c: ClassState
    done: float = 0.0
    taken: float = 0.0
    released: float = 0.0

    def __post_init__(self) -> None:
        self.done = self.c.done
        self.taken = self.c.taken
        self.released = max(self.c.released, self.c.taken)

    @property
    def left(self) -> float:
        return self.c.total - self.done

    @property
    def available(self) -> float:
        if self.c.total - self.done <= EPS:
            return 0.0
        return max(self.released - self.taken, 0.0)

    def release(self, all_cls: dict[str, "_Fluid"]) -> None:
        if not self.c.parents:
            self.released = self.c.total
            return
        frac = min(1.0 if all_cls[p].left <= EPS else all_cls[p].done / all_cls[p].c.total for p in self.c.parents)
        self.released = min(self.c.total, max(self.released, frac * self.c.total))


@dataclass
class _SlotSim:
    slot: Slot
    model: str | None
    loading: float
    phases: list[Phase]
    phase_idx: int = 0

    entered: int = -1

    def current_phase(self, cls: dict[str, _Fluid]) -> Phase | None:
        while self.phase_idx < len(self.phases):
            phase = self.phases[self.phase_idx]
            if any(f.left > EPS and f.c.model in phase.models and (phase.experiment is None or f.c.experiment == phase.experiment)
                   for f in cls.values()):
                if phase.fresh and self.entered != self.phase_idx:
                    self.model = None
                self.entered = self.phase_idx
                return phase
            self.phase_idx += 1
        return None

    def next_model(self, phase: Phase, avail: dict[str, float], remaining: dict[str, float]) -> str | None:
        cycle = list(phase.models)
        start = (cycle.index(self.model) + 1) if self.model in cycle else 0
        ordered = cycle[start:] + cycle[:start]
        for m in ordered:
            if avail[m] > EPS:
                return m
        if self.model in cycle and remaining.get(self.model, 0) > EPS:
            return self.model  # wait for more of the resident model's work
        for m in ordered:
            if remaining[m] > EPS:
                return m  # load what comes next even though its inputs are not ready yet
        return None


def _available(order: list[_Fluid], model: str, exp: str | None) -> float:
    return sum(f.available for f in order if f.c.model == model and (exp is None or f.c.experiment == exp))


def _remaining(order: list[_Fluid], model: str, exp: str | None) -> float:
    return sum(f.left for f in order if f.c.model == model and (exp is None or f.c.experiment == exp))


# ---- search -------------------------------------------------------------------------------------------------------

def _sequences(models: list[str]) -> list[tuple[str, ...]]:
    seqs: list[tuple[str, ...]] = []
    for k in range(1, len(models) + 1):
        seqs.extend(itertools.permutations(models, k))
    return seqs


def _key(result: SimResult) -> tuple[float, ...]:
    # plans that never finish lose; then tiers compare strictly in order, rounded to a second so float noise cannot
    # decide between equal plans; then fewer loads
    unfinished = sum(1 for x in result.tier_finish if math.isinf(x))
    return (unfinished,) + tuple(round(x, 0) for x in result.tier_finish) + (result.loads,)


def optimise(problem: Problem, limit: int = CANDIDATE_LIMIT) -> Plan:
    """Search per-slot model cycles exhaustively when the space is small enough, by local search otherwise."""
    models = problem.models
    if not models:
        return Plan(phases={s.id: [] for s in problem.slots}, predicted_tiers=tuple(0.0 for _ in problem.tiers),
                    predicted_makespan=0.0, note="nothing left to run")
    per_slot = {s.id: [()] + [seq for seq in _sequences(models) if all(problem.fits(m, s) for m in seq)]
                for s in problem.slots}  # () leaves the slot idle
    groups: dict[tuple, list[Slot]] = {}
    for s in problem.slots:  # slots that look the same to the simulator are interchangeable
        groups.setdefault((s.hw, s.size, s.mem_gb, problem.resident.get(s.id), problem.loading_left.get(s.id, 0) > 0),
                          []).append(s)
    group_choices = [list(itertools.combinations_with_replacement(range(len(per_slot[g[0].id])), len(g)))
                     for g in groups.values()]
    total = math.prod(len(c) for c in group_choices)
    candidates: Iterable[Plan]
    if total <= limit:
        candidates = (_plan_from(problem, groups, per_slot, combo) for combo in itertools.product(*group_choices))
        how = f"exhaustive over {total} plans"
    else:
        candidates = _local_search_candidates(problem, per_slot, limit)
        how = f"local search ({total} plans in the full space)"
    coarse_dt = problem._default_dt() * 8  # screen every candidate cheaply, then refine the leaders
    scored: list[tuple[tuple[float, ...], Plan]] = []
    for plan in candidates:
        covered = {m for phases in plan.phases.values() for p in phases for m in p.models}
        if not set(models) <= covered:
            continue
        scored.append((_key(problem.simulate(plan, dt=coarse_dt)), plan))
    scored.sort(key=lambda kv: kv[0])
    best: tuple[tuple[float, ...], Plan, SimResult] | None = None
    for _, plan in scored[:12]:  # refine the leaders with a finer time step
        result = problem.simulate(plan)
        if best is None or _key(result) < best[0]:
            best = (_key(result), plan, result)
    if best is None:
        raise RuntimeError("no plan can run every model on the given GPUs (check memory_gb against the GPUs)")
    _, plan, result = best
    plan.predicted_tiers = result.tier_finish
    plan.predicted_makespan = result.makespan
    plan.predicted_loads = result.loads
    plan.note = how
    return plan


def _plan_from(problem: Problem, groups: dict[tuple, list[Slot]], per_slot: dict[str, list[tuple[str, ...]]],
               combo: tuple[tuple[int, ...], ...]) -> Plan:
    phases: dict[str, list[Phase]] = {}
    for (key, slots), picks in zip(groups.items(), combo):
        for slot, idx in zip(slots, picks):
            seq = per_slot[slot.id][idx]
            phases[slot.id] = [Phase(models=seq)] if seq else []
    return Plan(phases=phases)


def _local_search_candidates(problem: Problem, per_slot: dict[str, list[tuple[str, ...]]], limit: int) -> Iterable[Plan]:
    """Start from every slot cycling all models, then try changing one slot at a time."""
    base = {s.id: [seq for seq in per_slot[s.id] if len(seq) == len(problem.models)][:1] or per_slot[s.id][-1:]
            for s in problem.slots}
    current = {sid: seqs[0] for sid, seqs in base.items()}
    yield Plan(phases={sid: [Phase(seq)] if seq else [] for sid, seq in current.items()})
    produced = 1
    for s in problem.slots:
        for seq in per_slot[s.id]:
            if produced >= limit:
                return
            trial = dict(current)
            trial[s.id] = seq
            produced += 1
            yield Plan(phases={sid: [Phase(sq)] if sq else [] for sid, sq in trial.items()})


def naive_plan(problem: Problem, experiment_order: list[str]) -> Plan:
    """What a typical script does: each GPU takes whole experiments in turn, loading each model as the experiment
    reaches it (one job per experiment, experiments spread round-robin over the GPUs)."""
    phases: dict[str, list[Phase]] = {s.id: [] for s in problem.slots}
    for i, exp in enumerate(experiment_order):
        slot = problem.slots[i % len(problem.slots)]
        models_in_order = [c.model for c in sorted(problem.classes.values(), key=lambda c: c.depth)
                           if c.experiment == exp]
        cycle = tuple(dict.fromkeys(models_in_order))
        if cycle:
            phases[slot.id].append(Phase(models=cycle, experiment=exp, fresh=True))
    plan = Plan(phases=phases, note="naive: experiments one after another on each GPU")
    result = problem.simulate(plan)
    plan.predicted_tiers, plan.predicted_makespan, plan.predicted_loads = result.tier_finish, result.makespan, result.loads
    return plan
