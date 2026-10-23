"""Splitting one graph across sites that cannot share work at run time (no shared filesystem, separate schedulers).

`split()` assigns every node to a site so that each query goes to a site that can host its model, data only flows
forward from earlier sites to later ones, and the busiest site has as little work as possible. The result is plain
data (JSON): which nodes each site runs and which values cross from one site to the next. Each site then runs the same
script with `qp.run(..., site_plan=plan, site="name", imports=[...])`, after copying in the values it needs.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .graph import Graph, LocalModel, Node


@dataclass
class SitePlan:
    order: list[str]  # sites in the order their stages run
    nodes: dict[str, list[str]]  # site -> node ids it runs
    boundary: list[dict[str, Any]] = field(default_factory=list)  # values that cross sites
    work: dict[str, float] = field(default_factory=dict)  # estimated output tokens per site

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "SitePlan":
        return cls(**json.loads(Path(path).read_text()))


def _components(graph: Graph) -> list[list[Node]]:
    parent = {n: n for n in graph.nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for node in graph.nodes.values():
        for dep in node.deps:
            parent[find(dep)] = find(node.id)
    groups: dict[str, list[Node]] = {}
    for node in graph.nodes.values():  # creation order is a topological order
        groups.setdefault(find(node.id), []).append(node)
    return list(groups.values())


def split(graph: Graph, sites: dict[str, Iterable[LocalModel | str]], out_tokens: float = 256.0) -> SitePlan:
    """Assign nodes to sites. `sites` maps a site name to the models it can host (handles or model names).

    Independent pieces of work (connected components, usually one experiment unit each) go whole to one site where
    possible, balanced by size; a piece needing models no single site has is chained across sites in a global order.
    """
    hostable = {s: {m.model if isinstance(m, LocalModel) else m for m in models} for s, models in sites.items()}
    pieces = sorted(_components(graph), key=lambda p: -sum(1 for n in p if n.kind == "query"))
    best: tuple[tuple[float, int], SitePlan] | None = None
    for order in itertools.permutations(hostable):
        rank = {s: i for i, s in enumerate(order)}
        site_of: dict[str, str] = {}
        work = {s: 0.0 for s in order}
        feasible = True
        for piece in pieces:
            models = {n.model.model for n in piece if n.kind == "query"}
            whole = [s for s in order if models <= hostable[s]]
            size = out_tokens * sum(1 for n in piece if n.kind == "query")
            if whole:
                site = min(whole, key=lambda s: work[s])
                for node in piece:
                    site_of[node.id] = site
                work[site] += size
                continue
            for node in piece:  # chain: each node at the earliest site, not before its inputs, that can run it
                floor = max((rank[site_of[d]] for d in node.deps), default=0)
                if node.kind == "step":
                    site_of[node.id] = order[floor]
                    continue
                options = [s for s in order[floor:] if node.model.model in hostable[s]]
                if not options:
                    feasible = False
                    break
                site_of[node.id] = options[0]
                work[options[0]] += out_tokens
            if not feasible:
                break
        if not feasible:
            continue
        crossing = [(d, n.id) for n in graph.nodes.values() for d in n.deps if site_of[d] != site_of[n.id]]
        key = (max(work.values()), len(crossing))
        if best is None or key < best[0]:
            plan = SitePlan(order=list(order),
                            nodes={s: [n for n in graph.nodes if site_of[n] == s] for s in order},
                            boundary=[{"value": d, "from": site_of[d], "to": site_of[n]} for d, n in crossing],
                            work=work)
            best = (key, plan)
    if best is None:
        raise ValueError("no split exists: some chain needs a model at an earlier site after a later one")
    return best[1]
