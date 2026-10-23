"""Predicted wall time, naive vs planned, across workload shapes, using the planner's own simulator (no GPUs).

Each experiment generates N items with one of two generator models and has a judge model score each item. The cost
model is either a saved one (measured on real hardware) or the parameters given on the command line.

    python examples/simulate_scenarios.py --cost-model runs/cost.json --hw GH200-96GB --out scenarios.md
"""

from __future__ import annotations

import argparse
import itertools

import queryplan as qp
from queryplan.planner import ClassState, Problem, naive_plan, optimise

GEN = ["gen-a", "gen-b"]
JUDGE = "judge"


def synthetic_cost(load_s: float, theta: float, hw: str) -> qp.CostModel:
    obs = []
    for m in GEN + [JUDGE]:
        obs.append({"type": "load", "model": m, "hw": hw, "gpus": 1, "seconds": load_s})
        for b in (8, 32, 128, 192):
            for _ in range(4):
                obs.append({"type": "request", "model": m, "hw": hw, "gpus": 1, "kind": f"x|{m}", "n_in": 200,
                            "n_out": 150, "seconds": 150 * (b + 8) / theta, "concurrency": b})
    return qp.CostModel(obs)


def problem(n_exp: int, n_items: int, n_gpus: int, cost: qp.CostModel, hw: str, models: dict[str, str],
            gen_out: float, judge_out: float) -> Problem:
    classes = {}
    for e in range(n_exp):
        gen = models[GEN[e % 2]]
        classes[f"e{e}|gen"] = ClassState(id=f"e{e}|gen", experiment=f"e{e}", tier=0, model=gen, depth=1,
                                          total=n_items, out_tokens=gen_out, in_tokens=60)
        classes[f"e{e}|judge"] = ClassState(id=f"e{e}|judge", experiment=f"e{e}", tier=0, model=models[JUDGE],
                                            depth=2, total=n_items, out_tokens=judge_out,
                                            in_tokens=gen_out + 60, parents=(f"e{e}|gen",))
    return Problem(classes, qp.gpus(list(range(n_gpus)), hw=hw, mem_gb=96), cost)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cost-model", default=None)
    parser.add_argument("--hw", default="GPU")
    parser.add_argument("--models", nargs=3, default=GEN + [JUDGE], metavar=("GEN_A", "GEN_B", "JUDGE"),
                        help="model keys in the cost model for the two generators and the judge")
    parser.add_argument("--load", type=float, default=120, help="load seconds (synthetic cost model)")
    parser.add_argument("--theta", type=float, default=3000, help="peak tokens/s (synthetic cost model)")
    parser.add_argument("--gen-out", type=float, default=150)
    parser.add_argument("--judge-out", type=float, default=300)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cost = qp.CostModel.load(args.cost_model) if args.cost_model else synthetic_cost(args.load, args.theta, args.hw)
    models = dict(zip(GEN + [JUDGE], args.models))
    rows = ["| experiments | items each | GPUs | naive (min) | planned (min) | speed-up | loads naive / planned |",
            "|---:|---:|---:|---:|---:|---:|---|"]
    for n_exp, n_items, n_gpus in itertools.product([2, 6, 12], [100, 1000], [1, 2, 4]):
        p = problem(n_exp, n_items, n_gpus, cost, args.hw, models, args.gen_out, args.judge_out)
        naive = naive_plan(p, [f"e{e}" for e in range(n_exp)])
        best = optimise(p)
        rows.append(f"| {n_exp} | {n_items} | {n_gpus} | {naive.predicted_makespan / 60:.1f} | "
                    f"{best.predicted_makespan / 60:.1f} | {naive.predicted_makespan / best.predicted_makespan:.2f}x | "
                    f"{naive.predicted_loads} / {best.predicted_loads} |")
    text = "\n".join(rows)
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
