"""A generate-then-judge pipeline: several experiments share two generator models and one judge model.

Each experiment asks one generator to write a short advert for every product in a list, in one style, and asks the
judge to rate how persuasive each advert is. This is the shape of many LLM studies (conditions x items, an LLM judge),
and it is where running experiments one at a time wastes time: every experiment reloads the same models.

    python examples/judge_pipeline.py --policy plan  --store runs/plan  --gpus 0,1,2,3 --hw H100-80GB --mem 80
    python examples/judge_pipeline.py --policy naive --store runs/naive --gpus 0,1,2,3 --hw H100-80GB --mem 80
    python examples/judge_pipeline.py --fake --units 30 --store /tmp/qp-smoke     # no GPUs: simulated servers
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import queryplan as qp

QWEN = qp.local("Qwen/Qwen3.8-27B", temperature=0.8, top_p=0.95, max_tokens=400, seed=0,
                chat_template_kwargs={"enable_thinking": False},
                serve_args=["--max-model-len", "4096", "--max-num-seqs", "192"], label="qwen")
GEMMA = qp.local("google/gemma-4-26B-A4B-it", temperature=0.8, top_p=0.95, max_tokens=400, seed=0,
                 chat_template_kwargs={"enable_thinking": False},
                 serve_args=["--max-model-len", "4096", "--max-num-seqs", "192"], label="gemma")
JUDGE = qp.local("google/gemma-4-31B-it", temperature=0.1, max_tokens=300, seed=0,
                 chat_template_kwargs={"enable_thinking": False},
                 serve_args=["--max-model-len", "4096", "--max-num-seqs", "192"], label="gemma-judge")

STYLES = {
    "plain": "Write in a plain, factual style.",
    "warm": "Write in a warm, friendly style.",
    "urgent": "Write with a sense of urgency.",
}
ITEMS = ["kettle", "bicycle helmet", "notebook", "rain jacket", "desk lamp", "water bottle", "backpack", "blender",
         "running shoes", "headphones", "phone case", "frying pan", "yoga mat", "umbrella", "board game", "toaster",
         "sunscreen", "office chair", "travel mug", "garden hose"]
QUALITIES = ["budget", "premium", "eco-friendly", "compact", "family-size", "lightweight", "durable", "smart",
             "handmade", "classic"]


def products(n: int) -> list[str]:
    combos = [f"{q} {i}" for q in QUALITIES for i in ITEMS]
    return combos[:n]


def scored(text: str) -> bool:
    return re.search(r"Score:\s*[1-9]", text) is not None


@qp.step
def score(verdict: str) -> int | None:
    match = re.search(r"Score:\s*([1-9])", verdict)
    return int(match.group(1)) if match else None


def make_experiment(model: qp.LocalModel, style: str, n: int) -> qp.Experiment:
    @qp.experiment(f"{model.name}-{style}")
    def run():
        for product in products(n):
            advert = qp.llm(model, f"Write a social media advert of at most 80 words for a {product}. "
                                   f"{STYLES[style]} Reply with the advert only.")
            verdict = qp.llm(JUDGE, "Rate how persuasive this advert is, from 1 (not at all) to 9 (extremely). "
                                    f"End your reply with 'Score: N'.\n\nAdvert:\n{advert}",
                             validate=scored, attempts=3)
            yield {"product": product, "advert": advert, "score": score(verdict)}

    return run


def experiments(n: int, count: int) -> list[qp.Experiment]:
    pairs = [(m, s) for s in STYLES for m in (QWEN, GEMMA)]
    return [make_experiment(m, s, n) for m, s in pairs[:count]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", choices=["plan", "naive"], default="plan")
    parser.add_argument("--store", required=True)
    parser.add_argument("--units", type=int, default=500, help="products per experiment")
    parser.add_argument("--experiments", type=int, default=6)
    parser.add_argument("--gpus", default=None, help="GPU indices, e.g. 0,1,2,3 (default: CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--hw", default="GPU")
    parser.add_argument("--mem", type=float, default=80)
    parser.add_argument("--cost-model", default=None, help="load (and save) a cost model at this path")
    parser.add_argument("--first", default=None, help="experiment to put in its own top tier")
    parser.add_argument("--replan", type=float, default=120)
    parser.add_argument("--fake", action="store_true", help="simulated servers instead of vLLM")
    parser.add_argument("--local-snapshots", default=None,
                        help="serve models from this Hugging Face hub cache directory, with no hub lookups")
    args = parser.parse_args()

    exps = experiments(args.units, args.experiments)
    tiers: list = exps
    if args.first:
        first = [e for e in exps if e.name == args.first]
        if not first:
            parser.error(f"--first {args.first} is not one of {[e.name for e in exps]}")
        tiers = [first, [e for e in exps if e.name != args.first]]
    else:
        tiers = [exps]
    slots = (qp.gpus([int(g) for g in args.gpus.split(",")], hw=args.hw, mem_gb=args.mem) if args.gpus
             else qp.visible_gpus(hw=args.hw, mem_gb=args.mem))
    backend = "vllm"
    if args.fake:
        backend = qp.FakeBackend(load_s={QWEN.model: 120, GEMMA.model: 100, JUDGE.model: 140},
                                 theta={QWEN.model: 2500, GEMMA.model: 4000, JUDGE.model: 3500},
                                 out_tokens={QWEN.model: 120, GEMMA.model: 120, JUDGE.model: 250}, speedup=50,
                                 reply=lambda model, text, rng: f"{model.name} says Score: {rng.randint(1, 9)}")
    started = time.time()
    result = qp.run(tiers, slots=slots, store=args.store, cost_model=args.cost_model, backend=backend,
                    policy=args.policy, save_cost_model=args.cost_model, replan_seconds=args.replan,
                    vllm_options={"local_snapshots": args.local_snapshots} if args.local_snapshots else None)
    report = result.report.as_dict()
    report.update({"started": started, "units": args.units, "experiments": [e.name for e in exps],
                   "first": args.first, "fake": args.fake, "hw": args.hw,
                   "scores": {e: sum(1 for u in v if u["score"] is not None) for e, v in result.values.items()}})
    Path(args.store).mkdir(parents=True, exist_ok=True)
    (Path(args.store) / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("policy", "wall_seconds", "loads", "load_seconds", "experiment_seconds")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
