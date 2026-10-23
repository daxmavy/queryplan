# queryplan

Run a fixed set of LLM queries on your own GPUs in the order and placement that finishes soonest.

You write experiments as ordinary Python that calls `llm()`. queryplan traces that code into a graph of queries and
the computations between them, learns how long each kind of query and each model load takes, and decides which model
each GPU holds and when. It never changes *what* runs; it only changes the order and the placement.

```python
import queryplan as qp

writer = qp.local("Qwen/Qwen3.8-27B", temperature=0.8, max_tokens=400)
judge = qp.local("openai/gpt-oss-120b", temperature=0.1, reasoning_effort="low")

@qp.step                                    # reads an output, so it runs once the output exists
def score(verdict):
    return int(re.search(r"Score: (\d)", verdict).group(1))

def condition(style):
    @qp.experiment(f"ads-{style}")
    def run():
        for product in PRODUCTS:
            advert = qp.llm(writer, f"Write an advert for a {product}. {style}")
            verdict = qp.llm(judge, f"Rate this advert 1-9, ending 'Score: N'.\n{advert}",
                             validate=lambda t: "Score:" in t, attempts=3)
            yield {"advert": advert, "score": score(verdict)}
    return run

result = qp.run([condition("Be brief."), condition("Be warm."), condition("Be urgent.")],
                slots=qp.visible_gpus(hw="H100-80GB", mem_gb=80), store="runs/ads")
```

## The problem

LLM experiments usually run as scripts, one experiment at a time: load a model, send its queries, load the judge,
judge the outputs, move on. On a machine with a few GPUs and several experiments that share models, this wastes a
lot of wall-clock time:

- **The same model is loaded again and again.** Loading a 30–120B model takes one to several minutes. Ten experiments
  that each use the same generator and the same judge pay for twenty loads when two would do.
- **Work that could overlap waits.** A judge could score adverts on one GPU while another GPU is still writing them;
  in a script it waits for the whole batch.
- **GPUs sit idle or are badly shared.** Splitting experiments over GPUs by hand balances them by count, not by how
  long each takes, and the long tail of one experiment keeps a GPU busy while the others are idle.
- **Priorities are manual.** "Get experiment 1 done first, then the rest" means editing scripts.

Doing better by hand means reasoning about load times, throughput and dependencies that you only learn by running
the work. queryplan learns them and does the reasoning.

## How it works

1. **Trace.** Experiments are generator functions. While they are traced, `llm()` returns a placeholder instead of
   text, and functions marked `@step` are deferred, so the code builds a graph of queries and steps without running
   any of them. The graph's structure must not depend on outputs; a validator with retries inside one query is
   allowed.
2. **Model the costs.** A small learned cost model predicts each call site's output length, each model's throughput
   on each GPU type (as a function of how many requests run at once), and each model's load time. With no data it
   runs a short pilot first: a few real units from each experiment, which are part of the declared work, not extra.
3. **Plan.** Queries are grouped into classes (experiment × call site × model × depth). Each GPU slot is given a
   cycle of models; a simulator drains the classes as fluids under the cost model and every candidate plan is
   simulated. The best plan by priority tier, then by finish time, wins. On a few GPUs with a few models the search
   is exhaustive.
4. **Run and re-plan.** A driver process starts `vllm serve` on each slot as the plan says, sends queries in priority
   order, and records every request and load. Periodically it refits the cost model and re-plans the whole run,
   switching only when the new plan is predicted to finish earlier by more than a margin.

Results are written to a store directory as they arrive, so a run cut short (say, by a job's time limit) resumes
where it stopped when the same script is run again.

### What it never does

- Run a query twice, skip one, or merge identical queries: every declared query runs exactly once, as declared.
- Change anything that affects outputs: the model, its sampling settings, chat template or reasoning setting are fixed
  by the declaration. The planner only changes speed-only choices (which GPU, when, how many at once), which can make
  tiny numerical differences in outputs, as any change of batch does in vLLM.
- Discover or claim GPUs on its own. You say which GPUs it may use.

## Writing experiments

| You write | What happens while tracing |
|---|---|
| `qp.local(model, temperature=..., max_tokens=..., chat_template_kwargs=..., serve_args=[...])` | declares a locally served model; everything here is fixed |
| `qp.llm(model, prompt, system=None, validate=None, attempts=1)` | adds a query; returns a placeholder for its reply |
| f-strings, `+`, `"".join(...)` with placeholders | work; the placeholder is filled in later |
| `len(x)`, `x.split()`, `x in y`, `if x:`, `re.search(..., x)` on a placeholder | raise `OutputReadError`: move that code into a `@step` |
| `@qp.step` (or `@qp.step(outputs=n)`) | the call is deferred until its inputs exist; returns a placeholder (or n of them) |
| `@qp.experiment(name)` on a generator | each `yield` is one unit; `run()` returns the yielded values with outputs filled in |

`qp.run(tiers, slots=..., store=...)` takes a list of tiers, highest priority first: `[e1, e2, e3]` runs them in
strict order of priority, `[e1, [e2, e3]]` finishes e1 first and then e2 and e3 together as early as possible.

Resources are explicit: `qp.gpus([0, 1, 2], hw="H100-80GB", mem_gb=80)` for a fixed machine, or
`qp.visible_gpus(...)` for whatever `CUDA_VISIBLE_DEVICES` allows (inside a Slurm job, the job's GPUs). `per_slot=2`
groups GPUs in pairs for models that need tensor parallelism.

The cost model is an object you can keep: `save_cost_model="cost.json"` writes its parameters together with every
observation it was fitted on; `qp.CostModel.load(path)` and `qp.CostModel.merge(a, b)` reuse it in later runs.

## Results

See [docs/results.md](docs/results.md).

## Status and limits

This is a first version.

- Local models served by vLLM only. API models and other model types are planned, not built.
- One model per slot at a time; a slot's size (GPUs per model) is fixed per run, not chosen by the planner.
- Models are stopped and started rather than put to sleep.
- Within a Slurm job it uses the job's GPUs; it does not submit extra jobs.
- The whole graph is held in memory while tracing (fine for tens of thousands of queries).
- The planner's search is exhaustive only over its plan family (a cycle of models per slot, drained in priority
  order); it is optimal against the cost model within that family, not over every possible schedule.

## Install

```bash
pip install git+https://github.com/daxmavy/queryplan
```

vLLM is not a dependency: install it where your GPUs are. queryplan runs `vllm serve` from the same environment, or
from `QUERYPLAN_VLLM`.
