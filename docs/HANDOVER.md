# Handover brief

Written 2026-10-25 for the next agent taking over this repository. It covers the goal, what exists, what has been
verified on real GPUs, what is known to be broken, and what is left, in the order to do it.

## 1. The goal

The operator asked for three things:

1. **A package** (this repo, `queryplan`): a standalone pip package that takes a fixed set of LLM queries written as
   ordinary Python, compiles them into a computation graph, and decides which model each GPU holds and when, to finish
   declared results as early as possible. **Done as v0.1**, with known bugs (section 5).
2. **A simple experiment demonstrating its utility on two sites**: a shared GPU server (site A) and a Slurm GPU
   cluster (site B), described in section 4. **Not done**: smoke tests ran, no timed comparison has finished.
3. **Public documentation** in this repo: the intended utility, the gap it fills, and experimental results.
   **Partly done**: `README.md` and `docs/design.md` exist. `docs/results.md` is still to be written.

### Hard constraint for everything committed here

**Never name the two sites anywhere in this repository.** That covers code, docs, commit messages and PR text. They
are examples of a fixed GPU server and a Slurm cluster, and the package must work in general. In results, describe
each site only by its properties: GPU type and count, scheduler, filesystem, how models are loaded. Before every push,
grep the tracked files for the site names, the cluster project id, the shared-disk path and the server host names
(the operator's skills list them) and remove any hits. Site-specific settings stay in untracked files (`site/` is in
`.gitignore`).

## 2. Requirements agreed with the operator

These were settled in a long design discussion; `docs/design.md` has the maths.

- **Interface:** experiments are ordinary Python generators. `llm()` returns a placeholder (a `str` subclass).
  f-strings, `+` and `join` work on placeholders; reading content raises unless the code is in a `@step`.
  Regex is guarded. This is "option B" and was explicitly chosen.
- **The graph never depends on outputs.** The one exception is a validator with retries inside a query node.
- **The tool only reorders and places queries.**
  - Every declared query runs exactly once, as declared.
  - It never deduplicates queries and never caches across runs.
  - It has no spending objective.
  - A query's identity is its position in the graph, used only for resume.
- **Fixed by the declaration:**
  - whether a query is served locally or through an API, and the API provider;
  - everything that changes outputs: weights, quantisation, sampling, chat template, reasoning setting, max tokens,
    seeds.
- **The planner chooses only speed settings:** GPU count and type, tensor parallelism versus replicas, batch size,
  placement and order. Small output differences caused by these choices are accepted.
- **Objective:** ordered tiers, each minimising the finish time of its last item.
  - `[[e1], [e2], [e3]]` means strict order.
  - `[[e1], [rest]]` means e1 first, then everything else as early as possible.
- **Optimal means optimal against the current cost model.** The problem is solved at the level of query classes.
- **Cost model:**
  - It learns output length per call site, throughput per model and hardware (including concurrency limited by KV
    cache), load times, and API latency.
  - It explores with a small real pilot first, then updates during the run.
  - A `CostModel` object holds its form, parameters and training data, and can be saved, loaded and merged.
- **Re-planning:** the whole plan is re-solved periodically. It switches only when the gain beats the reload cost.
- **Resources are an explicit input**, either a fixed GPU list or a Slurm allocation. Nothing backs off automatically.
- **Sites:** a graph can be split into per-site subgraphs, saved as data. There is no cross-site balancing at run
  time.
- **Packaging:** standalone pip package, no service. A driver process may run on site for one graph; on site B it
  runs inside the Slurm job.
- **Built in v0.1:** local vLLM models only.
- **Not built**, and listed as limits in the README: API nodes, other model types, vLLM sleep mode, choosing GPUs per
  model, submitting extra Slurm jobs, streaming the trace instead of holding it in memory.

## 3. What exists

| Path | What it is |
|---|---|
| `src/queryplan/refs.py`, `graph.py` | placeholders, tracing, experiments, steps |
| `costmodel.py` | learned output lengths, throughput curve, load times; save/load/merge |
| `planner.py` | fluid simulator, exhaustive search over per-slot model cycles, naive baseline |
| `executor.py`, `backends.py`, `store.py`, `resources.py`, `__init__.py` (`qp.run`) | driver, vLLM and fake backends, store, resume |
| `split.py` | split a graph across sites; run one site's share with imports |
| `examples/judge_pipeline.py` | the demo: N experiments × M products; two generator models write adverts, a judge scores them |
| `examples/slurm_demo.sh` | runs a sequence of steps (smoke, naive, plan, plan-warm, first) in one Slurm job, or anywhere `CUDA_VISIBLE_DEVICES` is set |
| `examples/summarise_runs.py` | markdown table and GPU timeline PNG from finished runs |
| `examples/simulate_scenarios.py` | predicted naive vs planned across workload shapes, from the planner's simulator |
| `examples/site.env.example` | template for the untracked site environment file |
| `tests/` | 46 tests (`fake_vllm.py` stands in for `vllm serve`); `uv venv && uv pip install -e '.[dev]' pytest-timeout && .venv/bin/pytest -q --timeout 120` |

All work went through PRs #1–#9, merged to `main`. The repo is in self-merge mode: branch, open a PR, merge it
yourself, never push `main`. The operator has not yet applied branch rulesets to this repo.

**Demo models**, cached on both sites:

| Role | Model |
|---|---|
| Generator | `Qwen/Qwen3.8-27B`, thinking off |
| Generator | `google/gemma-4-26B-A4B-it`, thinking off |
| Judge | `google/gemma-4-31B-it`, thinking off, temperature 0.1 |

All three use `--max-model-len 4096 --max-num-seqs 192`. The judge was gpt-oss-120b until site A turned out not to
have its weights.

## 4. The two sites, and what we learned on each

You will recognise them from the operator's skills: one covers the shared GPU servers, the other the GPU cluster.
The operator's CLAUDE.md says which skills to load before touching the cluster.

### Site A: shared GPU servers

- **Hardware:** two machines that share one NFS workspace.
  - **Machine 1:** 4 × H100 80 GB. Exclusive GPU rule: never touch a GPU that has any process on it, and never take
    all 4. It was fully occupied by two other users all afternoon on 2026-10-23.
  - **Machine 2:** 2 × A100 80 GB plus 2 × L40S 46 GB. GPU 0 (A100) was usually free.
- **No scheduler:** jobs run directly, through the operator's GPU-server tool (tmux), never through its Slurm.
- **Access:** only over the operator's VPN from the laptop. The VPN dropped once, which made the servers unreachable.
  A jump host also works if the operator has put the key on it.
- **Shared Hugging Face cache problems:**
  - It is incomplete: some model folders have no weights, and snapshots lack README or licence files.
  - It is partly not writable.
  - Online, vLLM tries to fetch missing files and fails on lock permissions. Offline (`HF_HUB_OFFLINE=1`) is worse:
    Hugging Face refuses incomplete snapshots.
  - **Fix:** pass `--local-snapshots <hub dir>`. The backend then serves each model straight from its complete
    snapshot folder (checked against `model.safetensors.index.json`) under its usual name.
- **The shared disk is about 99% full.** Do not download models.
- **`vllm` script:** in the shared conda env it has a shebang that does not exist on machine 2. The backend now always
  runs `python -m vllm.entrypoints.cli.main`.
- **Load times** (process start to first completion):

  | Load | Model | Seconds |
  |---|---|---:|
  | Cold, over NFS | Qwen3.8-27B | 1060 |
  | Cold, over NFS | Gemma-4-26B-A4B | 1179 |
  | Cold, over NFS | Gemma-4-31B | 1561 |
  | Warm, from page cache | Qwen | 185 |
  | Warm, from page cache | Gemma-4-26B-A4B | 201 |

  The page cache is about 430 GB. Run a warm-up before timed runs so both policies see warm caches, and say so in the
  results.
- **Smoke test passed on the A100**, 2026-10-23: `--units 10 --experiments 2`, all three models on one GPU,
  3 loads, finished.

### Site B: Slurm cluster

- **Hardware:** GH200 nodes, 4 × 96 GB GPUs each, aarch64, with a Lustre scratch filesystem.
- **Limits:**
  - The project may use at most 32 GPUs at once, shared with colleagues whose jobs you cannot see.
  - Walltime is at most 24 h.
- **Login:** a 12-hour SSH certificate that the operator approves in a browser. Show them the link in-session.
  Jobs keep running after it expires, but you cannot check them.
- **Scheduling:** our fair-share priority was 1, so jobs only start through backfill. Use `--time-min` and keep jobs
  small.
  - On 2026-10-23 the queue was full (919 nodes allocated, 3 idle).
  - The access host briefly refused connections once. Treat a failed connection as "unknown", never as "job
    finished".
  - Poll no faster than every 2 minutes.
- **Environment:**
  - Reuse the existing aarch64 vLLM 0.29.0+cu129 virtualenv from an earlier project in the cluster home directory.
    Look for a `.venv` that has `vllm`. Do not build a new one.
  - Put this repo on `PYTHONPATH`.
  - Required environment variables: `HF_HOME=<scratch>/hf`, `HF_HUB_OFFLINE=1`, `VLLM_USE_FLASHINFER_SAMPLER=0`,
    and `TIKTOKEN_ENCODINGS_BASE=$HF_HOME/tiktoken_encodings`.
  - See `examples/site.env.example`.
  - The repo copy lives in `~/queryplan` on the cluster, synced with rsync (excluding `.venv`, `.git`, `runs`). Its
    untracked `site/cluster.env` has the real values.
- **First smoke run failed:** Gemma-4-31B at `--max-model-len 16384` needed 13.8 GiB of KV cache, and only 12.9 GiB
  was free next to its weights. Fixed by using 4096 (PR #9). This is not yet re-verified on the cluster.

### Jobs submitted and not yet checked (2026-10-23 ~16:35 BST)

| Job | GPUs | Sequence | Output dir (on the cluster, in `~/queryplan`) |
|---|---:|---|---|
| 7663943 `qp-full` | 4 | smoke naive plan naive plan-warm first, 6 experiments × "500" products | `runs/full3`, log `logs/qp-full_7663943.out` |
| 7663344 `qp-small` | 2 | smoke naive plan, 4 experiments × "300" products | `runs/small1`, log `logs/qp-small_7663344.out` |

- **Check them first:** `sacct -X -j 7663943,7663344 -o JobID,State,Start,Elapsed`.
- Both use the code as of PR #9 (4096 context), synced to the cluster before they could start.
- **Products are capped at 200.** `products()` in the demo builds only 10 qualities × 20 items, so "500" and "300"
  actually ran 200 per experiment (review finding below). Treat any results from these jobs as 200-product runs, or
  rerun after fixing.
- Other jobs on the account (`glm-smoke`, `nvidia-smi`) belong to another project. Leave them alone.

## 5. Code review findings (2026-10-23)

- **How the review worked:** six reviewers each took one subsystem, and three skeptics then tried to refute each
  finding.
- **Reading the table:** "3/3" means all three skeptics confirmed it; "not verified" means the run stopped before
  that finding was checked.
- **What to fix first:** the high-severity rows, and every row that affects the demo's timings.

| sev | location | finding | upheld |
|---|---|---|---|
| high | src/queryplan/planner.py:230 | In-flight queries never complete in the simulator, so mid-run replans predict inf: the learned cost model is never used, and a replan can drop tier 0's priority | 5/5 |
| high | src/queryplan/planner.py:324 | optimise builds the full candidate list before checking the limit: OOM/hang at the very first plan with 5 models on 4 GPUs | not verified |
| high | src/queryplan/costmodel.py:145 | Prompt cost is counted twice: theta is fitted on seconds/n_out, then the planner adds n_in/PREFILL_RATIO on top | 3/3 |
| high | src/queryplan/backends.py:108 | Concurrent runs pick the same port, and readiness accepts whichever server answers on it, so queries can go to another run's server | 3/3 |
| high | src/queryplan/__init__.py:87, executor.py:501, backends.py:120 | Ctrl-C, SIGTERM, SIGHUP or any exception escaping the main loop leaves loaded vLLM servers running (and drops buffered results) | 3/3 |
| high | src/queryplan/graph.py:83/86, executor.py:131 | Resume keeps stale downstream results: a node's fingerprint ignores its inputs' fingerprints and step code, and node ids are positional | 3/3 |
| high | src/queryplan/graph.py:156 | llm() prompts and @step args are kept by reference, so mutating a list after the call changes earlier nodes' inputs and makes their deps wrong | 3/3 |
| high | src/queryplan/refs.py:15 | The marker delimiter \x1e is whitespace and a line break to Python, so .strip()/.split()/.splitlines()/\s on a prompt template silently destroy markers | 3/3 |
| high | src/queryplan/refs.py:62 | repr-based formatting (f'{x!r}', a list or dict of outputs in an f-string, multi-output tuples) and json.dumps embed placeholder text with no marker and no dependency | 3/3 |
| medium | examples/judge_pipeline.py:45 | --units above 200 silently runs only 200 products per experiment while the report records the requested number | 3/3 |
| medium | examples/slurm_demo.sh:3 | #SBATCH --output=logs/... fails the job before the script's `mkdir -p logs` can run (create `logs/` before submitting, or drop the directory from --output) | 3/3 |
| medium | src/queryplan/backends.py:104 | One socket per in-flight request with no raised file limit: a 4-GPU run aborts under the common soft limit of 1024 open files | 3/3 |
| medium | src/queryplan/executor.py:518, 296 | Dispatch and switching stop for the whole solve: the main loop awaits optimise, so GPUs drain and sit idle at every replan (the comment claiming otherwise is wrong) | 3/3 |
| medium | src/queryplan/executor.py:478 | Validator retries are not in the plan: each query is charged one request's work although it may send up to `attempts` requests | 3/3 |
| medium | src/queryplan/costmodel.py:149 | Fallback and slope<=0 speed fits leave out `half` and return throughput at the observed concurrency, which is then used as peak theta | 3/3 |
| medium | src/queryplan/costmodel.py:66 | Output-length estimate can far exceed any observed reply: the sd prior includes spread between call sites, and exp(mu + sd^2/2) amplifies it | 3/3 |
| medium | src/queryplan/executor.py:138 | On resume the cost model loses the killed run's observations and the pilot is skipped, so the first plan uses default constants | 2/3 |
| medium | src/queryplan/graph.py:88 | Step fingerprint is only __qualname__ + repr(args): editing a step's code, closure, or outputs= (or a query's validate) resumes old values | 3/3 |
| medium | src/queryplan/refs.py:139 | The regex guard wraps only module-level re functions; precompiled patterns silently fail to match a placeholder or silently break markers | 3/3 |
| medium | src/queryplan/refs.py:68 | Hash-based membership on a Ref (`in set`, `in dict`, `dict.get`, Counter, set dedup) silently returns not-found instead of raising | 3/3 |
| medium | src/queryplan/__init__.py:79 | Site imports are not checked against the graph's fingerprints, so stale or mismatched upstream values are used silently | 3/3 |
| medium | src/queryplan/store.py:21, executor.py:132 | Results are serialised with json default=str, so resumed or imported step values silently change type and can crash dependents | 3/3 |
| medium | src/queryplan/executor.py:541 | run() raises a connection error from servers it is shutting down instead of the real failure | 3/3 |
| medium | src/queryplan/planner.py:367 | 'Local search' only tries one-slot changes to a single base, so it misses obvious plans and can be worse than naive | not verified |
| medium | src/queryplan/planner.py:309 | When every plan exceeds the fixed 7-day horizon, the loads tie-break picks the plan that uses the fewest GPUs | not verified |
| medium | src/queryplan/planner.py:389 | naive_plan ignores memory_gb and places a model on a GPU it does not fit | not verified |
| medium | src/queryplan/backends.py:69 | stop() escalates to SIGKILL only while the group leader is alive, so workers that outlive it keep holding GPU memory | 1/3 |
| low | src/queryplan/executor.py:404 | Every 'unload' event names the wrong model (the next target, or null); this skews the timeline figure | 3/3 |
| low | src/queryplan/executor.py:464 | Request timing includes failed attempts and retry back-off sleeps, which corrupts speed observations | 3/3 |
| low | src/queryplan/executor.py:266 | loading_left is ~1 s while a slot is still stopping its old server, because load_started is stale | 3/3 |
| low | src/queryplan/executor.py:113 | With a site plan, pilot units can be drawn from other sites, so the pilot never completes and the pilot replan never fires | 3/3 |
| low | src/queryplan/executor.py:542 | Per-site reports count the whole graph's queries and mark experiments run elsewhere as finished at ~0 s | 2/3 |
| low | src/queryplan/executor.py:536 | wall_seconds (and tier_seconds for unfinished tiers) include tearing down every server, one after another | 1/3 |
| low | src/queryplan/graph.py:128 | Experiment names containing '#' or ':' break node_of/marker parsing: silent None results or a crash after the whole run | 3/3 |
| low | src/queryplan/backends.py:114 | A user's --gpu-memory-utilization in `=` or underscore form is not detected and is silently overridden | 3/3 |
| low | src/queryplan/backends.py:226 | Any unexpected line from nvidia-smi crashes the load and aborts the run, even though the share is only a best-effort guess | 3/3 |
| low | src/queryplan/costmodel.py:77 | Load-time 'median' is the upper value for even counts, so a first load that included the weight download sticks | 1/3 |
| low | src/queryplan/planner.py:247 | A class whose queries hang off different parent classes is released at the slowest parent's rate, so times are badly overestimated | not verified |
| low | src/queryplan/backends.py:116 | The server's CUDA_VISIBLE_DEVICES replaces the driver's inherited restriction instead of indexing into it | 0/3 (refuted) |

**Must fix before the timed runs**, because these change or invalidate the demo's numbers. **All fixed on 2026-10-25** by the `train/mustfix/*` PR train, each with red/green tests:

- the products cap;
- servers left running on failure;
- port collisions;
- the in-flight simulator bug (replans never use what was learned);
- prompt cost counted twice;
- planning blocking dispatch;
- the file-descriptor limit on 4 GPUs;
- `logs/` for the Slurm output;
- unload events naming the wrong model.

Also fixed on the way: the speed fit's fallbacks now report peak throughput, `wall_seconds` no longer includes server teardown, and the plan-versus-naive test no longer flakes. The other findings matter for users, but not for the demo.

Note for the results: the demo's products now vary by audience as well ("budget kettle for students"), so prompts differ from the 2026-10-23 runs; the demo refuses more than 1000 products.

## 6. What is left, in order

1. **Check the two cluster jobs** (section 4). If either finished, pull `runs/*/*/report.json`, `events.jsonl`,
   `observations.jsonl` and `cost.json` back. They are useful as a first cost model, even if the runs are discarded.
2. ~~Fix the must-fix list~~ — done 2026-10-25 (section 5).
3. **Smoke test on both sites again** with `SEQUENCE=smoke`.
   - Site A: on a free GPU, with `EXTRA_ARGS="--local-snapshots <shared hub dir>"`, `HW` and `MEM` set.
   - Site B: 2 GPUs, `--time 00:45:00 --time-min 00:30:00`.
4. **Timed runs.** Same models and the same workload on both sites. Run naive and planned on the same GPUs, one after
   the other in one job or session. Warm the cache first, and interleave the policies (`naive plan naive plan-warm
   first`) so cache or queue drift does not favour one.
   - Site B: one node, `EXPS=6 UNITS=500` (after fixing the products cap), `--time 04:00:00 --time-min 02:30:00`.
     The budget agreed with the operator was about 8 node-hours in total.
   - Site A: whatever free GPUs exist, at most 3 of the H100s, else the free A100. `EXPS=4 UNITS=300`,
     `FIRST=gemma-warm`. Say in the results which hardware was used.
5. **Summaries.**
   - Run `examples/summarise_runs.py` per site to get a markdown table and a timeline PNG; put them in `docs/img/`.
   - Rerun `examples/simulate_scenarios.py --cost-model <measured cost.json> --models <three model keys>` to get
     predicted gains across workload shapes from real costs.
6. **Write `docs/results.md` and link it from the README.** Describe the sites only by their properties, for example:
   - "a shared server with 1–4 NVIDIA H100/A100 80 GB GPUs, models read from NFS, no scheduler, other users on the
     machine";
   - "a Slurm cluster of GH200 nodes (4 × 96 GB), models on a Lustre filesystem, low queue priority".

   Report for each run:
   - wall time, model loads and time spent loading;
   - when the first-priority experiment finished;
   - predicted vs actual times.

   Also report the cold vs warm load numbers, and say where the tool does not help (the scenario table shows 1.00×
   for 2 experiments on 2 GPUs). State n and caveats: one run per policy per site unless you repeat it, and warm
   caches.
7. **Name scan, then push** (section 1).

## 7. Gotchas that cost time

- **The agent's Bash tool is zsh:** arrays are 1-indexed, and some `for` loops behave differently. Run array logic
  under `bash`.
- **macOS vs Linux shell differences:**
  - macOS `date` has no `-Is`; use `date +%FT%T`.
  - macOS bash 3.2 treats an empty array as unbound under `set -u`; use the `${A[@]+"${A[@]}"}` idiom.
- **`set -o pipefail` with `ls … | wc -l`:** this kills the script when `ls` matches nothing.
- **Scratch directories disappear on a session restart.** Keep watchers and notes under the untracked `site/`.
- **"Ready" is not "serving".** The backend sends one real completion before any work.
- **GPU launchers (like the operator's GPU-server tool) set `CUDA_VISIBLE_DEVICES` and a port base.** The backend
  reads `QUERYPLAN_PORT_BASE`; inside Slurm it defaults from `SLURM_JOB_ID`. Runs sharing a base are safe: each
  server gets its own API key, and the backend only uses a server that accepts that key and refuses requests without it.
- **Write output unbuffered.** Driver progress is flushed (`print(..., flush=True)`); keep it that way, or log files
  stay empty for the whole run.
