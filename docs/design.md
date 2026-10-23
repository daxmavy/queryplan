# Design

## Graph

Tracing produces a graph $G = (V, E)$ of queries $Q$ and steps $S$. An edge $u \to v$ means $v$ reads $u$'s output.
Every query $q$ has a model handle $m(q)$ (model plus every setting that affects outputs), a call site, and a depth
$\delta(q)$: the number of queries on the longest chain ending at $q$.

Queries are grouped into **classes** $\kappa = (\text{experiment}, \text{call site}, m, \delta, \text{tier})$. The
planner works on classes; queries are only used for dispatch.

## Cost model

For a call site $k$ (one `llm()` line with one model), output length is log-normal:

$$\log n^{out}_q \sim \mathcal N(\mu_k, \sigma_k^2)$$

with $\mu_k$ shrunk toward the model's mean by 3 pseudo-observations, so a call site with little data borrows from the
model's other call sites. Expected work per query is $n^{out} + n^{in}/20$ output-token equivalents.

For a model $m$ on GPU type $h$ with $g$ GPUs, total decode throughput at concurrency $B$ is

$$\Theta(B) = \Theta^{max} \frac{B}{B + B^{1/2}}$$

fitted by least squares on per-request seconds per output token, $(B + B^{1/2})/\Theta^{max}$, against the mean number
of requests in flight. Load time $t^{load}_{m,h,g}$ is the median of observed loads. An unseen configuration borrows an
observed configuration of the same model and is marked unknown, which triggers a pilot.

## Plans

A plan gives each GPU slot a list of phases; a phase is a cycle of models. A slot serves its resident model while it
has ready work, then switches to the next model in its cycle that has ready work (paying $t^{load}$), and waits if
nothing is ready but its resident model still has work coming.

## Simulation and search

Each class is a fluid. Its released amount grows with its parents' progress,

$$r_\kappa(t) = N_\kappa \min_{p \in \text{parents}(\kappa)} \frac{d_p(t)}{N_p}$$

and a slot serving model $m$ drains the ready work of $m$'s classes in priority order (tier, then deeper first) at
$\Theta(B)$, with $B$ the ready work divided among the slots serving $m$ (capped at 256).

Every assignment of a cycle to each slot is simulated (slots that look alike are interchangeable, so combinations,
not products, are enumerated). Plans are compared by

$$\big(\#\text{unfinished tiers},\; C(T_1),\; C(T_2),\; \dots,\; \#\text{loads}\big)$$

lexicographically, where $C(T) = \max_{d \in T} \text{finish}(d)$ is the time the last item of tier $T$ finishes.
Candidates are screened with a coarse time step and the twelve best are re-simulated with a fine one.

## Exploration and re-planning

When a call site or a model configuration has no data, a pilot of two random units per experiment runs first as a
top-priority tier. When the pilot finishes, and every `replan_seconds` after that, the cost model is refitted and the
whole plan re-solved from the current state (what each slot holds, what is done). The new plan $\pi'$ replaces the
current $\pi$ only if, at the first tier where they differ, it finishes earlier by more than 3% and 30 s.

## What the planner controls

| Fixed by the declaration | Chosen by the planner |
|---|---|
| model, weights, sampling settings, chat template, reasoning setting, max tokens, seed, validator, attempts | which slot serves which model, when models load, the order queries are sent, how many are in flight |

Every declared query runs exactly once (a failed validation is retried inside the same query, up to its attempts).
