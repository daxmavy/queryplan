import queryplan as qp
from queryplan.costmodel import CostModel
from queryplan.planner import ClassState, Problem, naive_plan, optimise


def cost_with(load_s: float, theta: float, models: list[str], hw: str = "H100") -> CostModel:
    obs = []
    for m in models:
        obs.append({"type": "load", "model": m, "hw": hw, "gpus": 1, "seconds": load_s})
        for b in (4, 16, 64, 128):
            for _ in range(5):
                obs.append({"type": "request", "model": m, "hw": hw, "gpus": 1, "kind": f"k|{m}", "n_in": 100,
                            "n_out": 200, "seconds": 200 * (b + 8) / theta, "concurrency": b})
    return CostModel(obs)


def slots(n: int):
    return qp.gpus(list(range(n)), hw="H100", mem_gb=80)


def test_independent_queries_one_gpu_loads_each_model_once():
    models = ["a", "b", "c"]
    classes = {}
    for e in range(4):  # four experiments, each using all three models independently
        for m in models:
            cid = f"e{e}|{m}"
            classes[cid] = ClassState(id=cid, experiment=f"e{e}", tier=0, model=m, depth=1, total=200,
                                      out_tokens=200, in_tokens=100)
    cost = cost_with(load_s=60, theta=2000, models=models)
    problem = Problem(classes, slots(1), cost)
    plan = optimise(problem)
    naive = naive_plan(problem, [f"e{e}" for e in range(4)])
    assert plan.predicted_loads == 3
    assert naive.predicted_loads == 12
    assert plan.predicted_makespan < naive.predicted_makespan


def test_generate_then_judge_on_two_gpus_keeps_both_models_resident():
    classes = {
        "gen": ClassState(id="gen", experiment="e", tier=0, model="g", depth=1, total=2000, out_tokens=300),
        "judge": ClassState(id="judge", experiment="e", tier=0, model="j", depth=2, total=2000, out_tokens=300,
                            parents=("gen",)),
    }
    # judging is as heavy as generating and loads are slow: one GPU per model, both resident throughout
    cost = cost_with(load_s=300, theta=1500, models=["g", "j"])
    plan = optimise(Problem(classes, slots(2), cost))
    assert plan.predicted_loads == 2
    assert sorted(phases[0].models for phases in plan.phases.values()) == [("g",), ("j",)]


def test_generate_then_light_judge_uses_both_gpus_for_generation_first():
    classes = {
        "gen": ClassState(id="gen", experiment="e", tier=0, model="g", depth=1, total=2000, out_tokens=300),
        "judge": ClassState(id="judge", experiment="e", tier=0, model="j", depth=2, total=2000, out_tokens=50,
                            parents=("gen",)),
    }
    cost = cost_with(load_s=120, theta=1500, models=["g", "j"])
    problem = Problem(classes, slots(2), cost)
    plan = optimise(problem)
    split = problem.simulate(qp.Plan(phases={"gpu0": [qp.planner.Phase(("g",))], "gpu1": [qp.planner.Phase(("j",))]}))
    assert plan.predicted_makespan <= split.makespan


def test_strict_tiers_finish_first_tier_sooner_than_one_tier():
    def build(tier_of_e1):
        classes = {}
        for e, tier in (("e1", tier_of_e1), ("e2", 1)):
            classes[f"{e}|a"] = ClassState(id=f"{e}|a", experiment=e, tier=tier, model="a", depth=1, total=500,
                                           out_tokens=200)
            classes[f"{e}|b"] = ClassState(id=f"{e}|b", experiment=e, tier=tier, model="b", depth=2, total=500,
                                           out_tokens=200, parents=(f"{e}|a",))
        return classes

    cost = cost_with(load_s=60, theta=2000, models=["a", "b"])
    prioritised = optimise(Problem(build(0), slots(2), cost))
    assert prioritised.predicted_tiers[0] <= prioritised.predicted_tiers[1]


def test_memory_limits_are_respected():
    classes = {"big": ClassState(id="big", experiment="e", tier=0, model="big", depth=1, total=10, memory_gb=70)}
    cost = cost_with(load_s=10, theta=1000, models=["big"])
    small = qp.gpus([0], hw="L40S", mem_gb=46) + qp.gpus([1], hw="H100", mem_gb=80)
    plan = optimise(Problem(classes, small, cost))
    assert plan.phases["gpu0"] == []
    assert plan.phases["gpu1"][0].models == ("big",)
