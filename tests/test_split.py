import queryplan as qp

from test_executor import backend, judge, make_experiments


def test_split_sends_judging_to_the_site_with_the_judge_and_runs_in_two_halves(tmp_path):
    exps = make_experiments(6)
    graph = qp.trace(exps)
    plan = qp.split(graph, {"gen-site": ["gen-a", "gen-b"], "judge-site": ["judge"]})
    assert plan.order == ["gen-site", "judge-site"]
    assert all(b["from"] == "gen-site" and b["to"] == "judge-site" for b in plan.boundary)
    plan.save(tmp_path / "split.json")
    first = qp.run(make_experiments(6), slots=qp.gpus([0], hw="sim", mem_gb=80), store=tmp_path / "a",
                   backend=backend(), site_plan=tmp_path / "split.json", site="gen-site", log=lambda _: None)
    assert all(u["score"] is None and u["text"] for u in first.values["e1"])
    second = qp.run(make_experiments(6), slots=qp.gpus([0], hw="sim", mem_gb=80), store=tmp_path / "b",
                    backend=backend(), site_plan=tmp_path / "split.json", site="judge-site",
                    imports=[tmp_path / "a" / "results.jsonl"], log=lambda _: None)
    assert all(isinstance(u["score"], int) for u in second.values["e1"])


def test_split_balances_whole_units_when_sites_are_alike():
    graph = qp.trace(make_experiments(10))
    plan = qp.split(graph, {"x": ["gen-a", "gen-b", "judge"], "y": ["gen-a", "gen-b", "judge"]})
    assert plan.boundary == []
    assert abs(plan.work["x"] - plan.work["y"]) <= 2 * 256
