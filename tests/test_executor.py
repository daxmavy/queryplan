import json

import queryplan as qp

gen_a = qp.local("gen-a", temperature=0.7, max_tokens=400)
gen_b = qp.local("gen-b", temperature=0.7, max_tokens=400)
judge = qp.local("judge", temperature=0.0, max_tokens=50)


@qp.step
def parse(verdict):
    return int(verdict.rsplit(" ", 1)[-1])


def make_experiments(n_units=30):
    def build(name, model):
        @qp.experiment(name)
        def e():
            for i in range(n_units):
                text = qp.llm(model, f"{name} prompt {i}")
                verdict = qp.llm(judge, f"Rate: {text}", validate=lambda t: t.endswith(("7", "8", "9")), attempts=3)
                yield {"text": text, "score": parse(verdict)}
        return e

    return [build("e1", gen_a), build("e2", gen_b), build("e3", gen_a), build("e4", gen_b)]


def backend(speedup=400):
    return qp.FakeBackend(load_s={"gen-a": 60, "gen-b": 60, "judge": 60},
                          theta={"gen-a": 2000, "gen-b": 2000, "judge": 3000},
                          out_tokens={"gen-a": 300, "gen-b": 300, "judge": 20}, speedup=speedup)


def test_runs_everything_and_fills_values(tmp_path):
    exps = make_experiments()
    result = qp.run(exps, slots=qp.gpus([0, 1], hw="sim", mem_gb=80), store=tmp_path / "s", backend=backend(),
                    replan_seconds=0.5, log=lambda _: None)
    assert set(result.values) == {"e1", "e2", "e3", "e4"}
    for unit in result.values["e1"]:
        assert unit["text"].startswith("gen-a reply")
        assert isinstance(unit["score"], int)
    assert result.report.queries == 4 * 30 * 2
    observed = [o for o in result.cost_model.observations if o["type"] == "request"]
    assert len(observed) >= 240  # retries add requests


def test_validator_retries_until_valid_or_attempts_run_out(tmp_path):
    result = qp.run(make_experiments(10), slots=qp.gpus([0], hw="sim", mem_gb=80), store=tmp_path / "s",
                    backend=backend(), log=lambda _: None)
    lines = [json.loads(l) for l in (tmp_path / "s" / "results.jsonl").read_text().splitlines()]
    judged = [r for r in lines if str(r["value"]).startswith("judge reply")]
    assert all(1 <= r["attempts"] <= 3 for r in judged)
    assert any(r["attempts"] > 1 for r in judged)
    for r in judged:
        if r["attempts"] < 3:
            assert r["value"].endswith(("7", "8", "9"))


def test_plan_beats_naive_on_shared_models(tmp_path):
    slots = qp.gpus([0, 1], hw="sim", mem_gb=80)
    naive = qp.run(make_experiments(), slots=slots, store=tmp_path / "naive", backend=backend(100), policy="naive",
                   log=lambda _: None)
    planned = qp.run(make_experiments(), slots=slots, store=tmp_path / "plan", backend=backend(100),
                     cost_model=naive.cost_model, log=lambda _: None)
    assert planned.report.loads < naive.report.loads
    assert planned.report.wall_seconds < naive.report.wall_seconds


def test_first_tier_finishes_before_the_rest(tmp_path):
    exps = make_experiments()
    result = qp.run([exps[3], exps[:3]], slots=qp.gpus([0, 1], hw="sim", mem_gb=80), store=tmp_path / "s",
                    backend=backend(), log=lambda _: None)
    seconds = result.report.experiment_seconds
    assert seconds["e4"] <= min(seconds["e1"], seconds["e2"], seconds["e3"]) + 1e-6


def test_resume_skips_completed_nodes(tmp_path):
    store = tmp_path / "s"
    first = qp.run(make_experiments(5), slots=qp.gpus([0], hw="sim", mem_gb=80), store=store, backend=backend(),
                   log=lambda _: None)
    fake = backend()
    again = qp.run(make_experiments(5), slots=qp.gpus([0], hw="sim", mem_gb=80), store=store, backend=fake,
                   log=lambda _: None)
    assert fake.starts == []
    assert again.values == first.values


def test_unload_events_name_the_model_that_was_unloaded(tmp_path):
    qp.run(make_experiments(5), slots=qp.gpus([0], hw="sim", mem_gb=80), store=tmp_path / "s", backend=backend(),
           policy="naive", log=lambda _: None)
    events = [json.loads(l) for l in (tmp_path / "s" / "events.jsonl").read_text().splitlines()]
    resident = {}
    unloads = 0
    for e in events:
        if e["event"] == "load_done":
            resident[e["slot"]] = e["model"]
        elif e["event"] == "unload":
            assert e["model"] == resident.pop(e["slot"])
            unloads += 1
    assert unloads >= 3


def test_dispatch_carries_on_while_the_planner_solves(tmp_path, monkeypatch):
    import time
    from queryplan import executor

    windows = []
    real = executor.optimise

    def slow(problem, *args, **kwargs):
        if windows:  # every replan after the first plan takes half a second
            started = time.time()
            time.sleep(0.5)
            windows.append((started, time.time()))
        else:
            windows.append(None)
        return real(problem, *args, **kwargs)

    @qp.experiment("only")
    def only():
        for i in range(400):
            yield qp.llm(gen_a, f"prompt {i}")

    monkeypatch.setattr(executor, "optimise", slow)
    qp.run([only], slots=qp.gpus([0], hw="sim", mem_gb=80), store=tmp_path / "s", backend=backend(100),
           replan_seconds=0.3, max_inflight=4, log=lambda _: None)
    starts = [o["t"] - o["seconds"] for o in map(json.loads, (tmp_path / "s" / "observations.jsonl").open())
              if o["type"] == "request"]
    solving = [(a, b) for a, b in windows[1:] if b < max(starts)]  # a solve can outlast the work
    assert solving, "no replan happened"
    assert all(any(a + 0.1 < s < b for s in starts) for a, b in solving)


def test_cost_model_round_trips(tmp_path):
    result = qp.run(make_experiments(5), slots=qp.gpus([0], hw="sim", mem_gb=80), store=tmp_path / "s",
                    backend=backend(), save_cost_model=tmp_path / "cost.json", log=lambda _: None)
    loaded = qp.CostModel.load(tmp_path / "cost.json")
    assert loaded.observations == result.cost_model.observations
    merged = qp.CostModel.merge(loaded, loaded)
    assert len(merged.observations) == 2 * len(loaded.observations)


def test_snapshot_dir_picks_a_complete_snapshot(tmp_path):
    from queryplan.backends import snapshot_dir
    root = tmp_path / "models--org--m" / "snapshots"
    (root / "partial").mkdir(parents=True)
    (root / "partial" / "README.md").write_text("x")
    full = root / "full"
    full.mkdir()
    (full / "config.json").write_text("{}")
    (full / "model.safetensors").write_text("")
    assert snapshot_dir(tmp_path, "org/m") == full
    assert snapshot_dir(tmp_path, "org/other") is None


def test_snapshot_dir_skips_snapshots_missing_indexed_shards(tmp_path):
    import json as _json
    from queryplan.backends import snapshot_dir
    snap = tmp_path / "models--org--m" / "snapshots" / "a"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model-1.safetensors").write_text("")
    (snap / "model.safetensors.index.json").write_text(_json.dumps({"weight_map": {"x": "model-1.safetensors",
                                                                                   "y": "model-2.safetensors"}}))
    assert snapshot_dir(tmp_path, "org/m") is None
    (snap / "model-2.safetensors").write_text("")
    assert snapshot_dir(tmp_path, "org/m") == snap
