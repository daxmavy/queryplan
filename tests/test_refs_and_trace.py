import re

import pytest

import queryplan as qp
from queryplan.refs import resolve

gen = qp.local("gen-model", temperature=0.7, max_tokens=100)
judge = qp.local("judge-model", temperature=0.0)


@qp.step
def score(text):
    return int(re.search(r"(\d)", text).group(1))


@qp.step(outputs=2)
def split_pair(a, b):
    return [a + "!", b + "?"]


def test_formatting_placeholders_works_and_resolves():
    @qp.experiment("e")
    def e():
        a = qp.llm(gen, "write something")
        b = qp.llm(judge, f"rate this: {a}\n" + "--" + "".join(["[", a, "]"]))
        yield {"a": a, "b": b, "s": score(b)}

    graph = qp.trace([e])
    q1, q2, s = list(graph.nodes.values())
    assert q2.deps == (q1.id,)
    assert q2.depth == 2 and s.depth == 2
    values = {q1.id: "hello", q2.id: "7 points"}
    assert resolve(q2.prompt, values.__getitem__) == "rate this: hello\n--[hello]"


def test_format_spec_is_applied_later():
    @qp.experiment("e")
    def e():
        a = qp.llm(gen, "x")
        n = score(a)
        yield qp.llm(gen, f"share {n:.0%}")

    graph = qp.trace([e])
    last = list(graph.nodes.values())[-1]
    lookup = {"e/2": 0.25}.__getitem__
    assert resolve(last.prompt, lookup) == "share 25%"


@pytest.mark.parametrize("read", [
    lambda r: len(r), lambda r: "x" in r, lambda r: r[0], lambda r: r.split(), lambda r: r.strip(),
    lambda r: bool(r), lambda r: r == "a", lambda r: re.search("a", r), lambda r: re.findall("a", "b" + r),
    lambda r: list(r), lambda r: r.upper(),
])
def test_reading_an_output_raises(read):
    @qp.experiment("e")
    def e():
        a = qp.llm(gen, "x")
        read(a)
        yield a

    with pytest.raises(qp.OutputReadError):
        qp.trace([e])


def test_steps_without_placeholders_run_immediately():
    @qp.experiment("e")
    def e():
        yield score("value 3")

    graph = qp.trace([e])
    assert graph.nodes == {}
    assert graph.unit_outputs["e"] == [3]


def test_multi_output_step_can_be_unpacked():
    @qp.experiment("e")
    def e():
        a = qp.llm(gen, "x")
        b = qp.llm(gen, "y")
        first, second = split_pair(a, b)
        yield qp.llm(judge, f"{second} then {first}")

    graph = qp.trace([e])
    last = list(graph.nodes.values())[-1]
    assert set(last.deps) == {"e/3"}
    assert resolve(last.prompt, lambda key: ["A!", "B?"][int(key.split("#")[1])]) == "B? then A!"


def test_llm_outside_experiment_is_an_error():
    with pytest.raises(RuntimeError):
        qp.llm(gen, "x")


def test_model_key_changes_with_output_settings():
    assert qp.local("m", temperature=0.1).key != qp.local("m", temperature=0.2).key
    assert qp.local("m", temperature=0.1).key == qp.local("m", temperature=0.1).key
