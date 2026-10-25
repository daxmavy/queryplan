import importlib.util
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_builds_as_many_distinct_products_as_asked():
    demo = load("judge_pipeline")
    names = demo.products(500)
    assert len(names) == 500
    assert len(set(names)) == 500


def test_demo_refuses_more_products_than_it_can_build():
    demo = load("judge_pipeline")
    with pytest.raises(ValueError):
        demo.products(10 ** 6)


def test_slurm_output_does_not_need_a_directory_to_exist():
    script = (EXAMPLES / "slurm_demo.sh").read_text()
    outputs = [line for line in script.splitlines() if line.startswith("#SBATCH --output")]
    assert outputs and all("/" not in line for line in outputs)
