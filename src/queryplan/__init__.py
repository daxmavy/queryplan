"""queryplan: run a fixed set of LLM queries on local GPUs in the order and placement that finishes soonest.

Write experiments as ordinary Python generators that call `llm()`; queryplan traces them into a graph, learns how long
each kind of query and each model load takes, and decides which model each GPU holds and when.
"""

from .costmodel import CostModel
from .graph import Experiment, Graph, LocalModel, experiment, llm, local, step, trace
from .planner import Plan, Problem, Slot, naive_plan, optimise
from .refs import OutputReadError, Ref
from .resources import gpus, visible_gpus

__all__ = [
    "CostModel", "Experiment", "Graph", "LocalModel", "OutputReadError", "Plan", "Problem", "Ref", "Slot",
    "experiment", "gpus", "llm", "local", "naive_plan", "optimise", "step", "trace", "visible_gpus",
]

__version__ = "0.1.0"
