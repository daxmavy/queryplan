"""queryplan: run a fixed set of LLM queries on local GPUs in the order and placement that finishes soonest.

Write experiments as ordinary Python generators that call `llm()`; queryplan traces them into a graph, learns how long
each kind of query and each model load takes, and decides which model each GPU holds and when.
"""

from .graph import Experiment, Graph, LocalModel, experiment, llm, local, step, trace
from .refs import OutputReadError, Ref

__all__ = ["Experiment", "Graph", "LocalModel", "OutputReadError", "Ref", "experiment", "llm", "local", "step", "trace"]

__version__ = "0.1.0"
