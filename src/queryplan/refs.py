"""Placeholders for values that do not exist yet.

While a graph is being traced, `llm()` and `@step` calls return a `Ref` instead of a value. A `Ref` is a
`str` whose characters are a marker naming the value, so formatting and concatenation (f-strings, `+`,
`str.join`, `%`) work unchanged and leave the marker inside the resulting string. Anything that would read
the value raises `OutputReadError`, which tells the user to move that code into a `@step`.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Callable, Iterator

MARK = "\x1e"  # ASCII record separator: never appears in normal prompt text
_MARKER = re.compile(MARK + r"qp:([^:\x1e]+):([^\x1e]*)" + MARK)


class OutputReadError(TypeError):
    """Raised when code reads an LLM output (or a step result) while the graph is being built."""


def marker(key: str, spec: str = "") -> str:
    return f"{MARK}qp:{key}:{spec}{MARK}"


def _refuse(name: str) -> Callable[..., Any]:
    def method(self: "Ref", *args: Any, **kwargs: Any) -> Any:
        raise OutputReadError(
            f"`{name}` reads the value of {self.describe()}, which does not exist until the graph runs. "
            "Formatting it into a string is fine; to read or transform it, put that code in a function "
            "decorated with @step."
        )

    method.__name__ = name
    return method


class Ref(str):
    """A placeholder for the output of node `node_id` (element `index` of it, for multi-output steps)."""

    node_id: str
    index: int | None

    def __new__(cls, node_id: str, index: int | None = None) -> "Ref":
        key = node_id if index is None else f"{node_id}#{index}"
        self = str.__new__(cls, marker(key))
        self.node_id = node_id
        self.index = index
        return self

    @property
    def key(self) -> str:
        return self.node_id if self.index is None else f"{self.node_id}#{self.index}"

    def describe(self) -> str:
        return f"output {self.key}"

    def __format__(self, spec: str) -> str:
        return marker(self.key, spec)

    def __repr__(self) -> str:
        return f"<queryplan.Ref {self.key}>"

    def __reduce__(self) -> tuple:
        return (Ref, (self.node_id, self.index))

    __hash__ = str.__hash__  # Refs are stored in dicts and sets internally; hashing the marker reads nothing


# Every str method or operator that would look at the characters is refused. Formatting (__format__, __str__),
# concatenation (__add__, __radd__ via str) and %-interpolation keep working because they only copy the marker.
_ALLOWED = {
    "__new__", "__init__", "__format__", "__str__", "__repr__", "__add__", "__rmod__",
    "__getattribute__", "__setattr__", "__delattr__", "__reduce__", "__reduce_ex__", "__getnewargs__",
    "__class__", "__init_subclass__", "__subclasshook__", "__sizeof__", "__dir__", "__doc__", "__hash__",
    "__getstate__",
}
for _name in dir(str):
    if _name in _ALLOWED or _name in Ref.__dict__:
        continue
    if callable(getattr(str, _name)):
        setattr(Ref, _name, _refuse(_name))
setattr(Ref, "__bool__", _refuse("__bool__"))


def keys_in_text(text: str) -> list[str]:
    """Marker keys embedded in a string, in order."""
    return [m.group(1) for m in _MARKER.finditer(text)]


def find_keys(value: Any) -> set[str]:
    """Every value key referenced anywhere inside `value` (nested containers and strings included)."""
    found: set[str] = set()
    _walk(value, found)
    return found


def _walk(value: Any, found: set[str]) -> None:
    if isinstance(value, Ref):
        found.add(value.key)
    elif isinstance(value, str):
        if MARK in value:
            found.update(keys_in_text(value))
    elif isinstance(value, dict):
        for k, v in value.items():
            _walk(k, found)
            _walk(v, found)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for v in value:
            _walk(v, found)


def node_of(key: str) -> str:
    return key.split("#", 1)[0]


def resolve(value: Any, lookup: Callable[[str], Any]) -> Any:
    """Replace every Ref and embedded marker in `value` with the real value from `lookup(key)`."""
    if isinstance(value, Ref):
        return lookup(value.key)
    if isinstance(value, str):
        if MARK not in value:
            return value
        return _MARKER.sub(lambda m: format(lookup(m.group(1)), m.group(2)), value)
    if isinstance(value, dict):
        return {resolve(k, lookup): resolve(v, lookup) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(resolve(v, lookup) for v in value)
    if isinstance(value, list):
        return [resolve(v, lookup) for v in value]
    if isinstance(value, (set, frozenset)):
        return type(value)(resolve(v, lookup) for v in value)
    return value


# Python's `re` reads a str subclass's raw characters, so a Ref passed to it cannot raise by itself. While tracing,
# the module-level functions are wrapped so that a string containing a marker raises instead of silently not matching.
_RE_FUNCS = ("match", "search", "fullmatch", "findall", "finditer", "split", "sub", "subn")


@contextmanager
def guard_regex() -> Iterator[None]:
    originals = {name: getattr(re, name) for name in _RE_FUNCS}

    def wrap(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def checked(pattern: Any, *args: Any, **kwargs: Any) -> Any:
            for arg in list(args) + list(kwargs.values()):
                if isinstance(arg, str) and MARK in arg:
                    raise OutputReadError(
                        f"re.{name} was given an LLM output (or a string containing one) while the graph is being "
                        "built. Put this code in a function decorated with @step."
                    )
            return fn(pattern, *args, **kwargs)

        checked.__name__ = name
        return checked

    try:
        for name, fn in originals.items():
            setattr(re, name, wrap(name, fn))
        yield
    finally:
        for name, fn in originals.items():
            setattr(re, name, fn)
