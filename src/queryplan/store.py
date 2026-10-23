"""A run's directory on the compute site's filesystem: results, observations and an event log, all append-only."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)
        self._results = open(self.path / "results.jsonl", "a")
        self._observations = open(self.path / "observations.jsonl", "a")
        self._events = open(self.path / "events.jsonl", "a")
        self._last_flush = time.monotonic()

    def _write(self, handle: Any, record: dict[str, Any]) -> None:
        handle.write(json.dumps(record, default=str) + "\n")
        if time.monotonic() - self._last_flush > 5:
            self.flush()

    def result(self, node_id: str, fingerprint: str, value: Any, **extra: Any) -> None:
        self._write(self._results, {"id": node_id, "fp": fingerprint, "value": value, **extra})

    def observation(self, record: dict[str, Any]) -> None:
        self._write(self._observations, record)

    def event(self, kind: str, **fields: Any) -> None:
        self._write(self._events, {"at": time.time(), "event": kind, **fields})

    def flush(self) -> None:
        for handle in (self._results, self._observations, self._events):
            handle.flush()
        self._last_flush = time.monotonic()

    def close(self) -> None:
        self.flush()
        for handle in (self._results, self._observations, self._events):
            handle.close()

    def read(self, name: str) -> Iterator[dict[str, Any]]:
        path = self.path / f"{name}.jsonl"
        if not path.exists():
            return iter(())
        with open(path) as handle:
            return iter([json.loads(line) for line in handle if line.strip()])
