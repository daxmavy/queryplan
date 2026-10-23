"""A small learned model of how long work takes: output lengths, model load times and serving throughput."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_LOAD_S = 120.0
DEFAULT_THETA = 1500.0  # output tokens per second for one saturated server
DEFAULT_HALF = 8.0  # concurrency at which a server reaches half its peak throughput
DEFAULT_SIGMA = 0.8
PREFILL_RATIO = 20.0  # input tokens are this many times cheaper than output tokens
SHRINK = 3.0  # pseudo-observations pulling a call site's length estimate toward its model's average
MIN_KIND_OBS = 5  # call sites with fewer observed replies count as unknown (the pilot samples them)


@dataclass
class Speed:
    theta: float  # peak output tokens/s of one server
    half: float  # concurrency at half of peak
    known: bool

    def rate(self, concurrency: float) -> float:
        """Total output tokens/s with `concurrency` requests in flight."""
        b = max(concurrency, 0.0)
        return self.theta * b / (b + self.half) if b > 0 else 0.0


class CostModel:
    """Fitted parameters plus the observations they were fitted on, saved together.

    Observation records (dicts) have a "type":
      request: model, hw, gpus, kind, n_in, n_out, seconds, concurrency (mean requests in flight on that server)
      load:    model, hw, gpus, seconds
    """

    def __init__(self, observations: Iterable[dict[str, Any]] = ()):
        self.observations: list[dict[str, Any]] = list(observations)
        self.fit()

    # ---- fitting -------------------------------------------------------------------------------------------------
    def add(self, *records: dict[str, Any]) -> None:
        self.observations.extend(records)

    def fit(self) -> "CostModel":
        requests = [o for o in self.observations if o.get("type") == "request" and o.get("n_out", 0) > 0]
        loads = [o for o in self.observations if o.get("type") == "load"]

        logs_by_kind: dict[str, list[float]] = {}
        logs_by_model: dict[str, list[float]] = {}
        for o in requests:
            v = math.log(o["n_out"])
            logs_by_kind.setdefault(o["kind"], []).append(v)
            logs_by_model.setdefault(o["model"], []).append(v)
        self._model_len = {m: _mean_sd(v) for m, v in logs_by_model.items()}
        self._kind_len: dict[str, tuple[float, float, int]] = {}
        for kind, values in logs_by_kind.items():
            model = kind.rsplit("|", 1)[-1]
            mu_m, sd_m = self._model_len.get(model, (math.log(256), DEFAULT_SIGMA))
            mu_k, sd_k = _mean_sd(values)
            n = len(values)
            mu = (n * mu_k + SHRINK * mu_m) / (n + SHRINK)
            sd = math.sqrt((max(n - 1, 0) * sd_k ** 2 + SHRINK * sd_m ** 2) / max(n - 1 + SHRINK, 1))
            self._kind_len[kind] = (mu, sd, n)

        by_config: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
        for o in requests:
            by_config.setdefault((o["model"], o["hw"], o["gpus"]), []).append(o)
        self._speed = {cfg: _fit_speed(obs) for cfg, obs in by_config.items()}

        load_times: dict[tuple[str, str, int], list[float]] = {}
        for o in loads:
            load_times.setdefault((o["model"], o["hw"], o["gpus"]), []).append(o["seconds"])
        self._load = {cfg: sorted(v)[len(v) // 2] for cfg, v in load_times.items()}
        return self

    # ---- predictions ---------------------------------------------------------------------------------------------
    def out_tokens(self, kind: str, model: str, max_tokens: int | None = None) -> tuple[float, float, bool]:
        """Expected output tokens for one query of `kind`, its log-scale sd, and whether the estimate rests on data."""
        if kind in self._kind_len:
            mu, sd, n = self._kind_len[kind]
            known = n >= MIN_KIND_OBS
        elif model in self._model_len:
            mu, sd = self._model_len[model]
            known = False
        else:
            guess = min(max_tokens or 1024, 1024) / 2
            mu, sd, known = math.log(guess), DEFAULT_SIGMA, False
        mean = math.exp(mu + sd * sd / 2)
        if max_tokens:
            mean = min(mean, max_tokens)
        return mean, sd, known

    def speed(self, model: str, hw: str, gpus: int = 1) -> Speed:
        if (model, hw, gpus) in self._speed:
            return self._speed[(model, hw, gpus)]
        same_model = [s for (m, _, _), s in self._speed.items() if m == model]
        if same_model:  # another GPU type or count: borrow the nearest thing we have, but mark it unknown
            s = same_model[0]
            return Speed(s.theta, s.half, known=False)
        return Speed(DEFAULT_THETA, DEFAULT_HALF, known=False)

    def load_seconds(self, model: str, hw: str, gpus: int = 1) -> tuple[float, bool]:
        if (model, hw, gpus) in self._load:
            return self._load[(model, hw, gpus)], True
        same_model = [v for (m, _, _), v in self._load.items() if m == model]
        if same_model:
            return sorted(same_model)[len(same_model) // 2], False
        return DEFAULT_LOAD_S, False

    def params(self) -> dict[str, Any]:
        return {
            "lengths": {k: {"mu": mu, "sd": sd, "n": n} for k, (mu, sd, n) in self._kind_len.items()},
            "speed": {"|".join(map(str, cfg)): {"theta": s.theta, "half": s.half} for cfg, s in self._speed.items()},
            "load_seconds": {"|".join(map(str, cfg)): v for cfg, v in self._load.items()},
        }

    # ---- persistence ---------------------------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"version": 1, "params": self.params(), "observations": self.observations}))

    @classmethod
    def load(cls, path: str | Path) -> "CostModel":
        data = json.loads(Path(path).read_text())
        return cls(data["observations"])

    @classmethod
    def merge(cls, *models: "CostModel") -> "CostModel":
        return cls([o for m in models for o in m.observations])


def _mean_sd(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, DEFAULT_SIGMA
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, max(math.sqrt(var), 0.05)


def _fit_speed(obs: list[dict[str, Any]]) -> Speed:
    """Fit seconds-per-output-token = (B + half) / theta by least squares on B (mean concurrency of each request)."""
    points = [(o["concurrency"], o["seconds"] / o["n_out"]) for o in obs
              if o.get("n_out", 0) >= 16 and o.get("seconds", 0) > 0 and o.get("concurrency", 0) > 0]
    if len(points) < 10:
        total_out = sum(o["n_out"] for o in obs)
        busy = sum(o["seconds"] / max(o.get("concurrency", 1), 1) for o in obs)
        theta = total_out / busy if busy > 0 else DEFAULT_THETA
        return Speed(theta, DEFAULT_HALF, known=len(obs) >= 10)
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    slope = sum((x - mx) * (y - my) for x, y in points) / sxx if sxx > 0 else 0.0
    if slope <= 0:  # concurrency barely varied: treat throughput as proportional to concurrency at the mean
        theta = mx / my
        return Speed(theta, DEFAULT_HALF, known=True)
    intercept = my - slope * mx
    theta = 1 / slope
    half = max(intercept * theta, 0.5)
    return Speed(theta, half, known=True)
