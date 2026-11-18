"""Timed forecast updates with rising reliability and waiting (Supplementary Section S10)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import flood as f
from . import social as so
from .model import Portfolio


@dataclass
class TimedModel(Portfolio):
    """``Portfolio`` with a release tick per query (earliest public time at which it can be asked)."""

    release: np.ndarray = field(default=None)
    group_features: np.ndarray = field(default=None)


def schedule(first: float, last: float, count: int) -> np.ndarray:
    """Reliabilities rising linearly from ``first`` (hour 0) to ``last`` (last update)."""
    return np.linspace(first, last, count) if count > 1 else np.array([first])


def build_timed(cfg: so.SocialConfig, reliabilities, hours=None) -> TimedModel:
    """Social model with forecast updates at ``hours`` (default 0, 1, ...) of the given reliabilities."""
    base = so.build_social_flood(cfg)
    rel = np.asarray(reliabilities, dtype=float)
    hours = np.arange(len(rel)) if hours is None else np.asarray(hours)
    _, _, depth0, _ = f._states(f.FloodConfig(lead_hours=cfg.lead_hours, hazard_prior=cfg.hazard_prior,
                                              preference_prior=cfg.preference_prior))
    riverside = depth0[np.repeat(np.arange(len(depth0)), 4), 0]
    flag = 0.15 + 0.7 * np.clip((riverside - 0.23) / 1.54, 0, 1)
    updates = np.column_stack([0.5 + r * (flag - 0.5) for r in rel])
    likelihood = np.column_stack((updates, base.likelihood[:, 1:]))
    durations = np.concatenate((np.ones(len(rel), dtype=int), base.durations[1:]))
    release = np.concatenate((2 * hours, np.zeros(len(base.durations) - 1, dtype=int))).astype(int)
    names = tuple(f"forecast_update_{int(h)}h" for h in hours) + base.queries[1:]
    return TimedModel(base.prior, likelihood, base.loss, base.features, base.feasible, durations,
                      cfg.query_cost * durations, base.indices, base.actions, names, base.config,
                      release=release, group_features=base.group_features)


class TimedSolver:
    """Exact history DP over (answers, tick) with release times and an optional wait action."""

    def __init__(self, model, allow_wait: bool = True, allowed=None, method: str = "optimal"):
        self.m = model
        self.allow_wait = allow_wait
        self.allowed = tuple(range(len(model.queries))) if allowed is None else tuple(allowed)
        self.method = method
        self.release = np.asarray(getattr(model, "release", np.zeros(len(model.queries), dtype=int)))
        self.durations = [int(d) for d in model.durations]
        self.deadline = int(model.config.deadline)
        self.lik1 = np.ascontiguousarray(model.likelihood.T)
        self.loss = model.loss
        self.root = ((-1,) * len(model.queries), 0)
        self.memo: dict = {}

    def mass(self, key) -> np.ndarray:
        h, _ = key
        b = self.m.prior.copy()
        for q, y in enumerate(h):
            if y != -1:
                b *= self.lik1[q] if y else 1 - self.lik1[q]
        return b

    def available(self, key):
        h, t = key
        return [q for q in self.allowed
                if h[q] == -1 and t >= self.release[q] and t + self.durations[q] <= self.deadline]

    def can_wait(self, key) -> bool:
        h, t = key
        if not self.allow_wait or t + 1 > self.deadline:
            return False
        return any(h[q] == -1 and self.release[q] > t and self.release[q] + self.durations[q] <= self.deadline
                   for q in self.allowed)

    def stop(self, key, mass):
        t = key[1]
        values = mass @ self.loss[:, t]
        feasible = np.flatnonzero(self.m.feasible[t])
        a = int(feasible[np.argmin(values[feasible])])
        return float(values[a]), ("act", a)

    def solve(self, key=None, mass=None):
        key = self.root if key is None else key
        if key in self.memo:
            return self.memo[key]
        mass = self.mass(key) if mass is None else mass
        ans = self.stop(key, mass)
        if self.method != "immediate":
            h, t = key
            total = float(mass.sum())
            for q in self.available(key):
                value = total * float(self.m.costs[q])
                for y in (0, 1):
                    hh = h[:q] + (y,) + h[q + 1:]
                    value += self.solve((hh, t + self.durations[q]), mass * (self.lik1[q] if y else 1 - self.lik1[q]))[0]
                if value < ans[0] - 1e-15:
                    ans = (value, ("query", q))
            if self.can_wait(key):
                value = self.solve((h, t + 1), mass)[0]
                if value < ans[0] - 1e-15:
                    ans = (value, ("wait", 0))
        self.memo[key] = ans
        return ans


def evaluate(model, policy, prior=None) -> dict:
    """Exact value of a solved ``TimedSolver`` policy in ``model`` (same public graph), with summaries."""
    lik1 = model.likelihood.T
    durations = [int(d) for d in model.durations]
    out = dict(value=0.0, waits=0.0, queries=np.zeros(len(model.queries)), commit_hours=0.0,
               actions={})
    stack = [(policy.root, (model.prior if prior is None else prior).copy())]
    while stack:
        key, mass = stack.pop()
        h, t = key
        kind, j = policy.solve(key, mass)[1] if key not in policy.memo else policy.memo[key][1]
        p = float(mass.sum())
        if kind == "act":
            assert model.feasible[t, j]
            out["value"] += float(mass @ model.loss[:, t, j])
            out["commit_hours"] += p * t / 2
            out["actions"][j] = out["actions"].get(j, 0.0) + p
        elif kind == "wait":
            out["waits"] += p
            stack.append(((h, t + 1), mass))
        else:
            out["value"] += p * float(model.costs[j])
            out["queries"][j] += p
            for y in (0, 1):
                hh = h[:j] + (y,) + h[j + 1:]
                stack.append(((hh, t + durations[j]), mass * (lik1[j] if y else 1 - lik1[j])))
    return out


def certificate_view(model) -> Portfolio:
    """Copy for the certificate routines: adds a zero-cost, uninformative half-hour wait query."""
    lik = np.column_stack((model.likelihood, np.full(len(model.prior), 0.5)))
    return Portfolio(model.prior, lik, model.loss, model.features, model.feasible,
                     np.concatenate((model.durations, [1])), np.concatenate((model.costs, [0.0])),
                     model.indices, model.actions, tuple(model.queries) + ("wait",), model.config)


def aggregate_timed(model, inverse) -> TimedModel:
    """Prior-conditional class means of scores and laws; release times unchanged."""
    from .reduction import aggregate_partition
    base = aggregate_partition(Portfolio(model.prior, model.likelihood, model.loss, model.features,
                                         model.feasible, model.durations, model.costs, model.indices,
                                         model.actions, model.queries, model.config), inverse)
    return TimedModel(base.prior, base.likelihood, base.loss, base.features, base.feasible, base.durations,
                      base.costs, base.indices, base.actions, base.queries, base.config,
                      release=model.release, group_features=None)
