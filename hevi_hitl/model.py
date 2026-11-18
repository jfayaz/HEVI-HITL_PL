"""Query-or-commit model, exact history recursion (Eq. 5) and the bounds of Proposition 1."""

from dataclasses import dataclass
from functools import lru_cache
import numpy as np


@dataclass
class Portfolio:
    """Finite query-or-commit model: configurations z, commitment ticks t (half-hours), portfolios a, queries q."""

    prior: np.ndarray        # [z]
    likelihood: np.ndarray   # [z, q] probability of answer 1
    loss: np.ndarray         # [z, t, a] normalised score
    features: np.ndarray     # [z, t, a, 4] damage, person-days, exposed people, expenditure
    feasible: np.ndarray     # [t, a]
    durations: np.ndarray    # [q] ticks
    costs: np.ndarray        # [q]
    indices: np.ndarray      # [z, coordinates]
    actions: tuple
    queries: tuple
    config: object


class PortfolioSolver:
    """Exact recursion over answer histories with unnormalised masses."""

    def __init__(self, model, allowed=None, method="optimal"):
        """Set up a solver; ``allowed`` restricts the query inventory, ``method`` the planner."""
        self.m = model
        self.allowed = tuple(range(len(model.queries))) if allowed is None else tuple(allowed)
        self.method = method
        self.root = (-1,) * len(model.queries)
        self.memo = {}
        self.massmemo = {}

    def time(self, h):
        """Elapsed time (half-hours) after the queries answered in history ``h``."""
        return int(sum(d for d, y in zip(self.m.durations, h) if y != -1))

    def mass(self, h):
        """Unnormalised state masses prior * likelihood of the answers in ``h`` (memoised)."""
        if h not in self.massmemo:
            b = self.m.prior.copy()
            for q, y in enumerate(h):
                if y != -1:
                    b *= self.m.likelihood[:, q] if y else 1 - self.m.likelihood[:, q]
            self.massmemo[h] = b
        return self.massmemo[h]

    def children(self, h, q):
        """Yield the two histories that extend ``h`` by answering query ``q``."""
        for y in (0, 1):
            hh = list(h)
            hh[q] = y
            yield tuple(hh)

    def available(self, h):
        """Unasked allowed queries that still fit before the public deadline."""
        t = self.time(h)
        return [q for q in self.allowed if h[q] == -1 and t + self.m.durations[q] <= self.m.config.deadline]

    def stop(self, h):
        """Best feasible immediate commitment at ``h``: (unnormalised value, ('act', a))."""
        vals = self.mass(h) @ self.m.loss[:, self.time(h)]
        a = min(np.flatnonzero(self.m.feasible[self.time(h)]), key=lambda a: vals[a])
        return float(vals[a]), ("act", int(a))

    def solve(self, h=None):
        """Exact (or planner-specific) value and decision at history ``h``; memoised."""
        h = self.root if h is None else h
        if h in self.memo:
            return self.memo[h]
        ans = self.stop(h)
        qs = self.available(h)
        if self.method == "immediate":
            qs = []
        elif self.method == "fixed_order":
            qs = qs[:1]
        for q in qs:
            children = list(self.children(h, q))
            cost = self.mass(h).sum() * self.m.costs[q]
            if self.method == "one_step":
                candidate = cost + sum(self.stop(hh)[0] for hh in children)
            else:
                candidate = cost + sum(self.solve(hh)[0] for hh in children)
            if candidate < ans[0] - 1e-15:
                ans = (candidate, ("query", q))
        self.memo[h] = ans
        return ans

    def evaluate(self, policy):
        """Evaluate a fixed ``policy`` in this model."""

        @lru_cache(None)
        def rec(h):
            """Unnormalised loss, query counts and features of the subtree at ``h``."""
            kind, j = policy.solve(h)[1]
            mass = self.mass(h)
            t = self.time(h)
            if kind == "act":
                assert self.m.feasible[t, j]
                return (
                    float(mass @ self.m.loss[:, t, j]),
                    np.zeros(len(self.m.queries)),
                    mass @ self.m.features[:, t, j],
                )
            assert j in self.available(h)
            cost = mass.sum() * self.m.costs[j]
            counts = np.eye(len(self.m.queries))[j] * mass.sum()
            f = np.zeros(4)
            for hh in self.children(h, j):
                cc, nn, ff = rec(hh)
                cost += cc
                counts += nn
                f += ff
            return cost, counts, f

        return rec(self.root)


def normalized_reference(model):
    """Normalised recursion; an independent check of the solver in the tests."""

    @lru_cache(None)
    def value(h):
        """Normalised Bellman value at history ``h``."""
        b = model.prior.copy()
        t = 0
        for q, y in enumerate(h):
            if y != -1:
                t += model.durations[q]
                b *= model.likelihood[:, q] if y else 1 - model.likelihood[:, q]
        b /= b.sum()
        best = min(float(b @ model.loss[:, t, a]) for a in range(len(model.actions)) if model.feasible[t, a])
        for q in range(len(model.queries)):
            if h[q] != -1 or t + model.durations[q] > model.config.deadline:
                continue
            prob = float(b @ model.likelihood[:, q])
            vv = float(model.costs[q])
            for y, py in ((0, 1 - prob), (1, prob)):
                hh = list(h)
                hh[q] = y
                vv += py * value(tuple(hh))
            best = min(best, vv)
        return best

    return value((-1,) * len(model.queries))


def certificates(model, reduced, inverse, policy, gamma=0.0):
    """Uniform and per-configuration post-solve regret bounds (Proposition 1)."""
    if gamma < 0:
        raise ValueError("Nonnegative surrogate optimality gap required")
    if not (
        np.array_equal(model.feasible, reduced.feasible)
        and np.array_equal(model.durations, reduced.durations)
        and np.array_equal(model.costs, reduced.costs)
        and model.queries == reduced.queries
        and model.actions == reduced.actions
        and model.config.deadline == reduced.config.deadline
    ):
        raise ValueError("The certificate requires a shared public decision graph")
    mass = np.bincount(inverse, weights=model.prior, minlength=len(reduced.prior))
    if mass.shape != reduced.prior.shape or not np.allclose(mass, reduced.prior, rtol=0, atol=1e-12):
        raise ValueError("The surrogate must retain original prior class masses")
    times = {0}
    for duration in model.durations:
        times |= {t + int(duration) for t in tuple(times) if t + duration <= model.config.deadline}
    endpoints = model.feasible.copy()
    for t in range(len(endpoints)):
        if t not in times:
            endpoints[t] = False
    difference = (model.loss - reduced.loss[inverse])[:, endpoints]
    lo = difference.min(axis=1)
    hi = difference.max(axis=1)
    midpoint = 0.5 * (lo + hi)
    eps = 0.5 * (hi - lo)
    tv = np.abs(model.likelihood - reduced.likelihood[inverse])
    qn = len(model.queries)
    allmask = (1 << qn) - 1

    def available(mask):
        """Queries still askable when the bitmask ``mask`` marks the unasked queries."""
        t = sum(int(model.durations[q]) for q in range(qn) if not (mask >> q & 1))
        return [q for q in range(qn) if mask >> q & 1 and t + model.durations[q] <= model.config.deadline]

    @lru_cache(None)
    def divergence(mask):
        """Per-state worst-case total-variation accumulation over remaining queries."""
        ans = np.zeros(len(model.prior))
        for q in available(mask):
            ans = np.maximum(ans, tv[:, q] + (1 - tv[:, q]) * divergence(mask ^ (1 << q)))
        return ans

    @lru_cache(None)
    def maxcost(mask):
        """Largest acquisition cost of any feasible query sequence from ``mask``."""
        return max([0] + [model.costs[q] + maxcost(mask ^ (1 << q)) for q in available(mask)])

    span = np.ptp(reduced.loss[:, endpoints], axis=1) + maxcost(allmask)
    uniform = float(model.prior @ (eps + span[inverse] * divergence(allmask)))
    original_value = PortfolioSolver(model).evaluate(policy)[0]
    surrogate_value = PortfolioSolver(reduced).evaluate(policy)[0]
    offset = float(model.prior @ midpoint)
    delta = float(original_value - offset - surrogate_value)
    assert abs(delta) <= uniform + 1e-10
    return dict(
        uniform_error=uniform,
        accounting_offset=offset,
        original_policy_value=float(original_value),
        surrogate_policy_value=float(surrogate_value),
        signed_discrepancy=delta,
        solver_gap=float(gamma),
        uniform_regret_bound=2 * uniform + gamma,
        postsolve_regret_bound=uniform + gamma + delta,
    )
