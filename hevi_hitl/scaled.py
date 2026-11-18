"""Scaled instance family (768 to 518,112 configurations) and a fast exact history recursion (Section 5.8)."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.special import expit

from . import flood as f
from . import social as so
from .model import Portfolio

SEVERITY_KNOTS_M = np.array([0.35, 0.85, 1.65])
LATTICE_ORDER = {10: 3, 15: 4, 21: 5}
LATTICE_SHRINK = 0.1
EXTRA_PREFERENCE_PAIR = np.array([[4_000_000., 4000., 100.], [3_000_000., 9000., 100.]])
# further comparison pairs for the query-scaling study (experiment 21)
MORE_PREFERENCE_PAIRS = np.array([
    [[3_200_000., 6000., 150.], [3_600_000., 6000., 100.]],
    [[3_000_000., 6000., 120.], [4_400_000., 6000., 80.]],
    [[3_500_000., 5000., 130.], [3_500_000., 7000., 90.]],
    [[3_200_000., 5000., 100.], [3_600_000., 3000., 100.]],
    [[3_000_000., 8000., 120.], [3_600_000., 5000., 100.]],
    [[3_800_000., 4000., 140.], [3_200_000., 9000., 90.]],
])
GAUGE_NAMES = ("depth_gauge_central", "depth_gauge_upstream")


@dataclass(frozen=True)
class ScaledConfig:
    severity_levels: int = 3
    footprints: int = 2
    preference_types: int = 4
    district_gauges: int = 0
    extra_preference_pair: bool = False
    lead_hours: int = 6
    mechanism_strength: float = 1.0
    social_query_reliability: float = 0.9
    preference_prior: object = "diffuse"
    query_cost: float = 0.00025
    hazard_prior: object = "elevated"
    original_reliability: float = 0.9
    preference_slope: float = 40.0
    more_pairs: int = 0  # further comparisons from MORE_PREFERENCE_PAIRS (experiment 21)
    vulnerability_levels: int = 2  # physical-vulnerability levels (experiment 28)
    # catalogue variants of experiment 28
    occupancy_levels: tuple = (0.8, 1.2)   # occupancy multipliers of the two occupancy levels (maximum 1.2)
    susceptibility_levels: object = None   # damage-fraction multipliers; None: linspace(0.85, 1.15, levels)
    restoration_levels: tuple = (0.75, 1.25)   # restoration-time multipliers
    severity_depths: object = None   # Riverside ground depths (m), with ``hazard_prior`` a vector of equal length
    damage_function: str = "benchmark"   # 'benchmark' or 'jrc_europe'

    @property
    def deadline(self) -> int:
        return 2 * int(self.lead_hours)

    @property
    def states(self) -> int:
        return self.severity_levels * self.footprints * 4 * self.vulnerability_levels * self.preference_types * 4

    @property
    def query_count(self) -> int:
        return 7 + self.district_gauges + int(self.extra_preference_pair) + int(self.more_pairs)


@dataclass
class ScaledModel(Portfolio):
    loss2d: np.ndarray = field(default=None)
    implementation_ticks: np.ndarray = field(default=None)


def severity_depths(n: int) -> np.ndarray:
    """Riverside peak depths (m) for ``n`` severity levels."""
    if n == 3:
        return SEVERITY_KNOTS_M.copy()
    return np.interp(np.linspace(0, 1, n), [0, 0.5, 1], SEVERITY_KNOTS_M)


def severity_prior(name_or_vector, n: int) -> np.ndarray:
    base = f._named(name_or_vector, f.HAZARD_PRIORS, 3)
    if n == 3:
        return base
    p = np.interp(np.linspace(0, 1, n), [0, 0.5, 1], base)
    return p / p.sum()


def preference_weights(n: int) -> np.ndarray:
    if n == 4:
        return f.PREFERENCE_WEIGHTS.copy()
    order = LATTICE_ORDER[n]
    lattice = np.array([(i, j, order - i - j) for i in range(order + 1) for j in range(order + 1 - i)], float)
    return (1 - LATTICE_SHRINK) * lattice / order + LATTICE_SHRINK / 3


def preference_prior(value, n: int) -> np.ndarray:
    if n == 4:
        return f._named(value, f.PREFERENCE_PRIORS, 4)
    if isinstance(value, str) and value == "diffuse":
        return np.full(n, 1.0 / n)
    p = np.asarray(value, dtype=float)
    if p.shape != (n,) or abs(p.sum() - 1) > 1e-9 or np.any(p < 0):
        raise ValueError("invalid preference prior")
    return p


def build_scaled(cfg: ScaledConfig = ScaledConfig(), with_features: bool = False) -> ScaledModel:
    """Build one member of the scaled family."""
    if (cfg.footprints < 2 or cfg.severity_levels < 2 or not 0 <= cfg.district_gauges <= 2
            or cfg.vulnerability_levels < 2):
        raise ValueError("unsupported scaled configuration")
    ns, nf, npref, nv = cfg.severity_levels, cfg.footprints, cfg.preference_types, cfg.vulnerability_levels
    idx = np.array(list(product(range(ns), range(nf), range(2), range(nv), range(2), range(npref),
                                range(2), range(2))))
    if cfg.severity_depths is None:
        hz, base_depth = severity_prior(cfg.hazard_prior, ns), severity_depths(ns)
    else:   # explicit severity catalogue, for example with a level without flooding
        hz, base_depth = np.asarray(cfg.hazard_prior, dtype=float), np.asarray(cfg.severity_depths, dtype=float)
        if base_depth.shape != (ns,) or hz.shape != (ns,) or abs(hz.sum() - 1) > 1e-9 or np.any(hz < 0):
            raise ValueError("severity depths and hazard prior must have one entry per severity level")
    occupancy = np.asarray(cfg.occupancy_levels, dtype=float)
    if occupancy.shape != (2,) or occupancy.max() > 1.2 + 1e-12 or len(cfg.restoration_levels) != 2:
        raise ValueError("unsupported occupancy or restoration levels (reservations assume at most 1.2)")
    pref = preference_prior(cfg.preference_prior, npref)
    weights = preference_weights(npref)
    factor = (0.5 ** 4 if nf == 2 else 0.125 / nf) * 2 / nv
    prior = hz[idx[:, 0]] * factor * pref[idx[:, 5]] / 4
    depth = base_depth[idx[:, 0], None] * np.array([1., 0.8, 0.6])
    footprint = np.linspace(-1, 1, nf) if nf != 2 else np.array([-1., 1.])
    depth += footprint[idx[:, 1], None] * np.array([-0.12, 0.04, 0.08])
    occupants = f.BASE_OCCUPANTS * occupancy[idx[:, 2, None]]
    social = idx[:, 6:8]
    population = occupants[:, :, None] * so.GROUP_SHARES
    strength = cfg.mechanism_strength
    rates = np.zeros((len(idx), 3, 2))
    rates[:, 1, 0] = .9
    rates[:, 2, 0] = .9
    rates[:, 1, 1] = .9 - strength * so.ACCESS_SHORTFALL[social[:, 0]]
    rates[:, 2, 1] = .9 - strength * so.ASSISTED_RESIDUAL_SHORTFALL
    delay = np.column_stack((np.zeros(len(idx)), strength * so.RECOVERY_ACCESS_DAYS[social[:, 1]]))
    actions, costs, completion = so.action_inventory()
    floor = f.FLOOR_HEIGHT_M + np.linspace(.1, -.1, nv)[idx[:, 3, None]]
    factors = (np.linspace(.85, 1.15, nv) if cfg.susceptibility_levels is None
               else np.asarray(cfg.susceptibility_levels, dtype=float))
    if factors.shape != (nv,):
        raise ValueError("one susceptibility factor per physical-vulnerability level is required")
    susceptibility = factors[idx[:, 3, None]]
    restoration = np.asarray(cfg.restoration_levels, dtype=float)[idx[:, 4, None]]
    pref_w = weights[idx[:, 5]]
    loss = np.empty((len(idx), len(actions)))
    raw = np.empty((len(idx), len(actions), 4)) if with_features else None
    damage_cache = {}
    knots, structure_curve, contents_curve = f.damage_curves(cfg.damage_function)
    for j, action in enumerate(actions):
        a = np.array(action)
        key = tuple(a[:, 0])
        defended = np.where((a[:, 0] == 1) & (depth <= 1.2), np.maximum(0, depth - .65), depth)
        internal = np.maximum(0, defended - floor)
        if key not in damage_cache:
            structure_fraction = np.minimum(1, susceptibility * np.interp(internal, knots, structure_curve))
            contents_fraction = np.minimum(1, susceptibility * np.interp(internal, knots, contents_curve))
            contents_fraction *= np.where(a[:, 0] == 2, 0.4, 1.)
            damage_cache[key] = (f.BUILDINGS * (f.STRUCTURE_VALUE_EUR * structure_fraction
                                                + f.CONTENTS_VALUE_EUR * contents_fraction)).sum(axis=1)
        damage = damage_cache[key]
        structure = np.minimum(1, susceptibility * np.interp(internal, knots, structure_curve))
        technical = np.where(internal > 0, restoration * (2 + 50 * structure + 10 * internal), 0.)
        uninhabitable = np.clip((internal - .05) / .45, 0, 1)
        evac_rate = rates[:, a[:, 1], :]
        displacement = (population * (uninhabitable[:, :, None] * (technical[:, :, None] + delay[:, None, :])
                                      + evac_rate * (1 - uninhabitable[:, :, None]) * .75)).sum(axis=1)
        exposed = (population * (1 - evac_rate) * (defended[:, :, None] >= .3)).sum(axis=1)
        normalized = np.column_stack((damage, displacement.sum(axis=1), exposed.sum(axis=1)))
        normalized[:, 0] += costs[j]
        normalized /= f.SCALES
        loss[:, j] = np.einsum("zq,zq->z", normalized, pref_w)
        if with_features:
            raw[:, j] = np.column_stack((damage, displacement.sum(axis=1), exposed.sum(axis=1),
                                         np.full(len(idx), costs[j])))
    r = cfg.original_reliability
    gauge = lambda d: 0.15 + 0.7 * np.clip((d - 0.23) / 1.54, 0, 1)
    physical = np.column_stack((gauge(depth[:, 0]), 0.15 + 0.7 * idx[:, 2], 0.15 + 0.7 * idx[:, 3] / (nv - 1),
                                0.15 + 0.7 * idx[:, 4]))
    difference = (f.PREFERENCE_PAIRS[:, 0] - f.PREFERENCE_PAIRS[:, 1]) / f.SCALES
    preference = expit(cfg.preference_slope * (pref_w @ difference.T))
    base = 0.5 + r * (np.column_stack((physical, preference)) - 0.5)
    p_social = .5 + cfg.social_query_reliability * (.1 + .4 * social[:, 0] + .4 * social[:, 1] - .5)
    columns = [base[:, :4], p_social[:, None], base[:, 4:]]
    names = list(so.QUERIES)
    for k in range(cfg.district_gauges):
        columns.append((0.5 + r * (gauge(depth[:, k + 1]) - 0.5))[:, None])
        names.append(GAUGE_NAMES[k])
    if cfg.extra_preference_pair:
        extra = (EXTRA_PREFERENCE_PAIR[0] - EXTRA_PREFERENCE_PAIR[1]) / f.SCALES
        columns.append((0.5 + r * (expit(cfg.preference_slope * (pref_w @ extra)) - 0.5))[:, None])
        names.append("economic_vs_displacement")
    if not 0 <= cfg.more_pairs <= len(MORE_PREFERENCE_PAIRS):
        raise ValueError("unsupported number of further comparisons")
    for k in range(cfg.more_pairs):
        extra = (MORE_PREFERENCE_PAIRS[k, 0] - MORE_PREFERENCE_PAIRS[k, 1]) / f.SCALES
        columns.append((0.5 + r * (expit(cfg.preference_slope * (pref_w @ extra)) - 0.5))[:, None])
        names.append(f"further_comparison_{k + 1}")
    likelihood = np.column_stack(columns)
    durations = np.array([1, 1, 2, 2, 1, 1, 1] + [1] * (len(names) - 7))
    feasible = np.arange(cfg.deadline + 1)[:, None] + completion <= cfg.deadline
    shape = (len(idx), cfg.deadline + 1, len(actions))
    features = np.broadcast_to(raw[:, None], shape + (4,)) if with_features else None
    return ScaledModel(prior, likelihood, np.broadcast_to(loss[:, None, :], shape), features, feasible,
                       durations, cfg.query_cost * durations, idx, actions, tuple(names), cfg,
                       loss2d=loss, implementation_ticks=completion)


def aggregate_scaled(model, inverse) -> ScaledModel:
    """Prior-conditional class means (sparse, no ``[z, t, a]`` copies)."""
    inverse = np.asarray(inverse, dtype=int)
    nc = int(inverse.max()) + 1
    n = len(model.prior)
    mass = np.bincount(inverse, weights=model.prior, minlength=nc)
    if not np.all(mass > 0):
        raise ValueError("empty class")
    weights = sparse.csr_matrix((model.prior, (inverse, np.arange(n))), shape=(nc, n))
    loss = np.asarray(weights @ model.loss2d) / mass[:, None]
    likelihood = np.asarray(weights @ model.likelihood) / mass[:, None]
    shape = (nc,) + model.feasible.shape
    return ScaledModel(mass, likelihood, np.broadcast_to(loss[:, None, :], shape), None, model.feasible,
                       model.durations, model.costs, np.arange(nc)[:, None], model.actions, model.queries,
                       model.config, loss2d=loss, implementation_ticks=getattr(model, "implementation_ticks", None))


class FastSolver:
    """Exact finite-history DP with batched commitment values."""

    def __init__(self, model, method: str = "optimal", time_limit: float | None = None):
        self.time_limit = time_limit
        self.started = perf_counter()
        self.prior = model.prior
        self.lik1 = np.ascontiguousarray(model.likelihood.T)
        self.lik0 = 1.0 - self.lik1
        self.loss = np.ascontiguousarray(model.loss2d if getattr(model, "loss2d", None) is not None
                                         else model.loss[:, 0, :])
        self.penalty = np.where(model.feasible, 0.0, np.inf)
        self.durations = [int(d) for d in model.durations]
        self.costs = [float(c) for c in model.costs]
        self.deadline = int(model.config.deadline)
        self.method = method
        self.root = (-1,) * len(self.durations)
        self.memo: dict = {}
        self.stops: dict = {}

    def time(self, h) -> int:
        return sum(d for d, y in zip(self.durations, h) if y != -1)

    def available(self, h, t):
        return [q for q, y in enumerate(h) if y == -1 and t + self.durations[q] <= self.deadline]

    def _stop_rows(self, keys, masses, times):
        values = masses @ self.loss
        for key, row, t in zip(keys, values, times):
            row = row + self.penalty[t]
            a = int(np.argmin(row))
            self.stops[key] = (float(row[a]), ("act", a))

    def solve(self, h=None, mass=None):
        """Optimal (or one-step) value and decision at history ``h``."""
        if h is None:
            h, mass = self.root, self.prior
        if h in self.memo:
            return self.memo[h]
        if mass is None:
            mass = self.mass(h)
        t = self.time(h)
        if h not in self.stops:
            self._stop_rows([h], mass[None, :], [t])
        ans = self.stops[h]
        queries = [] if self.method == "immediate" else self.available(h, t)
        if queries:
            keys, rows, times = [], [], []
            children = {}
            for q in queries:
                for y, lik in ((0, self.lik0[q]), (1, self.lik1[q])):
                    hh = h[:q] + (y,) + h[q + 1:]
                    children[(q, y)] = (hh, mass * lik)
                    if hh not in self.stops and (self.method == "one_step" or hh not in self.memo):
                        keys.append(hh)
                        rows.append(children[(q, y)][1])
                        times.append(t + self.durations[q])
            if keys:
                self._stop_rows(keys, np.vstack(rows), times)
            total = float(mass.sum())
            for q in queries:
                cost = total * self.costs[q]
                (h0, m0), (h1, m1) = children[(q, 0)], children[(q, 1)]
                if self.method == "one_step":
                    candidate = cost + self.stops[h0][0] + self.stops[h1][0]
                else:
                    candidate = cost + self.solve(h0, m0)[0] + self.solve(h1, m1)[0]
                if candidate < ans[0] - 1e-15:
                    ans = (candidate, ("query", q))
        self.memo[h] = ans
        if self.time_limit is not None and len(self.memo) % 512 == 0                 and perf_counter() - self.started > self.time_limit:
            raise TimeoutError(f"exact DP exceeded {self.time_limit} s")
        return ans

    def mass(self, h) -> np.ndarray:
        b = self.prior.copy()
        for q, y in enumerate(h):
            if y != -1:
                b *= self.lik1[q] if y else self.lik0[q]
        return b

    def decision(self, h, mass=None):
        """Decision at ``h`` (``optimal`` policies: after ``solve()`` at the root)."""
        return self.solve(h, mass)[1]


def evaluate(model, decide, root=None) -> dict:
    """Exact original-model value of a history policy ``decide(h, mass)``."""
    loss = model.loss2d if getattr(model, "loss2d", None) is not None else model.loss[:, 0, :]
    lik = model.likelihood
    durations = [int(d) for d in model.durations]
    q_count = len(durations)
    value = 0.0
    queries = 0.0
    visited = 0
    stack = [((-1,) * q_count, model.prior.copy(), 0)]
    while stack:
        h, mass, t = stack.pop()
        visited += 1
        kind, j = decide(h, mass)
        if kind == "act":
            if not model.feasible[t, j]:
                raise AssertionError("infeasible commitment")
            value += float(mass @ loss[:, j])
        else:
            if h[j] != -1 or t + durations[j] > model.config.deadline:
                raise AssertionError("unavailable query")
            total = float(mass.sum())
            value += total * float(model.costs[j])
            queries += total
            for y in (0, 1):
                hh = h[:j] + (y,) + h[j + 1:]
                stack.append((hh, mass * (lik[:, j] if y else 1 - lik[:, j]), t + durations[j]))
    return dict(value=value, expected_queries=queries, visited_histories=visited)


def evaluate_solver(model, solver) -> dict:
    """Original-model value of a solved ``FastSolver`` policy (on any model with the same graph)."""
    if solver.method == "optimal":
        solver.solve()
        return evaluate(model, lambda h, mass: solver.memo[h][1])
    return evaluate(model, lambda h, mass: solver.solve(h, mass)[1])


LADDER = (
    ScaledConfig(3, 2, 4, 0, False),
    ScaledConfig(5, 2, 4, 1, False),
    ScaledConfig(5, 3, 10, 2, False),
    ScaledConfig(7, 3, 10, 2, True),
    ScaledConfig(7, 3, 15, 2, True),
    ScaledConfig(9, 3, 21, 2, True),
    ScaledConfig(17, 3, 21, 2, True),
    ScaledConfig(33, 3, 21, 2, True),
    ScaledConfig(65, 3, 21, 2, True),
    ScaledConfig(129, 3, 21, 2, True),
    ScaledConfig(257, 3, 21, 2, True),
)
