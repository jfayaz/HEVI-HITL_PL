"""Bayes-covering endpoints, class-level certificates, query relaxation and guided refinement (Sections 3 and 4)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush
from itertools import combinations

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

INCLUDE_TOL = 1e-9     # LP value accepted as "a is a best response" (superset side: safe)
EXCLUDE_MARGIN = 1e-12  # strict-domination margin verified directly on all rows


# Public decision graph: reachable times, endpoints, query sets
def reachable_times(model) -> np.ndarray:
    """Boolean vector over commitment times reachable from the root."""
    deadline = int(model.config.deadline)
    times = {0}
    for duration in model.durations:
        times |= {t + int(duration) for t in tuple(times) if t + duration <= deadline}
    out = np.zeros(model.feasible.shape[0], dtype=bool)
    out[sorted(times)] = True
    return out


def endpoint_mask(model) -> np.ndarray:
    """Reachable feasible endpoints ``(t, a)`` as a ``[T, A]`` mask (endpoint set E)."""
    return model.feasible & reachable_times(model)[:, None]


def loss2d(model):
    """Time-invariant ``[z, a]`` score view, or ``None`` if scores depend on time."""
    if getattr(model, "loss2d", None) is not None:
        return model.loss2d
    loss = model.loss
    if loss.strides[1] == 0 or bool(np.all(loss == loss[:, :1, :])):
        return loss[:, 0, :]
    return None


def endpoint_columns(model, mask: np.ndarray, loss=None) -> np.ndarray:
    """Score matrix ``[z, m]`` with one column per distinct endpoint in ``mask``."""
    flat = loss2d(model) if loss is None else loss
    if flat is not None:
        return flat[:, np.flatnonzero(mask.any(axis=0))]
    return model.loss[:, mask]


def feasible_query_sets(model):
    """Maximal query sets whose total duration fits the deadline, and C_max."""
    q = len(model.queries)
    deadline = int(model.config.deadline)
    durations = np.asarray(model.durations, dtype=int)
    costs = np.asarray(model.costs, dtype=float)
    subsets = ((np.arange(1 << q)[:, None] >> np.arange(q)) & 1).astype(bool)
    feasible = subsets @ durations <= deadline
    cmax = float(np.max(subsets[feasible] @ costs))
    sets = subsets[feasible]
    maximal = []
    for row in sets:
        free = ~row
        extendable = any(row @ durations + durations[j] <= deadline for j in np.flatnonzero(free))
        if not extendable:
            maximal.append(row)
    return np.array(maximal, dtype=bool), cmax


def divergence(tv: np.ndarray, sets: np.ndarray) -> np.ndarray:
    """Root divergence probability upsilon_z from per-query TV distances ``[z, Q]``."""
    keep = 1.0 - np.clip(tv, 0.0, 1.0)
    worst = np.ones(len(tv))
    for row in sets:
        worst = np.minimum(worst, np.prod(keep[:, row], axis=1))
    return 1.0 - worst


def divergence_recursion(model, tv: np.ndarray) -> np.ndarray:
    """Backward recursion for the divergence probability; checks ``divergence`` in the tests."""
    qn = len(model.queries)

    def available(mask):
        t = sum(int(model.durations[q]) for q in range(qn) if not (mask >> q & 1))
        return [q for q in range(qn) if mask >> q & 1 and t + model.durations[q] <= model.config.deadline]

    @lru_cache(None)
    def rec(mask):
        ans = np.zeros(len(tv))
        for q in available(mask):
            ans = np.maximum(ans, tv[:, q] + (1 - tv[:, q]) * rec(mask ^ (1 << q)))
        return ans

    return rec((1 << qn) - 1)


# Undominated endpoints
def weak_dominance_survivors(scores: np.ndarray, sample: int = 2048) -> np.ndarray:
    """Columns not weakly pointwise dominated (strict partial order, index tie-break)."""
    n, m = scores.shape
    rows = np.unique(np.linspace(0, n - 1, min(n, sample)).astype(int))
    rows = np.union1d(rows, np.argmin(scores, axis=0))
    probe = scores[rows]
    alive = np.ones(m, dtype=bool)
    for a in range(m):
        candidates = np.flatnonzero(np.all(probe <= probe[:, a:a + 1], axis=0))
        for b in candidates:
            if b == a:
                continue
            if np.all(scores[:, b] <= scores[:, a]) and (b < a or np.any(scores[:, b] < scores[:, a])):
                alive[a] = False
                break
    return alive


def _best_response_somewhere(sub: np.ndarray, local: int, row_gap: np.ndarray,
                             start_rows: int = 32, add_rows: int = 32):
    """Whether column ``local`` of ``sub`` is a best response to some belief."""
    n, m_all = sub.shape
    others = np.delete(np.arange(m_all), local)
    m = len(others)
    if m == 0 or np.any(row_gap >= 0):
        return True, 0
    target = sub[:, local]
    order = np.argsort(-row_gap, kind="stable")
    active = list(order[:min(n, start_rows)])
    in_active = np.zeros(n, dtype=bool)
    in_active[active] = True
    lps = 0
    while True:
        gaps = sub[np.ix_(active, others)] - target[active, None]
        k = len(active)
        c = np.zeros(k + 1)
        c[-1] = -1.0
        a_ub = np.hstack((-gaps.T, np.ones((m, 1))))
        a_eq = np.zeros((1, k + 1))
        a_eq[0, :k] = 1.0
        res = linprog(c, A_ub=a_ub, b_ub=np.zeros(m), A_eq=a_eq, b_eq=[1.0],
                      bounds=[(0, None)] * k + [(None, None)], method="highs")
        lps += 1
        if res.status != 0:
            return True, lps  # inconclusive: keep (a superset of U is always valid)
        if -res.fun >= -INCLUDE_TOL:
            return True, lps
        lam = np.maximum(-res.ineqlin.marginals, 0.0)
        if lam.sum() <= 0:
            return True, lps
        weights = np.zeros(m_all)
        weights[others] = lam / lam.sum()
        advantage = sub @ weights - target
        if advantage.max() < -EXCLUDE_MARGIN:
            return False, lps
        violated = np.flatnonzero(~in_active & (advantage >= -EXCLUDE_MARGIN))
        if len(violated) == 0:
            violated = np.flatnonzero(~in_active)
            if len(violated) == 0:
                return True, lps
        pick = violated[np.argsort(-advantage[violated], kind="stable")[:add_rows]]
        active.extend(pick.tolist())
        in_active[pick] = True


def undominated_actions(scores: np.ndarray):
    """Undominated subset of the columns of ``scores`` (one feasible set)."""
    survivors = weak_dominance_survivors(scores)
    idx = np.flatnonzero(survivors)
    sub = np.ascontiguousarray(scores[:, idx])
    keep = np.zeros(scores.shape[1], dtype=bool)
    if len(idx) == 1:
        keep[idx] = True
        return keep, dict(columns=int(scores.shape[1]), weak_survivors=1, vertex_optimal=1,
                          lp_candidates=0, lp_solves=0, undominated=1)
    part = np.argpartition(sub, 1, axis=1)[:, :2]
    rows = np.arange(len(sub))
    first = sub[rows, part[:, 0]]
    second = sub[rows, part[:, 1]]
    swap = second < first
    smallest = np.where(swap, second, first)
    runner = np.where(swap, first, second)
    arg_small = np.where(swap, part[:, 1], part[:, 0])
    vertex = np.zeros(len(idx), dtype=bool)
    vertex[np.unique(np.flatnonzero((sub <= smallest[:, None]).ravel()) % len(idx))] = True
    keep[idx[vertex]] = True
    lps = 0
    for local in np.flatnonzero(~vertex):
        best_other = np.where(arg_small == local, runner, smallest)
        member, count = _best_response_somewhere(sub, int(local), best_other - sub[:, local])
        lps += count
        keep[idx[local]] = member
    stats = dict(columns=int(scores.shape[1]), weak_survivors=int(survivors.sum()),
                 vertex_optimal=int(vertex.sum()), lp_candidates=int((~vertex).sum()),
                 lp_solves=int(lps), undominated=int(keep.sum()))
    return keep, stats


def undominated_endpoints(model, loss=None):
    """Undominated endpoint mask ``U`` ``[T, A]`` over reachable feasible endpoints."""
    mask = endpoint_mask(model)
    flat = loss2d(model) if loss is None else loss
    out = np.zeros_like(mask)
    cache = {}
    stats = []
    for t in np.flatnonzero(mask.any(axis=1)):
        feasible = np.flatnonzero(mask[t])
        key = feasible.tobytes() if flat is not None else (t, feasible.tobytes())
        if key not in cache:
            scores = flat[:, feasible] if flat is not None else model.loss[:, t, feasible]
            keep, st = undominated_actions(np.ascontiguousarray(scores))
            cache[key] = keep
            stats.append(dict(time=int(t), **st))
        out[t, feasible[cache[key]]] = True
    return out, stats


def policy_endpoints(model, policy) -> np.ndarray:
    """Endpoints a history policy commits to on every history consistent with its own queries."""
    out = np.zeros_like(model.feasible)
    seen = set()
    stack = [tuple([-1] * len(model.queries))]
    while stack:
        h = stack.pop()
        if h in seen:
            continue
        seen.add(h)
        kind, j = policy.solve(h)[1]
        t = sum(int(model.durations[q]) for q, y in enumerate(h) if y != -1)
        if kind == "act":
            out[t, j] = True
        else:
            for y in (0, 1):
                hh = list(h)
                hh[j] = y
                stack.append(tuple(hh))
    return out


# Bound terms and certificates
@dataclass
class Terms:
    """Per-configuration terms for one endpoint set S."""
    lo: np.ndarray      # min_{xi in S} J(z,xi) - R~(g(z),xi)
    hi: np.ndarray      # max_{xi in S} ...
    span: np.ndarray    # M_c^S per class (range of R~ over S plus C_max)

    @property
    def eps(self):
        return 0.5 * (self.hi - self.lo)

    @property
    def mid(self):
        return 0.5 * (self.hi + self.lo)


def bound_terms(J: np.ndarray, R: np.ndarray, inverse: np.ndarray, cmax: float,
                chunk: int = 20000) -> Terms:
    """Terms over one set of endpoint columns; ``J`` ``[z, m]``, ``R`` ``[c, m]``."""
    n = len(J)
    lo = np.empty(n)
    hi = np.empty(n)
    for start in range(0, n, chunk):
        sl = slice(start, start + chunk)
        diff = J[sl] - R[inverse[sl]]
        lo[sl] = diff.min(axis=1)
        hi[sl] = diff.max(axis=1)
    span = np.ptp(R, axis=1) + cmax
    return Terms(lo, hi, span)


def answer_vector_tv(likelihood: np.ndarray, prior: np.ndarray, inverse: np.ndarray,
                     reduced_likelihood: np.ndarray, chunk: int = 4096):
    """Total-variation distances between full answer-vector laws."""
    n, q = likelihood.shape
    answers = ((np.arange(1 << q)[:, None] >> np.arange(q)) & 1).astype(float)
    # floor of 1e-300 avoids -inf * 0 for answer probabilities of exactly 0 or 1
    floor = np.log(1e-300)
    with np.errstate(divide="ignore"):
        log1 = np.where(likelihood > 0, np.log(likelihood), floor)
        log0 = np.where(likelihood < 1, np.log1p(-likelihood), floor)
        rlog1 = np.where(reduced_likelihood > 0, np.log(reduced_likelihood), floor)
        rlog0 = np.where(reduced_likelihood < 1, np.log1p(-reduced_likelihood), floor)
    order = np.argsort(inverse, kind="stable")
    labels = inverse[order]
    mass = np.bincount(inverse, weights=prior, minlength=len(reduced_likelihood))
    tv_z = np.empty(n)
    tv_c = np.zeros(len(reduced_likelihood))
    carry_label = -1
    carry = None
    for start in range(0, n, chunk):
        rows = order[start:start + chunk]
        lab = labels[start:start + chunk]
        probs = np.exp(log1[rows] @ answers.T + log0[rows] @ (1 - answers).T)
        cls, first = np.unique(lab, return_index=True)
        ref = np.exp(rlog1[cls] @ answers.T + rlog0[cls] @ (1 - answers).T)
        local = np.searchsorted(cls, lab)
        tv_z[rows] = 0.5 * np.abs(probs - ref[local]).sum(axis=1)
        partial = np.add.reduceat(probs * prior[rows, None], first, axis=0)
        if carry is not None and cls[0] == carry_label:
            partial[0] += carry
        elif carry is not None:
            raise RuntimeError("unsorted class stream")
        carry = None
        last_open = start + chunk < n and labels[min(start + chunk, n - 1)] == cls[-1]
        done = len(cls) - 1 if last_open else len(cls)
        for j in range(done):
            tv_c[cls[j]] = 0.5 * np.abs(partial[j] / mass[cls[j]] - ref[j]).sum()
        if last_open:
            carry_label = cls[-1]
            carry = partial[-1]
    return tv_z, tv_c


def class_mean(values: np.ndarray, prior: np.ndarray, inverse: np.ndarray, n_classes: int) -> np.ndarray:
    """Prior-conditional class means of the rows of ``values`` (sparse product)."""
    weights = sparse.csr_matrix((prior, (inverse, np.arange(len(prior)))), shape=(n_classes, len(prior)))
    mass = np.asarray(weights.sum(axis=1)).ravel()
    return np.asarray(weights @ values) / mass[:, None]


@dataclass
class ClassTerms:
    """Class-level quantities for one endpoint set S (paper, Proposition 2)."""
    lower: np.ndarray   # a_c^-(S): lower bound on sum_{z in c} b0(z) E_z[J - R~] over policies in S
    upper: np.ndarray   # a_c^+(S)
    span: np.ndarray    # M_c^S


def class_terms(t: Terms, prior: np.ndarray, inverse: np.ndarray, tv_z: np.ndarray, n_classes: int) -> ClassTerms:
    """Combine the pathwise and the second-order (prior-mean representative) bounds per class."""
    lo = np.bincount(inverse, weights=prior * t.lo, minlength=n_classes)
    hi = np.bincount(inverse, weights=prior * t.hi, minlength=n_classes)
    second = np.bincount(inverse, weights=prior * t.eps * 2 * np.minimum(tv_z, 1.0), minlength=n_classes)
    return ClassTerms(np.maximum(lo, -second), np.minimum(hi, second), t.span)


def refined_certificate(model, reduced, inverse, original_value: float, surrogate_value: float,
                        U: np.ndarray, U_tilde: np.ndarray | None = None, gamma: float = 0.0,
                        lower_bound: float | None = None, sets=None, cmax=None,
                        J_all=None, J_U=None, class_level: bool = True, prior_mean: bool = True) -> dict:
    """Certificates of a surrogate policy: Proposition 1 (prop1_*), covering set (U_*), Proposition 2 (refined_*)."""
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    inverse = np.asarray(inverse)
    prior = model.prior
    nc = len(reduced.prior)
    if sets is None or cmax is None:
        sets, cmax = feasible_query_sets(model)
    tv = np.abs(model.likelihood - reduced.likelihood[inverse])
    ups = divergence(tv, sets)
    E_mask = endpoint_mask(model)
    out = {}

    def terms(mask, J=None):
        J = endpoint_columns(model, mask) if J is None else J
        R = endpoint_columns(reduced, mask)
        if prior_mean:
            mean = class_mean(J, prior, inverse, nc)
            if not np.allclose(mean, R, rtol=0, atol=1e-10):
                raise ValueError("surrogate scores are not prior-conditional class means")
        return bound_terms(J, R, inverse, cmax)

    t_all = terms(E_mask, J_all)
    t_U = terms(U, J_U)
    E_all = float(prior @ (t_all.eps + t_all.span[inverse] * ups))
    E_U = float(prior @ (t_U.eps + t_U.span[inverse] * ups))
    delta_all = float(original_value - prior @ t_all.mid - surrogate_value)
    delta_U = float(original_value - prior @ t_U.mid - surrogate_value)
    raw = float(original_value - surrogate_value)
    out.update(prop1_error=E_all, prop1_uniform=2 * E_all + gamma, prop1_postsolve=E_all + gamma + delta_all,
               prop1_delta=delta_all, U_error=E_U, U_delta=delta_U, U_postsolve=E_U + gamma + delta_U,
               original_policy_value=float(original_value), surrogate_policy_value=float(surrogate_value),
               raw_discrepancy=raw, mean_divergence=float(prior @ ups), gamma=float(gamma))
    lower_U = float(prior @ t_U.mid + surrogate_value - gamma - E_U)
    if class_level:
        tv_z, tv_c = answer_vector_tv(model.likelihood, prior, inverse, reduced.likelihood)
        T_z = np.minimum(ups, tv_z)
        tv_star = np.minimum(tv_c, np.bincount(inverse, weights=prior * T_z, minlength=nc) / reduced.prior)
        c_U = class_terms(t_U, prior, inverse, T_z, nc)
        div_U = float(reduced.prior @ (c_U.span * tv_star))
        pi_star_side = float(-c_U.lower.sum() + div_U)
        out.update(refined_postsolve=raw + gamma + pi_star_side, refined_pi_star_term=pi_star_side,
                   class_divergence=float(reduced.prior @ tv_star), refined_divergence_term=div_U,
                   refined_terminal_term=float(-c_U.lower.sum()))
        lower = surrogate_value - gamma - pi_star_side
    else:
        lower = lower_U
    if lower_bound is not None:
        out["lb_binds"] = bool(lower_bound > lower)
        out["lb_binds_U"] = bool(lower_bound > lower_U)
        lower = max(lower, float(lower_bound))
        out["U_lb"] = float(original_value - max(lower_U, float(lower_bound)))
    out["refined_lb"] = float(original_value - lower)
    if U_tilde is not None:
        t_Ut = terms(U_tilde)
        asym = float(prior @ (t_Ut.hi - t_U.lo + (t_U.span + t_Ut.span)[inverse] * ups)) + gamma
        t_V = terms(U | U_tilde)
        E_V = float(prior @ (t_V.eps + t_V.span[inverse] * ups))
        out.update(U_uniform=asym, U_uniform_symmetric=2 * E_V + gamma, union_error=E_V)
        if class_level:
            c_Ut = class_terms(t_Ut, prior, inverse, T_z, nc)
            out["refined_uniform"] = float(c_Ut.upper.sum() - c_U.lower.sum()
                                           + reduced.prior @ ((c_U.span + c_Ut.span) * tv_star)) + gamma
    return out


def class_constants(model, reduced, inverse, U: np.ndarray, U_tilde: np.ndarray, sets=None, cmax=None,
                    J_U=None) -> np.ndarray:
    """Per-class constants ``chi_c`` of the pre-solve class-level certificate (forecast-free form)."""
    prior = model.prior
    nc = len(reduced.prior)
    if sets is None or cmax is None:
        sets, cmax = feasible_query_sets(model)
    tv = np.abs(model.likelihood - reduced.likelihood[inverse])
    ups = divergence(tv, sets)
    tv_z, tv_c = answer_vector_tv(model.likelihood, prior, inverse, reduced.likelihood)
    T_z = np.minimum(ups, tv_z)
    tv_star = np.minimum(tv_c, np.bincount(inverse, weights=prior * T_z, minlength=nc) / reduced.prior)
    J_U = endpoint_columns(model, U) if J_U is None else J_U
    t_U = bound_terms(J_U, endpoint_columns(reduced, U), inverse, cmax)
    t_Ut = bound_terms(endpoint_columns(model, U_tilde), endpoint_columns(reduced, U_tilde), inverse, cmax)
    c_U = class_terms(t_U, prior, inverse, T_z, nc)
    c_Ut = class_terms(t_Ut, prior, inverse, T_z, nc)
    return (c_Ut.upper - c_U.lower) / reduced.prior + (c_U.span + c_Ut.span) * tv_star


def perfect_information_bound(model, U=None) -> float:
    """sum_z b0(z) min over reachable feasible endpoints of J(z, xi) (a lower bound on L*)."""
    mask = endpoint_mask(model) if U is None else U
    J = endpoint_columns(model, mask)
    return float(model.prior @ J.min(axis=1))


def differing_queries(likelihood: np.ndarray, inverse: np.ndarray, atol: float = 1e-12) -> tuple:
    """Queries whose response law is not constant within some class of the partition ``inverse``."""
    lik = np.asarray(likelihood, dtype=float)
    inverse = np.asarray(inverse, dtype=int)
    nc = int(inverse.max()) + 1
    lo = np.full((nc, lik.shape[1]), np.inf)
    hi = np.full((nc, lik.shape[1]), -np.inf)
    np.minimum.at(lo, inverse, lik)
    np.maximum.at(hi, inverse, lik)
    return tuple(int(q) for q in np.flatnonzero((hi - lo).max(axis=0) > atol))


def query_relaxation_bound(model, inverse, queries=None) -> float:
    """Lower bound on L* from the answers of ``queries`` revealed at the warning without cost."""
    from itertools import product
    from types import SimpleNamespace

    from .moments import MomentSolver

    lik = np.asarray(model.likelihood, dtype=float)
    inverse = np.asarray(inverse, dtype=int)
    n, n_queries = lik.shape
    nc = int(inverse.max()) + 1
    queries = differing_queries(lik, inverse) if queries is None else tuple(sorted(int(q) for q in queries))
    rest = [q for q in range(n_queries) if q not in queries]
    if rest and differing_queries(lik[:, rest], inverse):
        raise ValueError("a query outside `queries` distinguishes the members of a class")
    flat = loss2d(model)
    durations = np.asarray(model.durations)
    costs = np.asarray(model.costs, dtype=float)
    total = 0.0
    for answers in product((0, 1), repeat=len(queries)):
        mass = np.asarray(model.prior, dtype=float).copy()
        for q, y in zip(queries, answers):
            mass = mass * (lik[:, q] if y else 1.0 - lik[:, q])
        class_mass = np.bincount(inverse, weights=mass, minlength=nc)
        live = class_mass > 0
        if not live.any():
            continue
        if flat is not None:
            weights = sparse.csr_matrix((mass, (inverse, np.arange(n))), shape=(nc, n))
            scores = np.asarray(weights @ flat)[live] / class_mass[live, None]
            laws = (np.asarray(weights @ lik[:, rest])[live] / class_mass[live, None] if rest
                    else np.zeros((int(live.sum()), 0)))
            if not rest:
                feasible_now = np.asarray(model.feasible, dtype=bool)[0]
                total += float(np.min((class_mass[live] @ scores)[feasible_now]))
                continue
            relaxed = SimpleNamespace(prior=class_mass[live], likelihood=laws, loss2d=scores,
                                      feasible=model.feasible, durations=durations[rest], costs=costs[rest],
                                      config=model.config)
            total += MomentSolver(relaxed).solve()[0]
        else:
            from dataclasses import replace

            from .model import PortfolioSolver
            from .reduction import aggregate_partition

            members = live[inverse]
            dummy = lik.copy()
            dummy[:, list(queries)] = 0.5
            free = costs.copy()
            free[list(queries)] = 0.0
            relabel = np.cumsum(live) - 1
            sub = replace(model, prior=mass[members], likelihood=dummy[members], loss=model.loss[members],
                          features=model.features[members], indices=model.indices[members], costs=free)
            total += float(PortfolioSolver(aggregate_partition(sub, relabel[inverse[members]])).solve()[0])
    return float(total)


# Certificate-guided refinement
class GuidedRefiner:
    """Certificate-guided hierarchical refinement (nested partitions)."""

    def __init__(self, model, U: np.ndarray | None = None, J_U: np.ndarray | None = None,
                 sets=None, cmax=None, split: str | None = None):
        self.split = DEFAULT_SPLIT if split is None else split
        if self.split not in ("median", "2means"):
            raise ValueError("split must be 'median' or '2means'")
        if J_U is None:
            if U is None:
                U, _ = undominated_endpoints(model)
            J_U = endpoint_columns(model, U)
        if sets is None or cmax is None:
            sets, cmax = feasible_query_sets(model)
        self.prior = model.prior
        self.lik = model.likelihood
        self.sets = sets
        self.cmax = cmax
        self.contrasts = J_U - J_U[:, :1]
        self.heap = []
        self.leaves = {}
        self.serial = 0
        self.trace = []
        self._insert(np.arange(len(self.prior)))

    def _evaluate(self, members):
        w = self.prior[members]
        mass = w.sum()
        block = self.contrasts[members]
        rbar = (w @ block) / mass
        diff = block - rbar
        eps = 0.5 * (diff.max(axis=1) - diff.min(axis=1))
        span = float(np.ptp(rbar)) + self.cmax
        kbar = (w @ self.lik[members]) / mass
        ups = divergence(np.abs(self.lik[members] - kbar), self.sets)
        return float(w @ (eps + span * ups)), span

    def _insert(self, members):
        uid = self.serial
        self.serial += 1
        contribution, span = self._evaluate(members)
        self.leaves[uid] = (members, contribution, span)
        if len(members) > 1:
            heappush(self.heap, (-contribution, uid))

    def presolve_error(self) -> float:
        """Current sum of class contributions (the pre-solve E_U of this partition)."""
        return float(sum(leaf[1] for leaf in self.leaves.values()))

    def refine_to(self, k: int) -> np.ndarray:
        n = len(self.prior)
        k = min(n, int(k))
        while len(self.leaves) < k:
            _, uid = heappop(self.heap)
            members, contribution, span = self.leaves.pop(uid)
            scores = np.concatenate((np.ptp(self.contrasts[members], axis=0) * 0.5,
                                     np.ptp(self.lik[members], axis=0) * span))
            j = int(np.argmax(scores))
            if scores[j] <= 0:
                values = np.zeros(len(members))
            elif j < self.contrasts.shape[1]:
                values = self.contrasts[members, j]
            else:
                values = self.lik[members, j - self.contrasts.shape[1]]
            ordered = members[np.lexsort((members, values))]
            cut = len(ordered) // 2
            first, second = ordered[:cut], ordered[cut:]
            if self.split == "2means" and scores[j] > 0 and len(members) > 2:
                first, second = self._two_means(first, second, span)
            self.trace.append((len(self.leaves) + 1, contribution, j))
            self._insert(first)
            self._insert(second)
        inverse = np.empty(n, dtype=int)
        for label, uid in enumerate(sorted(self.leaves)):
            inverse[self.leaves[uid][0]] = label
        return inverse

    def _two_means(self, first, second, span):
        """Prior-weighted Lloyd iterations on the weighted signature, from the median halving."""
        members = np.concatenate((first, second))
        X = np.column_stack((0.5 * self.contrasts[members], span * self.lik[members]))
        return two_means_halves(X, self.prior[members], first, second)


DEFAULT_SPLIT = "2means"


def lloyd_two_means(X: np.ndarray, w: np.ndarray, side: np.ndarray, iterations: int = 100):
    """Prior-weighted Lloyd iterations for two groups from the assignment ``side``; None if one empties."""
    for _ in range(iterations):
        if side.all() or not side.any():
            return None
        c0 = (w[~side] @ X[~side]) / w[~side].sum()
        c1 = (w[side] @ X[side]) / w[side].sum()
        new = ((X - c1) ** 2).sum(axis=1) < ((X - c0) ** 2).sum(axis=1)
        if np.array_equal(new, side):
            break
        side = new
    if side.all() or not side.any():
        return None
    return side


def two_means_halves(X: np.ndarray, w: np.ndarray, first: np.ndarray, second: np.ndarray):
    """Refine the halving ``(first, second)`` (rows of ``X`` in that order) by ``lloyd_two_means``."""
    members = np.concatenate((first, second))
    side = np.concatenate((np.zeros(len(first), dtype=bool), np.ones(len(second), dtype=bool)))
    side = lloyd_two_means(X, w, side)
    if side is None:
        return first, second
    return np.sort(members[~side]), np.sort(members[side])


def guided_hierarchy(model, class_counts, U: np.ndarray | None = None, J_U: np.ndarray | None = None,
                     sets=None, cmax=None, split: str | None = None) -> dict:
    """Nested certificate-guided partitions at the requested class counts."""
    refiner = GuidedRefiner(model, U, J_U, sets, cmax, split=split)
    return {k: refiner.refine_to(k) for k in sorted(set(min(len(model.prior), int(k)) for k in class_counts))}


def marginal_partition(indices: np.ndarray, coordinates) -> np.ndarray:
    """Merge configurations that differ only in ``coordinates`` (marginalise them)."""
    keep = [j for j in range(indices.shape[1]) if j not in set(coordinates)]
    _, inverse = np.unique(indices[:, keep], axis=0, return_inverse=True)
    return inverse.ravel().astype(int)


def coordinate_subsets(n_coordinates: int, pairs: bool = True):
    """Singletons then (optionally) pairs of latent coordinates."""
    out = [(j,) for j in range(n_coordinates)]
    if pairs:
        out += list(combinations(range(n_coordinates), 2))
    return out
