"""Experiment 25: numerical check of the exact reduction, Propositions 1 and 2 and Eq. 19 on random models.

Appendix A; Supplementary Section S13.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from hevi_hitl import certificates as C
from hevi_hitl.model import Portfolio, PortfolioSolver
from hevi_hitl.reduction import aggregate_partition

RESULTS = ROOT / "results"
SEED = 2035
TOL = 1e-10


def random_model(rng, exact: bool):
    n = int(rng.integers(2, 9))
    q = int(rng.integers(1, 4))
    a = int(rng.integers(2, 5))
    deadline = int(rng.integers(2, 6))
    durations = rng.integers(1, 3, size=q)
    completion = rng.integers(0, deadline + 1, size=a)
    completion[0] = 0  # one portfolio (no intervention) is always feasible
    feasible = np.arange(deadline + 1)[:, None] + completion[None, :] <= deadline
    prior = rng.dirichlet(np.ones(n))
    classes = int(rng.integers(1, n + 1))
    inverse = np.concatenate((np.arange(classes), rng.integers(0, classes, size=n - classes)))
    rng.shuffle(inverse)
    if exact:  # equal contrasts and laws within classes, plus a configuration-specific offset
        base = rng.uniform(0, 1, size=(classes, a))
        loss2d = base[inverse] + rng.uniform(-0.5, 0.5, size=n)[:, None]
        lik = rng.uniform(0, 1, size=(classes, q))[inverse]
    else:
        loss2d = rng.uniform(0, 1, size=(n, a))
        lik = rng.uniform(0, 1, size=(n, q))
    mask = rng.random(lik.shape) < 1 / 3
    lik = np.where(mask, rng.integers(0, 2, size=lik.shape).astype(float), lik)
    loss = np.repeat(loss2d[:, None, :], deadline + 1, axis=1)
    costs = rng.uniform(0, 0.05, size=q) * durations
    cfg = SimpleNamespace(deadline=deadline)
    model = Portfolio(prior, lik, loss, np.zeros(loss.shape + (4,)), feasible, durations, costs,
                      np.arange(n)[:, None], tuple(range(a)), tuple(range(q)), cfg)
    if exact:  # laws must be identical within classes after the 0/1 masking
        first = {c: np.flatnonzero(inverse == c)[0] for c in range(classes)}
        model.likelihood = np.vstack([model.likelihood[first[c]] for c in inverse])
    return model, inverse


class RandomPolicy:
    """Deterministic history policy with decisions drawn once per history (lazily, seeded)."""

    def __init__(self, model, rng):
        self.s = PortfolioSolver(model)
        self.root = self.s.root
        self.rng = rng
        self.memo = {}

    def solve(self, h=None):
        h = self.root if h is None else h
        if h not in self.memo:
            t = self.s.time(h)
            options = [("act", int(j)) for j in np.flatnonzero(self.s.m.feasible[t])]
            options += [("query", q) for q in self.s.available(h)]
            self.memo[h] = (float("nan"), options[int(self.rng.integers(len(options)))])
        return self.memo[h]


def history_laws(model, policy):
    """Terminal-history probabilities per configuration: dict history -> vector over z."""
    out = {}
    stack = [(policy.root, np.ones(len(model.prior)))]
    while stack:
        h, p = stack.pop()
        kind, j = policy.solve(h)[1]
        if kind == "act":
            out[h] = out.get(h, 0) + p
            continue
        for y in (0, 1):
            hh = list(h)
            hh[j] = y
            stack.append((tuple(hh), p * (model.likelihood[:, j] if y else 1 - model.likelihood[:, j])))
    return out


def audit_instance(seed_and_count):
    seed, policies = seed_and_count
    rng = np.random.default_rng([SEED, seed])
    exact = seed % 4 == 0
    model, inverse = random_model(rng, exact)
    nc = int(inverse.max() + 1)
    counts = {k: 0 for k in ("E", "P1", "D", "Dmix", "P2", "C", "R")}
    checks = {k: 0 for k in counts}
    slack = {k: np.inf for k in counts}

    def record(name, gap):
        checks[name] += 1
        if not np.isfinite(gap):  # a NaN or infinite slack counts as a violation
            counts[name] += 1
            return
        slack[name] = min(slack[name], gap)
        if gap < -TOL:
            counts[name] += 1

    mean = aggregate_partition(model, inverse)
    rnd = Portfolio(mean.prior, rng.uniform(0, 1, size=mean.likelihood.shape),
                    np.repeat(rng.uniform(0, 1, size=(nc, len(model.actions)))[:, None, :], model.config.deadline + 1, axis=1),
                    mean.features, mean.feasible, mean.durations, mean.costs, mean.indices, mean.actions, mean.queries,
                    mean.config)
    sets, cmax = C.feasible_query_sets(model)
    E = C.endpoint_mask(model)
    J = C.endpoint_columns(model, E)
    prior = model.prior
    omega = prior / np.bincount(inverse, weights=prior, minlength=nc)[inverse]

    def prop1_terms(sur):
        R = C.endpoint_columns(sur, E)
        diff = J - R[inverse]
        lo, hi = diff.min(axis=1), diff.max(axis=1)
        ups = C.divergence(np.abs(model.likelihood - sur.likelihood[inverse]), sets)
        span = np.ptp(R, axis=1) + cmax
        return float(prior @ (0.5 * (hi - lo) + span[inverse] * ups)), float(prior @ (0.5 * (hi + lo))), ups, R

    t1 = {name: prop1_terms(sur) for name, sur in (("mean", mean), ("random", rnd))}
    # Proposition 2 terms (prior-conditional means)
    E_mean, _, ups, R = t1["mean"]
    e = J - R[inverse]
    eps = 0.5 * (e.max(axis=1) - e.min(axis=1))
    tv_z, tv_c = C.answer_vector_tv(model.likelihood, prior, inverse, mean.likelihood)
    hat = np.minimum(ups, tv_z)
    star = np.minimum(tv_c, np.bincount(inverse, weights=prior * hat, minlength=nc) / mean.prior)
    span = np.ptp(R, axis=1) + cmax
    T_minus = np.minimum(-np.bincount(inverse, weights=prior * e.min(axis=1), minlength=nc),
                         2 * np.bincount(inverse, weights=prior * eps * hat, minlength=nc))
    T_plus = np.minimum(np.bincount(inverse, weights=prior * e.max(axis=1), minlength=nc),
                        2 * np.bincount(inverse, weights=prior * eps * hat, minlength=nc))
    div = float(mean.prior @ (span * star))
    lower2, upper2 = -float(T_minus.sum()) - div, float(T_plus.sum()) + div

    opt, sopt = PortfolioSolver(model), PortfolioSolver(mean)
    opt.solve()
    sopt.solve()
    pols = [opt, sopt] + [RandomPolicy(model, np.random.default_rng([SEED, seed, k])) for k in range(policies)]
    for pi in pols:
        L = PortfolioSolver(model).evaluate(pi)[0]
        for name, sur in (("mean", mean), ("random", rnd)):
            E_v, offset, ups_v, _ = t1[name]
            Ls = PortfolioSolver(sur).evaluate(pi)[0]
            record("P1", E_v - abs(L - offset - Ls))
        Lm = PortfolioSolver(mean).evaluate(pi)[0]
        record("P2", (L - Lm) - lower2)
        record("P2", upper2 - (L - Lm))
        orig_laws = history_laws(model, pi)
        sur_laws = history_laws(mean, pi)
        keys = set(orig_laws) | set(sur_laws)
        P = np.array([orig_laws.get(k, np.zeros(len(prior))) for k in keys])
        Q = np.array([sur_laws.get(k, np.zeros(nc)) for k in keys])
        tv = 0.5 * np.abs(P - Q[:, inverse]).sum(axis=0)
        for z in range(len(prior)):
            record("D", ups[z] - tv[z])
        mix = np.zeros((len(keys), nc))
        np.add.at(mix.T, inverse, (P * omega).T)
        tvc = 0.5 * np.abs(mix - Q).sum(axis=0)
        for c in range(nc):
            record("Dmix", star[c] - tvc[c])
        if exact:
            loss2d = model.loss[:, 0, :]
            B = loss2d[:, 0]  # portfolio 0 is feasible at every time
            first = np.array([np.flatnonzero(inverse == c)[0] for c in range(nc)])
            centred = Portfolio(mean.prior, mean.likelihood,
                                np.repeat((loss2d - B[:, None])[first][:, None, :], model.config.deadline + 1, axis=1),
                                mean.features, mean.feasible, mean.durations, mean.costs, mean.indices, mean.actions,
                                mean.queries, mean.config)
            assert np.allclose((loss2d - B[:, None]), (loss2d - B[:, None])[first][inverse])
            record("E", -abs(L - float(prior @ B) - PortfolioSolver(centred).evaluate(pi)[0]))
    if exact:
        loss2d = model.loss[:, 0, :]
        B = loss2d[:, 0]
        first = np.array([np.flatnonzero(inverse == c)[0] for c in range(nc)])
        centred = Portfolio(mean.prior, mean.likelihood,
                            np.repeat((loss2d - B[:, None])[first][:, None, :], model.config.deadline + 1, axis=1),
                            mean.features, mean.feasible, mean.durations, mean.costs, mean.indices, mean.actions,
                            mean.queries, mean.config)
        record("E", -abs(opt.solve()[0] - float(prior @ B) - PortfolioSolver(centred).solve()[0]))
    # Certificate for the surrogate-optimal policy
    U, _ = C.undominated_endpoints(model)
    value = PortfolioSolver(model).evaluate(sopt)[0]
    surrogate = PortfolioSolver(mean).evaluate(sopt)[0]
    cert = C.refined_certificate(model, mean, inverse, value, surrogate, U, None,
                                 lower_bound=C.perfect_information_bound(model), sets=sets, cmax=cmax)
    regret = value - opt.solve()[0]
    for key in ("prop1_uniform", "prop1_postsolve", "U_postsolve", "refined_postsolve", "refined_lb"):
        record("C", cert[key] - regret)
    # Query relaxation: a lower bound on the optimum that is at least the perfect-information bound
    relaxed = C.query_relaxation_bound(model, inverse)
    record("R", opt.solve()[0] - relaxed)
    record("R", relaxed - C.perfect_information_bound(model))
    timed = replace(model, loss=model.loss * rng.uniform(0.5, 1.5, size=model.loss.shape))
    relaxed_timed = C.query_relaxation_bound(timed, inverse)
    record("R", PortfolioSolver(timed).solve()[0] - relaxed_timed)
    record("R", relaxed_timed - C.perfect_information_bound(timed))
    return dict(seed=seed, exact=exact, configurations=len(prior), classes=nc, queries=len(model.queries),
                policies=len(pols), violations=counts, checks=checks,
                min_slack={k: (None if not np.isfinite(v) else float(v)) for k, v in slack.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=4000)
    parser.add_argument("--policies", type=int, default=80)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    start = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(audit_instance, [(s, args.policies) for s in range(args.instances)], chunksize=20):
            rows.append(r)
    total = {k: sum(r["checks"][k] for r in rows) for k in rows[0]["checks"]}
    bad = {k: sum(r["violations"][k] for r in rows) for k in rows[0]["violations"]}
    minslack = {k: min((r["min_slack"][k] for r in rows if r["min_slack"][k] is not None), default=None)
                for k in rows[0]["min_slack"]}
    out = dict(instances=len(rows), exact_instances=sum(r["exact"] for r in rows),
               policies=sum(r["policies"] for r in rows), checks=total, violations=bad, min_slack=minslack,
               zero_one_share=1 / 3, runtime_seconds=time.perf_counter() - start)
    (RESULTS / "proof_audit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
