"""Experiment 20: regret and re-certification after each answer.

Section 5.6; Supplementary Section S6.
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
from itertools import product
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from hevi_hitl import certificates as C
from hevi_hitl import flood as fl
from hevi_hitl import social as so
from hevi_hitl.model import Portfolio, PortfolioSolver
from hevi_hitl.reduction import aggregate_partition

RESULTS = ROOT / "results"
CLASS_COUNTS = {"flood": (1,) + tuple(range(12, 193, 12)), "social": tuple(range(48, 769, 48))}
EPS = 1e-10


def configs():
    out = []
    for domain, module in (("flood", fl), ("social", so)):
        grid = module.GRID
        cls = fl.FloodConfig if domain == "flood" else so.SocialConfig
        for cid, values in enumerate(product(*grid.values())):
            out.append((domain, cid, cls(**dict(zip(grid, values)))))
    return out


def build(domain, cfg):
    return fl.build_flood(cfg) if domain == "flood" else so.build_social_flood(cfg)


def certify(model, inverse, cache=None):
    """Aggregate, solve, evaluate and certify one partition; returns (policy solver, value, certificate)."""
    cache = {} if cache is None else cache
    if "U" not in cache:
        cache["U"], _ = C.undominated_endpoints(model)
        cache["sets"], cache["cmax"] = C.feasible_query_sets(model)
        cache["J_U"] = C.endpoint_columns(model, cache["U"])
        cache["lower"] = C.perfect_information_bound(model)
    reduced = aggregate_partition(model, inverse)
    policy = PortfolioSolver(reduced)
    policy.solve()
    value = PortfolioSolver(model).evaluate(policy)[0]
    surrogate = PortfolioSolver(reduced).evaluate(policy)[0]
    cert = C.refined_certificate(model, reduced, inverse, value, surrogate, cache["U"], None,
                                 lower_bound=cache["lower"], sets=cache["sets"], cmax=cache["cmax"],
                                 J_U=cache["J_U"])
    return policy, float(value), float(cert["refined_lb"])


def submodel(model, h):
    """The decision problem that remains after history h: posterior prior, unasked queries, shifted clock."""
    mass = model.prior.copy()
    for q, y in enumerate(h):
        if y != -1:
            mass = mass * (model.likelihood[:, q] if y else 1 - model.likelihood[:, q])
    t = int(sum(model.durations[q] for q, y in enumerate(h) if y != -1))
    keep = [q for q, y in enumerate(h) if y == -1]
    cfg = SimpleNamespace(deadline=int(model.config.deadline) - t)
    return Portfolio(mass / mass.sum(), model.likelihood[:, keep], model.loss[:, t:], model.features[:, t:],
                     model.feasible[t:], model.durations[keep], model.costs[keep], model.indices,
                     model.actions, tuple(model.queries[q] for q in keep), cfg), float(mass.sum())


def continue_value(model, policy, h):
    """Normalised expected cost, from history h, of following the frozen ``policy`` in ``model``."""
    solver = PortfolioSolver(model)

    def rec(hh):
        kind, j = policy.solve(hh)[1]
        mass = solver.mass(hh)
        t = solver.time(hh)
        if kind == "act":
            return float(mass @ model.loss[:, t, j])
        value = float(mass.sum() * model.costs[j])
        for child in solver.children(hh, j):
            value += rec(child)
        return value

    return rec(h) / float(solver.mass(h).sum())


def reached(model, policy):
    """Histories visited by ``policy`` with their probabilities under the prior."""
    solver = PortfolioSolver(model)
    out, stack = [], [policy.root]
    while stack:
        h = stack.pop()
        out.append((h, float(solver.mass(h).sum())))
        kind, j = policy.solve(h)[1]
        if kind == "query":
            stack.extend(solver.children(h, j))
    return out


def analyse(task):
    domain, cid, cfg, tol = task
    start = time.perf_counter()
    model = build(domain, cfg)
    full = PortfolioSolver(model)
    optimum = full.solve()[0]
    hierarchy = C.guided_hierarchy(model, CLASS_COUNTS[domain])
    cache = {}
    accepted = None
    for k in CLASS_COUNTS[domain]:
        inverse = hierarchy[min(k, len(model.prior))]
        policy, value, bound = certify(model, inverse, cache)
        if bound <= tol:
            accepted = (k, inverse, policy, value, bound)
            break
    if accepted is None:
        return []
    k, inverse, policy, value, bound = accepted
    rows = []
    for h, prob in reached(model, policy):
        depth = int(sum(y != -1 for y in h))
        w_star = full.solve(h)[0] / prob
        cont = continue_value(model, policy, h)
        kind = policy.solve(h)[1][0]
        row = dict(domain=domain, configuration_id=cid, classes=k, prior_certificate=bound,
                   prior_regret=float(value - optimum), history=str(h), depth=depth, probability=prob,
                   decision=kind, continue_regret=cont - w_star)
        tick = time.perf_counter()
        sub, _ = submodel(model, h)
        sub_policy, sub_value, sub_bound = certify(sub, inverse)
        row.update(resolve_regret=sub_value - w_star, recertificate=sub_bound,
                   recertify_seconds=time.perf_counter() - tick,
                   sub_optimum_check=abs(PortfolioSolver(sub).solve()[0] - w_star))
        assert row["sub_optimum_check"] < 1e-9, row
        assert sub_bound >= row["resolve_regret"] - 1e-9, row
        assert row["continue_regret"] >= -1e-9 and row["resolve_regret"] >= -1e-9
        rows.append(row)
    for r in rows:
        r["instance_seconds"] = time.perf_counter() - start
    return rows


def summarise(df: pd.DataFrame, tol: float) -> dict:
    out = {}
    nonroot = df[df.depth > 0]
    for domain, g in nonroot.groupby("domain"):
        exceed = g[g.continue_regret > tol]
        per_instance = g.groupby("configuration_id")
        out[domain] = dict(
            instances=int(df[df.domain == domain].configuration_id.nunique()),
            querying_instances=int(g.configuration_id.nunique()),
            histories=int(len(g)),
            max_continue_regret=float(g.continue_regret.max()),
            histories_continue_above_tol=int(len(exceed)),
            instances_continue_above_tol=int(exceed.configuration_id.nunique()),
            max_probability_above_tol=float(per_instance.apply(
                lambda x: x.loc[x.continue_regret > tol, "probability"].sum(), include_groups=False).max()),
            max_resolve_regret=float(g.resolve_regret.max()),
            recertified_share=float((g.recertificate <= tol).mean()),
            max_recertificate=float(g.recertificate.max()),
            median_recertificate=float(g.recertificate.median()),
            median_recertify_seconds=float(g.recertify_seconds.median()),
            max_recertify_seconds=float(g.recertify_seconds.max()))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=0.005)
    args = parser.parse_args()
    start = time.perf_counter()
    work = [t + (args.tolerance,) for t in configs()]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, part in enumerate(pool.map(analyse, work, chunksize=1)):
            rows.extend(part)
            print(f"{k + 1}/{len(work)}: {time.perf_counter() - start:.1f} s", flush=True)
    df = pd.DataFrame(rows)
    suffix = "" if abs(args.tolerance - 0.005) < 1e-12 else f"_tol{args.tolerance:g}"
    df.to_csv(RESULTS / f"conditional_regret{suffix}.csv", index=False)
    summary = dict(tolerance=args.tolerance, by_domain=summarise(df, args.tolerance),
                   runtime_seconds=time.perf_counter() - start)
    (RESULTS / f"conditional_regret{suffix}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
