"""Experiment 32: decision relevance by HEVI coordinate and class constants for every forecast.

Sections 3.4 and 5.6, Table 5; Supplementary Section S15.
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
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from hevi_hitl import certificates as C
from hevi_hitl import flood as fl
from hevi_hitl import social as so
from hevi_hitl.insight import revealed_value
from hevi_hitl.moments import MomentSolver
from hevi_hitl.reduction import aggregate_partition
from hevi_hitl.scaled import evaluate

RESULTS = ROOT / "results"
TOL = 0.005
SHARE = 0.25        # VOI-relative tolerance: a quarter of the instance's V_J
EPS = 1e-10
# name: (latent axes, queries that observe it, separable: independent and observed only through these queries)
GROUPS = {
    "flood": {"severity": ((0,), (0,), False), "footprint": ((1,), (0,), False), "occupancy": ((2,), (1,), True),
              "vulnerability": ((3,), (2,), True), "restoration": ((4,), (3,), True), "preference": ((5,), (4, 5), True),
              "others": ((1, 2, 3, 4), (0, 1, 2, 3), False)},
    "social": {"severity": ((0,), (0,), False), "footprint": ((1,), (0,), False), "occupancy": ((2,), (1,), True),
               "vulnerability": ((3,), (2,), True), "restoration": ((4,), (3,), True), "preference": ((5,), (5, 6), True),
               "mobility": ((6,), (4,), False), "recovery": ((7,), (4,), False), "access": ((6, 7), (4,), True),
               "others": ((1, 2, 3, 4, 6, 7), (0, 1, 2, 3, 4), False)},
}   # others: every coordinate except severity and preference type averaged out (12 configurations remain)


def forecasts():
    grid = [(i / 10, j / 10, (10 - i - j) / 10) for i in range(1, 9) for j in range(1, 10 - i)]
    return grid + [(0.98, 0.01, 0.01), (0.01, 0.98, 0.01), (0.01, 0.01, 0.98)]


def configs():
    out = []
    for domain, module in (("flood", fl), ("social", so)):
        grid = module.GRID
        cls = fl.FloodConfig if domain == "flood" else so.SocialConfig
        out += [(domain, cid, cls(**dict(zip(grid, values)))) for cid, values in enumerate(product(*grid.values()))]
    return out


def without(model, drop):
    keep = [q for q in range(len(model.queries)) if q not in drop]
    return replace(model, likelihood=model.likelihood[:, keep], durations=model.durations[keep],
                   costs=model.costs[keep], queries=tuple(model.queries[q] for q in keep))


def expected_counts(model, solver):
    out = np.zeros(len(model.queries))
    lik = model.likelihood
    stack = [(solver.root, model.prior.copy())]
    while stack:
        h, mass = stack.pop()
        kind, j = solver.decision(h)
        if kind == "act":
            continue
        out[j] += mass.sum()
        for y in (0, 1):
            stack.append((h[:j] + (y,) + h[j + 1:], mass * (lik[:, j] if y else 1 - lik[:, j])))
    return out


def merged_regret(model, inverse, U_tilde, optimum):
    """Exact regret of the merged model's optimal policy (commitments within its covering set)."""
    reduced = aggregate_partition(model, inverse)
    free = MomentSolver(reduced).solve()[0]
    restricted = replace(reduced, feasible=U_tilde)
    solver = MomentSolver(restricted)
    surrogate = solver.solve()[0]
    assert abs(surrogate - free) < 1e-12        # the covering set loses nothing in the surrogate
    value = evaluate(model, lambda h, mass: solver.decision(h))["value"]
    return float(value - optimum), float(value), float(surrogate), expected_counts(restricted, solver)


def analyse(task):
    domain, cid, cfg = task
    start = time.perf_counter()
    model = fl.build_flood(cfg) if domain == "flood" else so.build_social_flood(cfg)
    solver = MomentSolver(model)
    optimum = solver.solve()[0]
    counts = expected_counts(model, solver)
    immediate = revealed_value(model, ())
    U, _ = C.undominated_endpoints(model)
    sets, cmax = C.feasible_query_sets(model)
    J_U = C.endpoint_columns(model, U)
    lower = C.perfect_information_bound(model)
    severity = model.indices[:, 0]
    p_sev = np.array([model.prior[severity == s].sum() for s in range(3)])
    rows, grid_rows = [], []
    for name, (axes, queries, separable) in GROUPS[domain].items():
        inverse = C.marginal_partition(model.indices, axes)
        reduced = aggregate_partition(model, inverse)
        U_t, _ = C.undominated_endpoints(reduced)
        regret, value, surrogate, merged_counts = merged_regret(model, inverse, U_t, optimum)
        own = C.differing_queries(model.likelihood, inverse)
        assert set(own) <= set(queries)
        relaxed = C.query_relaxation_bound(model, inverse)
        assert lower - EPS <= relaxed <= optimum + EPS
        cert = C.refined_certificate(model, reduced, inverse, value, surrogate, U, U_t,
                                     lower_bound=max(lower, relaxed), sets=sets, cmax=cmax, J_U=J_U)
        kappa = C.class_constants(model, reduced, inverse, U, U_t, sets=sets, cmax=cmax, J_U=J_U)
        assert abs(float(reduced.prior @ kappa) - cert["refined_uniform"]) < 1e-10
        assert cert["refined_lb"] >= regret - 1e-10 and cert["refined_uniform"] >= regret - 1e-10
        query_value = float(MomentSolver(without(model, queries)).solve()[0] - optimum)
        asks_own = float(merged_counts[list(queries)].sum())
        row = dict(domain=domain, configuration_id=cid, coordinate=name, classes=int(inverse.max() + 1),
                   voi_joint=float(immediate - optimum),
                   evppi=float(immediate - revealed_value(model, axes)), query_value=query_value,
                   query_count=float(counts[list(queries)].sum()), regret=regret, dri=float(cert["refined_lb"]),
                   relaxation=float(value - relaxed), free_answer_value=float(optimum - relaxed),
                   certificate=float(min(cert["refined_postsolve"], value - lower)),
                   presolve=float(cert["refined_uniform"]), kappa_max=float(kappa.max()), separable=separable,
                   merged_asks_own=asks_own,
                   identity_error=abs(regret - query_value) if (separable or asks_own <= EPS) else np.nan)
        if 0 not in axes:   # classes keep their severity: bound linear in the severity forecast
            first = np.array([np.flatnonzero(inverse == c)[0] for c in range(len(kappa))])
            sev_c = severity[first]
            K = np.array([np.average(kappa[sev_c == s], weights=reduced.prior[sev_c == s]) for s in range(3)])
            row.update(K_low=float(K[0]), K_mid=float(K[1]), K_high=float(K[2]), K_max=float(K.max()))
            slack, worst, certified, worst_dri, quarter, querying = np.inf, 0.0, 0, 0.0, 0, 0
            for p in forecasts():
                p = np.asarray(p)
                shifted = replace(model, prior=model.prior * (p / p_sev)[severity])
                opt_p = MomentSolver(shifted).solve()[0]
                voi_p = float(MomentSolver(shifted, method="immediate").solve()[0] - opt_p)
                reg_p, val_p, _, _ = merged_regret(shifted, inverse, U_t, opt_p)
                dri_p = float(val_p - C.query_relaxation_bound(shifted, inverse))
                assert dri_p >= reg_p - 1e-10
                bound = float(p @ K)
                slack, worst, worst_dri = min(slack, bound - reg_p), max(worst, reg_p), max(worst_dri, dri_p)
                certified += bound <= TOL
                if voi_p > EPS:
                    querying += 1
                    quarter += dri_p <= SHARE * voi_p
                grid_rows.append(dict(domain=domain, configuration_id=cid, coordinate=name, p_low=p[0], p_mid=p[1],
                                      p_high=p[2], bound=bound, regret=reg_p, dri=dri_p, voi_joint=voi_p))
            row.update(grid_min_slack=float(slack), grid_max_regret=float(worst),
                       grid_certified_share=certified / len(forecasts()), grid_max_dri=float(worst_dri),
                       grid_querying=querying, grid_quarter=quarter)
        rows.append(row)
    for r in rows:
        r["seconds"] = time.perf_counter() - start
    return rows, grid_rows


def summarise(df: pd.DataFrame) -> dict:
    out = {}
    for (domain, name), g in df.groupby(["domain", "coordinate"]):
        q = g[g.voi_joint > EPS]
        share = q.dri / q.voi_joint
        s = dict(instances=int(len(g)), mean_evppi=float(g.evppi.mean()), max_evppi=float(g.evppi.max()),
                 mean_query_value=float(g.query_value.mean()), max_query_value=float(g.query_value.max()),
                 asked=int((g.query_count > 1e-10).sum()), mean_regret=float(g.regret.mean()),
                 max_regret=float(g.regret.max()), mean_dri=float(g.dri.mean()), max_dri=float(g.dri.max()),
                 mean_certificate=float(g.certificate.mean()), max_certificate=float(g.certificate.max()),
                 relaxation_binds=int((g.relaxation < g.certificate - 1e-12).sum()),
                 mean_free_answer_value=float(g.free_answer_value.mean()),
                 certified_at_prior=int((g.dri <= TOL).sum()),
                 certificate_certified_at_prior=int((g.certificate <= TOL).sum()),
                 querying=int(len(q)), certified_quarter=int((q.dri <= SHARE * q.voi_joint).sum()),
                 mean_share=float(share.mean()) if len(q) else None, max_share=float(share.max()) if len(q) else None,
                 mean_regret_share=float((q.regret / q.voi_joint).mean()) if len(q) else None,
                 max_regret_share=float((q.regret / q.voi_joint).max()) if len(q) else None,
                 max_kappa=float(g.kappa_max.max()), certified_every_prior=int((g.kappa_max <= TOL).sum()),
                 merged_asks_own=int((g.merged_asks_own > EPS).sum()))
        identity = g.identity_error.dropna()
        if len(identity):
            s["identity_cases"] = int(len(identity))
            s["max_identity_error"] = float(identity.max())
        if "K_max" in g and g.K_max.notna().all():
            s.update(mean_K=[float(g.K_low.mean()), float(g.K_mid.mean()), float(g.K_high.mean())],
                     max_K=[float(g.K_low.max()), float(g.K_mid.max()), float(g.K_high.max())],
                     max_K_max=float(g.K_max.max()), certified_every_forecast=int((g.K_max <= TOL).sum()),
                     mean_grid_certified_share=float(g.grid_certified_share.mean()),
                     min_grid_slack=float(g.grid_min_slack.min()), max_grid_regret=float(g.grid_max_regret.max()),
                     max_grid_dri=float(g.grid_max_dri.max()),
                     grid_querying=int(g.grid_querying.sum()), grid_quarter=int(g.grid_quarter.sum()))
        out[f"{domain}|{name}"] = s
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    if not args.analysis_only:
        rows, grid = [], []
        tasks = configs()[::-1]   # social instances (longer) first
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for k, (r, g) in enumerate(pool.map(analyse, tasks, chunksize=1)):
                rows.extend(r)
                grid.extend(g)
                print(f"{k + 1}/{len(tasks)} {r[0]['domain']} {r[0]['configuration_id']}: {r[0]['seconds']:.0f} s", flush=True)
        pd.DataFrame(rows).to_csv(RESULTS / "forecast_free.csv", index=False)
        pd.DataFrame(grid).to_csv(RESULTS / "forecast_free_grid.csv", index=False)
    df = pd.read_csv(RESULTS / "forecast_free.csv")
    summary = summarise(df)
    grid = pd.read_csv(RESULTS / "forecast_free_grid.csv")
    summary["grid"] = dict(cases=int(len(grid)), min_bound_minus_regret=float((grid.bound - grid.regret).min()),
                           min_dri_minus_regret=float((grid.dri - grid.regret).min()))
    summary["forecasts"] = len(forecasts())
    summary["tolerance"] = TOL
    summary["voi_share"] = SHARE
    summary["runtime_seconds"] = None if args.analysis_only else time.perf_counter() - start
    (RESULTS / "forecast_free.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for key, s in summary.items():
        if isinstance(s, dict) and "instances" in s:
            extra = (f" K {[round(1e3 * x, 2) for x in s['max_K']]} all-forecast {s['certified_every_forecast']}/{s['instances']}"
                     f" grid share {s['mean_grid_certified_share']:.2f} grid DRI {1e3 * s['max_grid_dri']:.2f}"
                     f" quarter {s['grid_quarter']}/{s['grid_querying']}") if "max_K" in s else ""
            print(f"{key:22s} EVPPI {1e3 * s['mean_evppi']:6.2f} qval {1e3 * s['mean_query_value']:6.3f} asked {s['asked']:2d} "
                  f"regret {1e3 * s['mean_regret']:6.3f} DRI {1e3 * s['mean_dri']:6.3f} (cert {1e3 * s['mean_certificate']:6.2f}) "
                  f"<=tol {s['certified_at_prior']:2d} quarter {s['certified_quarter']}/{s['querying']}{extra}"
                  + (f" id-err {s['max_identity_error']:.1e} ({s['identity_cases']})" if "max_identity_error" in s else ""))


if __name__ == "__main__":
    main()
