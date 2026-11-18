"""Experiment 28: finer catalogues, wider spreads, uncertain occurrence and the JRC damage function.

Section 5.7; Supplementary Section S12.
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
from hevi_hitl import social as so
from hevi_hitl.insight import revealed_value
from hevi_hitl.moments import MomentSolver
from hevi_hitl.scaled import ScaledConfig, aggregate_scaled, build_scaled, evaluate

RESULTS = ROOT / "results"
COORDINATES = ("severity", "footprint", "occupancy", "vulnerability", "restoration", "preference",
               "mobility", "recovery")
ALL = COORDINATES + ("others",)   # others: every coordinate except severity and preference type averaged out
RISK, PREFERENCE = (0, 1, 2, 3, 4), (5, 6)
TOL = 0.005
ELEVATED = (0.15, 0.35, 0.50)
NO_FLOOD = (0.3, 0.5, 0.7, 0.9, 0.95)
SPREADS = {"occupancy": dict(occupancy_levels=(0.6, 1.2)), "vulnerability": dict(susceptibility_levels=(0.5, 1.5)),
           "restoration": dict(restoration_levels=(0.5, 1.5))}


def catalogues():
    """(family, configuration changes, label); the label distinguishes variants of equal resolution."""
    out = [("published", dict(), "published")]
    out += [("severity", dict(severity_levels=n), f"{n} severities") for n in (5, 9, 17, 33)]
    out += [("footprint", dict(footprints=n), f"{n} footprints") for n in (3, 5, 9)]
    out += [("vulnerability", dict(vulnerability_levels=n), f"{n} vulnerability levels") for n in (3, 5, 9)]
    out += [("combined", dict(severity_levels=9, footprints=5, vulnerability_levels=5), "combined")]
    out += [("spread", change, name) for name, change in SPREADS.items()]
    out += [("spread", {k: v for change in SPREADS.values() for k, v in change.items()}, "all three")]
    out += [("occurrence", dict(severity_levels=4, severity_depths=(0.0, 0.35, 0.85, 1.65),
                                hazard_prior=(p,) + tuple((1 - p) * x for x in ELEVATED)), f"no flood {p:g}")
            for p in NO_FLOOD]
    out += [("damage", dict(damage_function="jrc_europe"), "JRC damage function")]
    return out


def settings():
    return [dict(lead_hours=lead, mechanism_strength=s, preference_prior=p)
            for lead, s, p in product((4, 6), (0.5, 1.0), ("diffuse", "economic_concentrated"))]


def restrict(model, keep):
    keep = list(keep)
    return replace(model, likelihood=model.likelihood[:, keep], durations=model.durations[keep],
                   costs=model.costs[keep], queries=tuple(model.queries[q] for q in keep))


def counts(model, solver):
    """Expected number of times each query is asked by the solved policy."""
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


def run(task):
    sid, setting, family, change, label = task
    start = time.perf_counter()
    cfg = ScaledConfig(**{**dict(severity_levels=3, footprints=2, preference_types=4, district_gauges=0,
                                 extra_preference_pair=False, social_query_reliability=0.9), **setting, **change})
    model = build_scaled(cfg)
    joint = MomentSolver(model)
    optimum = joint.solve()[0]
    immediate = revealed_value(model, ())
    v_risk = immediate - MomentSolver(restrict(model, RISK)).solve()[0]
    v_pref = immediate - MomentSolver(restrict(model, PREFERENCE)).solve()[0]
    v_joint = immediate - optimum
    evpi = immediate - revealed_value(model, tuple(range(8)))
    evppi_eta = immediate - revealed_value(model, (0, 1, 2, 3, 4, 6, 7))
    evppi_theta = immediate - revealed_value(model, (5,))
    n = counts(model, joint)
    root = joint.decision(joint.root)
    at_once = MomentSolver(model, method="immediate")
    at_once.solve()
    measures = float(so.action_inventory()[1][at_once.decision(at_once.root)[1]])
    row = dict(setting=sid, **setting, family=family, label=label, immediate_measures_eur=measures,
               severity_levels=cfg.severity_levels,
               footprints=cfg.footprints, vulnerability_levels=cfg.vulnerability_levels, configurations=len(model.prior),
               immediate=immediate, optimum=optimum, v_joint=v_joint, v_risk=v_risk, v_pref=v_pref,
               v_interaction=v_joint - v_risk - v_pref, evpi=evpi, evppi_eta=evppi_eta, evppi_theta=evppi_theta,
               captured_share=v_joint / evpi if evpi > 0 else np.nan, queries_before_commit=bool(root[0] == "query"),
               first_query=model.queries[root[1]] if root[0] == "query" else "commit",
               risk_queries=float(n[list(RISK)].sum()), comparisons=float(n[list(PREFERENCE)].sum()),
               access_checks=float(n[4]))
    U, _ = C.undominated_endpoints(model)
    sets, cmax = C.feasible_query_sets(model)
    J_U = C.endpoint_columns(model, U)
    lower = C.perfect_information_bound(model, U)
    others = tuple(k for k, name in enumerate(COORDINATES) if name not in ("severity", "preference"))
    for k, name in list(enumerate(COORDINATES)) + [(others, "others")]:
        inverse = C.marginal_partition(model.indices, (k,) if name != "others" else others)
        reduced = aggregate_scaled(model, inverse)
        rs = MomentSolver(reduced)
        surrogate = rs.solve()[0]
        value = evaluate(model, lambda h, mass: rs.decision(h))["value"]
        guard = max(lower, C.query_relaxation_bound(model, inverse))
        cert = C.refined_certificate(model, reduced, inverse, value, surrogate, U, None, lower_bound=guard,
                                     sets=sets, cmax=cmax, J_U=J_U)["refined_lb"]
        assert cert >= value - optimum - 1e-10
        row[f"dri_{name}"] = float(cert)
        row[f"regret_{name}"] = float(value - optimum)
        row[f"classes_{name}"] = int(inverse.max() + 1)
    row["seconds"] = time.perf_counter() - start
    return row


def summarise(df):
    out = {}
    base = df[df.family == "published"].set_index("setting")
    for (family, label, sev, foot, vul), g in df.groupby(["family", "label", "severity_levels", "footprints",
                                                          "vulnerability_levels"], sort=False):
        g = g.set_index("setting")
        b = base.loc[g.index]
        key = f"{family}|{sev}|{foot}|{vul}" if family not in ("spread", "occurrence") else f"{family}|{label}"
        out[key] = dict(
            configurations=int(g.configurations.iloc[0]), settings=int(len(g)),
            querying=int(g.queries_before_commit.sum()), comparisons_used=int((g.comparisons > 1e-10).sum()),
            interaction_positive=int((g.v_interaction > 1e-10).sum()), v_pref_positive=int((g.v_pref > 1e-10).sum()),
            access_check_used=int((g.access_checks > 1e-10).sum()),
            same_first_decision=int((g.first_query == b.first_query).sum()),
            mean_v_joint=float(g.v_joint.mean()), mean_v_joint_base=float(b.v_joint.mean()),
            max_abs_change_v_joint=float((g.v_joint - b.v_joint).abs().max()),
            mean_captured_share=float(g.captured_share.mean()),
            v_joint_over_base=[float((g.v_joint / b.v_joint).min()), float((g.v_joint / b.v_joint).max())],
            first_queries=sorted(set(g.first_query)),
            immediate_measures_eur=[float(g.immediate_measures_eur.min()), float(g.immediate_measures_eur.max())],
            max_dri_share={c: float((g[f"dri_{c}"] / g.v_joint).max()) for c in ALL},
            quarter={c: int((g[f"dri_{c}"] <= 0.25 * g.v_joint).sum()) for c in ALL},
            mean_regret={c: float(g[f"regret_{c}"].mean()) for c in ALL},
            mean_dri={c: float(g[f"dri_{c}"].mean()) for c in ALL},
            classes_others=int(g.classes_others.iloc[0]),
            certified=[c for c in COORDINATES if (g[f"dri_{c}"] <= TOL).all()],
            never_certified=[c for c in COORDINATES if (g[f"dri_{c}"] > TOL).all()],
            max_dri={c: float(g[f"dri_{c}"].max()) for c in ALL},
            max_regret={c: float(g[f"regret_{c}"].max()) for c in ALL})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--only", default=None, help="recompute one family and keep the saved rows of the others")
    args = parser.parse_args()
    start = time.perf_counter()
    tasks = [(sid, s, family, change, label) for sid, s in enumerate(settings()) for family, change, label in catalogues()
             if args.only in (None, family)]
    kept = []
    if args.only is not None:
        saved = pd.read_csv(RESULTS / "catalogue_discretisation.csv")
        kept = saved[saved.family != args.only].to_dict("records")
    tasks.sort(key=lambda t: -ScaledConfig(**{k: v for k, v in t[3].items() if k.endswith("levels") and isinstance(v, int)
                                              or k == "footprints"}).states)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run, tasks, chunksize=1):
            rows.append(row)
            pd.DataFrame(kept + rows).to_csv(RESULTS / "catalogue_discretisation.csv", index=False)
            print(f"{len(rows)}/{len(tasks)} {row['family']} ({row['configurations']}) setting {row['setting']}: "
                  f"VJ {1e3 * row['v_joint']:.3f} first {row['first_query']} ({row['seconds']:.0f} s)", flush=True)
    df = pd.DataFrame(kept + rows)
    out = summarise(df)
    previous = RESULTS / "catalogue_discretisation.json"
    out["runtime_seconds"] = (time.perf_counter() - start if args.only is None or not previous.exists()
                              else json.loads(previous.read_text(encoding="utf-8")).get("runtime_seconds"))
    (RESULTS / "catalogue_discretisation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out.items():
        if isinstance(v, dict):
            print(k, {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)})


if __name__ == "__main__":
    main()
