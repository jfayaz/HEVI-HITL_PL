"""Experiment 26: when an access check pays.

Section 5.7; Supplementary Section S11.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from hevi_hitl import access as ac
from hevi_hitl import social as so
from hevi_hitl.insight import revealed_value
from hevi_hitl.scaled import FastSolver

RESULTS = ROOT / "results"
STRUCTURES = ("common", "district")
EXTRA_EUR = (15000.0, 20000.0, 25000.0, 30000.0, 35000.0, 40000.0, 50000.0, 60000.0)
RELIABILITIES = (0.4, 0.7, 0.9, 1.0)
COST_FACTORS = (1.0, 4.0, 16.0)
EPS = 1e-10


def settings():
    out = []
    for lead, strength, pref in product((4, 6), (0.5, 1.0), ("diffuse", "economic_concentrated")):
        out.append(so.SocialConfig(lead_hours=lead, mechanism_strength=strength, preference_prior=pref,
                                    social_query_reliability=0.9))
    return out


def profile(model, solver) -> dict:
    """Exact expectations of the solved policy (every answer enumerated)."""
    root_value = solver.solve()[0]
    lik = model.likelihood
    out = dict(value=0.0, checks=0.0, queries=0.0, assisted=0.0, constrained_exposed=0.0,
               constrained_fraction=0.0, commitment_hours=0.0)
    by_district = np.zeros(3)
    stack = [(solver.root, model.prior.copy(), 0)]
    while stack:
        h, mass, t = stack.pop()
        kind, j = solver.memo[h][1]
        p = float(mass.sum())
        if kind == "act":
            assert model.feasible[t, j]
            out["value"] += float(mass @ model.loss2d[:, j])
            e = [x for _, x in model.actions[j]]
            if 2 in e:
                out["assisted"] += p
                by_district[e.index(2)] += p
            out["constrained_exposed"] += float(mass @ model.group[:, j, 1, 1])
            out["constrained_fraction"] += float(mass @ model.group[:, j, 1, 3])
            out["commitment_hours"] += p * t / 2
            continue
        out["value"] += p * float(model.costs[j])
        out["queries"] += p
        if j in model.checks:
            out["checks"] += p
        for y in (0, 1):
            hh = h[:j] + (y,) + h[j + 1:]
            stack.append((hh, mass * (lik[:, j] if y else 1 - lik[:, j]), t + int(model.durations[j])))
    assert abs(out["value"] - root_value) < 1e-12
    out.update({f"assisted_{d}": float(v) for d, v in zip(("riverside", "central", "upstream"), by_district)})
    return out


def run(task):
    sid, cfg, structure, extra = task
    start = time.perf_counter()
    base = ac.build_access(cfg, structure, extra_eur=extra)
    v_imm = revealed_value(base, ())
    evppi_access = v_imm - revealed_value(base, ac.ACCESS_AXES[structure])
    evppi_mobility = v_imm - revealed_value(base, ac.MOBILITY_AXES[structure])
    nocheck_model = ac.without_queries(base, base.checks)
    nocheck_solver = FastSolver(nocheck_model)
    nocheck = profile(nocheck_model, nocheck_solver)
    imm_solver = FastSolver(base, method="immediate")
    imm = profile(base, imm_solver)
    assert abs(imm["value"] - v_imm) < 1e-12
    rows = []
    for r, factor in product(RELIABILITIES, COST_FACTORS):
        model = ac.build_access(cfg, structure, extra_eur=extra, check_reliability=r, check_cost_factor=factor)
        solver = FastSolver(model)
        joint = profile(model, solver)
        rows.append(dict(setting=sid, lead_hours=cfg.lead_hours, mechanism_strength=cfg.mechanism_strength,
                         preference_prior=cfg.preference_prior, structure=structure, extra_eur=extra,
                         check_reliability=r, check_cost_factor=factor, configurations=len(model.prior),
                         portfolios=len(model.actions), voi_joint=v_imm - joint["value"],
                         value_of_checks=nocheck["value"] - joint["value"], evppi_access=evppi_access,
                         evppi_mobility=evppi_mobility,
                         **{f"joint_{k}": v for k, v in joint.items()},
                         **{f"nocheck_{k}": v for k, v in nocheck.items()},
                         **{f"immediate_{k}": v for k, v in imm.items()}))
    for row in rows:
        row["seconds"] = time.perf_counter() - start
    return rows


def summarise(df: pd.DataFrame) -> dict:
    out = {}
    for (structure, extra, r, factor), g in df.groupby(["structure", "extra_eur", "check_reliability",
                                                        "check_cost_factor"]):
        out[f"{structure}|{extra:g}|{r:g}|{factor:g}"] = dict(
            settings=int(len(g)), check_asked=int((g.joint_checks > EPS).sum()),
            mean_checks=float(g.joint_checks.mean()), mean_value_of_checks=float(g.value_of_checks.mean()),
            max_value_of_checks=float(g.value_of_checks.max()), mean_evppi_access=float(g.evppi_access.mean()),
            max_evppi_access=float(g.evppi_access.max()), mean_voi=float(g.voi_joint.mean()),
            mean_assisted=float(g.joint_assisted.mean()), mean_nocheck_assisted=float(g.nocheck_assisted.mean()),
            mean_constrained_fraction=float(g.joint_constrained_fraction.mean()),
            mean_nocheck_constrained_fraction=float(g.nocheck_constrained_fraction.mean()))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    start = time.perf_counter()
    tasks = [(sid, cfg, structure, extra) for structure in STRUCTURES[::-1] for extra in EXTRA_EUR
             for sid, cfg in enumerate(settings())]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, part in enumerate(pool.map(run, tasks, chunksize=1)):
            rows.extend(part)
            pd.DataFrame(rows).to_csv(RESULTS / "access_thresholds.csv", index=False)
            print(f"{k + 1}/{len(tasks)} {part[0]['structure']} extra={part[0]['extra_eur']:g} "
                  f"setting={part[0]['setting']}: max value of checks {1e3 * max(r['value_of_checks'] for r in part):.4f}, "
                  f"EVPPI {1e3 * part[0]['evppi_access']:.4f} ({part[0]['seconds']:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    out = summarise(df)
    out["runtime_seconds"] = time.perf_counter() - start
    (RESULTS / "access_thresholds.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for key, s in out.items():
        if isinstance(s, dict):
            print(f"{key:28s} asked {s['check_asked']}/8 value {1e3 * s['mean_value_of_checks']:.4f} "
                  f"(max {1e3 * s['max_value_of_checks']:.4f}) EVPPI {1e3 * s['mean_evppi_access']:.4f} "
                  f"assisted {s['mean_assisted']:.2f}/{s['mean_nocheck_assisted']:.2f} "
                  f"constrained {100 * s['mean_constrained_fraction']:.1f}/{100 * s['mean_nocheck_constrained_fraction']:.1f}%")


if __name__ == "__main__":
    main()
