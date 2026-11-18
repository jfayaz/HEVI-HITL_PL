"""Experiment 15: decision-relevance indices on the four largest scaled instances.

Section 5.8; Supplementary Section S2.
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
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from hevi_hitl import certificates as C
from hevi_hitl.insight import save_json
from hevi_hitl.moments import MomentSolver
from hevi_hitl.scaled import LADDER, FastSolver, ScaledConfig, aggregate_scaled, build_scaled, evaluate

RESULTS = ROOT / "results"
TOLERANCE = 0.005
COORDINATES = ("severity", "footprint", "occupancy", "physical_vulnerability", "restoration",
               "preference_type", "evacuation_access", "recovery_access")
OTHERS = tuple(j for j, name in enumerate(COORDINATES) if name not in ("severity", "preference_type"))


def optimum_of(rung: int) -> tuple[float, float]:
    recs = json.loads((RESULTS / "scaling.json").read_text(encoding="utf-8"))["records"]
    rec = recs[rung][0] if isinstance(recs[rung], list) else recs[rung]
    cfg = ScaledConfig(**{k: v for k, v in rec["config"].items() if k in {f.name for f in fields(ScaledConfig)}})
    assert cfg == LADDER[rung]
    return float(rec["optimal_value"]), float(rec.get("full_total_seconds", rec.get("full_dp_seconds")))


def analyse(task) -> dict:
    rung, j = task
    tick = time.perf_counter()
    model = build_scaled(LADDER[rung])
    build_s = time.perf_counter() - tick
    optimum, exact_s = optimum_of(rung)
    tick = time.perf_counter()
    U, _ = C.undominated_endpoints(model)
    sets, cmax = C.feasible_query_sets(model)
    J_U = C.endpoint_columns(model, U)
    lower = C.perfect_information_bound(model, U)
    setup_s = time.perf_counter() - tick
    immediate = MomentSolver(model, method="immediate").solve()[0]
    # j == len(COORDINATES): every coordinate except severity and preference type is averaged out
    axes = (j,) if j < len(COORDINATES) else OTHERS
    inverse = C.marginal_partition(model.indices, axes)
    tick = time.perf_counter()
    reduced = aggregate_scaled(model, inverse)
    solver = FastSolver(reduced)
    surrogate = solver.solve()[0]
    solve_s = time.perf_counter() - tick
    tick = time.perf_counter()
    ev = evaluate(model, lambda h, mass: solver.memo[h][1])
    guard = max(lower, C.query_relaxation_bound(model, inverse))
    cert = C.refined_certificate(model, reduced, inverse, ev["value"], surrogate, U, None,
                                 lower_bound=guard, sets=sets, cmax=cmax, J_U=J_U)
    certify_s = time.perf_counter() - tick
    regret = ev["value"] - optimum
    assert cert["refined_lb"] >= regret - 1e-10
    return dict(rung=rung, configurations=len(model.prior), queries=len(model.queries),
                coordinate=COORDINATES[j] if j < len(COORDINATES) else "others",
                classes=int(inverse.max() + 1), dri=cert["refined_lb"], class_level=cert["refined_postsolve"],
                certificate=float(min(cert["refined_postsolve"], ev["value"] - lower)),
                per_configuration=cert["prop1_postsolve"], regret=regret, voi_bound=ev["value"],
                voi_joint=float(immediate - optimum),
                build_seconds=build_s, setup_seconds=setup_s, solve_seconds=solve_s, certify_seconds=certify_s,
                exact_seconds=exact_s)


def summarise(df: pd.DataFrame) -> dict:
    out = []
    for rung, g in df.groupby("rung"):
        dri_s = float((g.solve_seconds + g.certify_seconds).sum() + g.setup_seconds.iloc[0] + g.build_seconds.iloc[0])
        out.append(dict(rung=int(rung), configurations=int(g.configurations.iloc[0]),
                        certified=sorted(g.loc[g.dri <= TOLERANCE, "coordinate"]),
                        dri={r.coordinate: float(r.dri) for r in g.itertuples()},
                        regret={r.coordinate: float(r.regret) for r in g.itertuples()},
                        all_dri_seconds=dri_s, exact_seconds=float(g.exact_seconds.iloc[0]),
                        min_certificate_minus_regret=float((g.dri - g.regret).min())))
    return dict(instances=out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rungs", type=int, nargs="+", default=[4, 5, 6, 7])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    csv = RESULTS / "scaled_relevance.csv"
    if not args.analysis_only:
        tasks = [(r, j) for r in sorted(args.rungs, reverse=True) for j in range(len(COORDINATES) + 1)]
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = []
            for row in pool.map(analyse, tasks):
                rows.append(row)
                print(f"rung {row['rung']} {row['coordinate']}: DRI {row['dri']:.5f} regret {row['regret']:.2e} "
                      f"solve {row['solve_seconds']:.0f}s", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(csv, index=False)
        runtime = time.perf_counter() - start
    else:
        df = pd.read_csv(csv)
        runtime = None
    summary = summarise(df)
    if runtime is not None:
        summary["runtime_seconds"] = runtime
    save_json(RESULTS / "scaled_relevance.json", summary)
    for rec in summary["instances"]:
        print(rec["configurations"], "certified:", rec["certified"], "DRI time %.0f s vs exact %.0f s"
              % (rec["all_dri_seconds"], rec["exact_seconds"]))


if __name__ == "__main__":
    main()
