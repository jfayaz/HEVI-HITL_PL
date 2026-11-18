"""Experiment 11: planned against true answer reliability on a 0.05 grid.

Section 5.4.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from hevi_hitl import social as so
from hevi_hitl.insight import misspecification, save_json, social_configs

RESULTS = ROOT / "results"
GRID = tuple(np.round(np.arange(0.50, 1.0001, 0.05), 2))
EPS = 1e-10


def variant(cfg: so.SocialConfig, case: str, rho: float) -> so.SocialConfig:
    if case == "joint":
        return replace(cfg, original_reliability=rho, comparison_reliability=None)
    if case == "diagnostic":
        return replace(cfg, original_reliability=rho, comparison_reliability=0.9)
    return replace(cfg, original_reliability=0.9, comparison_reliability=rho)


def analyse(task) -> list[dict]:
    cid, cfg = task
    rows = []
    for case in ("joint", "diagnostic", "comparison"):
        models = {rho: so.build_social_flood(variant(cfg, case, rho)) for rho in GRID}
        for rt in GRID:
            cache: dict = {}
            for rp in GRID:
                out = misspecification(models[rt], models[rp], cache)
                rows.append(dict(configuration_id=cid, case=case, planned=rp, true=rt, **out))
    return rows


def summarise(df: pd.DataFrame) -> dict:
    df = df.assign(harmful=df.versus_planned_immediate > EPS, shift=np.round(df.planned - df.true, 2))
    out = {}
    for case, g in df.groupby("case", sort=False):
        harm = g[g.harmful]
        by_shift = (g.groupby("shift").agg(cases=("harmful", "size"), harmful=("harmful", "sum"),
                                           max_harm=("versus_planned_immediate", "max"))
                    .reset_index().to_dict(orient="records"))
        mean_regret = g.groupby(["planned", "true"]).regret.mean().unstack()
        worst = mean_regret.max(axis=1)
        out[case] = dict(cases=int(len(g)), harmful_cases=int(len(harm)),
                         harmful_only_when_overstated=bool((harm["shift"] > 0).all()) if len(harm) else True,
                         smallest_harmful_overstatement=float(harm["shift"].min()) if len(harm) else None,
                         largest_harm=float(g.versus_planned_immediate.max()),
                         max_true_reliability_with_harm=float(harm["true"].max()) if len(harm) else None,
                         minimax_planned_reliability=float(worst.idxmin()),
                         minimax_worst_mean_regret=float(worst.min()),
                         worst_mean_regret_by_planned={f"{k:.2f}": float(v) for k, v in worst.items()},
                         by_shift=by_shift)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    csv = RESULTS / "reliability_grid.csv"
    if not args.analysis_only:
        start = time.perf_counter()
        tasks = list(enumerate(social_configs()))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = [r for part in pool.map(analyse, tasks) for r in part]
        df = pd.DataFrame(rows)
        df.to_csv(csv, index=False)
        runtime = time.perf_counter() - start
    else:
        df = pd.read_csv(csv)
        runtime = None
    summary = summarise(df)
    summary["grid"] = list(GRID)
    if runtime is not None:
        summary["runtime_seconds"] = runtime
    save_json(RESULTS / "reliability_grid.json", summary)
    for case, s in summary.items():
        if isinstance(s, dict):
            print(case, {k: v for k, v in s.items() if k not in ("by_shift", "worst_mean_regret_by_planned")})


if __name__ == "__main__":
    main()
