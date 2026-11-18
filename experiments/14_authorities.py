"""Experiment 14: 300 authorities with weights outside the preference types.

Section 5.7; Supplementary Section S3.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import importlib.util
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from hevi_hitl import flood as fl
from hevi_hitl import social as so
from hevi_hitl.insight import per_state_cost, plan, save_json, social_configs
from hevi_hitl.model import PortfolioSolver

RESULTS = ROOT / "results"
SEED = 2029
EPS = 1e-10
INSTANCES = (2, 3, 6, 7, 10, 11, 14, 15)          # access-check reliability 0.9
_spec = importlib.util.spec_from_file_location("exp13", ROOT / "experiments" / "13_benchmark_sensitivity.py")
exp13 = importlib.util.module_from_spec(_spec)
sys.modules["exp13"] = exp13
_spec.loader.exec_module(exp13)
TYPE_SETS = {4: None, 10: exp13.lattice(10)}


def authorities(n: int) -> np.ndarray:
    return np.random.default_rng(SEED).dirichlet(np.ones(3), size=n)


def analyse(task) -> list[dict]:
    cid, n_auth, first, last = task
    cfg = social_configs()[cid]
    thetas = authorities(n_auth)[first:last]
    planned = {}
    for k, weights in TYPE_SETS.items():
        pcfg = exp13.variant_config(cfg, {} if weights is None else dict(preference_weights=weights))
        model = so.build_social_flood(pcfg)
        planned[k] = (pcfg, plan(model, "joint"), plan(model, "immediate"), len(fl.weights_of(pcfg)))
    rows = []
    for a, theta in enumerate(thetas, start=first):
        oracle = None
        for k, (pcfg, joint, immediate, n_types) in planned.items():
            # The truth model repeats the authority's weights for every type, so the type prior is irrelevant.
            tcfg = replace(pcfg, preference_weights=tuple(tuple(float(x) for x in theta) for _ in range(n_types)))
            truth = so.build_social_flood(tcfg)
            score_joint = float(truth.prior @ per_state_cost(truth, joint))
            score_immediate = float(truth.prior @ per_state_cost(truth, immediate))
            if oracle is None or a == first:
                # The oracle does not depend on the type set; it is recomputed once per chunk as a check.
                found = (float(PortfolioSolver(truth).solve()[0]),
                         float(PortfolioSolver(truth, method="immediate").solve()[0]))
                if oracle is not None:
                    assert max(abs(found[0] - oracle[0]), abs(found[1] - oracle[1])) < 1e-9
                oracle = found
            rows.append(dict(configuration_id=cid, types=k, authority=a, w_money=theta[0], w_displacement=theta[1],
                             w_exposure=theta[2], joint=score_joint, immediate=score_immediate, oracle=oracle[0],
                             oracle_immediate=oracle[1], gain=score_immediate - score_joint,
                             regret=score_joint - oracle[0], oracle_voi=oracle[1] - oracle[0]))
    return rows


def summarise(df: pd.DataFrame) -> dict:
    out = {}
    for k, g in df.groupby("types"):
        harmed = g.gain < -EPS
        out[f"types_{k}"] = dict(cases=int(len(g)), harmed=int(harmed.sum()), harmed_share=float(harmed.mean()),
                                 mean_gain=float(g.gain.mean()), worst_gain=float(g.gain.min()),
                                 mean_regret=float(g.regret.mean()), max_regret=float(g.regret.max()),
                                 mean_oracle_voi=float(g.oracle_voi.mean()),
                                 harmed_mean_exposure_weight=float(g.loc[harmed, "w_exposure"].mean()) if harmed.any() else None,
                                 unharmed_mean_exposure_weight=float(g.loc[~harmed, "w_exposure"].mean()))
    paired = df.pivot_table(index=["configuration_id", "authority"], columns="types", values="joint")
    out["ten_types_better"] = int((paired[10] < paired[4] - EPS).sum())
    out["ten_types_worse"] = int((paired[10] > paired[4] + EPS).sum())
    out["pairs"] = int(len(paired))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorities", type=int, default=300)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--chunk", type=int, default=25)
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    csv = RESULTS / "authorities.csv"
    if not args.analysis_only:
        start = time.perf_counter()
        # Split each instance into chunks of authorities so that all workers stay busy.
        tasks = [(cid, args.authorities, s, min(args.authorities, s + args.chunk))
                 for cid in INSTANCES for s in range(0, args.authorities, args.chunk)]
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(analyse, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), start=1):
                rows.extend(future.result())
                if i % 8 == 0 or i == len(futures):     # save the finished prefix as the run proceeds
                    pd.DataFrame(rows).to_csv(csv, index=False)
                    print(f"{i}/{len(futures)} tasks: {time.perf_counter() - start:.0f} s", flush=True)
        df = pd.DataFrame(rows).sort_values(["configuration_id", "types", "authority"], ignore_index=True)
        df.to_csv(csv, index=False)
        runtime = time.perf_counter() - start
    else:
        df = pd.read_csv(csv)
        runtime = None
    summary = summarise(df)
    if runtime is not None:
        summary["runtime_seconds"] = runtime
    save_json(RESULTS / "authorities.json", summary)
    for k, v in summary.items():
        print(k, v)


if __name__ == "__main__":
    main()
