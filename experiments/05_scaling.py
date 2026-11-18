"""Experiment 05: history recursion and one-step planning on the scaled instances.

Section 5.8; Supplementary Sections S2 and S8.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from hevi_hitl.scaled import LADDER, FastSolver, build_scaled, evaluate

RESULTS = ROOT / "results"
TIME_LIMIT = 1800.0


def timed(function):
    start = time.perf_counter()
    value = function()
    return value, time.perf_counter() - start


def run_instance(cfg, full: bool = True) -> dict:
    """Time the history recursion and one-step planning on one instance."""
    rec = dict(config=asdict(cfg), states=cfg.states, queries=cfg.query_count)
    model, rec["build_seconds"] = timed(lambda: build_scaled(cfg))
    rec["actions"] = len(model.actions)
    optimum = None
    if full:
        solver = FastSolver(model, time_limit=TIME_LIMIT)
        try:
            value, rec["full_dp_seconds"] = timed(lambda: solver.solve()[0])
            optimum = float(value)
            rec["histories"] = len(solver.memo)
            rec["full_dp_completed"] = True
        except TimeoutError:
            rec["full_dp_seconds"] = TIME_LIMIT
            rec["full_dp_completed"] = False
        del solver
        gc.collect()
    rec["optimal_value"] = optimum
    one_step = FastSolver(model, method="one_step")
    out, rec["one_step_seconds"] = timed(lambda: evaluate(model, lambda h, mass: one_step.solve(h, mass)[1]))
    rec["one_step_value"] = out["value"]
    rec["one_step_regret"] = None if optimum is None else out["value"] - optimum
    del one_step
    if full and rec.get("full_dp_completed"):
        rec["full_total_seconds"] = rec["build_seconds"] + rec["full_dp_seconds"]
    del model
    gc.collect()
    return rec


def median_record(reps: list) -> dict:
    """Median timings across repetitions (values are identical across repetitions)."""
    out = dict(reps[0])
    out["repetitions"] = len(reps)
    for key in ("build_seconds", "full_dp_seconds", "one_step_seconds", "full_total_seconds"):
        if out.get(key) is not None:
            out[key] = float(np.median([r[key] for r in reps]))
            out[f"{key}_range"] = [float(min(r[key] for r in reps)), float(max(r[key] for r in reps))]
    return out


def main():
    global TIME_LIMIT
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rungs", type=int, default=len(LADDER))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--repeat-threshold", type=int, default=10080, help="larger instances are run once")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT,
                        help="time limit (s) of the exact recursion; the two largest instances used 36,000")
    parser.add_argument("--append", action="store_true", help="run only the instances missing from scaling.json")
    args = parser.parse_args()
    TIME_LIMIT = args.time_limit
    out = RESULTS / "scaling.json"
    records = json.loads(out.read_text(encoding="utf-8"))["records"] if args.append else []
    done = {r["states"] for r in records}
    with threadpool_limits(1):
        assert all(p["num_threads"] == 1 for p in threadpool_info() if p["user_api"] == "blas")
        for cfg in LADDER[:args.max_rungs]:
            if cfg.states in done:
                continue
            reps = []
            n_rep = args.repeats if cfg.states <= args.repeat_threshold else 1
            for rep in range(n_rep):
                print(f"instance {cfg.states} states, Q={cfg.query_count}, repetition {rep + 1}/{n_rep}", flush=True)
                reps.append(run_instance(cfg))
                r = reps[-1]
                print(f"  build {r['build_seconds']:.1f}s full {r.get('full_dp_seconds')}s "
                      f"one-step {r['one_step_seconds']:.2f}s", flush=True)
            records.append(median_record(reps))
            records[-1]["all_repetitions"] = [{k: v for k, v in r.items() if k != "config"} for r in reps]
            out.write_text(json.dumps(dict(records=records), indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
