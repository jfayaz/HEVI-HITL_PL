"""Experiment 21: time and memory against the number of queries.

Section 5.8; Supplementary Section S8.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import psutil

from hevi_hitl import certificates as C
from hevi_hitl.scaled import FastSolver, ScaledConfig, aggregate_scaled, build_scaled, evaluate

RESULTS = ROOT / "results"
OUT = RESULTS / "query_scaling.json"
LEAD_HOURS = 10
TIME_LIMIT = 4 * 3600.0
TOLERANCES = (0.02, 0.01, 0.005, 0.0025, 0.001)
SIZES = {  # name: (base configuration, further comparisons to add)
    "768": (ScaledConfig(3, 2, 4, 0, False, lead_hours=LEAD_HOURS), range(0, 7)),
    "4800": (ScaledConfig(5, 3, 10, 2, False, lead_hours=LEAD_HOURS), range(0, 5)),
    "10080": (ScaledConfig(7, 3, 15, 2, True, lead_hours=LEAD_HOURS), range(0, 4)),
}


def schedule(n: int):
    out, k = [], 48
    while k < n:
        out.append(k)
        k *= 2
    return out + [n]


def peak_mb() -> float:
    info = psutil.Process().memory_info()
    return float(getattr(info, "peak_wset", info.rss)) / 2 ** 20


def run(task: tuple) -> dict:
    """One measurement in a fresh process: ('exact' | 'reduction', size, more_pairs)."""
    kind, size, more = task
    cfg = replace(SIZES[size][0], more_pairs=more)
    rec = dict(kind=kind, size=size, more_pairs=more, states=cfg.states, queries=cfg.query_count,
               lead_hours=cfg.lead_hours, config=asdict(cfg))
    start = time.perf_counter()
    model = build_scaled(cfg)
    rec["build_seconds"] = time.perf_counter() - start
    rec["after_build_mb"] = float(psutil.Process().memory_info().rss) / 2 ** 20
    if kind == "exact":
        solver = FastSolver(model, time_limit=TIME_LIMIT)
        tick = time.perf_counter()
        try:
            rec["value"] = float(solver.solve()[0])
            rec["completed"] = True
        except TimeoutError:
            rec["value"] = None
            rec["completed"] = False
        rec["dp_seconds"] = time.perf_counter() - tick
        rec["histories"] = len(solver.memo)
        rec["total_seconds"] = rec["build_seconds"] + rec["dp_seconds"]
        rec["peak_mb"] = peak_mb()
        return rec
    tick = time.perf_counter()
    U, _ = C.undominated_endpoints(model)
    sets, cmax = C.feasible_query_sets(model)
    J_U = C.endpoint_columns(model, U)
    lower = C.perfect_information_bound(model, U)
    refiner = C.GuidedRefiner(model, J_U=J_U, sets=sets, cmax=cmax)
    setup = time.perf_counter() - tick
    elapsed = rec["build_seconds"] + setup
    rec["setup_seconds"] = setup
    candidates, accepted = [], {}
    for k in schedule(len(model.prior)):
        tick = time.perf_counter()
        inverse = refiner.refine_to(k)
        reduced = aggregate_scaled(model, inverse)
        rsolver = FastSolver(reduced)
        surrogate = rsolver.solve()[0]
        solve_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        ev = evaluate(model, lambda h, mass: rsolver.memo[h][1])
        cert = C.refined_certificate(model, reduced, inverse, ev["value"], surrogate, U, None,
                                     lower_bound=lower, sets=sets, cmax=cmax, J_U=J_U)
        cert_seconds = time.perf_counter() - tick
        elapsed += solve_seconds + cert_seconds
        candidates.append(dict(classes=k, solve_seconds=solve_seconds, certify_seconds=cert_seconds,
                               cumulative_seconds=elapsed, bound=cert["refined_lb"], value=ev["value"],
                               reduced_histories=len(rsolver.memo)))
        for tol in TOLERANCES:
            if tol not in accepted and cert["refined_lb"] <= tol:
                accepted[tol] = dict(classes=k, seconds=elapsed, bound=cert["refined_lb"], value=ev["value"])
        print(f"   {size}/{cfg.query_count}q K={k}: bound={cert['refined_lb']:.5f} "
              f"solve={solve_seconds:.1f}s cert={cert_seconds:.1f}s", flush=True)
        del reduced, rsolver
        gc.collect()
        if min(TOLERANCES) in accepted:
            break
    rec["candidates"] = candidates
    rec["accepted"] = {f"{t:g}": accepted.get(t) for t in TOLERANCES}
    rec["peak_mb"] = peak_mb()
    return rec


def load() -> list:
    return json.loads(OUT.read_text()) if OUT.exists() else []


def analyse(records: list) -> pd.DataFrame:
    exact = {(r["size"], r["more_pairs"]): r for r in records if r["kind"] == "exact"}
    rows = []
    for r in records:
        if r["kind"] != "reduction":
            continue
        e = exact.get((r["size"], r["more_pairs"]))
        row = dict(states=r["states"], queries=r["queries"], reduction_peak_mb=r["peak_mb"],
                   reduction_setup_seconds=r["build_seconds"] + r["setup_seconds"])
        if e is not None:
            row.update(exact_seconds=e["total_seconds"] if e["completed"] else None, exact_peak_mb=e["peak_mb"],
                       exact_histories=e["histories"], exact_completed=e["completed"])
        for tol in TOLERANCES:
            a = r["accepted"].get(f"{tol:g}")
            row[f"classes_{tol:g}"] = None if a is None else a["classes"]
            row[f"seconds_{tol:g}"] = None if a is None else a["seconds"]
            if a is not None and e is not None and e["completed"]:
                row[f"speedup_{tol:g}"] = e["total_seconds"] / a["seconds"]
                row[f"regret_{tol:g}"] = a["value"] - e["value"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["states", "queries"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    records = load()
    if not args.analysis_only:
        done = {(r["kind"], r["size"], r["more_pairs"]) for r in records}
        tasks = [(kind, size, more) for size, (_, mores) in SIZES.items() for more in mores
                 for kind in ("exact", "reduction") if (kind, size, more) not in done]
        tasks.sort(key=lambda t: (SIZES[t[1]][0].states * 3 ** (SIZES[t[1]][0].query_count + t[2])))
        print(f"{len(tasks)} runs to do", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
            futures = {pool.submit(run, t): t for t in tasks}
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    rec = fut.result()
                except Exception as exc:  # keep the other runs going; record the failure
                    rec = dict(kind=t[0], size=t[1], more_pairs=t[2], error=repr(exc))
                records.append(rec)
                OUT.write_text(json.dumps(records, indent=1))
                print(f"done {t}: {rec.get('total_seconds', rec.get('peak_mb'))}", flush=True)
    df = analyse([r for r in records if "error" not in r])
    df.to_csv(RESULTS / "query_scaling.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
