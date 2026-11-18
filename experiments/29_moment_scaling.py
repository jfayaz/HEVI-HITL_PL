"""Experiment 29: moment recursion and certified reduction against configurations and queries.

Section 5.8; Supplementary Section S8, Fig. S2.
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
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import psutil

from hevi_hitl import certificates as C
from hevi_hitl.moments import MomentSolver
from hevi_hitl.scaled import LADDER, ScaledConfig, aggregate_scaled, build_scaled, evaluate

RESULTS = ROOT / "results"
OUT = RESULTS / "moment_scaling.json"
LEAD_HOURS = 10
QUERY_SIZES = {
    "768": (ScaledConfig(3, 2, 4, 0, False, lead_hours=LEAD_HOURS), range(0, 7)),
    "4800": (ScaledConfig(5, 3, 10, 2, False, lead_hours=LEAD_HOURS), range(0, 5)),
    "10080": (ScaledConfig(7, 3, 15, 2, True, lead_hours=LEAD_HOURS), range(0, 4)),
}
TOLERANCES = {"ladder": (0.01, 0.005), "queries": (0.02, 0.01, 0.005, 0.0025, 0.001)}


def schedule(n: int):
    out, k = [], 48
    while k < n:
        out.append(k)
        k *= 2
    return out + [n]


def peak_mb() -> float:
    info = psutil.Process().memory_info()
    return float(getattr(info, "peak_wset", info.rss)) / 2 ** 20


def instances():
    out = [("ladder", f"{cfg.states}", cfg) for cfg in LADDER]
    for size, (cfg, more) in QUERY_SIZES.items():
        out += [("queries", f"{size}+{m}", replace(cfg, more_pairs=m)) for m in more]
    return out


def run(task) -> dict:
    kind, part, name, cfg = task
    rec = dict(kind=kind, part=part, name=name, states=cfg.states, queries=cfg.query_count,
               lead_hours=cfg.lead_hours, config=asdict(cfg))
    start = time.perf_counter()
    model = build_scaled(cfg)
    rec["build_seconds"] = time.perf_counter() - start
    solver = MomentSolver(model)
    tick = time.perf_counter()
    optimum = solver.solve()[0]
    rec["moment_seconds"] = time.perf_counter() - tick
    rec["moment_parts"] = dict(solver.timings)
    rec["value"] = float(optimum)
    rec["histories"] = solver.histories
    if kind == "moment":
        rec["total_seconds"] = rec["build_seconds"] + rec["moment_seconds"]
        rec["peak_mb"] = peak_mb()
        return rec
    del solver
    gc.collect()
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
    tolerances = TOLERANCES[part]
    for k in schedule(len(model.prior)):
        tick = time.perf_counter()
        inverse = refiner.refine_to(k)
        refine_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        reduced = aggregate_scaled(model, inverse)
        rsolver = MomentSolver(reduced)
        surrogate = rsolver.solve()[0]
        solve_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        ev = evaluate(model, lambda h, mass: rsolver.decision(h))
        cert = C.refined_certificate(model, reduced, inverse, ev["value"], surrogate, U, None,
                                     lower_bound=lower, sets=sets, cmax=cmax, J_U=J_U)
        cert_seconds = time.perf_counter() - tick
        elapsed += refine_seconds + solve_seconds + cert_seconds
        regret = ev["value"] - optimum
        assert cert["refined_lb"] >= regret - 1e-10
        candidates.append(dict(classes=k, refine_seconds=refine_seconds, solve_seconds=solve_seconds,
                               certify_seconds=cert_seconds, cumulative_seconds=elapsed,
                               bound=cert["refined_lb"], regret=regret, evaluated_histories=ev["visited_histories"]))
        for tol in tolerances:
            if tol not in accepted and cert["refined_lb"] <= tol:
                accepted[tol] = dict(classes=k, seconds=elapsed, bound=cert["refined_lb"], regret=regret)
        print(f"   {name}/{cfg.query_count}q K={k}: bound={cert['refined_lb']:.5f} regret={regret:.2e} "
              f"solve={solve_seconds:.1f}s cert={cert_seconds:.1f}s", flush=True)
        del reduced, rsolver
        gc.collect()
        if min(tolerances) in accepted:
            break
    rec["candidates"] = candidates
    rec["accepted"] = {f"{t:g}": accepted.get(t) for t in tolerances}
    rec["peak_mb"] = peak_mb()
    return rec


def load() -> list:
    return json.loads(OUT.read_text()) if OUT.exists() else []


def history_times() -> dict:
    """History-recursion times (build plus DP) of experiments 5 and 21, keyed like ``name``."""
    out = {}
    scaling = RESULTS / "scaling.json"
    if scaling.exists():
        for r in json.loads(scaling.read_text())["records"]:
            if r.get("full_dp_completed"):
                out[("ladder", f"{r['states']}")] = (r["full_total_seconds"], r["optimal_value"])
    qs = RESULTS / "query_scaling.json"
    if qs.exists():
        for r in json.loads(qs.read_text()):
            if r["kind"] == "exact" and r.get("completed"):
                out[("queries", f"{r['size']}+{r['more_pairs']}")] = (r["total_seconds"], r["value"])
    return out


def analyse(records: list) -> pd.DataFrame:
    hist = history_times()
    moment = {(r["part"], r["name"]): r for r in records if r["kind"] == "moment"}
    rows = []
    for r in records:
        if r["kind"] != "reduction":
            continue
        m = moment.get((r["part"], r["name"]))
        h = hist.get((r["part"], r["name"]))
        row = dict(part=r["part"], name=r["name"], states=r["states"], queries=r["queries"],
                   moment_total_seconds=None if m is None else m["total_seconds"],
                   moment_dp_seconds=None if m is None else m["moment_seconds"],
                   moment_peak_mb=None if m is None else m["peak_mb"],
                   build_seconds=r["build_seconds"], reduction_peak_mb=r["peak_mb"],
                   history_total_seconds=None if h is None else h[0],
                   history_value_difference=None if h is None else r["value"] - h[1])
        for tol, a in r["accepted"].items():
            row[f"classes_{tol}"] = None if a is None else a["classes"]
            row[f"seconds_{tol}"] = None if a is None else a["seconds"]
            row[f"regret_{tol}"] = None if a is None else a["regret"]
            if a is not None and m is not None:
                row[f"speedup_vs_moment_{tol}"] = m["total_seconds"] / a["seconds"]
            if a is not None and h is not None:
                row[f"speedup_vs_history_{tol}"] = h[0] / a["seconds"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["part", "states", "queries"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--min-states", type=int, default=0)
    parser.add_argument("--max-states", type=int, default=10 ** 9)
    parser.add_argument("--part", choices=("ladder", "queries", "both"), default="both")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--figure", action="store_true", help="draw figures/scalability from saved results")
    args = parser.parse_args()
    if args.figure:
        make_figure()
        return
    records = load()
    if not args.analysis_only:
        done = {(r["kind"], r["part"], r["name"]) for r in records}
        tasks = [(kind, part, name, cfg) for part, name, cfg in instances() for kind in ("moment", "reduction")
                 if (args.part in (part, "both")) and args.min_states <= cfg.states <= args.max_states
                 and (kind, part, name) not in done]
        tasks.sort(key=lambda t: -t[3].states * 3 ** t[3].query_count)
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
            for rec in pool.map(run, tasks, chunksize=1):
                records.append(rec)
                OUT.write_text(json.dumps(records, indent=1, default=float), encoding="utf-8")
                print(f"done {rec['kind']} {rec['part']} {rec['name']} ({rec['queries']}q): "
                      f"{rec.get('total_seconds', rec.get('setup_seconds', 0)):.1f} s, peak {rec['peak_mb']:.0f} MB "
                      f"[{time.perf_counter() - start:.0f} s]", flush=True)
    df = analyse(records)
    df.to_csv(RESULTS / "moment_scaling.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(df.to_string())


def make_figure():
    """Figure 'scalability': wall-clock time against configurations (a) and against queries (b)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from hevi_hitl.plotting import FULL_WIDTH_IN, PALETTE, apply_style, panel_label, save

    apply_style()
    records = load()
    moment = {(r["part"], r["name"]): r for r in records if r["kind"] == "moment"}
    reduction = {(r["part"], r["name"]): r for r in records if r["kind"] == "reduction"}
    ladder = json.loads((RESULTS / "scaling.json").read_text())["records"]
    queries = json.loads((RESULTS / "query_scaling.json").read_text())
    colours = {"history": "black", "moment": PALETTE["joint"], "reduction": PALETTE["refined"],
               "one_step": PALETTE["one_step"]}
    colours["reduction"] = PALETTE["regret"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.55), constrained_layout=True)
    # (a) against configurations
    hist = sorted((r["states"], r["full_total_seconds"]) for r in ladder if r.get("full_dp_completed"))
    ax.plot(*zip(*hist), "-o", color=colours["history"], label="History recursion")
    mom = sorted((r["states"], r["total_seconds"]) for k, r in moment.items() if k[0] == "ladder")
    ax.plot(*zip(*mom), "-s", color=colours["moment"], label="Moment recursion")
    red = sorted((r["states"], r["accepted"]["0.005"]["seconds"]) for k, r in reduction.items()
                 if k[0] == "ladder" and r["accepted"].get("0.005"))
    ax.plot(*zip(*red), "-^", color=colours["reduction"], label="Certified reduction")
    one = sorted((r["states"], r["one_step_seconds"] + r["build_seconds"]) for r in ladder)
    ax.plot(*zip(*one), ":", marker="D", color=colours["one_step"], label="One-step planning")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Latent configurations (log scale)")
    ax.set_ylabel("Wall-clock time (s, log scale)")
    panel_label(ax, "a", x=-0.02, y=1.03)
    # (b) against queries
    styles = {"768": "-", "4800": "--", "10080": ":"}
    for size, ls in styles.items():
        h = sorted((r["queries"], r["total_seconds"]) for r in queries
                   if r["kind"] == "exact" and r["size"] == size and r.get("completed"))
        bx.plot(*zip(*h), ls, marker="o", color=colours["history"], markersize=3)
        m = sorted((r["queries"], r["total_seconds"]) for k, r in moment.items()
                   if k[0] == "queries" and k[1].split("+")[0] == size)
        bx.plot(*zip(*m), ls, marker="s", color=colours["moment"], markersize=3)
        c = sorted((r["queries"], r["accepted"]["0.005"]["seconds"]) for k, r in reduction.items()
                   if k[0] == "queries" and k[1].split("+")[0] == size and r["accepted"].get("0.005"))
        if c:
            bx.plot(*zip(*c), ls, marker="^", color=colours["reduction"], markersize=3)
    bx.set_yscale("log")
    bx.set_xlabel("Binary queries (ten-hour lead)")
    bx.set_ylabel("Wall-clock time (s, log scale)")
    panel_label(bx, "b", x=-0.02, y=1.03)
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color="grey", ls=ls, label=f"{int(size):,} configurations") for size, ls in styles.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, handlelength=1.8, columnspacing=1.0,
               labelspacing=0.3)
    save(fig, "scalability")


if __name__ == "__main__":
    main()
