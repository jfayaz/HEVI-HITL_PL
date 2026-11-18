"""Experiment 16: joint variation of the benchmark assumptions.

Section 5.7; Supplementary Section S3.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from hevi_hitl import social as so
from hevi_hitl.insight import SEVERE, decision_value, per_state_cost, plan, save_json, social_configs


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp13 = _load("exp13", "13_benchmark_sensitivity.py")
exp02 = _load("exp02", "02_random_instances.py")

RESULTS = ROOT / "results"
TOLERANCE = exp13.TOLERANCE
EPS = exp13.EPS
SEED = 2030
BOOT_SEED = 2031
BLOCK = 48
COORDINATES = exp13.COORDINATES
RETAINED = ("severity", "preference_type")   # coordinates that the paper keeps; the others are averaged out
QUARTER = 0.25                               # the screen certifies a DRI within this share of V_J

# Assumption levels (first level = benchmark) and the SocialConfig keywords each sets.
ASSUMPTIONS = {
    "exposure": {"0.3 m": {}, "0.2 m": dict(exposure_depth=0.2), "0.5 m": dict(exposure_depth=0.5),
                 "graded 0.2-0.6 m": dict(exposure_depth=0.2, exposure_depth_full=0.6)},
    "shelter": {0.9: {}, 0.6: dict(shelter_share=0.6), 0.3: dict(shelter_share=0.3)},
    "compliance": {0.9: {}, 0.8: dict(evacuation_rate=0.8), 0.7: dict(evacuation_rate=0.7)},
    "delay_loss": {0.0: {}, 0.02: dict(evacuation_decay=0.02), 0.05: dict(evacuation_decay=0.05),
                   0.10: dict(evacuation_decay=0.10)},
    "barrier_limit": {1.2: {}, 1.0: dict(barrier_limit=1.0), 1.4: dict(barrier_limit=1.4)},
    "types": {4: {}, 10: dict(preference_weights=exp13.lattice(10)),
              15: dict(preference_weights=exp13.lattice(15))},
    "stakes": {1.0: {}, 1.5: dict(comparison_stakes=1.5), 2.0: dict(comparison_stakes=2.0)},
}
GRID = ("lead_hours", "mechanism_strength", "social_query_reliability", "preference_prior")
BINARY = {"queries": "Any query", "comparisons_used": "Comparisons asked", "voi_preference_positive": "V_theta > 0",
          "interaction_positive": "V_Rtheta > 0", "some_type_worse_off": "Some type worse off",
          "severe_worse": "Worse in highest severity", "class_level_compressed": "Class-level compression",
          "class_below_per_configuration": "Class-level below per-configuration"}
CONTINUOUS = ("voi_joint", "voi_risk", "voi_preference", "interaction", "captured_share", "delta_exposed",
              "delta_damage_spend_eur", "class_level_share", "per_configuration_share", "commitment_hours")
INDEXED = ("voi_joint", "interaction", "comparisons_used", "some_type_worse_off", "delta_exposed", "class_level_share")


def design(blocks: int, seed: int = SEED) -> list[dict]:
    """Balanced random instance-assumption pairs in blocks of 48 (prefixes of complete blocks stay balanced)."""
    rng = np.random.default_rng(seed)
    pairs = []
    for b in range(blocks):
        cids = rng.permutation(np.repeat(np.arange(16), BLOCK // 16))
        cols = {name: rng.permutation(np.repeat(np.arange(len(levels)), BLOCK // len(levels)))
                for name, levels in ASSUMPTIONS.items()}
        for k in range(BLOCK):
            levels = {name: list(ASSUMPTIONS[name])[int(cols[name][k])] for name in ASSUMPTIONS}
            pairs.append(dict(pair=b * BLOCK + k, block=b, configuration_id=int(cids[k]), levels=levels))
    return pairs


def analyse(task) -> tuple[dict, list[dict]]:
    """Experiment 13's evaluation of one instance under one combination of assumption levels."""
    pair, cfg = task["pair"], task["cfg"]
    start = time.perf_counter()
    kw = {}
    for name, level in task["levels"].items():
        kw.update(ASSUMPTIONS[name][level])
    model = so.build_social_flood(exp13.variant_config(cfg, kw))
    dv = decision_value(model)
    d, mi = dv["decomposition"], dv["minus_immediate"]["joint"]
    joint, immediate = plan(model, "joint"), plan(model, "immediate")
    cj, ci = per_state_cost(model, joint), per_state_cost(model, immediate)
    sev = model.indices[:, 0] == SEVERE
    severe_diff = []
    for fp in (0, 1):
        w = model.prior * (sev & (model.indices[:, 1] == fp))
        severe_diff.append(float(w @ (cj - ci) / w.sum()))
    gains = exp13.type_values(model, ci) - exp13.type_values(model, cj)
    row = dict(pair=pair, block=task["block"], configuration_id=task["configuration_id"],
               **{name: level for name, level in task["levels"].items()},
               **{g: getattr(cfg, g) for g in GRID},
               configurations=len(model.prior), actions=len(model.actions),
               voi_joint=d["voi_joint"], voi_risk=d["voi_physical"], voi_preference=d["voi_preference"],
               interaction=d["interaction"], evpi=d["evpi"], captured_share=d["captured_share"],
               one_step_regret=d["one_step_regret"],
               queries=bool(dv["policies"]["joint"]["any_query"]),
               comparisons=dv["policies"]["joint"]["preference_queries"],
               delta_damage_spend_eur=mi["damage_plus_intervention_eur"],
               delta_displacement=mi["displacement_person_days"], delta_exposed=mi["exposed_people"],
               severe_footprint_1=severe_diff[0], severe_footprint_2=severe_diff[1],
               worst_type_gain=float(gains.min()), types_worse_off=int((gains < -EPS).sum()),
               commitment_hours=dv["policies"]["joint"]["commitment_hours"])
    cert, dri = exp13.certificates(model)
    row.update(cert)
    for r in dri:
        r.update(pair=pair)
    row["seconds"] = time.perf_counter() - start
    return row, dri


def outcomes(df: pd.DataFrame, dri: pd.DataFrame) -> pd.DataFrame:
    """Per-pair findings of the benchmark (binary) and continuous outcomes."""
    out = df.copy()
    certified = dri[dri.dri <= TOLERANCE].groupby("pair").coordinate.agg(frozenset)
    screens = out.pair.map(certified).apply(lambda s: s if isinstance(s, frozenset) else frozenset())
    out["certified"] = screens.apply(lambda s: "+".join(c for c in COORDINATES if c in s))
    out["comparisons_used"] = out.comparisons > EPS
    out["voi_preference_positive"] = out.voi_preference > EPS
    out["interaction_positive"] = out.interaction > EPS
    out["some_type_worse_off"] = out.types_worse_off > 0
    out["severe_worse"] = (out.severe_footprint_1 > EPS) & (out.severe_footprint_2 > EPS)
    out["class_level_compressed"] = out.class_level_classes < out.configurations
    out["class_below_per_configuration"] = out.class_level_classes < out.per_configuration_classes
    out["class_level_share"] = out.class_level_classes / out.configurations
    out["per_configuration_share"] = out.per_configuration_classes / out.configurations
    out["preference_prior"] = (out.preference_prior != "diffuse").astype(int)
    for b in BINARY:
        out[b] = out[b].astype(float)
    return out


def codes(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric level codes for the first-order index (each level is one group)."""
    cols = {name: df[name].map({str(v): k for k, v in enumerate(levels)} if df[name].dtype == object
                               else {v: k for k, v in enumerate(levels)})
            for name, levels in ASSUMPTIONS.items()}
    cols.update({g: df[g].astype(float) for g in GRID})
    return pd.DataFrame(cols)


def crosstab(o: pd.DataFrame, row: pd.Series, col: pd.Series, value: str) -> dict:
    """Mean of ``value`` and pair counts by two groupings, with string keys."""
    g = o.groupby([row, col])[value]
    return {f"{r}|{c}": dict(mean=float(m), pairs=int(n)) for (r, c), m, n in zip(g.mean().index, g.mean(), g.size())}


def interactions(o: pd.DataFrame) -> dict:
    """Two-way summaries quoted in the supplement."""
    delay5 = (o.delay_loss.astype(float) >= 0.05).map({True: "delay>=5%", False: "delay<5%"})
    prior = o.preference_prior.map({0: "diffuse", 1: "economic_concentrated"})
    barrier, shelter = o.barrier_limit.astype(str), o.shelter.astype(str)
    silent = o[o.queries == 0]
    return dict(
        queries_by_barrier_delay=crosstab(o, barrier, delay5, "queries"),
        not_querying=dict(pairs=int(len(silent)), barrier_1_4=int((silent.barrier_limit == 1.4).sum()),
                          delay_at_least_5=int((silent.delay_loss.astype(float) >= 0.05).sum())),
        voi_joint_by_barrier_shelter=crosstab(o, barrier, shelter, "voi_joint"),
        voi_joint_by_barrier_delay=crosstab(o, barrier, delay5, "voi_joint"),
        comparisons_by_delay_prior=crosstab(o, delay5, prior, "comparisons_used"),
        comparisons_by_barrier_delay=crosstab(o, barrier, delay5, "comparisons_used"),
        voi_preference_positive_by_barrier_prior=crosstab(o, barrier, prior, "voi_preference_positive"),
        interaction_positive_by_barrier_delay=crosstab(o, barrier, delay5, "interaction_positive"),
        worse_by_shelter_barrier=crosstab(o, shelter, barrier, "some_type_worse_off"),
        exposed_by_shelter_barrier=crosstab(o, shelter, barrier, "delta_exposed"),
    )


def summarise(df: pd.DataFrame, dri: pd.DataFrame, runtime: float | None) -> dict:
    o = outcomes(df, dri)
    factors = codes(o)
    rng = np.random.default_rng(BOOT_SEED)
    exp02.FACTORS = tuple(factors.columns)
    frame = pd.concat([factors, o[list(INDEXED)].astype(float)], axis=1)
    indices = {y: exp02.sensitivity(frame, y, rng) for y in INDEXED}
    floors = {y: exp02.permutation_floor(frame, y, rng) for y in INDEXED}
    by_level = []
    for name, levels in ASSUMPTIONS.items():
        for level in levels:
            g = o[o[name].astype(str) == str(level)]
            by_level.append(dict(assumption=name, level=str(level), benchmark=level == next(iter(levels)),
                                 pairs=int(len(g)), **{b: float(g[b].mean()) for b in BINARY},
                                 **{f"mean_{c}": float(g[c].mean()) for c in CONTINUOUS},
                                 median_class_level_share=float(g.class_level_share.median()),
                                 median_per_configuration_share=float(g.per_configuration_share.median())))
    certified_counts = (dri[dri.dri <= TOLERANCE].groupby("coordinate").size()
                        .reindex(COORDINATES, fill_value=0))
    asking = dri[dri.voi_joint > EPS]
    quarter = (asking[asking.dri <= QUARTER * asking.voi_joint].groupby("coordinate").size()
               .reindex(COORDINATES, fill_value=0))
    averaged = dri[~dri.coordinate.isin(RETAINED)]
    summary = dict(
        pairs=int(len(o)), blocks=int(o.block.nunique()), seed=SEED, bootstrap_seed=BOOT_SEED,
        bootstrap_resamples=1000, permutations=200, tolerance=TOLERANCE,
        assumptions={k: [str(v) for v in levels] for k, levels in ASSUMPTIONS.items()},
        shares={b: float(o[b].mean()) for b in BINARY},
        counts={b: int(o[b].sum()) for b in BINARY},
        continuous={c: dict(mean=float(o[c].mean()), median=float(o[c].median()), min=float(o[c].min()),
                            max=float(o[c].max())) for c in CONTINUOUS},
        certified_counts={k: int(v) for k, v in certified_counts.items()},
        certified_sets={k: int(v) for k, v in o.certified.value_counts().items()},
        querying_pairs=int(asking.pair.nunique()),
        quarter_certified={k: int(v) for k, v in quarter.items()},
        max_regret={c: float(dri[dri.coordinate == c].regret.max()) for c in COORDINATES},
        max_dri={c: float(dri[dri.coordinate == c].dri.max()) for c in COORDINATES},
        max_regret_averaged=float(averaged.regret.max()),
        max_dri_averaged=float(averaged.dri.max()),
        by_level=by_level,
        first_order_sensitivity=indices,
        permutation_floor_95=floors,
        first_order_sums={y: float(sum(v["estimate"] for v in indices[y].values())) for y in INDEXED},
        interactions=interactions(o),
        mean_seconds=float(o.seconds.mean()),
    )
    if runtime is not None:
        summary["runtime_seconds"] = runtime
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--blocks", type=int, default=12, help="blocks of 48 instance-assumption pairs")
    parser.add_argument("--append", action="store_true", help="keep finished pairs and run the rest")
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    csv, dcsv = RESULTS / "joint_sensitivity.csv", RESULTS / "joint_sensitivity_dri.csv"
    runtime = None
    if not args.analysis_only:
        configs = social_configs()
        rows, dris, done, elapsed = [], [], set(), 0.0
        if args.append and csv.exists():
            old = pd.read_csv(csv)
            rows, dris = old.to_dict("records"), pd.read_csv(dcsv).to_dict("records")
            done = set(old.pair)
            elapsed = float(old.seconds.sum())
        tasks = [dict(p, cfg=configs[p["configuration_id"]]) for p in design(args.blocks) if p["pair"] not in done]
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for k, (row, dri) in enumerate(pool.map(analyse, tasks, chunksize=1)):
                rows.append(row)
                dris.extend(dri)
                if (k + 1) % 16 == 0 or k + 1 == len(tasks):
                    pd.DataFrame(rows).sort_values("pair").to_csv(csv, index=False)
                    pd.DataFrame(dris).sort_values(["pair"], kind="stable").to_csv(dcsv, index=False)
                    print(f"{k + 1}/{len(tasks)} pairs: {time.perf_counter() - start:.0f} s", flush=True)
        runtime = time.perf_counter() - start
        print(f"wall {runtime:.0f} s; task seconds before this run {elapsed:.0f}", flush=True)
    out = RESULTS / "joint_sensitivity.json"
    if runtime is None and out.exists():   # keep the wall time of the run that produced the saved pairs
        runtime = json.loads(out.read_text()).get("runtime_seconds")
    df, dri = pd.read_csv(csv), pd.read_csv(dcsv)
    complete = df.groupby("block").size()
    df = df[df.block.isin(complete[complete == BLOCK].index)]
    dri = dri[dri.pair.isin(df.pair)]
    summary = summarise(df, dri, runtime)
    save_json(out, summary)
    print(f"{summary['pairs']} pairs in {summary['blocks']} complete blocks")
    for b, share in summary["shares"].items():
        print(f"  {BINARY[b]:<38s} {100 * share:5.1f}%")
    print(pd.DataFrame(summary["by_level"])[["assumption", "level", "comparisons_used", "voi_preference_positive",
                                              "interaction_positive", "some_type_worse_off", "mean_voi_joint",
                                              "mean_delta_exposed"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
