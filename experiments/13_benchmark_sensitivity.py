"""Experiment 13: one changed benchmark assumption at a time.

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
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from hevi_hitl import certificates as C
from hevi_hitl import scaled as sc
from hevi_hitl import social as so
from hevi_hitl.insight import (SEVERE, decision_value, misspecification, per_state_cost, plan,
                               save_json, social_configs)
from hevi_hitl.model import PortfolioSolver

RESULTS = ROOT / "results"
TOLERANCE = 0.005
EPS = 1e-10
_spec = importlib.util.spec_from_file_location("exp04", ROOT / "experiments" / "04_certificates.py")
exp04 = importlib.util.module_from_spec(_spec)
sys.modules["exp04"] = exp04
_spec.loader.exec_module(exp04)
COORDINATES = exp04.COORDINATES["social"]


def lattice(n: int) -> tuple:
    return tuple(map(tuple, sc.preference_weights(n)))


VARIANTS = {
    "base": {},
    "exposure 0.2 m": dict(exposure_depth=0.2),
    "exposure 0.5 m": dict(exposure_depth=0.5),
    "graded exposure 0.2-0.6 m": dict(exposure_depth=0.2, exposure_depth_full=0.6),
    "shelter 60%": dict(shelter_share=0.6),
    "shelter 30%": dict(shelter_share=0.3),
    "compliance 0.8": dict(evacuation_rate=0.8),
    "compliance 0.7": dict(evacuation_rate=0.7),
    "delay loss 2%/h": dict(evacuation_decay=0.02),
    "delay loss 5%/h": dict(evacuation_decay=0.05),
    "delay loss 10%/h": dict(evacuation_decay=0.10),
    "barrier limit 1.0 m": dict(barrier_limit=1.0),
    "barrier limit 1.4 m": dict(barrier_limit=1.4),
    "10 preference types": dict(preference_weights=lattice(10)),
    "15 preference types": dict(preference_weights=lattice(15)),
    "comparison stakes x1.5": dict(comparison_stakes=1.5),
    "comparison stakes x2": dict(comparison_stakes=2.0),
}


def preference_prior(cfg: so.SocialConfig, weights) -> object:
    """Map the grid's two preference priors to K lattice types (diffuse; 0.7 on the most economic type)."""
    if weights is None or cfg.preference_prior == "diffuse":
        return cfg.preference_prior
    w = np.asarray(weights)
    k = int(np.argmin(np.linalg.norm(w - np.array([0.75, 0.15, 0.10]), axis=1)))
    p = np.full(len(w), 0.3 / (len(w) - 1))
    p[k] = 0.7
    return tuple(float(x) for x in p)


def variant_config(cfg: so.SocialConfig, kw: dict) -> so.SocialConfig:
    out = replace(cfg, **kw)
    if "preference_weights" in kw:
        out = replace(out, preference_prior=preference_prior(cfg, kw["preference_weights"]))
    return out


def type_values(model, cost: np.ndarray) -> np.ndarray:
    types = model.indices[:, 5]
    return np.array([float(model.prior[types == k] @ cost[types == k] / model.prior[types == k].sum())
                     for k in range(int(types.max()) + 1)])


def certificates(model) -> tuple[dict, list[dict]]:
    """Smallest certified class counts at TOLERANCE and single-coordinate DRIs."""
    full = PortfolioSolver(model)
    value = full.solve()[0]
    _, _, features = full.evaluate(full)
    optimum = dict(value=float(value), physical=exp04.physical(features))
    immediate_value = float(PortfolioSolver(model, method="immediate").solve()[0])
    U, _ = C.undominated_endpoints(model)
    sets, cmax = C.feasible_query_sets(model)
    J_all = C.endpoint_columns(model, C.endpoint_mask(model))
    J_U = C.endpoint_columns(model, U)
    lower = C.perfect_information_bound(model)
    n = len(model.prior)
    counts = sorted({max(1, round(n * k / 16)) for k in range(1, 17)})
    hierarchy = C.guided_hierarchy(model, counts, J_U=J_U, sets=sets, cmax=cmax)
    out = dict(configurations=n, class_level_classes=n, per_configuration_classes=n)
    for k in sorted(hierarchy):
        cert = exp04.candidate(model, hierarchy[k], U, sets, cmax, lower, optimum, J_all, J_U, with_tilde=False)
        if cert["refined_lb"] <= TOLERANCE and out["class_level_classes"] == n:
            out["class_level_classes"] = cert["classes"]
        if cert["prop1_postsolve"] <= TOLERANCE:
            out["per_configuration_classes"] = cert["classes"]
            break
    dri = []
    for j, name in enumerate(COORDINATES):
        inverse = C.marginal_partition(model.indices, (j,))
        guard = max(lower, C.query_relaxation_bound(model, inverse))
        cert = exp04.candidate(model, inverse, U, sets, cmax, guard, optimum, J_all, J_U, with_tilde=False)
        dri.append(dict(coordinate=name, dri=cert["refined_lb"], regret=cert["regret"], classes=cert["classes"],
                        certificate=float(min(cert["refined_postsolve"], cert["original_policy_value"] - lower)),
                        voi_joint=float(immediate_value - value)))
    return out, dri


def analyse(task) -> tuple[dict, list[dict]]:
    variant, cid, cfg = task
    start = time.perf_counter()
    vcfg = variant_config(cfg, VARIANTS[variant])
    model = so.build_social_flood(vcfg)
    dv = decision_value(model)
    d, mi = dv["decomposition"], dv["minus_immediate"]["joint"]
    joint, immediate = plan(model, "joint"), plan(model, "immediate")
    cj, ci = per_state_cost(model, joint), per_state_cost(model, immediate)
    sev = model.indices[:, 0] == SEVERE
    severe_diff = []
    for fp in (0, 1):
        mask = sev & (model.indices[:, 1] == fp)
        w = model.prior * mask
        severe_diff.append(float(w @ (cj - ci) / w.sum()))
    gains = type_values(model, ci) - type_values(model, cj)
    row = dict(variant=variant, configuration_id=cid, configurations=len(model.prior), actions=len(model.actions),
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
    if variant.startswith("barrier"):
        base_model = so.build_social_flood(cfg)
        mis = misspecification(model, base_model)
        row.update(misspecified_regret=mis["regret"], misspecified_versus_planned_immediate=mis["versus_planned_immediate"])
    cert, dri = certificates(model)
    row.update(cert)
    for r in dri:
        r.update(variant=variant, configuration_id=cid)
    row["seconds"] = time.perf_counter() - start
    return row, dri


def summarise(df: pd.DataFrame, dri: pd.DataFrame) -> dict:
    out = []
    for variant, g in df.groupby("variant", sort=False):
        d = dri[dri.variant == variant]
        cert = d[d.dri <= TOLERANCE].groupby("coordinate").size().reindex(COORDINATES, fill_value=0)
        rec = dict(variant=variant, instances=int(len(g)), querying=int(g.queries.sum()),
                   comparisons_used=int((g.comparisons > EPS).sum()),
                   mean_voi_joint=float(g.voi_joint.mean()), max_voi_joint=float(g.voi_joint.max()),
                   voi_preference_positive=int((g.voi_preference > EPS).sum()),
                   interaction_positive=int((g.interaction > EPS).sum()),
                   mean_interaction=float(g.interaction.mean()),
                   mean_captured_share=float(g.captured_share.mean()),
                   mean_delta_exposed=float(g.delta_exposed.mean()),
                   min_delta_exposed=float(g.delta_exposed.min()), max_delta_exposed=float(g.delta_exposed.max()),
                   mean_delta_damage_spend_eur=float(g.delta_damage_spend_eur.mean()),
                   worse_in_highest_severity=int(((g.severe_footprint_1 > EPS) & (g.severe_footprint_2 > EPS)).sum()),
                   worse_in_highest_severity_any=int(((g.severe_footprint_1 > EPS) | (g.severe_footprint_2 > EPS)).sum()),
                   some_type_worse_off=int((g.types_worse_off > 0).sum()),
                   median_class_level_share=float((g.class_level_classes / g.configurations).median()),
                   median_per_configuration_share=float((g.per_configuration_classes / g.configurations).median()),
                   class_level_compressed=int((g.class_level_classes < g.configurations).sum()),
                   dri_certified={k: int(v) for k, v in cert.items()})
        if "misspecified_regret" in g:
            rec.update(misspecified_max_regret=float(g.misspecified_regret.max()),
                       misspecified_harmful=int((g.misspecified_versus_planned_immediate > EPS).sum()))
        out.append(rec)
    return dict(variants=out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    csv, dcsv = RESULTS / "benchmark_sensitivity.csv", RESULTS / "benchmark_sensitivity_dri.csv"
    if not args.analysis_only:
        tasks = [(v, cid, cfg) for v in VARIANTS for cid, cfg in enumerate(social_configs())]
        # Large models first so that the pool ends with short tasks.
        tasks.sort(key=lambda t: 0 if "types" in t[0] or "delay" in t[0] else 1)
        start = time.perf_counter()
        rows, dris = [], []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for k, (row, dri) in enumerate(pool.map(analyse, tasks, chunksize=1)):
                rows.append(row)
                dris.extend(dri)
                if (k + 1) % 16 == 0:
                    pd.DataFrame(rows).to_csv(csv, index=False)
                    pd.DataFrame(dris).to_csv(dcsv, index=False)
                    print(f"{k + 1}/{len(tasks)} tasks: {time.perf_counter() - start:.0f} s", flush=True)
        df, dri = pd.DataFrame(rows), pd.DataFrame(dris)
        df.to_csv(csv, index=False)
        dri.to_csv(dcsv, index=False)
        runtime = time.perf_counter() - start
    else:
        df, dri = pd.read_csv(csv), pd.read_csv(dcsv)
        runtime = None
    summary = summarise(df, dri)
    if runtime is not None:
        summary["runtime_seconds"] = runtime
    save_json(RESULTS / "benchmark_sensitivity.json", summary)
    cols = ["variant", "querying", "comparisons_used", "mean_voi_joint", "interaction_positive", "mean_delta_exposed",
            "worse_in_highest_severity", "some_type_worse_off", "median_class_level_share"]
    print(pd.DataFrame(summary["variants"])[cols].to_string(index=False))


if __name__ == "__main__":
    main()
