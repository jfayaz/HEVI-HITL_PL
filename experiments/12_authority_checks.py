"""Experiment 12: value by preference type, agreed-type mismatch and timed interaction.

Sections 5.2 and 5.6.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from hevi_hitl import certificates as C
from hevi_hitl import flood as fl
from hevi_hitl import social as so
from hevi_hitl.insight import (decision_value, flood_configs, per_state_cost, plan, preference_axis,
                               save_json, social_configs, with_prior)
from hevi_hitl.model import PortfolioSolver

RESULTS = ROOT / "results"
EPS = 1e-10
_spec = importlib.util.spec_from_file_location("exp04", ROOT / "experiments" / "04_certificates.py")
exp04 = importlib.util.module_from_spec(_spec)
sys.modules["exp04"] = exp04
_spec.loader.exec_module(exp04)


def type_values(model, cost: np.ndarray) -> np.ndarray:
    types = model.indices[:, preference_axis(model)]
    out = np.zeros(types.max() + 1)
    for k in range(len(out)):
        w = model.prior * (types == k)
        out[k] = float(w @ cost / w.sum()) if w.sum() > 0 else np.nan
    return out


def per_type(task) -> dict:
    domain, cid, cfg = task
    model = fl.build_flood(cfg) if domain == "flood" else so.build_social_flood(cfg)
    joint = type_values(model, per_state_cost(model, plan(model, "joint")))
    immediate = type_values(model, per_state_cost(model, plan(model, "immediate")))
    out = dict(domain=domain, configuration_id=cid, joint=joint.tolist(), immediate=immediate.tolist(),
               gain=(immediate - joint).tolist(),
               type_prior=[float(model.prior[model.indices[:, 5] == k].sum()) for k in range(len(joint))])
    if domain == "social":
        types = model.indices[:, 5]
        agreed = {}
        for k in range(len(joint)):
            prior_k = model.prior * (types == k)
            policy = plan(with_prior(model, prior_k / prior_k.sum()), "joint")
            agreed[k] = type_values(model, per_state_cost(model, policy)).tolist()
        out["agreed"] = agreed
        # (c) prior-specific exact lumping of mixed access levels
        idx = model.indices
        access = np.where(idx[:, 6] + idx[:, 7] == 1, 1, idx[:, 6] + idx[:, 7])
        key = np.column_stack((idx[:, :6], access))
        inverse = np.unique(key, axis=0, return_inverse=True)[1].ravel()
        full = PortfolioSolver(model)
        value = full.solve()[0]
        _, _, features = full.evaluate(full)
        optimum = dict(value=float(value), physical=exp04.physical(features))
        U, _ = C.undominated_endpoints(model)
        sets, cmax = C.feasible_query_sets(model)
        J_all = C.endpoint_columns(model, C.endpoint_mask(model))
        J_U = C.endpoint_columns(model, U)
        cert = exp04.candidate(model, inverse, U, sets, cmax, C.perfect_information_bound(model), optimum,
                               J_all, J_U, with_tilde=False)
        out["mixed_access_lumping"] = dict(classes=int(inverse.max() + 1), regret=cert["regret"],
                                           certificate=cert["refined_lb"], class_level=cert["refined_postsolve"])
        # (d) timed interaction without query cost and with a longer lead
        for name, variant in (("free_queries", replace(cfg, query_cost=0.0)), ("lead_8h", replace(cfg, lead_hours=8))):
            d = decision_value(so.build_social_flood(variant))["decomposition"]
            out[name] = dict(voi_joint=d["voi_joint"], voi_physical=d["voi_physical"],
                             voi_preference=d["voi_preference"], interaction=d["interaction"])
    return out


def summarise(results: list[dict]) -> dict:
    rows = []
    for r in results:
        for k, (g, p) in enumerate(zip(r["gain"], r["type_prior"])):
            rows.append(dict(domain=r["domain"], configuration_id=r["configuration_id"], type=k,
                             type_prior=p, gain=g))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "authority_checks_types.csv", index=False)
    summary = {}
    for d, g in df.groupby("domain"):
        q = g[g.groupby("configuration_id").gain.transform(lambda s: (s.abs() > EPS).any())]
        summary[f"per_type_{d}"] = dict(
            querying_instances=int(q.configuration_id.nunique()),
            type_cases=int(len(q)), worse_than_immediate=int((q.gain < -EPS).sum()),
            largest_loss=float(-q.gain.min()) if len(q) else 0.0,
            largest_gain=float(q.gain.max()) if len(q) else 0.0)
    social = [r for r in results if r["domain"] == "social"]
    mismatch, worse_than_joint, worse_than_imm, losses = 0, 0, 0, []
    for r in social:
        for k, vals in r["agreed"].items():
            for j, v in enumerate(vals):
                if int(k) == j:
                    continue
                mismatch += 1
                worse_than_joint += int(v > r["joint"][j] + EPS)
                worse_than_imm += int(v > r["immediate"][j] + EPS)
                losses.append(v - r["joint"][j])
    oracle_gain = [r["joint"][j] - r["agreed"][j][j] for r in social for j in range(len(r["joint"]))]
    summary["agreed_type"] = dict(mismatched_cases=mismatch, worse_than_joint=worse_than_joint,
                                  worse_than_immediate=worse_than_imm, largest_loss_vs_joint=float(max(losses)),
                                  min_gain_when_correct=float(min(oracle_gain)))
    lump = [r["mixed_access_lumping"] for r in social]
    summary["mixed_access_lumping"] = dict(classes=sorted({x["classes"] for x in lump}),
                                           max_abs_regret=float(max(abs(x["regret"]) for x in lump)),
                                           max_certificate=float(max(x["certificate"] for x in lump)))
    for name in ("free_queries", "lead_8h"):
        inter = np.array([r[name]["interaction"] for r in social])
        base_pos = [r["configuration_id"] for r in social]
        summary[name] = dict(interaction_positive=int((inter > EPS).sum()), mean_interaction=float(inter.mean()),
                             max_interaction=float(inter.max()),
                             mean_voi_joint=float(np.mean([r[name]["voi_joint"] for r in social])),
                             instances=len(base_pos))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    tasks = [("flood", i, c) for i, c in enumerate(flood_configs())] + \
            [("social", i, c) for i, c in enumerate(social_configs())]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(per_type, tasks))
    summary = summarise(results)
    save_json(RESULTS / "authority_checks.json", dict(summary=summary, instances=results))
    for k, v in summary.items():
        print(k, v)


if __name__ == "__main__":
    main()
