"""Experiment 35: joint planning under a cap on expected exposed people.

Sections 5.2 and 5.7; Supplementary Section S16.
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
from dataclasses import replace
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from hevi_hitl import flood as fl
from hevi_hitl import social as so
from hevi_hitl.moments import MomentSolver
from hevi_hitl.scaled import ScaledConfig, build_scaled

RESULTS = ROOT / "results"
# name: (changes of the social configuration, depth-damage function)
VARIANTS = {"benchmark": ({}, "benchmark"), "shelter 60%": (dict(shelter_share=0.6), "benchmark"),
            "overtopping 1.4 m": (dict(barrier_limit=1.4), "benchmark"), "JRC damage function": ({}, "jrc_europe")}
NAMES = ("economic", "displacement", "exposure", "equal")
PREFERENCE_QUERIES = (5, 6)
DAMAGE, PERSON_DAYS, EXPOSED, SPEND = 0, 1, 2, 3      # columns of ``features``
EXPOSED_SCALE = fl.SCALES[2]    # 1,000 people, the normalisation of the score
EPS = 1e-10
TOL = 1e-9                      # people; a policy satisfies the cap within this tolerance


def social_configs():
    grid = so.GRID
    return [so.SocialConfig(**dict(zip(grid, v))) for v in product(*grid.values())]


def scaled_twin(cfg, damage_function="benchmark"):
    """Scaled-family model of a social instance (the smallest member reproduces it), with consequences."""
    return build_scaled(ScaledConfig(severity_levels=3, footprints=2, preference_types=4, district_gauges=0,
                                     extra_preference_pair=False, lead_hours=cfg.lead_hours,
                                     mechanism_strength=cfg.mechanism_strength,
                                     social_query_reliability=cfg.social_query_reliability,
                                     preference_prior=cfg.preference_prior, query_cost=cfg.query_cost,
                                     hazard_prior=cfg.hazard_prior, original_reliability=cfg.original_reliability,
                                     preference_slope=cfg.preference_slope, damage_function=damage_function),
                        with_features=True)


def build(variant, cfg):
    changes, damage_function = VARIANTS[variant]
    if damage_function == "benchmark":
        return so.build_social_flood(replace(cfg, **changes))
    assert not changes
    return scaled_twin(cfg, damage_function)


def scores(model):
    return model.loss2d if getattr(model, "loss2d", None) is not None else model.loss[:, 0, :]


def with_scores(model, values):
    """Copy of ``model`` whose time-invariant scores are ``values`` (configurations x portfolios)."""
    if getattr(model, "loss2d", None) is not None:
        return replace(model, loss2d=values, loss=np.broadcast_to(values[:, None, :], model.loss.shape))
    return replace(model, loss=np.repeat(values[:, None, :], model.loss.shape[1], axis=1))


def evaluate(model, solver, loss, features):
    """Per-configuration expected cost and consequences of a solved policy, and its expected queries."""
    n = len(model.prior)
    cost, outcome = np.zeros(n), np.zeros((n, features.shape[2]))
    asked = np.zeros(len(model.queries))
    lik = model.likelihood
    stack = [(solver.root, np.ones(n), 0.0)]
    while stack:
        h, prob, spent = stack.pop()
        kind, j = solver.decision(h)
        if kind == "act":
            cost += prob * (loss[:, j] + spent)
            outcome += prob[:, None] * features[:, j, :]
            continue
        asked[j] += model.prior @ prob
        for y in (0, 1):
            stack.append((h[:j] + (y,) + h[j + 1:], prob * (lik[:, j] if y else 1 - lik[:, j]),
                          spent + float(model.costs[j])))
    return cost, outcome, asked


def type_means(model, cost):
    types = model.indices[:, 5]
    return np.array([np.average(cost[types == k], weights=model.prior[types == k]) for k in range(4)])


def instance(task):
    variant, cid, cfg = task
    start = time.perf_counter()
    model = build(variant, cfg)
    prior, loss, features = model.prior, scores(model), np.asarray(model.features[:, 0])
    assert np.ptp(model.loss, axis=1).max() < 1e-12, "scores must not depend on the commitment time"
    exposed = features[:, :, EXPOSED]
    empty = [k for k, a in enumerate(model.actions) if not np.any(np.asarray(a))]
    assert len(empty) == 1 and np.all(features[:, empty[0], SPEND] == 0)
    s_none = float(prior @ loss[:, empty[0]])

    def point(mu, method="optimal"):
        solver = MomentSolver(with_scores(model, loss + mu * exposed / EXPOSED_SCALE) if mu else model, method=method)
        solver.solve()
        cost, outcome, asked = evaluate(model, solver, loss, features)
        mean = prior @ outcome
        return dict(mu=mu, cost=float(prior @ cost), people=float(mean[EXPOSED]), asked=asked, per_state=cost,
                    money=float(mean[DAMAGE] + mean[SPEND]), person_days=float(mean[PERSON_DAYS]))

    imm = point(0.0, method="immediate")
    S, X0 = imm["cost"], imm["people"]
    joint = point(0.0)
    v_joint = S - joint["cost"]
    row = dict(variant=variant, configuration_id=cid, lead_hours=cfg.lead_hours,
               mechanism_strength=cfg.mechanism_strength, preference_prior=cfg.preference_prior,
               access_reliability=cfg.social_query_reliability,
               score_none=s_none, score_immediate=S, score_joint=joint["cost"], voi_joint=v_joint,
               exposed_immediate=X0, exposed_joint=joint["people"],
               money_immediate=imm["money"], person_days_immediate=imm["person_days"],
               delta_money_joint=joint["money"] - imm["money"],
               delta_person_days_joint=joint["person_days"] - imm["person_days"],
               measures_gain=s_none - S, voi_share_of_measures=v_joint / (s_none - S),
               voi_share_of_immediate_score=v_joint / S)
    if joint["people"] <= X0 + TOL:        # the cap does not bind
        lo = hi = joint
        weight = 0.0
    else:
        mu_hi = 0.01
        hi = point(mu_hi)
        while hi["people"] > X0 + TOL:
            mu_hi *= 2
            assert mu_hi < 1e4, "no weight satisfies the cap"
            hi = point(mu_hi)
        lo = joint
        for _ in range(60):
            if hi["mu"] - lo["mu"] <= 1e-10 * max(1.0, hi["mu"]):
                break
            mid = point(0.5 * (lo["mu"] + hi["mu"]))
            if mid["people"] <= X0 + TOL:
                hi = mid
            else:
                lo = mid
        weight = hi["mu"]
    # deterministic capped policy (just above the breakpoint) and the optimal mixture at the breakpoint
    w_lo = (X0 - hi["people"]) / (lo["people"] - hi["people"]) if lo["people"] - hi["people"] > TOL else 0.0
    w_lo = float(np.clip(w_lo, 0.0, 1.0))
    mixed_cost = w_lo * lo["cost"] + (1 - w_lo) * hi["cost"]
    mixed_per_state = w_lo * lo["per_state"] + (1 - w_lo) * hi["per_state"]
    dual = hi["cost"] + weight * (hi["people"] - X0) / EXPOSED_SCALE    # Lagrangian bound at the breakpoint
    assert mixed_cost >= joint["cost"] - 1e-12 and hi["cost"] >= mixed_cost - 1e-12
    assert abs(mixed_cost - dual) <= 1e-6 * max(1.0, abs(dual)) + 1e-9, (mixed_cost, dual)
    gains_joint = type_means(model, imm["per_state"]) - type_means(model, joint["per_state"])
    gains_cap = type_means(model, imm["per_state"]) - type_means(model, hi["per_state"])
    gains_mix = type_means(model, imm["per_state"]) - type_means(model, mixed_per_state)
    row.update(cap_binds=bool(joint["people"] > X0 + TOL), weight=weight,
               voi_capped=S - hi["cost"], exposed_capped=hi["people"],
               delta_money_capped=hi["money"] - imm["money"],
               delta_person_days_capped=hi["person_days"] - imm["person_days"],
               voi_capped_mixed=S - mixed_cost, mixture_weight_uncapped=w_lo,
               exposed_below_breakpoint=lo["people"], voi_below_breakpoint=S - lo["cost"],
               capped_queries=float(hi["asked"].sum()),
               capped_comparisons=float(hi["asked"][list(PREFERENCE_QUERIES)].sum()),
               capped_risk_queries=float(hi["asked"][:5].sum()),
               joint_queries=float(joint["asked"].sum()),
               joint_comparisons=float(joint["asked"][list(PREFERENCE_QUERIES)].sum()),
               worst_type_gain_joint=float(gains_joint.min()), worst_type_gain_capped=float(gains_cap.min()),
               worst_type_gain_mixed=float(gains_mix.min()))
    for name, a, b in zip(NAMES, gains_joint, gains_cap):
        row[f"joint_gain_{name}"] = float(a)
        row[f"capped_gain_{name}"] = float(b)
    row["seconds"] = time.perf_counter() - start
    return row


def span(x):
    return [float(x.min()), float(x.mean()), float(x.max())]


def summarise(df):
    out = {}
    for variant, g in df.groupby("variant", sort=False):
        q = g[g.voi_joint > EPS]
        out[variant] = dict(
            instances=int(len(g)), querying=int(len(q)), cap_binds=int(g.cap_binds.sum()),
            mean_voi_joint=float(g.voi_joint.mean()), mean_voi_capped=float(g.voi_capped.mean()),
            mean_voi_capped_mixed=float(g.voi_capped_mixed.mean()),
            share_capped=span(q.voi_capped / q.voi_joint),
            share_capped_mixed=span(q.voi_capped_mixed / q.voi_joint),
            share_of_means=float(g.voi_capped.mean() / g.voi_joint.mean()),
            share_of_means_mixed=float(g.voi_capped_mixed.mean() / g.voi_joint.mean()),
            weight=[float(g.weight.min()), float(g.weight.max())],
            exposed_added_joint=span(g.exposed_joint - g.exposed_immediate),
            exposed_added_joint_percent=span(100 * (g.exposed_joint / g.exposed_immediate - 1)),
            exposed_added_capped=span(g.exposed_capped - g.exposed_immediate),
            delta_money_joint=span(g.delta_money_joint), delta_money_capped=span(g.delta_money_capped),
            delta_money_joint_percent=span(100 * g.delta_money_joint / g.money_immediate),
            delta_person_days_joint=span(g.delta_person_days_joint),
            delta_person_days_capped=span(g.delta_person_days_capped),
            capped_policies_querying=int((g.capped_queries > EPS).sum()),
            capped_policies_with_comparisons=int((g.capped_comparisons > EPS).sum()),
            joint_policies_with_comparisons=int((g.joint_comparisons > EPS).sum()),
            instances_with_type_worse_off_joint=int((g.worst_type_gain_joint < -EPS).sum()),
            instances_with_type_worse_off_capped=int((g.worst_type_gain_capped < -EPS).sum()),
            worst_type_gain_joint=float(g.worst_type_gain_joint.min()),
            worst_type_gain_capped=float(g.worst_type_gain_capped.min()),
            exposure_type_gain_capped=[float(g.capped_gain_exposure.min()), float(g.capped_gain_exposure.max())],
            voi_share_of_measures=span(g.voi_share_of_measures),
            voi_share_of_immediate_score=span(g.voi_share_of_immediate_score),
            measures_gain=span(g.measures_gain),
            measures_gain_share_of_none=float((g.measures_gain / g.score_none).mean()),
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    start = time.perf_counter()
    # the scaled family reproduces the social benchmark, so the JRC variant differs by the damage function only
    cfg0 = social_configs()[10]
    twin, base = scaled_twin(cfg0), so.build_social_flood(cfg0)
    assert np.allclose(scores(twin), scores(base), rtol=0, atol=1e-12) and np.allclose(twin.prior, base.prior)
    assert np.allclose(np.asarray(twin.features[:, 0]), base.features[:, 0], rtol=0, atol=1e-6)
    tasks = [(v, cid, cfg) for v in VARIANTS for cid, cfg in enumerate(social_configs())]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(instance, tasks, chunksize=1))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "exposure_cap.csv", index=False)
    out = summarise(df)
    out["runtime_seconds"] = time.perf_counter() - start
    (RESULTS / "exposure_cap.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    cols = ["instances", "cap_binds", "mean_voi_joint", "mean_voi_capped", "share_of_means", "share_of_means_mixed",
            "capped_policies_querying", "instances_with_type_worse_off_joint", "instances_with_type_worse_off_capped"]
    print(pd.DataFrame({v: {c: s[c] for c in cols} for v, s in out.items() if isinstance(s, dict)}).T.to_string())


if __name__ == "__main__":
    main()
