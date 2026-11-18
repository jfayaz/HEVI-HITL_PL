"""Experiment 34: one trade settled before the warning, answer durations and answer laws.

Sections 5.2 and 5.7; Supplementary Section S16.
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
from dataclasses import replace
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from hevi_hitl import social as so
from hevi_hitl.moments import MomentSolver

RESULTS = ROOT / "results"
SEED = 2029                       # authorities of experiment 14
DISTINCT = (2, 3, 6, 7, 10, 11, 14, 15)   # social instances with access-check reliability 0.9
DURATIONS = (1, 2, 3)             # half-hours per comparison
LAWS = ("logistic", "probit", "step")
NAMES = ("economic", "displacement", "exposure", "equal")
EPS = 1e-10


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp33 = load("33_question_design")
exp19 = load("19_response_models")


def social_configs():
    grid = so.GRID
    return [so.SocialConfig(**dict(zip(grid, v))) for v in product(*grid.values())]


def type_means(model, cost):
    types = model.indices[:, 5]
    return np.array([np.average(cost[types == k], weights=model.prior[types == k]) for k in range(4)])


def expected_comparisons(model, solver, prior):
    """Expected number of comparisons (queries 5 and 6) asked by a solved policy under ``prior``."""
    total = 0.0
    stack = [(solver.root, prior.copy())]
    while stack:
        h, mass = stack.pop()
        kind, j = solver.decision(h)
        if kind == "act":
            continue
        if j >= 5:
            total += mass.sum()
        for y in (0, 1):
            stack.append((h[:j] + (y,) + h[j + 1:], mass * (model.likelihood[:, j] if y else 1 - model.likelihood[:, j])))
    return float(total)


def groups_of(model, d):
    """Catalogue types that prefer the first disputed portfolio (lower score) when the trade is d."""
    return exp33.weights(model) @ d < 0


def settled_solvers(model, accept):
    """Joint policies planned for the accepting and the rejecting types (None for an empty group)."""
    types = model.indices[:, 5]
    out = {}
    for side in (True, False):
        mask = accept[types] == side
        if model.prior[mask].sum() <= 0:
            out[side] = None
            continue
        solver = MomentSolver(replace(model, prior=model.prior * mask))
        solver.solve()
        out[side] = solver
    return out


def instance(task):
    cid, cfg = task
    start = time.perf_counter()
    model = so.build_social_flood(cfg)
    types = model.indices[:, 5]
    p_type = np.array([model.prior[types == k].sum() for k in range(4)])
    imm = MomentSolver(model, method="immediate")
    S = imm.solve()[0]
    c_imm = exp33.per_configuration_cost(model, imm)
    joint = MomentSolver(model)
    L = joint.solve()[0]
    c_joint = exp33.per_configuration_cost(model, joint)
    risk_model = exp33.with_comparisons(model, [])
    L_risk = MomentSolver(risk_model).solve()[0]
    pref_model = replace(model, likelihood=model.likelihood[:, 5:], durations=model.durations[5:], costs=model.costs[5:],
                         queries=tuple(model.queries[5:]))
    L_pref = MomentSolver(pref_model).solve()[0]
    # preference type known before the warning
    c_agreed = np.zeros(len(model.prior))
    for k in range(4):
        solver = MomentSolver(replace(model, prior=model.prior * (types == k)))
        solver.solve()
        c_agreed[types == k] = exp33.per_configuration_cost(model, solver)[types == k]
    row = dict(configuration_id=cid, lead_hours=cfg.lead_hours, mechanism_strength=cfg.mechanism_strength,
               preference_prior=cfg.preference_prior, access_reliability=cfg.social_query_reliability,
               voi_risk=float(S - L_risk), voi_preference=float(S - L_pref), voi_joint=float(S - L),
               voi_agreed=float(model.prior @ (c_imm - c_agreed)),
               joint_comparisons=expected_comparisons(model, joint, model.prior))
    for name, g in zip(NAMES, type_means(model, c_imm) - type_means(model, c_joint)):
        row[f"joint_gain_{name}"] = float(g)
    for name, g in zip(NAMES, type_means(model, c_imm) - type_means(model, c_agreed)):
        row[f"agreed_gain_{name}"] = float(g)
    d, info = exp33.trade_direction(model)
    row["trade_found"] = d is not None
    if d is not None:
        accept = groups_of(model, d)
        solvers = settled_solvers(model, accept)
        c_settled = np.zeros(len(model.prior))
        comparisons = 0.0
        for side, solver in solvers.items():
            if solver is None:
                continue
            mask = accept[types] == side
            c_settled[mask] = exp33.per_configuration_cost(model, solver)[mask]
            comparisons += expected_comparisons(model, solver, model.prior * mask)
        gains = type_means(model, c_imm) - type_means(model, c_settled)
        row.update(voi_settled=float(model.prior @ (c_imm - c_settled)), settled_comparisons=comparisons,
                   accepting_types="+".join(n for n, a in zip(NAMES, accept) if a),
                   rejecting_types="+".join(n for n, a in zip(NAMES, accept) if not a),
                   rejecting_mass=float(p_type[~accept].sum()),
                   rejecting_commits_at_once=bool(solvers[False] is not None
                                                  and solvers[False].decision(solvers[False].root)[0] == "act"),
                   trade_eur=info["eur"], trade_person_days=info["person_days"], trade_exposed=info["exposed"])
        for name, g in zip(NAMES, gains):
            row[f"settled_gain_{name}"] = float(g)
    # answer duration
    for ticks in DURATIONS:
        slow = replace(model, durations=np.concatenate((model.durations[:5], np.full(2, ticks, dtype=int))))
        slow_pref = replace(pref_model, durations=np.full(2, ticks, dtype=int))
        vj = S - MomentSolver(slow).solve()[0]
        vp = S - MomentSolver(slow_pref).solve()[0]
        row[f"voi_joint_{30 * ticks}min"] = float(vj)
        row[f"interaction_{30 * ticks}min"] = float(vj - (S - L_risk) - vp)
    row["seconds"] = time.perf_counter() - start
    return row


def authorities(n):
    return np.random.default_rng(SEED).dirichlet(np.ones(3), size=n)


def authority_instance(task):
    cid, cfg, n_auth = task
    model = so.build_social_flood(cfg)
    types = model.indices[:, 5]
    phi = exp33.normalized_features(model)
    W = exp33.weights(model)
    assert np.allclose(np.einsum("zak,zk->za", phi, W[types]), model.loss[:, 0, :], atol=1e-12)
    imm = MomentSolver(model, method="immediate")
    imm.solve()
    joint = MomentSolver(model)
    joint.solve()
    d, _ = exp33.trade_direction(model)
    accept = groups_of(model, d)
    solvers = settled_solvers(model, accept)
    rho, slope = cfg.original_reliability, cfg.preference_slope
    rows = []
    for a, theta in enumerate(authorities(n_auth)):
        loss = phi @ theta
        lik = model.likelihood.copy()
        for k in range(2):
            lik[:, 5 + k] = 0.5 + rho * (1.0 / (1.0 + np.exp(-slope * float(theta @ exp33.BENCHMARK_PAIRS[k]))) - 0.5)
        truth = replace(model, loss=np.broadcast_to(loss[:, None, :], model.loss.shape), likelihood=lik)
        score = {name: float(truth.prior @ exp33.per_configuration_cost(truth, solver))
                 for name, solver in (("immediate", imm), ("joint", joint))}
        side = bool(theta @ d < 0)
        chosen = solvers[side] if solvers[side] is not None else joint
        score["settled"] = float(truth.prior @ exp33.per_configuration_cost(truth, chosen))
        oracle = float(MomentSolver(truth).solve()[0])
        rows.append(dict(configuration_id=cid, authority=a, w_money=theta[0], w_displacement=theta[1],
                         w_exposure=theta[2], accepts=side, immediate=score["immediate"], joint=score["joint"],
                         settled=score["settled"], oracle=oracle,
                         oracle_immediate=float(MomentSolver(truth, method="immediate").solve()[0])))
    return rows


def law_instance(task):
    sid, cfg = task
    model = so.build_social_flood(cfg)
    immediate = MomentSolver(model, method="immediate").solve()[0]
    W = exp33.weights(model)
    rho, slope = cfg.original_reliability, cfg.preference_slope
    d, _ = exp33.trade_direction(model)
    unit = d / np.linalg.norm(d)
    rows = []
    for law in LAWS:
        def answer(delta):
            return exp19.answer_probabilities(law, W, np.asarray(delta)[None, :], slope, rho)[:, 0]

        def value(deltas):
            return exp33.voi(exp33.with_comparisons(model, [answer(x) for x in deltas]), immediate)

        row = dict(setting=sid, law=law, lead_hours=cfg.lead_hours, mechanism_strength=cfg.mechanism_strength,
                   preference_prior=cfg.preference_prior, voi_none=value([]),
                   voi_pair0=value([exp33.BENCHMARK_PAIRS[0]]), voi_pair1=value([exp33.BENCHMARK_PAIRS[1]]),
                   voi_both=value(list(exp33.BENCHMARK_PAIRS)), voi_direct=value([d]))
        for k, m in enumerate(exp33.MAGNITUDES):
            row[f"voi_trade{k}"] = value([m * unit])
        rows.append(row)
    return rows


def ratio_range(num, den):
    r = num / den
    return [float(r.min()), float(r.max())]


def summarise(df, auth, laws):
    s = {}
    t = df[df.trade_found]
    s["instances"] = dict(
        n=int(len(df)), trade_found=int(len(t)),
        mean_voi={k: float(df[f"voi_{k}"].mean()) for k in ("risk", "joint", "agreed")},
        mean_voi_settled=float(t.voi_settled.mean()),
        joint_share_of_agreed_gain=float((df.voi_joint - df.voi_risk).sum() / (df.voi_agreed - df.voi_risk).sum()),
        settled_share_of_agreed_gain=float((t.voi_settled - t.voi_risk).sum() / (t.voi_agreed - t.voi_risk).sum()),
        settled_over_agreed=ratio_range(t.voi_settled, t.voi_agreed),
        settled_mean_over_agreed_mean=float(t.voi_settled.mean() / t.voi_agreed.mean()),
        joint_mean_over_agreed_mean=float(df.voi_joint.mean() / df.voi_agreed.mean()),
        rejecting_types=sorted(set(t.rejecting_types)), accepting_types=sorted(set(t.accepting_types)),
        rejecting_commits_at_once=int(t.rejecting_commits_at_once.sum()),
        settled_comparisons_max=float(t.settled_comparisons.max()),
        settled_instances_asking_comparisons=int((t.settled_comparisons > EPS).sum()),
        joint_instances_asking_comparisons=int((df.joint_comparisons > EPS).sum()),
        joint_types_worse_off=int((df[[f"joint_gain_{n}" for n in NAMES]].min(axis=1) < -EPS).sum()),
        settled_types_worse_off=int((t[[f"settled_gain_{n}" for n in NAMES]].min(axis=1) < -EPS).sum()),
        settled_min_gain=float(t[[f"settled_gain_{n}" for n in NAMES]].min().min()),
        joint_min_gain=float(df[[f"joint_gain_{n}" for n in NAMES]].min().min()),
        trade_eur=[float(t.trade_eur.abs().min()), float(t.trade_eur.abs().max())],
        trade_person_days=[float(t.trade_person_days.abs().min()), float(t.trade_person_days.abs().max())],
        trade_exposed=[float(t.trade_exposed.abs().min()), float(t.trade_exposed.abs().max())])
    s["duration"] = {}
    for ticks in DURATIONS:
        m = 30 * ticks
        vj, inter = df[f"voi_joint_{m}min"], df[f"interaction_{m}min"]
        s["duration"][f"{m}min"] = dict(
            mean_voi_joint=float(vj.mean()), mean_interaction=float(inter.mean()),
            interaction_positive=int((inter > EPS).sum()), comparisons_add_value=int((vj > df.voi_risk + EPS).sum()),
            share_of_agreed_gain=float((vj - df.voi_risk).sum() / (df.voi_agreed - df.voi_risk).sum()),
            interaction_positive_4h=int(((inter > EPS) & (df.lead_hours == 4)).sum()),
            interaction_positive_6h=int(((inter > EPS) & (df.lead_hours == 6)).sum()))
    if auth is not None and len(auth):
        a = auth.assign(gain_joint=auth.immediate - auth.joint, gain_settled=auth.immediate - auth.settled,
                        gain_oracle=auth.immediate - auth.oracle)
        high, low = a[a.w_exposure > 0.4], a[a.w_exposure < 0.3]
        s["authorities"] = dict(
            pairs=int(len(a)), authorities=int(a.authority.nunique()),
            joint_harmed_share=float((a.gain_joint < -EPS).mean()),
            settled_harmed_share=float((a.gain_settled < -EPS).mean()),
            settled_worst_gain=float(a.gain_settled.min()), joint_worst_gain=float(a.gain_joint.min()),
            joint_share_of_oracle_gain=float(a.gain_joint.sum() / a.gain_oracle.sum()),
            settled_share_of_oracle_gain=float(a.gain_settled.sum() / a.gain_oracle.sum()),
            mean_gain=dict(joint=float(a.gain_joint.mean()), settled=float(a.gain_settled.mean()),
                           oracle=float(a.gain_oracle.mean())),
            accepting_share=float(a.accepts.mean()),
            rejecting_max_oracle_voi=float((a[~a.accepts].oracle_immediate - a[~a.accepts].oracle).max()) if (~a.accepts).any() else None,
            exposure_above_0p4=dict(pairs=int(len(high)), joint_harmed_share=float((high.gain_joint < -EPS).mean()),
                                    settled_harmed_share=float((high.gain_settled < -EPS).mean()),
                                    accepting_share=float(high.accepts.mean()),
                                    max_oracle_voi=float((high.oracle_immediate - high.oracle).max())),
            exposure_below_0p3=dict(pairs=int(len(low)), joint_harmed_share=float((low.gain_joint < -EPS).mean()),
                                    settled_harmed_share=float((low.gain_settled < -EPS).mean()),
                                    accepting_share=float(low.accepts.mean())),
            rejecting_exposure_weight=[float(a[~a.accepts].w_exposure.min()), float(a[~a.accepts].w_exposure.max())] if (~a.accepts).any() else None,
            accepting_exposure_weight=[float(a[a.accepts].w_exposure.min()), float(a[a.accepts].w_exposure.max())])
    s["laws"] = {}
    for law, g in laws.groupby("law"):
        s["laws"][law] = dict(
            settings=int(len(g)),
            mean_voi={k: float(g[f"voi_{k}"].mean()) for k in ("none", "pair0", "pair1", "both", "direct", "trade0", "trade1")},
            trade1_over_both=ratio_range(g.voi_trade1, g.voi_both), trade1_over_pair1=ratio_range(g.voi_trade1, g.voi_pair1),
            trade0_over_both=ratio_range(g.voi_trade0, g.voi_both), trade0_over_pair0=ratio_range(g.voi_trade0, g.voi_pair0),
            trade1_beats_both=int((g.voi_trade1 > g.voi_both + EPS).sum()),
            trade1_equals_pair1=int(((g.voi_trade1 - g.voi_pair1).abs() <= EPS).sum()),
            trade1_beats_pair1=int((g.voi_trade1 > g.voi_pair1 + EPS).sum()),
            trade0_beats_pair0=int((g.voi_trade0 > g.voi_pair0 + EPS).sum()),
            trade0_beats_both=int((g.voi_trade0 > g.voi_both + EPS).sum()))
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--authorities", type=int, default=300)
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    configs = social_configs()
    if not args.analysis_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(instance, list(enumerate(configs))))
            laws = [r for part in pool.map(law_instance, list(enumerate(exp33.settings()))) for r in part]
            auth = [r for part in pool.map(authority_instance, [(cid, configs[cid], args.authorities) for cid in DISTINCT])
                    for r in part]
        pd.DataFrame(rows).to_csv(RESULTS / "settled_trade.csv", index=False)
        pd.DataFrame(laws).to_csv(RESULTS / "settled_trade_laws.csv", index=False)
        pd.DataFrame(auth).to_csv(RESULTS / "settled_trade_authorities.csv", index=False)
    df = pd.read_csv(RESULTS / "settled_trade.csv")
    laws = pd.read_csv(RESULTS / "settled_trade_laws.csv")
    auth = pd.read_csv(RESULTS / "settled_trade_authorities.csv")
    summary = summarise(df, auth, laws)
    summary["runtime_seconds"] = None if args.analysis_only else time.perf_counter() - start
    (RESULTS / "settled_trade.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
