"""Experiment 33: one designed comparison and a type-protective planning prior.

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
from scipy.special import expit

from hevi_hitl import flood as fl
from hevi_hitl import social as so
from hevi_hitl.moments import MomentSolver

RESULTS = ROOT / "results"
BENCHMARK_PAIRS = (fl.PREFERENCE_PAIRS[:, 0] - fl.PREFERENCE_PAIRS[:, 1]) / fl.SCALES   # normalized differences
MAGNITUDES = tuple(float(np.linalg.norm(d)) for d in BENCHMARK_PAIRS)
LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 32.0)
EXPOSURE_TYPE = 2
EPS = 1e-10


def settings():
    return [so.SocialConfig(lead_hours=lead, mechanism_strength=s, preference_prior=p, social_query_reliability=0.9)
            for lead, s, p in product((4, 6), (0.5, 1.0), ("diffuse", "economic_concentrated"))]


def sphere(n: int) -> np.ndarray:
    """Near-uniform unit vectors (Fibonacci lattice)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)))


def weights(model) -> np.ndarray:
    return fl.weights_of(fl.FloodConfig(preference_prior=model.config.preference_prior))


def answer_law(model, d: np.ndarray) -> np.ndarray:
    """Probability, by preference type, of preferring the second profile when the first minus the second is d."""
    cfg = model.config
    return 0.5 + cfg.original_reliability * (expit(cfg.preference_slope * (weights(model) @ d)) - 0.5)


def with_comparisons(model, laws) -> object:
    """Social model whose comparisons are replaced by the given per-type answer laws (possibly none)."""
    cols = [model.likelihood[:, :5]] + [law[model.indices[:, 5], None] for law in laws]
    return replace(model, likelihood=np.column_stack(cols),
                   durations=np.concatenate((model.durations[:5], np.ones(len(laws), dtype=int))),
                   costs=np.concatenate((model.costs[:5], np.full(len(laws), model.costs[5]))),
                   queries=tuple(model.queries[:5]) + tuple(f"comparison_{j}" for j in range(len(laws))))


def voi(model, immediate: float) -> float:
    return immediate - MomentSolver(model).solve()[0]


def normalized_features(model) -> np.ndarray:
    """phi[z, a, 3]: (damage plus expenditure, displacement, exposed people) on the scales of Appendix C."""
    f = model.features[:, 0]
    return np.stack(((f[:, :, 0] + f[:, :, 3]) / fl.SCALES[0], f[:, :, 1] / fl.SCALES[1], f[:, :, 2] / fl.SCALES[2]), axis=2)


def trade_direction(model):
    """Consequence difference between the two portfolios that the preference types disagree about."""
    risk_only = with_comparisons(model, [])
    solver = MomentSolver(risk_only)
    solver.solve()
    loss = model.loss[:, 0, :]
    types = model.indices[:, 5]
    candidates = [((-1,) * 5, model.prior.copy(), 0)]
    kind, q = solver.decision(solver.root)
    if kind == "query":
        for y in (0, 1):
            lik = model.likelihood[:, q] if y else 1 - model.likelihood[:, q]
            h = tuple(y if j == q else -1 for j in range(5))
            candidates.append((h, model.prior * lik, int(model.durations[q])))
    best = None
    for h, mass, t in candidates:
        feasible = np.flatnonzero(model.feasible[t])
        pooled = (mass @ loss[:, feasible]).min()
        by_type = [(mass[types == k] @ loss[types == k][:, feasible]) for k in range(4)]
        separate = sum(v.min() for v in by_type)
        value = float(pooled - separate)
        if best is None or value > best[0] + EPS:
            choice = [int(feasible[np.argmin(v)]) for v in by_type]
            best = (value, h, mass, choice)
    value, h, mass, choice = best
    if value <= EPS or len(set(choice)) < 2:
        return None, dict(history=list(h), value_of_type=value, portfolios=choice)
    type_mass = np.array([mass[types == k].sum() for k in range(4)])
    groups = {}
    for k, a in enumerate(choice):
        groups[a] = groups.get(a, 0.0) + type_mass[k]
    (a1, _), (a2, _) = sorted(groups.items(), key=lambda kv: -kv[1])[:2]
    phi = normalized_features(model)
    d = (mass @ (phi[:, a1] - phi[:, a2])) / mass.sum()
    return d, dict(history=list(h), value_of_type=value, portfolios=[a1, a2], difference=d.tolist(),
                   eur=float(d[0] * fl.SCALES[0]), person_days=float(d[1] * fl.SCALES[1]), exposed=float(d[2] * fl.SCALES[2]))


def mutual_information(p_type: np.ndarray, law: np.ndarray) -> float:
    def h(p):
        p = np.clip(p, 1e-300, 1 - 1e-300)
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return float(h(p_type @ law) - p_type @ h(law))


def design(task):
    sid, cfg, n_dir = task
    start = time.perf_counter()
    model = so.build_social_flood(cfg)
    immediate = MomentSolver(model, method="immediate").solve()[0]
    types = model.indices[:, 5]
    p_type = np.array([model.prior[types == k].sum() for k in range(4)])
    out = dict(setting=sid, lead_hours=cfg.lead_hours, mechanism_strength=cfg.mechanism_strength,
               preference_prior=cfg.preference_prior)
    out["voi_none"] = voi(with_comparisons(model, []), immediate)
    out["voi_both"] = voi(model, immediate)
    for k in range(2):
        out[f"voi_pair{k}"] = voi(with_comparisons(model, [answer_law(model, BENCHMARK_PAIRS[k])]), immediate)
    d_trade, info = trade_direction(model)
    out["trade"] = info
    if d_trade is not None:   # the two portfolios' expected profiles themselves (no amplification)
        out["direct_magnitude"] = float(np.linalg.norm(d_trade))
        out["voi_direct"] = voi(with_comparisons(model, [answer_law(model, d_trade)]), immediate)
    dirs = sphere(n_dir)
    rows = []
    for m in MAGNITUDES:
        vals = np.array([voi(with_comparisons(model, [answer_law(model, m * u)]), immediate) for u in dirs])
        mi = np.array([mutual_information(p_type, answer_law(model, m * u)) for u in dirs])
        j_best, j_info = int(np.argmax(vals)), int(np.argmax(mi))
        row = dict(setting=sid, magnitude=m, voi_best=float(vals[j_best]), best_direction=dirs[j_best].tolist(),
                   voi_information=float(vals[j_info]), information_direction=dirs[j_info].tolist(),
                   voi_median=float(np.median(vals)), share_useful=float((vals > out["voi_none"] + EPS).mean()))
        if d_trade is not None:
            u = d_trade / np.linalg.norm(d_trade)
            row["voi_trade"] = voi(with_comparisons(model, [answer_law(model, m * u)]), immediate)
            row["cos_trade_best"] = float(abs(u @ dirs[j_best]))
            row["trade_direction"] = u.tolist()
        rows.append(row)
    out["seconds"] = time.perf_counter() - start
    return out, rows


def per_configuration_cost(model, solver) -> np.ndarray:
    out = np.zeros(len(model.prior))
    lik, loss = model.likelihood, model.loss[:, 0, :]
    stack = [(solver.root, np.ones(len(model.prior)), 0.0)]
    while stack:
        h, prob, spent = stack.pop()
        kind, j = solver.decision(h)
        if kind == "act":
            out += prob * (loss[:, j] + spent)
            continue
        for y in (0, 1):
            stack.append((h[:j] + (y,) + h[j + 1:], prob * (lik[:, j] if y else 1 - lik[:, j]), spent + float(model.costs[j])))
    return out


def tilt(task):
    cid, cfg = task
    model = so.build_social_flood(cfg)
    types = model.indices[:, 5]
    by_type = lambda cost: np.array([np.average(cost[types == k], weights=model.prior[types == k]) for k in range(4)])
    immediate = MomentSolver(model, method="immediate")
    immediate.solve()
    c_imm = per_configuration_cost(model, immediate)
    rows = []
    for lam in LAMBDAS:
        factor = 1 + lam * (types == EXPOSURE_TYPE)
        tilted = replace(model, prior=model.prior * factor / (model.prior @ factor))
        solver = MomentSolver(tilted)
        solver.solve()
        cost = per_configuration_cost(model, solver)
        gain = by_type(c_imm) - by_type(cost)
        rows.append(dict(configuration_id=cid, stress_weight=lam, voi=float(model.prior @ (c_imm - cost)),
                         queries=bool(solver.decision(solver.root)[0] == "query"),
                         **{f"gain_{name}": float(g) for name, g in zip(("economic", "displacement", "exposure", "equal"), gain)}))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--directions", type=int, default=400)
    args = parser.parse_args()
    start = time.perf_counter()
    grid = so.GRID
    social = [so.SocialConfig(**dict(zip(grid, v))) for v in product(*grid.values())]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        designs = list(pool.map(design, [(i, cfg, args.directions) for i, cfg in enumerate(settings())]))
        tilts = [r for part in pool.map(tilt, list(enumerate(social))) for r in part]
    heads = [d[0] for d in designs]
    rows = pd.DataFrame([r for d in designs for r in d[1]])
    head = pd.DataFrame([{k: v for k, v in h.items() if k != "trade"} for h in heads])
    df = rows.merge(head, on="setting")
    df.to_csv(RESULTS / "question_design.csv", index=False)
    tdf = pd.DataFrame(tilts)
    tdf.to_csv(RESULTS / "type_tilt.csv", index=False)
    summary = dict(directions=args.directions, magnitudes=list(MAGNITUDES), trade={h["setting"]: h["trade"] for h in heads})
    for m, g in df.groupby("magnitude"):
        t = g.dropna(subset=["voi_trade"]) if "voi_trade" in g else g.iloc[:0]
        summary[f"magnitude_{m:.3f}"] = dict(
            settings=int(len(g)), mean_voi_best=float(g.voi_best.mean()), mean_voi_both=float(g.voi_both.mean()),
            mean_voi_none=float(g.voi_none.mean()), mean_voi_information=float(g.voi_information.mean()),
            best_over_both=float((g.voi_best / g.voi_both).mean()),
            best_beats_both=int((g.voi_best > g.voi_both + EPS).sum()),
            trade_settings=int(len(t)), mean_voi_trade=float(t.voi_trade.mean()) if len(t) else None,
            trade_over_best=float((t.voi_trade / t.voi_best).mean()) if len(t) else None,
            min_trade_over_best=float((t.voi_trade / t.voi_best).min()) if len(t) else None,
            trade_over_both=[float((t.voi_trade / t.voi_both).min()), float((t.voi_trade / t.voi_both).max())] if len(t) else None,
            trade_beats_both=int((t.voi_trade > t.voi_both + EPS).sum()) if len(t) else None,
            trade_added_share=[float(((t.voi_trade - t.voi_none) / (t.voi_best - t.voi_none)).min()),
                               float(((t.voi_trade - t.voi_none) / (t.voi_best - t.voi_none)).max())] if len(t) else None,
            information_added_share=[float(((g.voi_information - g.voi_none) / (g.voi_best - g.voi_none)).min()),
                                     float(((g.voi_information - g.voi_none) / (g.voi_best - g.voi_none)).max())],
            information_adds_nothing=int((g.voi_information <= g.voi_none + EPS).sum()),
            information_over_best=float((g.voi_information / g.voi_best).mean()),
            min_cos_trade_best=float(t.cos_trade_best.min()) if len(t) else None,
            mean_share_useful=float(g.share_useful.mean()))
    if "voi_direct" in head:
        summary["direct"] = dict(mean_magnitude=float(head.direct_magnitude.mean()), mean_voi=float(head.voi_direct.mean()),
                                 over_both=[float((head.voi_direct / head.voi_both).min()), float((head.voi_direct / head.voi_both).max())],
                                 mean_voi_both=float(head.voi_both.mean()), mean_voi_none=float(head.voi_none.mean()))
    base = tdf[tdf.stress_weight == 0].set_index("configuration_id")
    tilt_summary = {}
    for lam, g in tdf.groupby("stress_weight"):
        g = g.set_index("configuration_id")
        tilt_summary[f"{lam:g}"] = dict(
            voi_retained=float(g.voi.sum() / base.voi.sum()), querying=int(g.queries.sum()),
            mean_gain_exposure=float(g.gain_exposure.mean()), min_gain_exposure=float(g.gain_exposure.min()),
            exposure_worse_off=int((g.gain_exposure < -EPS).sum()),
            some_type_worse_off=int((g[["gain_economic", "gain_displacement", "gain_exposure", "gain_equal"]].min(axis=1) < -EPS).sum()))
    summary["type_tilt"] = tilt_summary
    summary["runtime_seconds"] = time.perf_counter() - start
    (RESULTS / "question_design.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "trade"}, indent=1, default=float))
    for sid, info in summary["trade"].items():
        print(sid, {k: (np.round(v, 3).tolist() if isinstance(v, list) and k == "difference" else v) for k, v in info.items()})


if __name__ == "__main__":
    main()
