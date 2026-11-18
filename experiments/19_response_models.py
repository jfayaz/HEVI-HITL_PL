"""Experiment 19: answers that follow other response models.

Section 5.4; Supplementary Section S7.
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
from scipy.special import expit
from scipy.stats import norm

from hevi_hitl import flood as fl
from hevi_hitl import social as so
from hevi_hitl.insight import misspecification

RESULTS = ROOT / "results"
EPS = 1e-10
MC_SEED = 2032
MC_DRAWS = 200_000
MODELS = ("probit", "step", "lapses", "random_weights", "exposure_salience", "cost_salience",
          "first_profile_bias", "lexicographic")

_spec = importlib.util.spec_from_file_location("exp02", ROOT / "experiments" / "02_random_instances.py")
exp02 = importlib.util.module_from_spec(_spec)
sys.modules["exp02"] = exp02
_spec.loader.exec_module(exp02)


def social_config(p: dict) -> so.SocialConfig:
    return so.SocialConfig(lead_hours=p["lead_hours"], mechanism_strength=p["mechanism_strength"],
                           social_query_reliability=p["social_query_reliability"],
                           preference_prior=p["preference_prior"], query_cost=p["query_cost"],
                           hazard_prior=p["hazard_prior"], original_reliability=p["original_reliability"],
                           preference_slope=p["preference_slope"])


def answer_probabilities(name: str, weights: np.ndarray, delta: np.ndarray, slope: float,
                         rho: float) -> np.ndarray:
    """P(prefer the second profile) for each preference type (rows) and comparison (columns)."""
    u = weights @ delta.T
    if name == "logistic":
        base = expit(slope * u)
    elif name == "probit":
        base = norm.cdf(slope * np.sqrt(2 * np.pi) / 4 * u)
    elif name == "step":
        base = (u > 0).astype(float) + 0.5 * (u == 0)
    elif name == "lapses":
        base = 0.5 + 0.7 * (expit(slope * u) - 0.5)
    elif name == "random_weights":
        rng = np.random.default_rng(MC_SEED)
        base = np.empty_like(u)
        for k, w in enumerate(weights):
            draws = rng.dirichlet(20 * w, size=MC_DRAWS)
            base[k] = ((draws @ delta.T) > 0).mean(axis=0)
    elif name in ("exposure_salience", "cost_salience"):
        tilt = np.array([1.0, 1.0, 2.0]) if name == "exposure_salience" else np.array([2.0, 1.0, 1.0])
        w = weights * tilt
        w /= w.sum(axis=1, keepdims=True)
        base = expit(slope * (w @ delta.T))
    elif name == "first_profile_bias":
        base = expit(slope * u - 1.0)
    elif name == "lexicographic":
        base = np.empty_like(u)
        for k, w in enumerate(weights):
            if np.allclose(w, w[0]):
                base[k] = expit(slope * u[k])
                continue
            order = np.argsort(-w, kind="stable")
            for j, d in enumerate(delta):
                choice = 0.5
                for attr in order:
                    if abs(d[attr]) > 1e-12:
                        choice = float(d[attr] > 0)  # delta > 0: the second profile is better
                        break
                base[k, j] = choice
    else:
        raise ValueError(name)
    return 0.5 + rho * (base - 0.5)


def true_model(model, name: str):
    """Copy of a social model whose comparison answers follow response model ``name``."""
    cfg = model.config
    weights = fl.weights_of(cfg)
    delta = (fl.comparison_pairs(cfg)[:, 0] - fl.comparison_pairs(cfg)[:, 1]) / fl.SCALES
    rho = cfg.original_reliability if cfg.comparison_reliability is None else cfg.comparison_reliability
    table = answer_probabilities(name, weights, delta, cfg.preference_slope, rho)
    lik = model.likelihood.copy()
    lik[:, list(so.PREFERENCE_QUERIES)] = table[model.indices[:, 5]]
    return replace(model, likelihood=lik), table


def analyse(task):
    source, ident, cfg = task
    start = time.perf_counter()
    model = so.build_social_flood(cfg)
    planned, plan_table = true_model(model, "logistic")
    assert np.allclose(planned.likelihood, model.likelihood)
    rows = []
    for name in MODELS:
        truth, table = true_model(model, name)
        out = misspecification(truth, model, None)
        rows.append(dict(source=source, instance=ident, model=name,
                         reliability=cfg.original_reliability, slope=cfg.preference_slope,
                         lead_hours=cfg.lead_hours,
                         mean_abs_probability_change=float(np.abs(table - plan_table).mean()),
                         max_abs_probability_change=float(np.abs(table - plan_table).max()), **out))
    for r in rows:
        r["instance_seconds"] = time.perf_counter() - start
    return rows


def tasks(n: int) -> list:
    grid = so.GRID
    designed = [("designed", i, so.SocialConfig(**dict(zip(grid, values))))
                for i, values in enumerate(product(*grid.values()))]
    randomized = [("randomized", p["instance"], social_config(p)) for p in exp02.sample_parameters(n)]
    return designed + randomized


def summarise(df: pd.DataFrame) -> dict:
    df = df.assign(harmful=df.versus_planned_immediate > EPS, queries=df.preference_queries > EPS)
    out = {}
    for (source, name), g in df.groupby(["source", "model"]):
        out[f"{source}|{name}"] = dict(
            cases=int(len(g)), harmful_share=float(g.harmful.mean()), harmful_cases=int(g.harmful.sum()),
            max_harm=float(g.versus_planned_immediate.max()), mean_regret=float(g.regret.mean()),
            max_regret=float(g.regret.max()), median_regret=float(g.regret.median()),
            regret_share_of_voi=float((g.regret / np.maximum(g.true_immediate - g.true_optimum, EPS)).median()),
            mean_probability_change=float(g.mean_abs_probability_change.mean()),
            comparisons_asked_share=float(g.queries.mean()))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    csv = RESULTS / "response_models.csv"
    start = time.perf_counter()
    if not args.analysis_only:
        rows = []
        work = tasks(args.n)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for k, part in enumerate(pool.map(analyse, work, chunksize=1)):
                rows.extend(part)
                if (k + 1) % 40 == 0 or k + 1 == len(work):
                    pd.DataFrame(rows).to_csv(csv, index=False)
                    print(f"{k + 1}/{len(work)}: {time.perf_counter() - start:.1f} s", flush=True)
    df = pd.read_csv(csv)
    summary = summarise(df)
    summary["randomized_instances"] = int(df[df.source == "randomized"].instance.nunique())
    summary["runtime_seconds"] = None if args.analysis_only else time.perf_counter() - start
    (RESULTS / "response_models.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for key, s in summary.items():
        if isinstance(s, dict):
            print(f"{key:40s} harmful {s['harmful_share']:.3f} ({s['harmful_cases']}) max harm {s['max_harm']:.5f}"
                  f" mean regret {s['mean_regret']:.6f} max regret {s['max_regret']:.5f} dp {s['mean_probability_change']:.3f}")


if __name__ == "__main__":
    main()
