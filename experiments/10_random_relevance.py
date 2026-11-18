"""Experiment 10: decision-relevance indices in randomized instances.

Section 5.6.
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from hevi_hitl import certificates as C
from hevi_hitl import social as so
from hevi_hitl.insight import save_json
from hevi_hitl.model import PortfolioSolver

RESULTS = ROOT / "results"
TOLERANCES = (0.001, 0.0025, 0.005, 0.01)


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp02 = _load("exp02", "02_random_instances.py")
exp04 = _load("exp04", "04_certificates.py")
NAMES = exp04.COORDINATES["social"]


def config_of(p: dict) -> so.SocialConfig:
    return so.SocialConfig(lead_hours=p["lead_hours"], mechanism_strength=p["mechanism_strength"],
                           social_query_reliability=p["social_query_reliability"],
                           preference_prior=p["preference_prior"], query_cost=p["query_cost"],
                           hazard_prior=p["hazard_prior"], original_reliability=p["original_reliability"],
                           preference_slope=p["preference_slope"])


def analyse(p: dict) -> list[dict]:
    start = time.perf_counter()
    model = so.build_social_flood(config_of(p))
    full = PortfolioSolver(model)
    value = full.solve()[0]
    _, _, features = full.evaluate(full)
    optimum = dict(value=float(value), physical=exp04.physical(features))
    U, _ = C.undominated_endpoints(model)
    sets, cmax = C.feasible_query_sets(model)
    J_all = C.endpoint_columns(model, C.endpoint_mask(model))
    J_U = C.endpoint_columns(model, U)
    lower = C.perfect_information_bound(model)
    immediate = PortfolioSolver(model, method="immediate").solve()[0]
    rows = []
    others = tuple(j for j, name in enumerate(NAMES) if name not in ("severity", "preference_type"))
    for subset in C.coordinate_subsets(len(NAMES), pairs=False) + [others]:
        inverse = C.marginal_partition(model.indices, subset)
        guard = max(lower, C.query_relaxation_bound(model, inverse))
        cert = exp04.candidate(model, inverse, U, sets, cmax, guard, optimum, J_all, J_U, with_tilde=False)
        rows.append(dict(instance=p["instance"],
                         coordinate="others" if subset == others else "+".join(NAMES[j] for j in subset),
                         dri=cert["refined_lb"], class_level=cert["refined_postsolve"],
                         certificate=float(min(cert["refined_postsolve"], cert["original_policy_value"] - lower)),
                         per_configuration=cert["prop1_postsolve"], regret=cert["regret"], classes=cert["classes"],
                         voi_joint=float(immediate - value), lead_hours=p["lead_hours"],
                         p_high_severity=p["hazard_prior"][2], original_reliability=p["original_reliability"],
                         mechanism_strength=p["mechanism_strength"]))
    for r in rows:
        r["instance_seconds"] = time.perf_counter() - start
    return rows


def summarise(df: pd.DataFrame) -> dict:
    out = []
    for coord, g in df.groupby("coordinate", sort=False):
        row = dict(coordinate=coord, instances=int(len(g)), mean_dri=float(g.dri.mean()),
                   max_dri=float(g.dri.max()), median_dri=float(g.dri.median()),
                   mean_regret=float(g.regret.mean()), max_regret=float(g.regret.max()),
                   zero_dri_share=float((g.dri <= 1e-12).mean()))
        for tol in TOLERANCES:
            row[f"certified_share_{tol:g}"] = float((g.dri <= tol).mean())
        out.append(row)
    slack = float((df.dri - df.regret).min())
    return dict(by_coordinate=out, min_certificate_minus_regret=slack)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="instances analysed (the paper uses 1,000)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    csv = RESULTS / "random_relevance.csv"
    if not args.analysis_only:
        params = exp02.sample_parameters(args.n)
        start = time.perf_counter()
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for k, part in enumerate(pool.map(analyse, params, chunksize=1)):
                rows.extend(part)
                if (k + 1) % 20 == 0:
                    pd.DataFrame(rows).to_csv(csv, index=False)
                    print(f"{k + 1} instances: {time.perf_counter() - start:.1f} s", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(csv, index=False)
        runtime = dict(n=args.n, workers=args.workers, runtime_seconds=time.perf_counter() - start)
    else:
        df = pd.read_csv(csv)
        runtime = None
    summary = summarise(df)
    summary["instances"] = int(df.instance.nunique())
    if runtime is not None:
        summary["runtime"] = runtime
    save_json(RESULTS / "random_relevance.json", summary)
    print(pd.DataFrame(summary["by_coordinate"]).to_string(index=False))


if __name__ == "__main__":
    main()
