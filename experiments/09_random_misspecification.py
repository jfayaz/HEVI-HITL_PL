"""Experiment 09: misspecified response laws and priors in randomized instances.

Section 5.4.
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

from hevi_hitl import social as so
from hevi_hitl.insight import misspecification, save_json

RESULTS = ROOT / "results"
DRAW_SEED = 2028
EPS = 1e-10

_spec = importlib.util.spec_from_file_location("exp02", ROOT / "experiments" / "02_random_instances.py")
exp02 = importlib.util.module_from_spec(_spec)
sys.modules["exp02"] = exp02
_spec.loader.exec_module(exp02)


def config_of(p: dict) -> so.SocialConfig:
    return so.SocialConfig(lead_hours=p["lead_hours"], mechanism_strength=p["mechanism_strength"],
                           social_query_reliability=p["social_query_reliability"],
                           preference_prior=p["preference_prior"], query_cost=p["query_cost"],
                           hazard_prior=p["hazard_prior"], original_reliability=p["original_reliability"],
                           preference_slope=p["preference_slope"])


def cases(p: dict) -> list[tuple[str, str, float | None, so.SocialConfig]]:
    true = config_of(p)
    rng = np.random.default_rng([DRAW_SEED, p["instance"]])
    out = []
    rho = p["original_reliability"]
    for d in (0.15, 0.30, -0.15, -0.30):
        rp = float(min(1.0, max(0.5, rho + d)))
        if abs(rp - rho) > 1e-9:
            out.append(("reliability", "overstated" if d > 0 else "understated", d, replace(true, original_reliability=rp)))
    for f in (0.5, 2.0):
        out.append(("preference_slope", "doubled" if f > 1 else "halved", f,
                    replace(true, preference_slope=p["preference_slope"] * f)))
    out.append(("preference_prior", "uniform", None, replace(true, preference_prior=(0.25, 0.25, 0.25, 0.25))))
    out.append(("preference_prior", "independent draw", None,
                replace(true, preference_prior=tuple(float(x) for x in rng.dirichlet(np.ones(4))))))
    out.append(("hazard_prior", "uniform", None, replace(true, hazard_prior=(1 / 3, 1 / 3, 1 / 3))))
    out.append(("hazard_prior", "independent draw", None,
                replace(true, hazard_prior=tuple(float(x) for x in rng.dirichlet(np.ones(3))))))
    k = p["mechanism_strength"]
    for d in (0.5, -0.5):
        kp = float(min(1.0, max(0.0, k + d)))
        if abs(kp - k) > 1e-9:
            out.append(("mechanism_strength", "stronger" if d > 0 else "weaker", d, replace(true, mechanism_strength=kp)))
    return out


def analyse(p: dict) -> list[dict]:
    start = time.perf_counter()
    true_model = so.build_social_flood(config_of(p))
    cache: dict = {}
    rows = []
    for family, direction, delta, pcfg in cases(p):
        out = misspecification(true_model, so.build_social_flood(pcfg), cache)
        rows.append(dict(instance=p["instance"], family=family, direction=direction, delta=delta,
                         true_reliability=p["original_reliability"], lead_hours=p["lead_hours"],
                         p_high_severity=p["hazard_prior"][2], **out))
    for r in rows:
        r["instance_seconds"] = time.perf_counter() - start
    return rows


def summarise(df: pd.DataFrame) -> dict:
    df = df.assign(harmful=df.versus_planned_immediate > EPS,
                   queries=(df.risk_queries + df.preference_queries) > EPS)
    out = []
    for (fam, direction), g in df.groupby(["family", "direction"], sort=False):
        out.append(dict(family=fam, direction=direction, cases=int(len(g)), instances=int(g.instance.nunique()),
                        harmful_share=float(g.harmful.mean()), harmful_cases=int(g.harmful.sum()),
                        max_harm=float(g.versus_planned_immediate.max()),
                        mean_regret=float(g.regret.mean()), max_regret=float(g.regret.max()),
                        querying_share=float(g.queries.mean())))
    rel = df[df.family == "reliability"]
    by_delta = {f"{d:+.2f}": dict(cases=int(len(g)), harmful_share=float(g.harmful.mean()),
                                  max_harm=float(g.versus_planned_immediate.max()))
                for d, g in rel.groupby("delta")}
    return dict(by_case=out, reliability_by_shift=by_delta,
                harmful_outside_reliability=int(df[(df.family != "reliability") & df.harmful].shape[0]),
                harmful_understated_reliability=int(rel[(rel.direction == "understated") & rel.harmful].shape[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="instances analysed (the paper uses 1,000)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    csv = RESULTS / "random_misspecification.csv"
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
    save_json(RESULTS / "random_misspecification.json", summary)
    print(pd.DataFrame(summary["by_case"]).to_string(index=False))


if __name__ == "__main__":
    main()
