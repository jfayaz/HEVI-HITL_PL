"""Experiment 23: hourly forecast updates with rising reliability.

Section 5.7; Supplementary Section S10.
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
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from hevi_hitl import certificates as C
from hevi_hitl import forecast as fc
from hevi_hitl import social as so

RESULTS = ROOT / "results"
TOL = 0.005
SCHEDULES = ("published", "constant", "improving", "late")


def reliabilities(name: str, lead: int) -> np.ndarray:
    count = lead - 1
    return {"published": np.array([0.9]), "constant": np.full(count, 0.9),
            "improving": fc.schedule(0.6, 0.95, count), "late": fc.schedule(0.5, 0.9, count)}[name]


def summarise(model, solver) -> dict:
    ev = fc.evaluate(model, solver)
    k = int(np.sum([q.startswith("forecast_update") for q in model.queries]))
    barrier = sum(p for j, p in ev["actions"].items() if any(a[0] == 1 for a in model.actions[j]))
    out = dict(value=ev["value"], waits=ev["waits"], commit_hours=ev["commit_hours"],
               updates=float(ev["queries"][:k].sum()), barrier_probability=float(barrier),
               other_queries=float(ev["queries"][k:].sum()))
    for j in range(k):
        out[f"update_{j}h"] = float(ev["queries"][j])
    return out


def certify(model, optimum):
    """Guided refinement with doubling class counts until the certificate is at most TOL."""
    view = fc.certificate_view(model)
    U, _ = C.undominated_endpoints(view)
    sets, cmax = C.feasible_query_sets(view)
    J_U = C.endpoint_columns(view, U)
    lower = C.perfect_information_bound(view)
    refiner = C.GuidedRefiner(view, J_U=J_U, sets=sets, cmax=cmax)
    n, k, tried = len(model.prior), 48, []
    while True:
        k = min(k, n)
        inverse = refiner.refine_to(k)
        reduced = fc.aggregate_timed(model, inverse)
        rsolver = fc.TimedSolver(reduced)
        surrogate = rsolver.solve()[0]
        value = fc.evaluate(model, rsolver)["value"]
        cert = C.refined_certificate(view, fc.certificate_view(reduced), inverse, value, surrogate, U, None,
                                     lower_bound=lower, sets=sets, cmax=cmax, J_U=J_U)["refined_lb"]
        assert cert >= value - optimum - 1e-10
        tried.append(dict(classes=k, certificate=cert, regret=value - optimum))
        if cert <= TOL or k == n:
            return dict(classes=k, class_share=k / n, certificate=cert, regret=value - optimum, candidates=len(tried))
        k *= 2


def run(task):
    cid, cfg, name = task
    start = time.perf_counter()
    model = fc.build_timed(cfg, reliabilities(name, cfg.lead_hours))
    joint = fc.TimedSolver(model)
    v_joint = joint.solve()[0]
    v_nowait = fc.TimedSolver(model, allow_wait=False).solve()[0]
    v_imm = fc.TimedSolver(model, method="immediate").solve()[0]
    row = dict(configuration_id=cid, schedule=name, lead_hours=cfg.lead_hours,
               mechanism_strength=cfg.mechanism_strength, preference_prior=str(cfg.preference_prior),
               social_query_reliability=cfg.social_query_reliability, updates_offered=int(len(reliabilities(name, cfg.lead_hours))),
               joint_value=v_joint, nowait_value=v_nowait, immediate_value=v_imm, voi_joint=v_imm - v_joint,
               value_of_waiting=v_nowait - v_joint, states=len(joint.memo))
    row.update({f"joint_{k}": v for k, v in summarise(model, joint).items()})
    row.update({f"cert_{k}": v for k, v in certify(model, v_joint).items()})
    row["seconds"] = time.perf_counter() - start
    return row


def recertify(task):
    """Certification only (for a changed split rule); the optimum is read from the saved row."""
    cid, cfg, name, optimum = task
    model = fc.build_timed(cfg, reliabilities(name, cfg.lead_hours))
    return cid, name, certify(model, optimum)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--certify-only", action="store_true",
                        help="recompute the certification columns of results/forecast_skill.csv")
    args = parser.parse_args()
    grid = so.GRID
    configs = [so.SocialConfig(**dict(zip(grid, values))) for values in product(*grid.values())]
    start = time.perf_counter()
    if args.certify_only:
        df = pd.read_csv(RESULTS / "forecast_skill.csv")
        tasks = [(int(r.configuration_id), configs[int(r.configuration_id)], r.schedule,
                  float(r.immediate_value - r.voi_joint)) for r in df.itertuples()]
        tasks.sort(key=lambda t: -t[1].lead_hours)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for k, (cid, name, cert) in enumerate(pool.map(recertify, tasks, chunksize=1)):
                sel = (df.configuration_id == cid) & (df.schedule == name)
                for key, value in cert.items():
                    df.loc[sel, f"cert_{key}"] = value
                df.to_csv(RESULTS / "forecast_skill.csv", index=False)
                print(f"{k + 1}/{len(tasks)} {name} id={cid}: classes {cert['classes']} cert {cert['certificate']:.4f}",
                      flush=True)
    else:
        tasks = [(i, cfg, name) for name in SCHEDULES for i, cfg in enumerate(configs)]
        tasks.sort(key=lambda t: -t[1].lead_hours)  # long tasks first
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for row in pool.map(run, tasks, chunksize=1):
                rows.append(row)
                pd.DataFrame(rows).to_csv(RESULTS / "forecast_skill.csv", index=False)
                print(f"{len(rows)}/{len(tasks)} {row['schedule']} id={row['configuration_id']}: VOI {1e3 * row['voi_joint']:.3f} "
                      f"wait {1e3 * row['value_of_waiting']:.3f} cert {row['cert_certificate']:.4f} "
                      f"({row['seconds']:.0f} s)", flush=True)
        df = pd.DataFrame(rows)
    out = {}
    for (name, lead), g in df.groupby(["schedule", "lead_hours"]):
        out[f"{name}|{lead}"] = dict(
            instances=int(len(g)), mean_voi=float(g.voi_joint.mean()), max_voi=float(g.voi_joint.max()),
            mean_value_of_waiting=float(g.value_of_waiting.mean()), max_value_of_waiting=float(g.value_of_waiting.max()),
            mean_waits=float(g.joint_waits.mean()), mean_commit_hours=float(g.joint_commit_hours.mean()),
            mean_updates=float(g.joint_updates.mean()), mean_barrier=float(g.joint_barrier_probability.mean()),
            certified=int((g.cert_certificate <= TOL).sum()), median_class_share=float(g.cert_class_share.median()),
            max_regret=float(g.cert_regret.max()))
    out["runtime_seconds"] = None if args.certify_only else time.perf_counter() - start
    (RESULTS / "forecast_skill.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
