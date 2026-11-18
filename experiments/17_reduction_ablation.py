"""Experiment 17: ablation of the certificate.

Section 5.6; Supplementary Section S6.
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
from heapq import heappop, heappush
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from hevi_hitl import certificates as C
from hevi_hitl import flood as fl
from hevi_hitl import social as so
from hevi_hitl.model import PortfolioSolver
from hevi_hitl.reduction import aggregate_partition

RESULTS = ROOT / "results"
CLASS_COUNTS = {"flood": (1,) + tuple(range(12, 193, 12)), "social": tuple(range(48, 769, 48))}
TOLERANCES = (0.001, 0.0025, 0.005, 0.01)
LADDER = ("classical_uniform", "classical_postsolve", "prop1_uniform", "prop1_postsolve", "U_postsolve",
          "refined_postsolve", "refined_lb")
PARTITIONS = ("contrast", "absolute")


def configs():
    out = []
    for domain, module in (("flood", fl), ("social", so)):
        grid = module.GRID
        cls = fl.FloodConfig if domain == "flood" else so.SocialConfig
        for cid, values in enumerate(product(*grid.values())):
            out.append((domain, cid, cls(**dict(zip(grid, values)))))
    return out


def build(domain, cfg):
    return fl.build_flood(cfg) if domain == "flood" else so.build_social_flood(cfg)


class AbsoluteRefiner:
    """Guided refinement on absolute scores with the offset-free per-configuration error."""

    def __init__(self, model, J_U, sets, cmax):
        self.prior, self.lik, self.sets, self.cmax = model.prior, model.likelihood, sets, cmax
        self.scores = J_U
        self.heap, self.leaves, self.serial = [], {}, 0
        self._insert(np.arange(len(self.prior)))

    def _evaluate(self, members):
        w = self.prior[members]
        mass = w.sum()
        block = self.scores[members]
        rbar = (w @ block) / mass
        eps = np.abs(block - rbar).max(axis=1)
        span = float(np.ptp(rbar)) + self.cmax
        kbar = (w @ self.lik[members]) / mass
        ups = C.divergence(np.abs(self.lik[members] - kbar), self.sets)
        return float(w @ (eps + span * ups)), span

    def _insert(self, members):
        uid = self.serial
        self.serial += 1
        contribution, span = self._evaluate(members)
        self.leaves[uid] = (members, contribution, span)
        if len(members) > 1:
            heappush(self.heap, (-contribution, uid))

    def refine_to(self, k):
        n = len(self.prior)
        k = min(n, int(k))
        while len(self.leaves) < k:
            _, uid = heappop(self.heap)
            members, _, span = self.leaves.pop(uid)
            ranges = np.concatenate((np.ptp(self.scores[members], axis=0),
                                     np.ptp(self.lik[members], axis=0) * span))
            j = int(np.argmax(ranges))
            if ranges[j] <= 0:
                values = np.zeros(len(members))
            elif j < self.scores.shape[1]:
                values = self.scores[members, j]
            else:
                values = self.lik[members, j - self.scores.shape[1]]
            ordered = members[np.lexsort((members, values))]
            cut = len(ordered) // 2
            first, second = ordered[:cut], ordered[cut:]
            if C.DEFAULT_SPLIT == "2means" and ranges[j] > 0 and len(members) > 2:
                both = np.concatenate((first, second))
                X = np.column_stack((self.scores[both], span * self.lik[both]))
                first, second = C.two_means_halves(X, self.prior[both], first, second)
            self._insert(first)
            self._insert(second)
        inverse = np.empty(n, dtype=int)
        for label, uid in enumerate(sorted(self.leaves)):
            inverse[self.leaves[uid][0]] = label
        return inverse


def classical_error(model, reduced, inverse, J_all, sets, cmax):
    """E_abs: per-configuration absolute score error over all endpoints plus the divergence allowance."""
    R = C.endpoint_columns(reduced, C.endpoint_mask(model))
    diff = np.abs(J_all - R[inverse]).max(axis=1)
    span = np.ptp(R, axis=1) + cmax
    ups = C.divergence(np.abs(model.likelihood - reduced.likelihood[inverse]), sets)
    return float(model.prior @ (diff + span[inverse] * ups))


def candidate(model, inverse, U, sets, cmax, lower, optimum, J_all, J_U):
    reduced = aggregate_partition(model, inverse)
    policy = PortfolioSolver(reduced)
    policy.solve()
    value = PortfolioSolver(model).evaluate(policy)[0]
    surrogate = PortfolioSolver(reduced).evaluate(policy)[0]
    cert = C.refined_certificate(model, reduced, inverse, value, surrogate, U, None, lower_bound=lower,
                                 sets=sets, cmax=cmax, J_all=J_all, J_U=J_U)
    e_abs = classical_error(model, reduced, inverse, J_all, sets, cmax)
    out = {k: cert[k] for k in ("prop1_uniform", "prop1_postsolve", "U_postsolve", "refined_postsolve", "refined_lb")}
    out.update(classical_uniform=2 * e_abs, classical_postsolve=e_abs + float(value - surrogate),
               classical_error=e_abs, prop1_error=cert["prop1_error"], classes=int(inverse.max() + 1),
               regret=float(value - optimum))
    for name in LADDER:
        assert out[name] >= out["regret"] - 1e-10, (name, out[name], out["regret"])
    return out


def analyse(task):
    domain, cid, cfg = task
    start = time.perf_counter()
    model = build(domain, cfg)
    optimum = float(PortfolioSolver(model).solve()[0])
    U, _ = C.undominated_endpoints(model)
    sets, cmax = C.feasible_query_sets(model)
    J_all = C.endpoint_columns(model, C.endpoint_mask(model))
    J_U = C.endpoint_columns(model, U)
    lower = C.perfect_information_bound(model)
    hierarchies = {"contrast": C.guided_hierarchy(model, CLASS_COUNTS[domain], J_U=J_U, sets=sets, cmax=cmax)}
    refiner = AbsoluteRefiner(model, J_U, sets, cmax)
    hierarchies["absolute"] = {k: refiner.refine_to(k) for k in CLASS_COUNTS[domain]}
    rows = []
    for partition, hierarchy in hierarchies.items():
        for k in CLASS_COUNTS[domain]:
            inverse = hierarchy[min(k, len(model.prior))]
            rows.append(dict(domain=domain, configuration_id=cid, partition=partition, states=len(model.prior),
                             **candidate(model, inverse, U, sets, cmax, lower, optimum, J_all, J_U)))
    for r in rows:
        r["instance_seconds"] = time.perf_counter() - start
    return rows


def smallest_share(g: pd.DataFrame, bound: str, tol: float) -> float:
    ok = g.loc[g[bound] <= tol, "classes"]
    n = int(g.states.iloc[0])
    return float(ok.min() / n) if len(ok) else 1.0


def summarise(df: pd.DataFrame) -> dict:
    out = {}
    for (domain, partition), g in df.groupby(["domain", "partition"]):
        per = {}
        for bound in LADDER:
            for tol in TOLERANCES:
                shares = g.groupby("configuration_id").apply(lambda x: smallest_share(x, bound, tol),
                                                            include_groups=False)
                per[f"{bound}@{tol}"] = dict(median_share=float(shares.median()), min_share=float(shares.min()),
                                             max_share=float(shares.max()),
                                             compressed=int((shares < 1).sum()), instances=int(len(shares)))
        ratio = {}
        for bound in LADDER:
            r = (g[bound] / g["refined_lb"]).replace([np.inf, -np.inf], np.nan)
            ratio[bound] = float(np.nanmedian(r[g.classes < g.states]))
        out[f"{domain}|{partition}"] = dict(acceptance=per, median_ratio_to_certificate=ratio)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    csv = RESULTS / "reduction_ablation.csv"
    start = time.perf_counter()
    if not args.analysis_only:
        rows = []
        tasks = configs()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for k, part in enumerate(pool.map(analyse, tasks, chunksize=1)):
                rows.extend(part)
                print(f"{k + 1}/{len(tasks)} instances: {time.perf_counter() - start:.1f} s", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(csv, index=False)
    df = pd.read_csv(csv)
    summary = summarise(df)
    summary["runtime_seconds"] = time.perf_counter() - start if not args.analysis_only else None
    (RESULTS / "reduction_ablation.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for key, s in summary.items():
        if key == "runtime_seconds":
            continue
        print(key, {b: round(s["acceptance"][f"{b}@0.005"]["median_share"], 3) for b in LADDER})


if __name__ == "__main__":
    main()
