"""Experiment 27: certified reduction against value-directed compression.

Sections 3.4 and 5.6; Supplementary Section S9.
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from hevi_hitl import certificates as C
from hevi_hitl.reduction import aggregate_partition
from hevi_hitl.scaled import FastSolver, evaluate, evaluate_solver

RESULTS = ROOT / "results"
TOL = 1e-10

_spec = importlib.util.spec_from_file_location("exp17", ROOT / "experiments" / "17_reduction_ablation.py")
exp17 = importlib.util.module_from_spec(_spec)
sys.modules["exp17"] = exp17
_spec.loader.exec_module(exp17)
CLASS_COUNTS = exp17.CLASS_COUNTS


def krylov_basis(prior, rewards, operators, rtol=TOL):
    """Greedy prior-weighted Krylov basis (columns), in the order in which vectors are added."""
    n = len(prior)
    sw = np.sqrt(prior)
    # Work in the transformed space u = sqrt(b0) * v, where the weighted inner product is Euclidean;
    # the diagonal update operators commute with this scaling.
    ops = np.column_stack(operators)
    R = sw[:, None] * rewards  # residuals of the candidates
    scale = np.linalg.norm(R, axis=0)  # norms of the candidates themselves
    B = np.zeros((n, 0))
    while B.shape[1] < n:
        norms = np.linalg.norm(R, axis=0)
        live = norms > rtol * np.maximum(scale, 1e-300)
        if not live.any():
            break
        R, scale, norms = R[:, live], scale[live], norms[live]
        j = int(np.argmax(norms))
        u = R[:, j] / norms[j]
        u -= B @ (B.T @ u)  # re-orthogonalise for numerical stability
        u /= np.linalg.norm(u)
        B = np.column_stack((B, u))
        R = np.delete(R - np.outer(u, u @ R), j, axis=1)
        scale = np.delete(scale, j)
        images = ops * u[:, None]  # images of the new basis vector under every update operator
        R = np.column_stack((R, images - B @ (B.T @ images)))
        scale = np.concatenate((scale, np.linalg.norm(images, axis=0)))
    return B / sw[:, None]  # columns orthonormal in the prior-weighted inner product


class CompressedSolver:
    """Exact recursion of the compressed model; histories and feasibility as in ``FastSolver``."""

    def __init__(self, model, F):
        D = model.prior
        self.W = F.T @ (D[:, None] * model.loss[:, 0, :])           # k x A
        self.w1 = F.T @ D                                           # k
        lik = model.likelihood
        self.G1 = [F.T @ (D[:, None] * lik[:, q, None] * F) for q in range(lik.shape[1])]
        self.G0 = [F.T @ (D[:, None] * (1 - lik[:, q, None]) * F) for q in range(lik.shape[1])]
        self.penalty = np.where(model.feasible, 0.0, np.inf)
        self.durations = [int(d) for d in model.durations]
        self.costs = [float(c) for c in model.costs]
        self.deadline = int(model.config.deadline)
        self.root = (-1,) * lik.shape[1]
        self.memo = {}

    def solve(self, h=None, c=None, t=0):
        if h is None:
            h, c, t = self.root, self.w1.copy(), 0
        if h in self.memo:
            return self.memo[h]
        row = c @ self.W + self.penalty[t]
        a = int(np.argmin(row))
        ans = (float(row[a]), ("act", a))
        total = float(c @ self.w1)
        for q, y in enumerate(h):
            if y != -1 or t + self.durations[q] > self.deadline:
                continue
            value = total * self.costs[q]
            for yy, G in ((0, self.G0[q]), (1, self.G1[q])):
                hh = h[:q] + (yy,) + h[q + 1:]
                value += self.solve(hh, G @ c, t + self.durations[q])[0]
            if value < ans[0] - 1e-15:
                ans = (value, ("query", q))
        self.memo[h] = ans
        return ans


def analyse(task):
    domain, cid, cfg = task
    start = time.perf_counter()
    model = exp17.build(domain, cfg)
    optimum = FastSolver(model).solve()[0]
    lik = model.likelihood
    rewards = np.column_stack((np.ones(len(model.prior)), model.loss[:, 0, :]))
    tick = time.perf_counter()
    F_full = krylov_basis(model.prior, rewards, [lik[:, q] for q in range(lik.shape[1])])
    krylov_seconds = time.perf_counter() - tick
    lossless = F_full.shape[1]
    U, _ = C.undominated_endpoints(model)
    sets, cmax = C.feasible_query_sets(model)
    subsets = sorted({frozenset(t) for row in sets for t in _subsets(np.flatnonzero(row))}, key=len)
    sw = np.sqrt(model.prior)
    moments = np.array([sw * np.prod(lik[:, sorted(S)], axis=1) if S else sw for S in subsets])
    belief_rank = int(np.linalg.matrix_rank(moments, tol=TOL * np.linalg.norm(moments, 2)))
    J_U = C.endpoint_columns(model, U)
    refiner = C.GuidedRefiner(model, J_U=J_U, sets=sets, cmax=cmax)
    rows = []
    for k in CLASS_COUNTS[domain]:
        k = min(k, len(model.prior))
        kk = min(k, lossless)
        tick = time.perf_counter()
        solver = CompressedSolver(model, F_full[:, :kk])
        surrogate = solver.solve()[0]
        value = evaluate(model, lambda h, mass: solver.memo[h][1])["value"]
        vdc_seconds = time.perf_counter() - tick
        inverse = refiner.refine_to(k)
        reduced = aggregate_partition(model, inverse)
        rsolver = FastSolver(reduced)
        rsolver.solve()
        cvalue = evaluate_solver(model, rsolver)["value"]
        rows.append(dict(domain=domain, configuration_id=cid, states=len(model.prior), dimension=k,
                         vdc_dimension=kk, lossless_dimension=lossless, belief_rank=belief_rank,
                         query_subsets=len(subsets), vdc_regret=float(value - optimum),
                         vdc_surrogate_error=float(surrogate - value), class_regret=float(cvalue - optimum),
                         vdc_seconds=vdc_seconds, krylov_seconds=krylov_seconds,
                         seconds=time.perf_counter() - start))
        assert value - optimum > -1e-10 and cvalue - optimum > -1e-10
    return rows


def _subsets(s):
    s = tuple(s)
    for mask in range(1 << len(s)):
        yield tuple(s[i] for i in range(len(s)) if mask >> i & 1)


def summarise(df):
    out = {}
    for domain, g in df.groupby("domain"):
        per = []
        for cid, h in g.groupby("configuration_id"):
            n = int(h.states.iloc[0])
            row = dict(lossless_share=float(h.lossless_dimension.iloc[0] / n),
                       belief_rank=int(h.belief_rank.iloc[0]), query_subsets=int(h.query_subsets.iloc[0]))
            for name in ("vdc", "class"):
                for tol in (1e-6, 0.001, 0.005):
                    ok = h.loc[h[f"{name}_regret"] <= tol, "dimension"]
                    row[f"{name}_share_{tol:g}"] = float(ok.min() / n) if len(ok) else 1.0
            per.append(row)
        p = pd.DataFrame(per)
        compressed = g[g.dimension < g.states]
        out[domain] = dict(
            instances=int(len(p)), lossless_share_median=float(p.lossless_share.median()),
            lossless_share_min=float(p.lossless_share.min()), lossless_share_max=float(p.lossless_share.max()),
            belief_rank_max=int(p.belief_rank.max()), query_subsets_max=int(p.query_subsets.max()),
            **{f"median_{c}": float(p[c].median()) for c in p.columns
               if c not in ("lossless_share", "belief_rank", "query_subsets")},
            vdc_mean_regret=float(compressed.vdc_regret.mean()), vdc_max_regret=float(compressed.vdc_regret.max()),
            class_mean_regret=float(compressed.class_regret.mean()),
            class_max_regret=float(compressed.class_regret.max()),
            vdc_lower_regret_share=float((compressed.vdc_regret < compressed.class_regret - 1e-12).mean()),
            class_lower_regret_share=float((compressed.class_regret < compressed.vdc_regret - 1e-12).mean()),
            median_krylov_seconds=float(g.drop_duplicates("configuration_id").krylov_seconds.median()))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    start = time.perf_counter()
    tasks = exp17.configs()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, part in enumerate(pool.map(analyse, tasks, chunksize=1)):
            rows.extend(part)
            pd.DataFrame(rows).to_csv(RESULTS / "vdc.csv", index=False)
            print(f"{k + 1}/{len(tasks)} {part[0]['domain']} {part[0]['configuration_id']}: lossless "
                  f"{part[0]['lossless_dimension']}/{part[0]['states']} ({part[-1]['seconds']:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    out = summarise(df)
    out["runtime_seconds"] = time.perf_counter() - start
    (RESULTS / "vdc.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
