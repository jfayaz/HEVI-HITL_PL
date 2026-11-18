"""Experiment 24: guided refinement against other partition constructions.

Section 5.6; Supplementary Section S9.
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
from heapq import heappop, heappush
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.cluster import BisectingKMeans, KMeans

from hevi_hitl import certificates as C
from hevi_hitl.model import PortfolioSolver
from hevi_hitl.reduction import aggregate_partition, signature_hierarchy

RESULTS = ROOT / "results"
SEED = 2034
RANDOM_SEEDS = 5
TOLERANCES = (0.001, 0.005, 0.01)

_spec = importlib.util.spec_from_file_location("exp17", ROOT / "experiments" / "17_reduction_ablation.py")
exp17 = importlib.util.module_from_spec(_spec)
sys.modules["exp17"] = exp17
_spec.loader.exec_module(exp17)
CLASS_COUNTS = exp17.CLASS_COUNTS


class ModeRefiner(C.GuidedRefiner):
    """Guided refinement with the score term, the response term or both (``mode``)."""

    def __init__(self, model, J_U, sets, cmax, mode, split=None):
        self.mode = mode
        super().__init__(model, J_U=J_U, sets=sets, cmax=cmax, split=split)

    def _evaluate(self, members):
        w = self.prior[members]
        mass = w.sum()
        block = self.contrasts[members]
        rbar = (w @ block) / mass
        diff = block - rbar
        eps = 0.5 * (diff.max(axis=1) - diff.min(axis=1))
        span = float(np.ptp(rbar)) + self.cmax
        kbar = (w @ self.lik[members]) / mass
        ups = C.divergence(np.abs(self.lik[members] - kbar), self.sets)
        score = {"joint": eps + span * ups, "score_only": eps, "response_only": span * ups}[self.mode]
        return float(w @ score), span

    def refine_to(self, k):
        n = len(self.prior)
        k = min(n, int(k))
        while len(self.leaves) < k:
            if self.heap:
                _, uid = heappop(self.heap)
            else:  # every remaining contribution is zero: split the largest class by index
                uid = max((u for u in self.leaves if len(self.leaves[u][0]) > 1), key=lambda u: len(self.leaves[u][0]))
            members, contribution, span = self.leaves.pop(uid)
            parts = []
            if self.mode != "response_only":
                parts.append(np.ptp(self.contrasts[members], axis=0) * 0.5)
            else:
                parts.append(np.zeros(self.contrasts.shape[1]))
            if self.mode != "score_only":
                parts.append(np.ptp(self.lik[members], axis=0) * span)
            else:
                parts.append(np.zeros(self.lik.shape[1]))
            scores = np.concatenate(parts)
            j = int(np.argmax(scores))
            if scores[j] <= 0:
                values = np.zeros(len(members))
            elif j < self.contrasts.shape[1]:
                values = self.contrasts[members, j]
            else:
                values = self.lik[members, j - self.contrasts.shape[1]]
            ordered = members[np.lexsort((members, values))]
            cut = len(ordered) // 2
            first, second = ordered[:cut], ordered[cut:]
            if self.split == "2means" and scores[j] > 0 and len(members) > 2:
                both = np.concatenate((first, second))
                parts = []
                if self.mode != "response_only":
                    parts.append(0.5 * self.contrasts[both])
                if self.mode != "score_only":
                    parts.append(span * self.lik[both])
                first, second = C.two_means_halves(np.column_stack(parts), self.prior[both], first, second)
            self._insert(first)
            self._insert(second)
        inverse = np.empty(n, dtype=int)
        for label, uid in enumerate(sorted(self.leaves)):
            inverse[self.leaves[uid][0]] = label
        return inverse

    def _insert(self, members):
        uid = self.serial
        self.serial += 1
        contribution, span = self._evaluate(members)
        self.leaves[uid] = (members, contribution, span)
        if len(members) > 1 and contribution > 0:
            heappush(self.heap, (-contribution, uid))


def relabel(labels):
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.ravel().astype(int)


def kmeans(features, k, seed, weights=None):
    if k >= len(features):
        return np.arange(len(features))
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(features, sample_weight=weights)
    return relabel(km.labels_)


def bisecting(features, k, seed, weights=None):
    if k >= len(features):
        return np.arange(len(features))
    km = BisectingKMeans(n_clusters=k, n_init=4, random_state=seed, bisecting_strategy="biggest_inertia")
    return relabel(km.fit(features, sample_weight=weights).labels_)


def random_partition(n, k, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    labels = np.empty(n, dtype=int)
    for c, part in enumerate(np.array_split(perm, min(k, n))):
        labels[part] = c
    return labels


def evaluate_partition(model, inverse, U, sets, cmax, lower, optimum, J_U):
    reduced = aggregate_partition(model, inverse)
    policy = PortfolioSolver(reduced)
    policy.solve()
    value = PortfolioSolver(model).evaluate(policy)[0]
    surrogate = PortfolioSolver(reduced).evaluate(policy)[0]
    cert = C.refined_certificate(model, reduced, inverse, value, surrogate, U, None, lower_bound=lower,
                                 sets=sets, cmax=cmax, J_U=J_U)
    regret = float(value - optimum)
    assert cert["refined_lb"] >= regret - 1e-10
    return dict(classes=int(inverse.max() + 1), regret=regret, certificate=float(cert["refined_lb"]),
                per_configuration=float(cert["prop1_postsolve"]))


def analyse(task):
    domain, cid, cfg = task
    model = exp17.build(domain, cfg)
    optimum = float(PortfolioSolver(model).solve()[0])
    U, _ = C.undominated_endpoints(model)
    sets, cmax = C.feasible_query_sets(model)
    J_all = C.endpoint_columns(model, C.endpoint_mask(model))
    J_U = C.endpoint_columns(model, U)
    lower = C.perfect_information_bound(model)
    counts = [min(k, len(model.prior)) for k in CLASS_COUNTS[domain]]
    contrasts = J_U - J_U[:, :1]
    scale = max(float(np.ptp(contrasts, axis=0).max()), 1e-12)
    joint_signature = np.column_stack((contrasts, scale * model.likelihood))
    builders = {}

    def guided(mode, split=None):
        tick = time.perf_counter()
        refiner = (C.GuidedRefiner(model, J_U=J_U, sets=sets, cmax=cmax, split=split) if mode == "joint"
                   else ModeRefiner(model, J_U, sets, cmax, mode, split=split))
        out = {k: refiner.refine_to(k) for k in counts}
        return out, time.perf_counter() - tick

    for mode in ("joint", "score_only", "response_only"):
        builders[mode] = guided(mode)
    builders["joint_median"] = guided("joint", split="median")
    tick = time.perf_counter()
    builders["bisection"] = (signature_hierarchy(model, counts), time.perf_counter() - tick)
    tick = time.perf_counter()
    refiner = exp17.AbsoluteRefiner(model, J_U, sets, cmax)
    builders["absolute"] = ({k: refiner.refine_to(k) for k in counts}, time.perf_counter() - tick)
    tick = time.perf_counter()
    builders["kmeans_scores"] = ({k: kmeans(J_all, k, SEED) for k in counts}, time.perf_counter() - tick)
    tick = time.perf_counter()
    builders["kmeans_joint"] = ({k: kmeans(joint_signature, k, SEED) for k in counts}, time.perf_counter() - tick)
    tick = time.perf_counter()
    builders["kmeans_joint_w"] = ({k: kmeans(joint_signature, k, SEED, model.prior) for k in counts},
                                  time.perf_counter() - tick)
    tick = time.perf_counter()
    builders["bisecting_joint"] = ({k: bisecting(joint_signature, k, SEED, model.prior) for k in counts},
                                   time.perf_counter() - tick)
    for s in range(RANDOM_SEEDS):
        tick = time.perf_counter()
        builders[f"random_{s}"] = ({k: random_partition(len(model.prior), k, SEED + 100 * cid + s) for k in counts},
                                   time.perf_counter() - tick)
    return rows_of(domain, cid, model, builders, counts, U, sets, cmax, lower, optimum, J_U)


def rows_of(domain, cid, model, builders, counts, U, sets, cmax, lower, optimum, J_U):
    rows = []
    for family, (hierarchy, seconds) in builders.items():
        for k in counts:
            r = evaluate_partition(model, hierarchy[k], U, sets, cmax, lower, optimum, J_U)
            rows.append(dict(domain=domain, configuration_id=cid, family=family, requested=k,
                             states=len(model.prior), build_seconds=seconds, **r))
    return rows


def summarise(df):
    df = df.assign(base=df.family.str.replace(r"_\d+$", "", regex=True))
    out = {}
    for (domain, base), g in df.groupby(["domain", "base"]):
        per_instance = []
        for (cid, fam), h in g.groupby(["configuration_id", "family"]):
            n = int(h.states.iloc[0])
            row = dict(configuration_id=cid, family=fam)
            for tol in TOLERANCES:
                ok = h.loc[h.certificate <= tol, "classes"]
                row[f"certified_share_{tol:g}"] = float(ok.min() / n) if len(ok) else 1.0
                okr = h.loc[h.regret <= tol, "classes"]
                row[f"regret_share_{tol:g}"] = float(okr.min() / n) if len(okr) else 1.0
            per_instance.append(row)
        p = pd.DataFrame(per_instance)
        compressed = g[g.classes < g.states]
        out[f"{domain}|{base}"] = dict(
            partitions=int(len(g)),
            median_certified_share=float(p["certified_share_0.005"].median()),
            median_certified_share_0001=float(p["certified_share_0.001"].median()),
            median_regret_share=float(p["regret_share_0.005"].median()),
            mean_regret=float(compressed.regret.mean()), max_regret=float(compressed.regret.max()),
            share_regret_above_1e6=float((compressed.regret > 1e-6).mean()),
            median_certificate=float(compressed.certificate.median()),
            median_build_seconds=float(g.drop_duplicates(["configuration_id", "family"]).build_seconds.median()))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    csv = RESULTS / "aggregation_baselines.csv"
    start = time.perf_counter()
    if not args.analysis_only:
        rows = []
        tasks = exp17.configs()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for k, part in enumerate(pool.map(analyse, tasks, chunksize=1)):
                rows.extend(part)
                pd.DataFrame(rows).to_csv(csv, index=False)
                print(f"{k + 1}/{len(tasks)}: {time.perf_counter() - start:.1f} s", flush=True)
    df = pd.read_csv(csv)
    summary = summarise(df)
    summary["runtime_seconds"] = None if args.analysis_only else time.perf_counter() - start
    (RESULTS / "aggregation_baselines.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for key, s in summary.items():
        if isinstance(s, dict):
            print(f"{key:26s} cert-share {s['median_certified_share']:.3f} (0.001: {s['median_certified_share_0001']:.3f}) "
                  f"regret-share {s['median_regret_share']:.3f} mean regret {s['mean_regret']:.2e} max {s['max_regret']:.2e} "
                  f">1e-6 {s['share_regret_above_1e6']:.2f} median cert {s['median_certificate']:.4f}")


if __name__ == "__main__":
    main()
