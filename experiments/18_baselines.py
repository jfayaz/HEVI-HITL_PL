"""Experiment 18: decision-analysis and POMDP baselines against the exact policy.

Section 5.3; Supplementary Section S7.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import importlib.util
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from hevi_hitl import flood as fl
from hevi_hitl import social as so
from hevi_hitl.model import PortfolioSolver

RESULTS = ROOT / "results"
EPS = 1e-10
SEED = 2033
UCB = 0.1  # best of 0.002-0.5 on the 16 social instances at 10,000 simulations
BASELINES = ("immediate", "one_step", "fixed_order", "voi_ranked", "info_gain", "minimax_regret",
             "worst_case", "pomcp_1k", "pomcp_10k", "pomcp_100k")

_spec = importlib.util.spec_from_file_location("exp02", ROOT / "experiments" / "02_random_instances.py")
exp02 = importlib.util.module_from_spec(_spec)
sys.modules["exp02"] = exp02
_spec.loader.exec_module(exp02)


class RestrictedSolver(PortfolioSolver):
    """Exact optimal stopping when only the queries returned by ``chooser`` may be asked at a history."""

    def __init__(self, model, chooser):
        super().__init__(model)
        self.chooser = chooser

    def solve(self, h=None):
        h = self.root if h is None else h
        if h in self.memo:
            return self.memo[h]
        ans = self.stop(h)
        for q in self.chooser(self, h, self.available(h)):
            cost = self.mass(h).sum() * self.m.costs[q]
            candidate = cost + sum(self.solve(hh)[0] for hh in self.children(h, q))
            if candidate < ans[0] - 1e-15:
                ans = (candidate, ("query", q))
        self.memo[h] = ans
        return ans


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def info_gain_chooser(solver, h, available):
    if not available:
        return []
    mass = solver.mass(h)
    b = mass / mass.sum()
    best, pick = -1.0, None
    for q in available:
        lik = solver.m.likelihood[:, q]
        p1 = float(b @ lik)
        cond = float(b @ (-(lik * np.log(np.clip(lik, 1e-300, 1)) + (1 - lik) * np.log(np.clip(1 - lik, 1e-300, 1)))))
        marginal = -(p1 * math.log(max(p1, 1e-300)) + (1 - p1) * math.log(max(1 - p1, 1e-300)))
        gain = (marginal - cond) / float(solver.m.durations[q])
        if gain > best + 1e-15:
            best, pick = gain, q
    return [pick]


def ranked_chooser(order):
    def chooser(solver, h, available):
        for q in order:
            if q in available:
                return [q]
        return []
    return chooser


def voi_order(model):
    """Queries ordered by their one-step value at the prior (lowest expected cost first)."""
    s = PortfolioSolver(model)
    root = s.root
    stop = s.stop(root)[0]
    values = []
    for q in s.available(root):
        v = s.mass(root).sum() * model.costs[q] + sum(s.stop(hh)[0] for hh in s.children(root, q))
        values.append((v - stop, q))
    return [q for _, q in sorted(values)]


class CommitPolicy:
    """Commit at once to portfolio ``a`` (the interface of a solved PortfolioSolver)."""

    def __init__(self, model, a):
        self.root = (-1,) * len(model.queries)
        self.decision = ("act", int(a))

    def solve(self, h=None):
        return float("nan"), self.decision


def commit_policy(model, a):
    return CommitPolicy(model, a)


def type_values(model):
    """Expected score at t=0 by preference type and feasible portfolio: [types, actions]."""
    feasible = np.flatnonzero(model.feasible[0])
    types = model.indices[:, 5]
    out = np.full((types.max() + 1, len(model.actions)), np.inf)
    for k in range(types.max() + 1):
        w = model.prior * (types == k)
        out[k, feasible] = (w @ model.loss[:, 0, feasible]) / w.sum()
    return out


def minimax_regret_action(model):
    v = type_values(model)
    regret = v - v.min(axis=1, keepdims=True)
    return int(np.argmin(regret.max(axis=0)))


def worst_case_action(model):
    severe = model.indices[:, 0] == model.indices[:, 0].max()
    feasible = np.flatnonzero(model.feasible[0])
    w = model.prior * severe
    vals = w @ model.loss[:, 0, feasible]
    return int(feasible[np.argmin(vals)])


class POMCP:
    """Monte Carlo tree search over query sequences with exact posteriors (re-planned at each history)."""

    def __init__(self, model, simulations, seed, ucb=UCB):
        self.m, self.n, self.seed, self.ucb = model, simulations, seed, ucb
        self.base = PortfolioSolver(model)
        self.decisions = {}
        self.commit = {}  # history -> (normalised Bayes commitment cost, portfolio); no belief vectors kept

    def posterior_mass(self, h):
        mass = self.m.prior.copy()
        for q, y in enumerate(h):
            if y != -1:
                mass *= self.m.likelihood[:, q] if y else 1 - self.m.likelihood[:, q]
        return mass

    def commit_cost(self, h):
        if h not in self.commit:
            mass = self.posterior_mass(h)
            t = self.base.time(h)
            values = mass @ self.m.loss[:, t]
            a = min(np.flatnonzero(self.m.feasible[t]), key=lambda j: values[j])
            self.commit[h] = (float(values[a]) / float(mass.sum()), int(a))
        return self.commit[h][0]

    def solve(self, h=None):
        h = self.base.root if h is None else h
        if h not in self.decisions:
            self.decisions[h] = (float("nan"), self.search(h))
        return self.decisions[h]

    def search(self, h0):
        rng = np.random.default_rng([self.seed] + [y + 1 for y in h0])
        mass = self.posterior_mass(h0)
        p = mass / mass.sum()
        tree = {}
        draws = rng.choice(len(p), size=self.n, p=p)
        for z in draws:
            self.simulate(h0, int(z), tree, rng)
        node = tree[h0]
        best = min(node["stats"].items(), key=lambda kv: kv[1][1] / kv[1][0] if kv[1][0] else math.inf)
        a = best[0]
        if a == "commit":
            self.commit_cost(h0)
            return ("act", self.commit[h0][1])
        return ("query", int(a))

    def simulate(self, h, z, tree, rng):
        node = tree.get(h)
        if node is None:
            actions = ["commit"] + self.base.available(h)
            tree[h] = {"N": 0, "stats": {a: [0, 0.0] for a in actions}}
            return self.commit_cost(h)
        node["N"] += 1
        logn = math.log(node["N"] + 1)
        pick, score = None, None
        for a, (n, total) in node["stats"].items():
            s = -1e300 if n == 0 else total / n - self.ucb * math.sqrt(logn / n)
            if score is None or s < score:
                pick, score = a, s
        if pick == "commit":
            cost = self.commit_cost(h)
        else:
            q = pick
            y = int(rng.random() < self.m.likelihood[z, q])
            hh = list(h)
            hh[q] = y
            cost = float(self.m.costs[q]) + self.simulate(tuple(hh), z, tree, rng)
        stats = node["stats"][pick]
        stats[0] += 1
        stats[1] += cost
        return cost


def evaluate(model, policy):
    value, counts, _ = PortfolioSolver(model).evaluate(policy)
    return float(value), float(np.sum(counts))


def run(task):
    source, ident, model_builder, cfg = task
    start = time.perf_counter()
    model = model_builder(cfg)
    joint = PortfolioSolver(model)
    joint_value = float(joint.solve()[0])
    joint_queries = evaluate(model, joint)[1]
    immediate = float(PortfolioSolver(model, method="immediate").solve()[0])
    rows = []
    timings = {}
    for name in BASELINES:
        tick = time.perf_counter()
        if name == "immediate":
            policy = PortfolioSolver(model, method="immediate")
            policy.solve()
        elif name == "one_step":
            policy = PortfolioSolver(model, method="one_step")
        elif name == "fixed_order":
            policy = PortfolioSolver(model, method="fixed_order")
            policy.solve()
        elif name == "voi_ranked":
            policy = RestrictedSolver(model, ranked_chooser(voi_order(model)))
            policy.solve()
        elif name == "info_gain":
            policy = RestrictedSolver(model, info_gain_chooser)
            policy.solve()
        elif name == "minimax_regret":
            policy = commit_policy(model, minimax_regret_action(model))
        elif name == "worst_case":
            policy = commit_policy(model, worst_case_action(model))
        else:
            sims = {"pomcp_1k": 1000, "pomcp_10k": 10000, "pomcp_100k": 100000}[name]
            policy = POMCP(model, sims, seed=SEED + 7919 * ident + (0 if source == "randomized" else 1_000_003))
        value, queries = evaluate(model, policy)
        timings[name] = time.perf_counter() - tick
        rows.append(dict(source=source, instance=ident, baseline=name, value=value, regret=value - joint_value,
                         versus_immediate=value - immediate, expected_queries=queries,
                         joint_queries=joint_queries, joint_value=joint_value, immediate_value=immediate,
                         voi_joint=immediate - joint_value, seconds=timings[name]))
        assert value >= joint_value - 1e-10, (name, value, joint_value)
    for r in rows:
        r["instance_seconds"] = time.perf_counter() - start
    return rows


def tasks(n):
    out = []
    for domain, module, builder, cls in (("flood", fl, fl.build_flood, fl.FloodConfig),
                                          ("social", so, so.build_social_flood, so.SocialConfig)):
        grid = module.GRID
        for i, values in enumerate(product(*grid.values())):
            out.append((domain, i, builder, cls(**dict(zip(grid, values)))))
    for p in exp02.sample_parameters(n):
        cfg = so.SocialConfig(lead_hours=p["lead_hours"], mechanism_strength=p["mechanism_strength"],
                              social_query_reliability=p["social_query_reliability"],
                              preference_prior=p["preference_prior"], query_cost=p["query_cost"],
                              hazard_prior=p["hazard_prior"], original_reliability=p["original_reliability"],
                              preference_slope=p["preference_slope"])
        out.append(("randomized", p["instance"], so.build_social_flood, cfg))
    return out


def summarise(df):
    out = {}
    for (source, name), g in df.groupby(["source", "baseline"]):
        querying = g[g.voi_joint > EPS]
        out[f"{source}|{name}"] = dict(
            instances=int(len(g)), mean_regret=float(g.regret.mean()), max_regret=float(g.regret.max()),
            optimal_share=float((g.regret <= 1e-9).mean()),
            captured_share_of_voi=float((1 - querying.regret / querying.voi_joint).mean()) if len(querying) else None,
            worse_than_immediate_share=float((g.versus_immediate > EPS).mean()),
            max_worse_than_immediate=float(g.versus_immediate.max()),
            mean_queries=float(g.expected_queries.mean()), mean_seconds=float(g.seconds.mean()),
            max_seconds=float(g.seconds.max()))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="ignore saved rows and recompute every task")
    args = parser.parse_args()
    csv = RESULTS / "baselines.csv"
    start = time.perf_counter()
    if not args.analysis_only:
        rows = []
        work = tasks(args.n)
        if csv.exists() and not args.fresh:
            # resume: keep the tasks whose every baseline is saved and run the rest
            old = pd.read_csv(csv)
            complete = {key for key, g in old.groupby(["source", "instance"]) if len(g) == len(BASELINES)}
            rows = old[[(s, i) in complete for s, i in zip(old.source, old.instance)]].to_dict("records")
            work = [t for t in work if (t[0], t[1]) not in complete]
            print(f"resuming: {len(complete)} tasks saved, {len(work)} to run", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for k, part in enumerate(pool.map(run, work, chunksize=1)):
                rows.extend(part)
                if (k + 1) % 20 == 0 or k + 1 == len(work):
                    pd.DataFrame(rows).to_csv(csv, index=False)
                    print(f"{k + 1}/{len(work)}: {time.perf_counter() - start:.1f} s", flush=True)
    df = pd.read_csv(csv)
    summary = summarise(df)
    summary["randomized_instances"] = int(df[df.source == "randomized"].instance.nunique())
    summary["runtime_seconds"] = None if args.analysis_only else time.perf_counter() - start
    (RESULTS / "baselines.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for key, s in summary.items():
        if isinstance(s, dict):
            print(f"{key:28s} regret mean {s['mean_regret']:.5f} max {s['max_regret']:.5f} optimal {s['optimal_share']:.2f}"
                  f" captured {s['captured_share_of_voi']} worse {s['worse_than_immediate_share']:.2f}"
                  f" q {s['mean_queries']:.2f} t {s['mean_seconds']:.2f}")


if __name__ == "__main__":
    main()
