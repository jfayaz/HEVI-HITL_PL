"""Experiment 22: access information, assistance capacity and a distributive weight.

Section 5.7; Supplementary Section S3.
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

from hevi_hitl import flood as fl
from hevi_hitl import social as so
from hevi_hitl.insight import plan, revealed_value
from hevi_hitl.model import PortfolioSolver

RESULTS = ROOT / "results"
KAPPAS = (0.0, 1.0, 3.0)
CHECKS = ("published", "mobility_only")
TEAMS = (1, 2)
EPS = 1e-10


_ORIGINAL_INVENTORY = so.action_inventory


def inventory(teams: int):
    """Portfolio inventory with ``teams`` assistance teams of 150 places, one district each."""
    if teams == 1:
        return _ORIGINAL_INVENTORY

    def build(shelter_share=0.9):
        actions, costs, completion = [], [], []
        for action in product(tuple(product(range(3), range(3))), repeat=3):
            a = np.array(action)
            e = a[:, 1]
            cost = float((fl.PROPERTY_COST_EUR[a[:, 0]] + fl.EVACUATION_COST_EUR * (e > 0)
                          + so.ASSISTANCE_EXTRA_EUR * (e == 2)) @ fl.DISTRICT_COST_MULTIPLIER)
            shelter = float((shelter_share * 1.2 * fl.BASE_OCCUPANTS) @ (e > 0))
            need = so.GROUP_SHARES[1] * 1.2 * fl.BASE_OCCUPANTS
            if (cost > fl.BUDGET_EUR + 1e-9 or np.sum(a[:, 0] == 1) > 1 or np.sum(a[:, 0] == 2) > 2
                    or shelter > fl.SHELTER_CAPACITY or np.sum(e == 2) > teams
                    or np.any(need[e == 2] > so.ASSISTANCE_CAPACITY)):
                continue
            actions.append(action)
            costs.append(cost)
            completion.append(int(max(np.max(fl.IMPLEMENTATION_TICKS[a[:, 0]]),
                                      np.max(np.array([0, fl.EVACUATION_TICKS, so.ASSISTANCE_TICKS])[e]))))
        return tuple(actions), np.array(costs), np.array(completion)

    return build


def build_model(cfg, teams: int):
    so.action_inventory = inventory(teams)
    try:
        return so.build_social_flood(cfg)
    finally:
        so.action_inventory = _ORIGINAL_INVENTORY


def variant(model, check: str, kappa: float):
    """Copy of a social model with the chosen access check and distributive weight."""
    lik = model.likelihood.copy()
    if check == "mobility_only":
        rho = model.config.social_query_reliability
        lik[:, so.SOCIAL_QUERY] = 0.5 + rho * (0.1 + 0.8 * model.indices[:, 6] - 0.5)
    loss = model.loss
    if kappa:
        weights = fl.weights_of(model.config)[model.indices[:, 5]]
        extra = kappa * weights[:, 2, None] * model.group_features[:, :, 1, 1] / fl.SCALES[2]
        loss = model.loss + extra[:, None, :]
    return replace(model, likelihood=lik, loss=loss)


def summary(model, policy) -> dict:
    """Exact expectations of a frozen policy (all answers enumerated), for any assistance inventory."""
    score = check = assisted = districts = exposed = 0.0
    groups = np.zeros((2, 4))
    stack = [(policy.root, model.prior.copy())]
    while stack:
        h, mass = stack.pop()
        kind, j = policy.decisions[h]
        t = int(sum(model.durations[q] for q, y in enumerate(h) if y != -1))
        if kind == "act":
            p = float(mass.sum())
            e = [x for _, x in model.actions[j]]
            score += float(mass @ model.loss[:, t, j])
            assisted += p * (2 in e)
            districts += p * sum(x == 2 for x in e)
            groups += np.einsum("z,zgk->gk", mass, model.group_features[:, j])
            exposed += float(mass @ model.features[:, t, j, 2])
            continue
        score += float(mass.sum() * model.costs[j])
        if j == so.SOCIAL_QUERY:
            check += float(mass.sum())
        for y in (0, 1):
            hh = list(h)
            hh[j] = y
            stack.append((tuple(hh), mass * (model.likelihood[:, j] if y else 1 - model.likelihood[:, j])))
    return dict(score=score, access_check=check, assisted_probability=assisted, assisted_districts=districts,
                constrained_exposed_fraction=float(groups[1, 3]), independent_exposed_fraction=float(groups[0, 3]),
                exposed_people=exposed, constrained_exposed_people=float(groups[1, 1]))


def run(task):
    cid, cfg = task
    rows = []
    for teams in TEAMS:
      base = build_model(cfg, teams)
      for check, kappa in product(CHECKS, KAPPAS):
        if teams > 1 and check != "published":
            continue
        model = variant(base, check, kappa)
        joint = plan(model, "joint")
        immediate = plan(model, "immediate")
        without_check = PortfolioSolver(model, allowed=[q for q in range(len(model.queries)) if q != so.SOCIAL_QUERY])
        v_without = float(without_check.solve()[0])
        check_only = PortfolioSolver(model, allowed=[so.SOCIAL_QUERY])
        v_check_only = float(check_only.solve()[0])
        s_joint, s_imm = summary(model, joint), summary(model, immediate)
        v_imm = revealed_value(model, ())
        row = dict(configuration_id=cid, teams=teams, portfolios=len(model.actions), check=check, kappa=kappa, **{f"joint_{k}": v for k, v in s_joint.items()},
                   **{f"immediate_{k}": v for k, v in s_imm.items()},
                   voi_joint=v_imm - s_joint["score"], value_of_check_in_joint=v_without - s_joint["score"],
                   standalone_check_value=v_imm - v_check_only,
                   evppi_access=v_imm - revealed_value(model, (6, 7)),
                   evppi_mobility=v_imm - revealed_value(model, (6,)),
                   evppi_recovery=v_imm - revealed_value(model, (7,)),
                   evppi_severity=v_imm - revealed_value(model, (0,)))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    start = time.perf_counter()
    grid = so.GRID
    tasks = [(i, so.SocialConfig(**dict(zip(grid, values)))) for i, values in enumerate(product(*grid.values()))]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for part in pool.map(run, tasks):
            rows.extend(part)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "social_access.csv", index=False)
    out = {}
    for (teams, check, kappa), g in df.groupby(["teams", "check", "kappa"]):
        out[f"teams{teams}|{check}|{kappa:g}"] = dict(
            portfolios=int(g.portfolios.iloc[0]),
            mean_assisted_districts=float(g.joint_assisted_districts.mean()),
            access_check_asked_instances=int((g.joint_access_check > EPS).sum()),
            mean_access_check=float(g.joint_access_check.mean()),
            max_standalone_check_value=float(g.standalone_check_value.max()),
            max_value_of_check_in_joint=float(g.value_of_check_in_joint.max()),
            max_evppi_access=float(g.evppi_access.max()), mean_evppi_access=float(g.evppi_access.mean()),
            max_evppi_mobility=float(g.evppi_mobility.max()), max_evppi_recovery=float(g.evppi_recovery.max()),
            mean_evppi_severity=float(g.evppi_severity.mean()),
            mean_assisted_probability=float(g.joint_assisted_probability.mean()),
            min_assisted_probability=float(g.joint_assisted_probability.min()),
            mean_constrained_exposed_fraction=float(g.joint_constrained_exposed_fraction.mean()),
            mean_independent_exposed_fraction=float(g.joint_independent_exposed_fraction.mean()),
            mean_exposed_people=float(g.joint_exposed_people.mean()),
            mean_voi_joint=float(g.voi_joint.mean()))
    out["runtime_seconds"] = time.perf_counter() - start
    (RESULTS / "social_access.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
