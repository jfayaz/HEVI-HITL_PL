"""Planners, value-of-information decomposition and misspecification helpers (Sections 2.5 and 5)."""
from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np

from . import flood as fl
from . import social as so
from .model import PortfolioSolver
from .operational import FrozenPolicy, condition, evaluate, freeze

FLOOD_GRID = fl.GRID
SOCIAL_GRID = so.GRID
WORKED_SOCIAL_ID = 10
POLICIES = ("joint", "immediate", "physical_only", "preference_only", "one_step")
SEVERE = 2                                   # severity index of the 1.65 m scenario
TOL = 1e-10


def flood_configs() -> list[fl.FloodConfig]:
    """The 32 flood configurations in the fixed grid order."""
    keys = list(FLOOD_GRID)
    return [fl.FloodConfig(**dict(zip(keys, values)))
            for values in product(*(FLOOD_GRID[k] for k in keys))]


def social_configs() -> list[so.SocialConfig]:
    """The 16 social configurations in the fixed grid order (id 10 is the worked case)."""
    keys = list(SOCIAL_GRID)
    return [so.SocialConfig(**dict(zip(keys, values)))
            for values in product(*(SOCIAL_GRID[k] for k in keys))]


# Model helpers
def is_social(model) -> bool:
    return hasattr(model, "group_features")


def query_sets(model) -> dict[str, tuple[int, ...]]:
    """Risk (physical, plus social-access in the social model) and preference queries."""
    if is_social(model):
        return {"risk": (0, 1, 2, 3, 4), "preference": tuple(range(5, len(model.queries)))}
    return {"risk": (0, 1, 2, 3), "preference": tuple(range(4, len(model.queries)))}


def preference_axis(model) -> int:
    """Column of ``model.indices`` holding the preference type."""
    return 5


def physical_axes(model) -> tuple[int, ...]:
    """Columns of ``model.indices`` holding the physical (and social) state."""
    return tuple(k for k in range(model.indices.shape[1]) if k != preference_axis(model))


def same_public_graph(a, b) -> bool:
    """Two models share actions, queries, durations, costs, feasibility and deadline."""
    return (a.actions == b.actions and a.queries == b.queries
            and np.array_equal(a.durations, b.durations) and np.array_equal(a.costs, b.costs)
            and np.array_equal(a.feasible, b.feasible)
            and a.config.deadline == b.config.deadline)


def with_prior(model, prior: np.ndarray):
    """Copy of ``model`` with a replaced (possibly unnormalised) prior."""
    return replace(model, prior=np.asarray(prior, dtype=float))


# Planning
def complete(solver: PortfolioSolver) -> PortfolioSolver:
    """Ensure every reachable history has a memoised decision."""
    pending = [solver.root]
    seen = set()
    while pending:
        h = pending.pop()
        if h in seen:
            continue
        seen.add(h)
        kind, j = solver.solve(h)[1]
        if kind == "query":
            for child in solver.children(h, j):
                pending.append(child)
    return solver


def plan(model, name: str) -> FrozenPolicy:
    """Solve one of the named planners in ``model`` and freeze its policy."""
    sets = query_sets(model)
    allowed, method = {
        "joint": (None, "optimal"),
        "immediate": (None, "immediate"),
        "physical_only": (sets["risk"], "optimal"),
        "preference_only": (sets["preference"], "optimal"),
        "one_step": (None, "one_step"),
    }[name]
    solver = PortfolioSolver(model, allowed=allowed, method=method)
    solver.solve()
    return freeze(complete(solver))


def per_state_cost(model, policy: FrozenPolicy) -> np.ndarray:
    """Per-configuration expected total cost (terminal loss plus acquisition) of a policy."""
    out = np.zeros(len(model.prior))
    stack = [(policy.root, np.ones(len(model.prior)), 0.0)]
    while stack:
        h, prob, spent = stack.pop()
        kind, j = policy.decisions[h]
        t = int(sum(model.durations[q] for q, y in enumerate(h) if y != -1))
        if kind == "act":
            out += prob * (model.loss[:, t, j] + spent)
            continue
        for y in (0, 1):
            hh = list(h)
            hh[j] = y
            factor = model.likelihood[:, j] if y else 1 - model.likelihood[:, j]
            stack.append((tuple(hh), prob * factor, spent + float(model.costs[j])))
    return out


def policy_profile(model, policy: FrozenPolicy, prior=None) -> dict:
    """Flat expected-score and consequence summary of a frozen policy."""
    ev = evaluate(model, policy, prior=prior)
    f = np.asarray(ev["physical_features"])
    counts = np.asarray(ev["expected_query_counts"])
    sets = query_sets(model)
    barrier = sum(p for j, p in ev["action_probabilities"].items()
                  if any(a[0] == 1 for a in model.actions[int(j)]))
    out = dict(
        score=float(ev["expected_loss"]),
        query_cost=float(ev["expected_query_cost"]),
        property_damage_eur=float(f[0]),
        intervention_eur=float(f[3]),
        damage_plus_intervention_eur=float(f[0] + f[3]),
        displacement_person_days=float(f[1]),
        exposed_people=float(f[2]),
        risk_queries=float(counts[list(sets["risk"])].sum()),
        preference_queries=float(counts[list(sets["preference"])].sum()),
        commitment_hours=float(ev["expected_commitment_hours"]),
        root_decision=list(policy.decisions[policy.root]),
        any_query=bool(policy.decisions[policy.root][0] == "query"),
        barrier_probability=float(barrier),
    )
    if is_social(model):
        g = np.asarray(ev["group_features"])
        out.update(social_queries=float(counts[4]),
                   constrained_exposed_people=float(g[1, 1]),
                   constrained_exposed_fraction=float(g[1, 3]),
                   independent_exposed_fraction=float(g[0, 3]),
                   constrained_person_days=float(g[1, 0]))
    return out


# Information bounds
def revealed_value(model, axes: tuple[int, ...]) -> float:
    """Expected loss when the latent axes ``axes`` are revealed free at t=0."""
    feasible = np.flatnonzero(model.feasible[0])
    if len(axes) == 0:
        keys = np.zeros(len(model.prior), dtype=int)
    else:
        _, keys = np.unique(model.indices[:, list(axes)], axis=0, return_inverse=True)
        keys = keys.ravel()
    weighted = model.prior[:, None] * model.loss[:, 0, feasible]
    totals = np.zeros((keys.max() + 1, len(feasible)))
    np.add.at(totals, keys, weighted)
    return float(totals.min(axis=1).sum())


def perfect_information(model) -> dict:
    """Perfect-information value and its consequences (free full revelation)."""
    feasible = np.flatnonzero(model.feasible[0])
    best = feasible[np.argmin(model.loss[:, 0, feasible], axis=1)]
    zz = np.arange(len(model.prior))
    f = model.prior @ model.features[zz, 0, best]
    out = dict(score=float(model.prior @ model.loss[zz, 0, best]),
               property_damage_eur=float(f[0]), intervention_eur=float(f[3]),
               damage_plus_intervention_eur=float(f[0] + f[3]),
               displacement_person_days=float(f[1]), exposed_people=float(f[2]))
    if is_social(model):
        g = np.einsum("z,zgk->gk", model.prior, model.group_features[zz, best])
        out.update(constrained_exposed_people=float(g[1, 1]),
                   constrained_exposed_fraction=float(g[1, 3]))
    assert abs(out["score"] - revealed_value(model, tuple(range(model.indices.shape[1])))) < 1e-12
    return out


# Decision-value decomposition
CONSEQUENCES = ("damage_plus_intervention_eur", "property_damage_eur", "intervention_eur",
                "displacement_person_days", "exposed_people")


def decision_value(model, homogeneous_model=None) -> dict:
    """VOI decomposition, bounds and consequences for one configuration."""
    policies = {name: plan(model, name) for name in POLICIES}
    profiles = {name: policy_profile(model, p) for name, p in policies.items()}
    if homogeneous_model is not None:
        assert same_public_graph(model, homogeneous_model)
        hom = plan(homogeneous_model, "joint")
        profiles["homogeneous"] = policy_profile(model, hom)
        profiles["homogeneous"]["own_model_prediction"] = policy_profile(homogeneous_model, hom)["score"]
    pi = perfect_information(model)
    v = {name: profiles[name]["score"] for name in profiles}
    # Exact DP must be no worse than any restricted planner and no better than PI.
    for name in ("immediate", "physical_only", "preference_only", "one_step") + (
            ("homogeneous",) if homogeneous_model is not None else ()):
        assert v["joint"] <= v[name] + TOL, (name, v)
    assert v["joint"] >= pi["score"] - TOL
    evppi_pref = revealed_value(model, (preference_axis(model),))
    evppi_phys = revealed_value(model, physical_axes(model))
    assert abs(revealed_value(model, ()) - v["immediate"]) < 1e-10
    voi_joint = v["immediate"] - v["joint"]
    voi_phys = v["immediate"] - v["physical_only"]
    voi_pref = v["immediate"] - v["preference_only"]
    evpi = v["immediate"] - pi["score"]
    decomposition = dict(
        voi_joint=voi_joint, voi_physical=voi_phys, voi_preference=voi_pref,
        interaction=voi_joint - voi_phys - voi_pref,
        evpi=evpi, evppi_preference=v["immediate"] - evppi_pref,
        evppi_physical=v["immediate"] - evppi_phys,
        captured_share=voi_joint / evpi if evpi > 0 else float("nan"),
        one_step_regret=v["one_step"] - v["joint"],
    )
    differences = {name: {c: profiles[name][c] - profiles["immediate"][c] for c in
                          CONSEQUENCES + (("constrained_exposed_fraction", "constrained_exposed_people")
                                          if is_social(model) else ())}
                   for name in profiles if name != "immediate"}
    return dict(policies=profiles, perfect_information=pi, decomposition=decomposition,
                minus_immediate=differences)


# Misspecification
def misspecification(true_model, plan_model, cache: dict | None = None) -> dict:
    """Plan in ``plan_model``, freeze, evaluate in ``true_model``."""
    assert same_public_graph(true_model, plan_model)
    cache = {} if cache is None else cache
    if "joint" not in cache:
        cache["joint"] = PortfolioSolver(true_model).solve()[0]
        cache["immediate"] = PortfolioSolver(true_model, method="immediate").solve()[0]
    policy = plan(plan_model, "joint")
    true_profile = policy_profile(true_model, policy)
    # immediate commitment chosen under the assumed model, scored in the true model
    planned_immediate = float(true_model.prior @ per_state_cost(true_model, plan(plan_model, "immediate")))
    return dict(true_score=true_profile["score"], true_optimum=cache["joint"],
                true_immediate=cache["immediate"],
                planned_immediate_true_score=planned_immediate,
                regret=true_profile["score"] - cache["joint"],
                versus_immediate=true_profile["score"] - cache["immediate"],
                versus_planned_immediate=true_profile["score"] - planned_immediate,
                planned_score=policy_profile(plan_model, policy)["score"],
                risk_queries=true_profile["risk_queries"],
                preference_queries=true_profile["preference_queries"])


# Stress-weighted objective
def severe_indicator(model) -> np.ndarray:
    return (model.indices[:, 0] == SEVERE).astype(float)


def tilted_prior(model, lam: float) -> np.ndarray:
    """b_lambda(z) proportional to b0(z) (1 + lambda 1{severity high})."""
    w = model.prior * (1 + lam * severe_indicator(model))
    return w / w.sum()


def stress_policy(model, lam: float) -> tuple[FrozenPolicy, dict]:
    """Optimal policy for the stress-weighted objective (Section 2.5) and the identity of Appendix A.1."""
    s = severe_indicator(model)
    p_severe = float(model.prior @ s)
    tilted = with_prior(model, tilted_prior(model, lam))
    solver = PortfolioSolver(tilted)
    tilted_value = solver.solve()[0]
    policy = freeze(solver)
    weighted_value = PortfolioSolver(with_prior(model, model.prior * (1 + lam * s))).solve()[0]
    direct = float((model.prior * (1 + lam * s)) @ per_state_cost(model, policy))
    scaled = (1 + lam * p_severe) * tilted_value
    audit = dict(lam=lam, severe_probability=p_severe, tilted_value=tilted_value,
                 weighted_solver_value=weighted_value, reweighted_policy_value=direct,
                 identity_error_solver=abs(weighted_value - scaled),
                 identity_error_policy=abs(direct - scaled))
    assert audit["identity_error_solver"] < 1e-10 and audit["identity_error_policy"] < 1e-10
    return policy, audit


def stress_profile(model, policy: FrozenPolicy) -> dict:
    """Expected profile under b0 and conditional on each severe footprint."""
    out = dict(original=policy_profile(model, policy))
    severe = []
    for footprint in (0, 1):
        prior, prob = condition(model, SEVERE, footprint)
        prof = policy_profile(model, policy, prior=prior)
        prof["scenario_probability"] = prob
        out[f"severe_footprint_{footprint}"] = prof
        severe.append((prob, prof))
    total = sum(p for p, _ in severe)
    out["severe_any_footprint"] = {k: sum(p * prof[k] for p, prof in severe) / total
                                   for k in ("score",) + CONSEQUENCES}
    return out


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def save_json(path, obj) -> None:
    """Write ``obj`` as indented JSON, converting numpy scalars and arrays."""
    import json
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2))
