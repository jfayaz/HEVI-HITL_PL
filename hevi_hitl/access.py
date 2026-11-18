"""Access-check variants of the social model (Supplementary Section S11)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import numpy as np

from . import flood as f
from . import social as so
from .model import Portfolio

CHECK_COLUMNS = {"common": (so.SOCIAL_QUERY,), "district": (4, 5, 6)}
ACCESS_AXES = {"common": (6, 7), "district": (6, 7, 8, 9)}
MOBILITY_AXES = {"common": (6,), "district": (6, 7, 8)}


@dataclass
class AccessModel(Portfolio):
    """``Portfolio`` with group outcomes ``group[z, a, g, k]`` and time-invariant scores ``loss2d``."""

    group: np.ndarray = None
    loss2d: np.ndarray = None
    checks: tuple = ()


def inventory(extra_eur: float, shelter_share: float = 0.9):
    """Portfolio inventory of the social model with assisted evacuation costing ``extra_eur`` more."""
    actions, costs, completion = [], [], []
    for action in product(tuple(product(range(3), range(3))), repeat=3):
        a = np.array(action)
        e = a[:, 1]
        cost = float((f.PROPERTY_COST_EUR[a[:, 0]] + f.EVACUATION_COST_EUR * (e > 0) + extra_eur * (e == 2))
                     @ f.DISTRICT_COST_MULTIPLIER)
        shelter = float((shelter_share * 1.2 * f.BASE_OCCUPANTS) @ (e > 0))
        assistance = float((so.GROUP_SHARES[1] * 1.2 * f.BASE_OCCUPANTS) @ (e == 2))
        if (cost > f.BUDGET_EUR + 1e-9 or np.sum(a[:, 0] == 1) > 1 or np.sum(a[:, 0] == 2) > 2
                or shelter > f.SHELTER_CAPACITY or np.sum(e == 2) > 1 or assistance > so.ASSISTANCE_CAPACITY):
            continue
        actions.append(action)
        costs.append(cost)
        completion.append(int(max(np.max(f.IMPLEMENTATION_TICKS[a[:, 0]]),
                                  np.max(np.array([0, f.EVACUATION_TICKS, so.ASSISTANCE_TICKS])[e]))))
    return tuple(actions), np.array(costs), np.array(completion)


def _base_config(cfg: so.SocialConfig) -> f.FloodConfig:
    return f.FloodConfig(lead_hours=cfg.lead_hours, response_reliability=cfg.original_reliability,
                         query_cost=cfg.query_cost, hazard_prior=cfg.hazard_prior,
                         preference_prior=cfg.preference_prior, preference_slope=cfg.preference_slope)


def build_access(cfg: so.SocialConfig, structure: str = "district", extra_eur: float = so.ASSISTANCE_EXTRA_EUR,
                 check_reliability: float | None = None, check_cost_factor: float = 1.0) -> AccessModel:
    """Social model with the given access structure, assistance cost and check reliability and cost."""
    if structure not in CHECK_COLUMNS:
        raise ValueError("structure must be 'common' or 'district'")
    r = cfg.social_query_reliability if check_reliability is None else float(check_reliability)
    base_cfg = _base_config(cfg)
    base = f.build_flood(base_cfg)
    social = (np.array(list(product(range(2), repeat=2))) if structure == "common"
              else np.array(list(product(range(2), repeat=4))))
    ns = len(social)
    base_id = np.repeat(np.arange(len(base.prior)), ns)
    soc = np.tile(social, (len(base.prior), 1))
    idx = np.column_stack((base.indices[base_id], soc))
    prior = base.prior[base_id] / ns
    _, _, depth0, occupants0 = f._states(base_cfg)
    depth = depth0[base_id]
    population = occupants0[base_id][:, :, None] * so.GROUP_SHARES
    group_population = population.sum(axis=1)
    lam = cfg.mechanism_strength
    mobility = np.repeat(soc[:, :1], 3, axis=1) if structure == "common" else soc[:, :3]
    recovery = soc[:, -1]
    # rates[z, district, evacuation option, group]
    rates = np.zeros((len(idx), 3, 3, 2))
    rates[:, :, 1:, 0] = 0.9
    rates[:, :, 1, 1] = np.maximum(0.0, 0.9 - lam * so.ACCESS_SHORTFALL[mobility])
    rates[:, :, 2, 1] = 0.9 - lam * so.ASSISTED_RESIDUAL_SHORTFALL
    delay = np.column_stack((np.zeros(len(idx)), lam * so.RECOVERY_ACCESS_DAYS[recovery]))
    actions, costs, completion = inventory(extra_eur)
    original_actions = {a: j for j, a in enumerate(base.actions)}
    floor = f.FLOOR_HEIGHT_M + np.array([0.1, -0.1])[idx[:, 3, None]]
    susceptibility = np.array([0.85, 1.15])[idx[:, 3, None]]
    restoration = np.array([0.75, 1.25])[idx[:, 4, None]]
    raw = np.zeros((len(idx), len(actions), 4))
    group = np.zeros((len(idx), len(actions), 2, 4))
    districts = np.arange(3)
    for j, action in enumerate(actions):
        a = np.array(action)
        original = tuple((p, int(e > 0)) for p, e in action)
        raw[:, j, 0] = base.features[base_id, 0, original_actions[original], 0]
        raw[:, j, 3] = costs[j]
        defended = np.where((a[:, 0] == 1) & (depth <= 1.2), np.maximum(0, depth - 0.65), depth)
        internal = np.maximum(0, defended - floor)
        structure_fraction = np.minimum(1, susceptibility * np.interp(internal, f.DEPTH_KNOTS_M, f.STRUCTURE_DAMAGE))
        technical = np.where(internal > 0, restoration * (2 + 50 * structure_fraction + 10 * internal), 0.0)
        uninhabitable = np.clip((internal - 0.05) / 0.45, 0, 1)
        at_risk = f.exposure_factor(defended, base_cfg)[:, :, None]
        evac_rate = rates[:, districts, a[:, 1], :]
        displacement = (population * (uninhabitable[:, :, None] * (technical[:, :, None] + delay[:, None, :])
                                      + evac_rate * (1 - uninhabitable[:, :, None]) * 0.75)).sum(axis=1)
        exposed = (population * (1 - evac_rate) * at_risk).sum(axis=1)
        group[:, j, :, 0] = displacement
        group[:, j, :, 1] = exposed
        group[:, j, :, 2] = displacement / group_population
        group[:, j, :, 3] = exposed / group_population
        raw[:, j, 1] = displacement.sum(axis=1)
        raw[:, j, 2] = exposed.sum(axis=1)
    normalized = raw[:, :, :3].copy()
    normalized[:, :, 0] += raw[:, :, 3]
    normalized /= f.SCALES
    loss2d = np.einsum("zaq,zq->za", normalized, f.weights_of(base_cfg)[idx[:, 5]])
    if structure == "common":
        flags = (0.1 + 0.4 * soc[:, 0] + 0.4 * soc[:, 1])[:, None]
        names = ("social_access_check",)
    else:
        flags = 0.1 + 0.8 * soc[:, :3]
        names = ("access_check_riverside", "access_check_central", "access_check_upstream")
    checks = 0.5 + r * (flags - 0.5)
    likelihood = np.column_stack((base.likelihood[base_id, :4], checks, base.likelihood[base_id, 4:6]))
    nq = likelihood.shape[1]
    durations = np.array([1, 1, 2, 2] + [1] * (nq - 4))
    costs_q = cfg.query_cost * durations.astype(float)
    check_cols = CHECK_COLUMNS[structure]
    costs_q[list(check_cols)] *= check_cost_factor
    feasible = np.arange(cfg.deadline + 1)[:, None] + completion <= cfg.deadline
    shape = (len(idx), cfg.deadline + 1, len(actions))
    queries = tuple(f.QUERIES[:4]) + names + tuple(f.QUERIES[4:6])
    return AccessModel(prior, likelihood, np.broadcast_to(loss2d[:, None, :], shape),
                       np.broadcast_to(raw[:, None], shape + (4,)), feasible, durations, costs_q, idx, actions,
                       queries, replace(cfg, social_query_reliability=r), group=group, loss2d=loss2d,
                       checks=check_cols)


def without_queries(model: AccessModel, drop) -> AccessModel:
    """Copy of ``model`` without the queries in ``drop`` (for the value of the checks)."""
    keep = [q for q in range(len(model.queries)) if q not in set(drop)]
    return replace(model, likelihood=model.likelihood[:, keep], durations=model.durations[keep],
                   costs=model.costs[keep], queries=tuple(model.queries[q] for q in keep), checks=())
