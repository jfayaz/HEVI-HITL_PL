"""Social-access extension of the flood benchmark (Appendix C)."""

from dataclasses import dataclass
from itertools import product
import numpy as np
from . import flood as f
from .model import Portfolio


GROUPS = ("independent_access", "assistance_constrained")
GROUP_SHARES = np.array([0.7, 0.3])
ACCESS_SHORTFALL = np.array([0.15, 0.45])
RECOVERY_ACCESS_DAYS = np.array([2.0, 10.0])
ASSISTED_RESIDUAL_SHORTFALL = 0.05
ASSISTANCE_EXTRA_EUR = 15000.0
ASSISTANCE_TICKS = 4
ASSISTANCE_CAPACITY = 150.0
SOCIAL_QUERY = 4
PREFERENCE_QUERIES = (5, 6)
CLASS_COUNTS = (96, 192, 384, 576, 768)
TOLERANCES = (0.005, 0.025)
QUERIES = f.QUERIES[:4] + ("social_access_check",) + f.QUERIES[4:]
# the 16 designed instances
GRID = {
    "lead_hours": [4, 6],
    "mechanism_strength": [0.5, 1.0],
    "social_query_reliability": [0.4, 0.9],
    "preference_prior": ["diffuse", "economic_concentrated"],
}


@dataclass(frozen=True)
class SocialConfig:
    """Parameters of one social-access instance."""

    lead_hours: int = 6
    mechanism_strength: float = 1.0
    social_query_reliability: float = 0.9
    preference_prior: object = "diffuse"
    query_cost: float = 0.00025
    hazard_prior: object = "elevated"
    original_reliability: float = 0.9
    preference_slope: float = 40.0
    comparison_reliability: object = None  # None: comparisons use original_reliability
    # assumptions varied in Section 5.7, as in flood.FloodConfig
    exposure_depth: float = 0.3
    exposure_depth_full: object = None
    barrier_limit: float = 1.2
    shelter_share: float = 0.9
    evacuation_rate: float = 0.9
    evacuation_decay: float = 0.0
    preference_weights: object = None
    extra_comparisons: tuple = ()
    comparison_stakes: float = 1.0

    @property
    def deadline(self):
        """Public decision deadline in half-hour ticks."""
        return 2 * self.lead_hours


@dataclass
class SocialModel(Portfolio):
    """``Portfolio`` with group-level outcomes, group populations and implementation times."""

    group_features: np.ndarray
    group_population: np.ndarray
    implementation_ticks: np.ndarray


def action_inventory(shelter_share=0.9):
    """Enumerate publicly feasible portfolios with assisted evacuation."""
    actions, costs, completion = [], [], []
    for action in product(tuple(product(range(3), range(3))), repeat=3):
        a = np.array(action)
        e = a[:, 1]
        cost = float(
            (f.PROPERTY_COST_EUR[a[:, 0]] + f.EVACUATION_COST_EUR * (e > 0) + ASSISTANCE_EXTRA_EUR * (e == 2))
            @ f.DISTRICT_COST_MULTIPLIER
        )
        shelter = float((shelter_share * 1.2 * f.BASE_OCCUPANTS) @ (e > 0))
        assistance = float((GROUP_SHARES[1] * 1.2 * f.BASE_OCCUPANTS) @ (e == 2))
        if (
            cost > f.BUDGET_EUR + 1e-9
            or np.sum(a[:, 0] == 1) > 1
            or np.sum(a[:, 0] == 2) > 2
            or shelter > f.SHELTER_CAPACITY
            or np.sum(e == 2) > 1
            or assistance > ASSISTANCE_CAPACITY
        ):
            continue
        actions.append(action)
        costs.append(cost)
        completion.append(
            int(
                max(
                    np.max(f.IMPLEMENTATION_TICKS[a[:, 0]]),
                    np.max(np.array([0, f.EVACUATION_TICKS, ASSISTANCE_TICKS])[e]),
                )
            )
        )
    return tuple(actions), np.array(costs), np.array(completion)


def build_social_flood(cfg=SocialConfig(), homogeneous=False):
    """Build the 768-state social-access model."""
    if (
        int(cfg.lead_hours) != cfg.lead_hours
        or cfg.lead_hours < 2
        or not 0 <= cfg.mechanism_strength <= 1
        or not 0 <= cfg.social_query_reliability <= 1
    ):
        raise ValueError("Invalid social experiment parameter")
    base_cfg = f.FloodConfig(
        lead_hours=cfg.lead_hours,
        response_reliability=cfg.original_reliability,
        query_cost=cfg.query_cost,
        hazard_prior=cfg.hazard_prior,
        preference_prior=cfg.preference_prior,
        preference_slope=cfg.preference_slope,
        comparison_reliability=cfg.comparison_reliability,
        exposure_depth=cfg.exposure_depth,
        exposure_depth_full=cfg.exposure_depth_full,
        barrier_limit=cfg.barrier_limit,
        shelter_share=cfg.shelter_share,
        evacuation_rate=cfg.evacuation_rate,
        evacuation_decay=cfg.evacuation_decay,
        preference_weights=cfg.preference_weights,
        extra_comparisons=cfg.extra_comparisons,
        comparison_stakes=cfg.comparison_stakes,
    )
    base = f.build_flood(base_cfg)
    base_id = np.repeat(np.arange(len(base.prior)), 4)
    social = np.tile(np.array(list(product(range(2), repeat=2))), (len(base.prior), 1))
    idx = np.column_stack((base.indices[base_id], social))
    prior = base.prior[base_id] / 4
    _, _, depth0, occupants0 = f._states(base_cfg)
    depth = depth0[base_id]
    occupants = occupants0[base_id]
    population = occupants[:, :, None] * GROUP_SHARES
    group_population = population.sum(axis=1)
    strength = 0.0 if homogeneous else cfg.mechanism_strength
    rates = np.zeros((len(idx), 3, 2))
    rates[:, 1, 0] = cfg.evacuation_rate
    rates[:, 2, 0] = cfg.evacuation_rate
    rates[:, 1, 1] = np.maximum(0.0, cfg.evacuation_rate - strength * ACCESS_SHORTFALL[social[:, 0]])
    rates[:, 2, 1] = np.maximum(0.0, cfg.evacuation_rate - strength * ASSISTED_RESIDUAL_SHORTFALL)
    delay = np.column_stack((np.zeros(len(idx)), strength * RECOVERY_ACCESS_DAYS[social[:, 1]]))
    actions, costs, completion = action_inventory(cfg.shelter_share)
    scale = f.evacuation_scale(base_cfg)
    n_ticks = cfg.deadline + 1
    raw = np.zeros((len(idx), n_ticks, len(actions), 4))
    group = np.zeros((len(idx), len(actions), 2, 4))
    original_actions = {a: j for j, a in enumerate(base.actions)}
    floor = f.FLOOR_HEIGHT_M + np.array([0.1, -0.1])[idx[:, 3, None]]
    susceptibility = np.array([0.85, 1.15])[idx[:, 3, None]]
    restoration = np.array([0.75, 1.25])[idx[:, 4, None]]
    for j, action in enumerate(actions):
        a = np.array(action)
        original = tuple((p, int(e > 0)) for p, e in action)
        raw[:, :, j, 0] = base.features[base_id, 0, original_actions[original], 0][:, None]
        raw[:, :, j, 3] = costs[j]
        defended = np.where((a[:, 0] == 1) & (depth <= cfg.barrier_limit), np.maximum(0, depth - 0.65), depth)
        internal = np.maximum(0, defended - floor)
        structure = np.minimum(1, susceptibility * np.interp(internal, f.DEPTH_KNOTS_M, f.STRUCTURE_DAMAGE))
        technical = np.where(internal > 0, restoration * (2 + 50 * structure + 10 * internal), 0.0)
        uninhabitable = np.clip((internal - 0.05) / 0.45, 0, 1)
        at_risk = f.exposure_factor(defended, cfg)[:, :, None]
        # with decay, evacuation effectiveness falls with the commitment tick
        for tick in ((0,) if cfg.evacuation_decay == 0 else range(n_ticks)):
            evac_rate = rates[:, a[:, 1], :] * scale[tick]
            displacement = (
                population
                * (
                    uninhabitable[:, :, None] * (technical[:, :, None] + delay[:, None, :])
                    + evac_rate * (1 - uninhabitable[:, :, None]) * 0.75
                )
            ).sum(axis=1)
            exposed = (population * (1 - evac_rate) * at_risk).sum(axis=1)
            if tick == 0:
                # group outcomes refer to commitment at the warning
                group[:, j, :, 0] = displacement
                group[:, j, :, 1] = exposed
                group[:, j, :, 2] = displacement / group_population
                group[:, j, :, 3] = exposed / group_population
            ticks = slice(None) if cfg.evacuation_decay == 0 else tick
            raw[:, ticks, j, 1] = displacement.sum(axis=1)[:, None] if cfg.evacuation_decay == 0 else displacement.sum(axis=1)
            raw[:, ticks, j, 2] = exposed.sum(axis=1)[:, None] if cfg.evacuation_decay == 0 else exposed.sum(axis=1)
    normalized = raw[:, :, :, :3].copy()
    normalized[:, :, :, 0] += raw[:, :, :, 3]
    normalized /= f.SCALES
    terminal_loss = np.einsum("ztaq,zq->zta", normalized, f.weights_of(base_cfg)[idx[:, 5]])
    features = raw
    p_social = 0.5 + cfg.social_query_reliability * (0.1 + 0.4 * social[:, 0] + 0.4 * social[:, 1] - 0.5)
    likelihood = np.column_stack((base.likelihood[base_id, :4], p_social, base.likelihood[base_id, 4:]))
    durations = np.array([1, 1, 2, 2, 1, 1, 1] + [1] * len(cfg.extra_comparisons))
    feasible = np.arange(cfg.deadline + 1)[:, None] + completion <= cfg.deadline
    return SocialModel(
        prior,
        likelihood,
        terminal_loss,
        features,
        feasible,
        durations,
        cfg.query_cost * durations,
        idx,
        actions,
        QUERIES + f.extra_query_names(cfg),
        cfg,
        group,
        group_population,
        completion,
    )
