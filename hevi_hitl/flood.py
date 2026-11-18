"""Three-district flood-warning benchmark (Section 5.1 and Appendix B)."""

from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.special import expit
from .model import Portfolio


@dataclass(frozen=True)
class FloodConfig:
    """Parameters of one flood-warning instance."""

    lead_hours: int = 6
    response_reliability: float = 0.9
    query_cost: float = 0.00025
    hazard_prior: object = "elevated"  # name or probability triple (low, medium, high)
    preference_prior: object = "diffuse"  # name or probability vector over PREFERENCE_WEIGHTS
    preference_slope: float = 40.0  # logistic slope of the comparison response law
    comparison_reliability: object = None  # None: comparisons use response_reliability
    # assumptions varied in Section 5.7
    exposure_depth: float = 0.3  # ground depth (m) at which remaining occupants count as exposed
    exposure_depth_full: object = None  # None: step at exposure_depth; else linear ramp up to this depth
    barrier_limit: float = 1.2  # unprotected depth (m) above which the barrier is overtopped
    shelter_share: float = 0.9  # share of maximum occupants reserved in public shelter per evacuated district
    evacuation_rate: float = 0.9  # share of occupants that an evacuation moves (compliance)
    evacuation_decay: float = 0.0  # relative loss of evacuation effectiveness per hour of commitment delay
    preference_weights: object = None  # tuple of K weight triples; None: PREFERENCE_WEIGHTS
    extra_comparisons: tuple = ()  # additional comparison profile pairs ((EUR, person-days, exposed) x 2)
    comparison_stakes: float = 1.0  # scales each published profile pair about its midpoint

    @property
    def deadline(self):
        """Public decision deadline in half-hour ticks."""
        return int(2 * self.lead_hours)


BUILDINGS = np.array([140, 100, 60])
STRUCTURE_VALUE_EUR = np.array([190000.0, 170000.0, 150000.0])
CONTENTS_VALUE_EUR = np.array([40000.0, 35000.0, 30000.0])
BASE_OCCUPANTS = np.array([350.0, 250.0, 150.0])
FLOOR_HEIGHT_M = np.array([0.25, 0.35, 0.20])
DEPTH_KNOTS_M = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
STRUCTURE_DAMAGE = np.array([0.0, 0.05, 0.12, 0.28, 0.42, 0.55, 0.75])
CONTENTS_DAMAGE = np.array([0.0, 0.20, 0.40, 0.70, 0.90, 1.0, 1.0])
# JRC depth-damage function for European residential buildings (Huizinga et al., 2017)
JRC_DEPTH_KNOTS_M = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
JRC_EUROPE_RESIDENTIAL = np.array([0.0, 0.25, 0.40, 0.50, 0.60, 0.75, 0.85, 0.95, 1.00])


def damage_curves(name: str = "benchmark"):
    """Depth knots (m) and structure and contents damage fractions of a named depth-damage function."""
    if name == "benchmark":
        return DEPTH_KNOTS_M, STRUCTURE_DAMAGE, CONTENTS_DAMAGE
    if name == "jrc_europe":
        return JRC_DEPTH_KNOTS_M, JRC_EUROPE_RESIDENTIAL, JRC_EUROPE_RESIDENTIAL
    raise ValueError(f"unknown damage function {name!r}")


SCALES = np.array([20_000_000.0, 30_000.0, 1_000.0])
PREFERENCE_WEIGHTS = np.array(
    [[0.75, 0.15, 0.10], [0.15, 0.70, 0.15], [0.10, 0.15, 0.75], [1 / 3, 1 / 3, 1 / 3]]
)
DISTRICT_COST_MULTIPLIER = np.array([1.0, 0.85, 0.70])
PROPERTY_COST_EUR = np.array([0.0, 80000.0, 25000.0])
EVACUATION_COST_EUR = 20000.0
IMPLEMENTATION_TICKS = np.array([0, 6, 4])
EVACUATION_TICKS = 3
BUDGET_EUR = 180000.0
SHELTER_CAPACITY = 700.0
QUERIES = (
    "depth_update",
    "occupancy_check",
    "property_record_review",
    "restoration_resource_check",
    "economic_vs_exposure",
    "displacement_vs_exposure",
)
DURATIONS = np.array([1, 1, 2, 2, 1, 1])
# comparison profiles A and B: EUR, person-days, exposed people (answer 1 prefers B)
PREFERENCE_PAIRS = np.array(
    [
        [[3_000_000.0, 6000.0, 180.0], [3_800_000.0, 6000.0, 80.0]],
        [[3_500_000.0, 4000.0, 150.0], [3_500_000.0, 10000.0, 50.0]],
    ]
)
PROPERTY_LABELS = ("none", "temporary_barrier", "contents_protection")
# the 32 designed instances
GRID = {
    "query_cost": [0.00025, 0.002],
    "response_reliability": [0.6, 0.9],
    "lead_hours": [4, 6],
    "hazard_prior": ["mild", "elevated"],
    "preference_prior": ["diffuse", "economic_concentrated"],
}


HAZARD_PRIORS = {"mild": np.array([0.55, 0.35, 0.10]), "elevated": np.array([0.15, 0.35, 0.50])}
PREFERENCE_PRIORS = {"diffuse": np.full(4, 0.25), "economic_concentrated": np.array([0.7, 0.1, 0.1, 0.1])}


def _named(value, table, size):
    """Return a named prior or validate a supplied probability vector."""
    if isinstance(value, str) and value == "diffuse":
        p = np.full(size, 1.0 / size)
    else:
        p = table[value] if isinstance(value, str) else np.asarray(value, dtype=float)
    if p.shape != (size,) or np.any(p < 0) or abs(p.sum() - 1) > 1e-9:
        raise ValueError("Invalid prior vector")
    return p / p.sum()


def weights_of(cfg) -> np.ndarray:
    """Preference-type weight matrix (K x 3) of a configuration."""
    return PREFERENCE_WEIGHTS if cfg.preference_weights is None else np.asarray(cfg.preference_weights, dtype=float)


def comparison_pairs(cfg) -> np.ndarray:
    """Comparison profile pairs: the two published pairs plus any configured extras."""
    pairs = PREFERENCE_PAIRS
    if cfg.comparison_stakes != 1.0:
        mid = pairs.mean(axis=1, keepdims=True)
        pairs = mid + cfg.comparison_stakes * (pairs - mid)
        if np.any(pairs < 0):
            raise ValueError("comparison stakes make a profile negative")
    if not cfg.extra_comparisons:
        return pairs
    return np.concatenate((pairs, np.asarray(cfg.extra_comparisons, dtype=float)), axis=0)


def extra_query_names(cfg) -> tuple:
    return tuple(f"comparison_{k + 3}" for k in range(len(cfg.extra_comparisons)))


def exposure_factor(depth, cfg) -> np.ndarray:
    """Share of remaining occupants counted as exposed at ground depth ``depth``."""
    if cfg.exposure_depth_full is None:
        return (depth >= cfg.exposure_depth).astype(float)
    return np.clip((depth - cfg.exposure_depth) / (cfg.exposure_depth_full - cfg.exposure_depth), 0.0, 1.0)


def evacuation_scale(cfg) -> np.ndarray:
    """Evacuation effectiveness multiplier by commitment tick (half hours)."""
    hours = np.arange(cfg.deadline + 1) / 2.0
    return np.clip(1.0 - cfg.evacuation_decay * hours, 0.0, 1.0)


def _states(cfg):
    """Enumerate the 192 joint latent states; return indices, prior, district depths, occupants."""
    n_types = len(weights_of(cfg))
    idx = np.array(list(product(range(3), range(2), range(2), range(2), range(2), range(n_types))))
    hz = _named(cfg.hazard_prior, HAZARD_PRIORS, 3)
    pref = _named(cfg.preference_prior, PREFERENCE_PRIORS, n_types)
    prior = hz[idx[:, 0]] * 0.5**4 * pref[idx[:, 5]]
    depth = np.array([0.35, 0.85, 1.65])[idx[:, 0], None] * np.array([1.0, 0.8, 0.6])
    depth += (2 * idx[:, 1, None] - 1) * np.array([-0.12, 0.04, 0.08])
    occupants = BASE_OCCUPANTS * np.array([0.8, 1.2])[idx[:, 2, None]]
    return idx, prior, depth, occupants


def _action_data(shelter_share=0.9):
    """Feasible portfolios: (actions, intervention EUR, completion ticks)."""
    # each district takes one property measure and may evacuate
    candidates = tuple(product(tuple(product(range(3), range(2))), repeat=3))
    actions, costs, completion = [], [], []
    for action in candidates:
        a = np.array(action)
        cost = float((PROPERTY_COST_EUR[a[:, 0]] + EVACUATION_COST_EUR * a[:, 1]) @ DISTRICT_COST_MULTIPLIER)
        shelter = float((shelter_share * 1.2 * BASE_OCCUPANTS) @ a[:, 1])
        if (
            cost > BUDGET_EUR + 1e-10
            or np.sum(a[:, 0] == 1) > 1
            or np.sum(a[:, 0] == 2) > 2
            or shelter > SHELTER_CAPACITY
        ):
            continue
        actions.append(action)
        costs.append(cost)
        completion.append(int(max(np.max(IMPLEMENTATION_TICKS[a[:, 0]]), EVACUATION_TICKS * np.max(a[:, 1]))))
    return tuple(actions), np.array(costs), np.array(completion)


def build_flood(cfg=FloodConfig()):
    """Build the finite flood-warning ``Portfolio`` for configuration ``cfg``."""
    if (
        int(cfg.lead_hours) != cfg.lead_hours
        or cfg.lead_hours < 2
        or not 0 <= cfg.response_reliability <= 1
        or (cfg.comparison_reliability is not None and not 0 <= cfg.comparison_reliability <= 1)
        or cfg.query_cost < 0
        or not 0 < cfg.evacuation_rate <= 1
        or cfg.evacuation_decay < 0
        or not 0 < cfg.shelter_share <= 1
        or (cfg.exposure_depth_full is not None and cfg.exposure_depth_full <= cfg.exposure_depth)
    ):
        raise ValueError("Unsupported public lead or invalid response/cost parameter")
    idx, prior, depth, occupants = _states(cfg)
    actions, action_cost, completion = _action_data(cfg.shelter_share)
    weights = weights_of(cfg)
    scale = evacuation_scale(cfg)
    features = np.zeros((len(idx), cfg.deadline + 1, len(actions), 4))
    floor = FLOOR_HEIGHT_M + np.array([0.1, -0.1])[idx[:, 3, None]]
    susceptibility = np.array([0.85, 1.15])[idx[:, 3, None]]
    restoration = np.array([0.75, 1.25])[idx[:, 4, None]]
    for j, action in enumerate(actions):
        a = np.array(action)
        defended = np.where((a[:, 0] == 1) & (depth <= cfg.barrier_limit), np.maximum(0, depth - 0.65), depth)
        internal = np.maximum(0, defended - floor)
        structure_fraction = np.minimum(
            1, susceptibility * np.interp(internal, DEPTH_KNOTS_M, STRUCTURE_DAMAGE)
        )
        contents_fraction = np.minimum(
            1, susceptibility * np.interp(internal, DEPTH_KNOTS_M, CONTENTS_DAMAGE)
        )
        contents_fraction *= np.where(a[:, 0] == 2, 0.4, 1.0)
        damage = (
            BUILDINGS * (STRUCTURE_VALUE_EUR * structure_fraction + CONTENTS_VALUE_EUR * contents_fraction)
        ).sum(axis=1)
        repair_days = np.where(internal > 0, restoration * (2 + 50 * structure_fraction + 10 * internal), 0.0)
        uninhabitable = np.clip((internal - 0.05) / 0.45, 0, 1)
        at_risk = exposure_factor(defended, cfg)
        for ticks in ((slice(None),) if cfg.evacuation_decay == 0 else range(cfg.deadline + 1)):
            factor = 1.0 if cfg.evacuation_decay == 0 else scale[ticks]
            evacuated = occupants * (cfg.evacuation_rate * factor * a[:, 1])
            person_days = (occupants * uninhabitable * repair_days + 0.75 * evacuated * (1 - uninhabitable)).sum(
                axis=1
            )
            exposed = ((occupants - evacuated) * at_risk).sum(axis=1)
            f = np.column_stack((damage, person_days, exposed, np.full(len(idx), action_cost[j])))
            if cfg.evacuation_decay == 0:
                features[:, :, j] = f[:, None, :]
            else:
                features[:, ticks, j] = f
    normalized = features[:, :, :, :3].copy()
    normalized[:, :, :, 0] += features[:, :, :, 3]
    normalized /= SCALES
    loss = np.einsum("ztaq,zq->zta", normalized, weights[idx[:, 5]])
    physical = np.column_stack(
        (
            0.15 + 0.7 * np.clip((depth[:, 0] - 0.23) / 1.54, 0, 1),
            0.15 + 0.7 * idx[:, 2],
            0.15 + 0.7 * idx[:, 3],
            0.15 + 0.7 * idx[:, 4],
        )
    )
    pairs = comparison_pairs(cfg)
    comparison_difference = (pairs[:, 0] - pairs[:, 1]) / SCALES
    preference = expit(cfg.preference_slope * (weights[idx[:, 5]] @ comparison_difference.T))
    rho_comparison = cfg.response_reliability if cfg.comparison_reliability is None else cfg.comparison_reliability
    physical = 0.5 + cfg.response_reliability * (physical - 0.5)
    preference = 0.5 + rho_comparison * (preference - 0.5)
    likelihood = np.column_stack((physical, preference))
    feasible = np.arange(cfg.deadline + 1)[:, None] + completion[None, :] <= cfg.deadline
    durations = np.concatenate((DURATIONS, np.ones(len(cfg.extra_comparisons), dtype=DURATIONS.dtype)))
    return Portfolio(
        prior,
        likelihood,
        loss,
        features,
        feasible,
        durations,
        cfg.query_cost * durations,
        idx,
        actions,
        QUERIES + extra_query_names(cfg),
        cfg,
    )
