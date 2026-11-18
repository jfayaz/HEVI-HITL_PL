"""Reference values and invariants of the flood and social benchmarks."""
import numpy as np

from hevi_hitl import flood as f
from hevi_hitl.flood import (BUILDINGS, CONTENTS_VALUE_EUR, FLOOR_HEIGHT_M, STRUCTURE_VALUE_EUR, FloodConfig,
                             _action_data, _states, build_flood)
from hevi_hitl.model import PortfolioSolver, normalized_reference
from hevi_hitl.social import GROUP_SHARES, SocialConfig, build_social_flood


def validate_flood_model(model=None):
    """Physical and accounting invariants of the flood model."""
    model = build_flood() if model is None else model
    idx, prior, depth, occupants = _states(model.config)
    checks = {}
    checks["prior_normalized"] = bool(abs(prior.sum() - 1) < 1e-12)
    checks["finite_nonnegative_consequences"] = bool(
        np.isfinite(model.features).all() and (model.features >= 0).all()
    )
    checks["valid_response_probabilities"] = bool(((model.likelihood > 0) & (model.likelihood < 1)).all())
    checks["public_feasibility_nested"] = bool(np.all(model.feasible[1:] <= model.feasible[:-1]))
    checks["no_action_always_available"] = bool(model.feasible[:, 0].all())
    checks["property_loss_bounded_by_inventory"] = bool(
        np.max(model.features[:, :, :, 0]) <= np.sum(BUILDINGS * (STRUCTURE_VALUE_EUR + CONTENTS_VALUE_EUR))
    )
    # A separate hand calculation for state (0,0,0,0,0,0), with no action:
    # depths .47,.24,.13 m; only district 1 is wet above its .35 m floor.
    known = int(np.flatnonzero(np.all(idx == 0, axis=1))[0])
    hand_damage = 140 * (190000 * 0.85 * 0.05 * 0.12 / 0.25 + 40000 * 0.85 * 0.20 * 0.12 / 0.25)
    hand_displacement = (
        280 * ((0.12 - 0.05) / 0.45) * 0.75 * (2 + 50 * (0.85 * 0.05 * 0.12 / 0.25) + 10 * 0.12)
    )
    checks["independent_single_state_consequence_calculation"] = bool(
        np.allclose(
            model.features[known, 0, 0], [hand_damage, hand_displacement, 280.0, 0.0], rtol=0, atol=1e-8
        )
    )
    checks["preference_likelihood_depends_only_on_preference_state"] = all(
        np.all(np.ptp(model.likelihood[idx[:, 5] == j, 4:], axis=0) < 1e-14) for j in range(4)
    )
    checks["physical_likelihood_independent_of_preference_state"] = bool(
        np.all(np.ptp(model.likelihood[:, :4].reshape(-1, 4, 4), axis=1) < 1e-14)
    )
    actions = {a: j for j, a in enumerate(model.actions)}
    evacuation_error, conservation_error, barrier_excess, contents_excess = 0.0, 0.0, 0.0, 0.0
    displacement_error, contents_displacement_error = 0.0, 0.0
    for a, j in actions.items():
        for k in range(3):
            a0 = [list(x) for x in a]
            a1 = [list(x) for x in a]
            a0[k][1] = 0
            a1[k][1] = 1
            key0, key1 = tuple(map(tuple, a0)), tuple(map(tuple, a1))
            if key0 in actions and key1 in actions:
                f0 = model.features[:, 0, actions[key0]]
                f1 = model.features[:, 0, actions[key1]]
                evacuation_error = max(evacuation_error, float(np.max(np.abs(f0[:, 0] - f1[:, 0]))))
                local_depth = np.where(
                    (a[k][0] == 1) & (depth[:, k] <= 1.2), np.maximum(0, depth[:, k] - 0.65), depth[:, k]
                )
                local_floor = FLOOR_HEIGHT_M[k] + np.array([0.1, -0.1])[idx[:, 3]]
                local_uninhabitable = np.clip((np.maximum(0, local_depth - local_floor) - 0.05) / 0.45, 0, 1)
                expected_precaution = 0.75 * 0.9 * occupants[:, k] * (1 - local_uninhabitable)
                displacement_error = max(
                    displacement_error, float(np.max(np.abs(f1[:, 1] - f0[:, 1] - expected_precaution)))
                )
                assert np.all(f1[:, 2] <= f0[:, 2] + 1e-10)
                remaining = occupants[:, k] * (1 - 0.9 * a1[k][1])
                moved = occupants[:, k] * 0.9 * a1[k][1]
                conservation_error = max(
                    conservation_error, float(np.max(abs(remaining + moved - occupants[:, k])))
                )
            if a[k][0] == 0:
                for measure in (1, 2):
                    altered = list(a)
                    altered[k] = (measure, a[k][1])
                    altered = tuple(altered)
                    if altered in actions:
                        new = model.features[:, 0, actions[altered]]
                        old = model.features[:, 0, j]
                        excess = float(np.max(new[:, 0] - old[:, 0]))
                        if measure == 1:
                            barrier_excess = max(barrier_excess, excess)
                            assert np.all(new[:, 1:3] <= old[:, 1:3] + 1e-8)
                        else:
                            contents_excess = max(contents_excess, excess)
                            contents_displacement_error = max(
                                contents_displacement_error, float(np.max(abs(new[:, 1:3] - old[:, 1:3])))
                            )
    checks.update(
        evacuation_property_damage_unchanged=evacuation_error < 1e-8,
        evacuation_precaution_accounted_once=displacement_error < 1e-8,
        occupants_conserved=conservation_error < 1e-10,
        barrier_weakly_reduces_damage=barrier_excess < 1e-8,
        contents_weakly_reduces_damage=contents_excess < 1e-8,
        contents_leaves_displacement_and_people_unchanged=contents_displacement_error < 1e-8,
    )
    # No hidden-state feasibility: every entry depends only on the public action
    # inventory and commitment time, with shelter reserved at maximum occupancy.
    _, _, completion = _action_data()
    checks["implementation_finishes_by_public_deadline"] = bool(
        np.array_equal(
            model.feasible,
            np.arange(model.config.deadline + 1)[:, None] + completion <= model.config.deadline,
        )
    )
    if not all(checks.values()):
        raise AssertionError(checks)
    return {
        "checks": checks,
        "state_count": len(model.prior),
        "action_count": len(model.actions),
        "query_count": len(model.queries),
        "maximum_evacuation_damage_difference_eur": evacuation_error,
        "maximum_displacement_accounting_difference_person_days": displacement_error,
        "maximum_population_conservation_error": conservation_error,
    }


def validate_social(model=None):
    """Accounting and monotonicity invariants of the social model."""
    model = build_social_flood() if model is None else model
    checks = {
        "prior_normalized": abs(model.prior.sum() - 1) < 1e-12,
        "group_sum_matches_aggregate": np.allclose(
            model.group_features[:, :, :, :2].sum(axis=2), model.features[:, 0, :, 1:3], rtol=1e-12, atol=1e-8
        ),
        "finite_nonnegative_outcomes": np.isfinite(model.group_features).all()
        and (model.group_features >= 0).all(),
        "exposure_fraction_bounded": np.max(model.group_features[:, :, :, 3]) <= 1 + 1e-12,
        "public_feasibility_nested": np.all(model.feasible[1:] <= model.feasible[:-1]),
        "no_action_always_available": model.feasible[:, 0].all(),
    }
    ids = model.indices
    actions = {a: j for j, a in enumerate(model.actions)}
    physical_same = 0.0
    access_excess = 0.0
    independent_change = 0.0
    for a, j in actions.items():
        for k in range(3):
            if a[k][1] == 1:
                other = list(a)
                other[k] = (a[k][0], 2)
                other = tuple(other)
                if other in actions:
                    jj = actions[other]
                    physical_same = max(
                        physical_same,
                        float(np.max(abs(model.features[:, 0, j, 0] - model.features[:, 0, jj, 0]))),
                    )
                    access_excess = max(
                        access_excess,
                        float(np.max(model.group_features[:, jj, 1, 1] - model.group_features[:, j, 1, 1])),
                    )
                    independent_change = max(
                        independent_change,
                        float(np.max(abs(model.group_features[:, jj, 0] - model.group_features[:, j, 0]))),
                    )
    checks.update(
        assistance_preserves_property_damage=physical_same < 1e-8,
        assistance_weakly_reduces_constrained_exposure=access_excess < 1e-8,
        assistance_preserves_independent_group_outcomes=independent_change < 1e-8,
    )
    # Consecutive rows differ only by recovery access; only constrained repair absence changes.
    g = model.group_features.reshape(192, 2, 2, len(model.actions), 2, 4)
    checks["recovery_delay_monotone"] = np.all(g[:, :, 1, :, 1, 0] >= g[:, :, 0, :, 1, 0] - 1e-8)
    checks["recovery_does_not_change_exposure"] = np.allclose(
        g[:, :, 1, :, :, 1], g[:, :, 0, :, :, 1], atol=1e-10
    )
    checks["social_axes_do_not_change_property_damage"] = np.all(
        np.ptp(model.features[:, 0, :, 0].reshape(192, 4, -1), axis=1) < 1e-8
    )
    checks["worse_access_weakly_increases_exposure"] = np.all(
        g[:, 1, :, :, 1, 1] >= g[:, 0, :, :, 1, 1] - 1e-8
    )
    expected_population = f.BASE_OCCUPANTS.sum() * np.array([0.8, 1.2])[ids[:, 2]]
    checks["population_partition_conserved"] = np.allclose(
        model.group_population.sum(axis=1), expected_population, atol=1e-10
    )
    # Separate arithmetic for shallow state and no action: 0.12m internal depth in district1.
    pos = int(np.flatnonzero(np.all(ids == 0, axis=1))[0])
    s = model.config.mechanism_strength
    u = (0.12 - 0.05) / 0.45
    technical = 0.75 * (2 + 50 * (0.85 * 0.05 * 0.12 / 0.25) + 10 * 0.12)
    hand = np.array(
        [[0.7 * 280 * u * technical, 0.7 * 280], [0.3 * 280 * u * (technical + s * 2), 0.3 * 280]]
    )
    checks["independent_shallow_state_hand_check"] = np.allclose(
        model.group_features[pos, 0, :, :2], hand, rtol=0, atol=1e-8
    )
    # Standard evacuation in district1: precaution applies only to habitable homes.
    key = ((0, 1), (0, 0), (0, 0))
    j = actions[key]
    rates = np.array([0.9, 0.9 - s * 0.15])
    precaution = 280 * GROUP_SHARES * rates * (1 - u) * 0.75
    checks["disjoint_precaution_accounting"] = np.allclose(
        model.group_features[pos, j, :, 0] - model.group_features[pos, 0, :, 0], precaution, rtol=0, atol=1e-8
    )
    checks = {k: bool(v) for k, v in checks.items()}
    if not all(checks.values()):
        raise AssertionError(checks)
    return dict(
        checks=checks, states=len(model.prior), actions=len(model.actions), queries=len(model.queries)
    )


def neutral_check():
    """Zero mechanism strength reproduces the flood model."""
    cfg = SocialConfig(mechanism_strength=0.0, preference_prior="diffuse")
    model = build_social_flood(cfg)
    base = f.build_flood(
        f.FloodConfig(
            lead_hours=6,
            response_reliability=0.9,
            query_cost=cfg.query_cost,
            hazard_prior="elevated",
            preference_prior="diffuse",
        )
    )
    action_ids = [model.actions.index(a) for a in base.actions]
    feature_error = float(np.max(abs(model.features[::4, :, action_ids] - base.features)))
    score_error = float(np.max(abs(model.loss[::4, :, action_ids] - base.loss)))
    social_value = PortfolioSolver(model).solve()[0]
    base_value = PortfolioSolver(base).solve()[0]
    assert feature_error < 1e-7 and score_error < 1e-12 and abs(social_value - base_value) < 1e-10
    return dict(
        maximum_physical_error=feature_error,
        maximum_score_error=score_error,
        optimal_value_error=abs(social_value - base_value),
    )


def test_flood_reference_value():
    m = build_flood(FloodConfig())
    validate_flood_model(m)
    v = PortfolioSolver(m).solve()[0]
    assert abs(v - 0.36364206691311324) < 1e-12
    assert abs(v - normalized_reference(m)) < 1e-12


def test_social_reference_value():
    m = build_social_flood(SocialConfig())
    validate_social(m)
    assert abs(PortfolioSolver(m).solve()[0] - 0.3812798258682664) < 1e-12


def test_neutral_social_limit():
    out = neutral_check()
    assert out["optimal_value_error"] < 1e-10
