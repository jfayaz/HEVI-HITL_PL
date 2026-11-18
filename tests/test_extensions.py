"""Moment recursion, 2-means split, access variants, timed model and exposure cap."""
import numpy as np
import pytest

from hevi_hitl import access as ac
from hevi_hitl import certificates as C
from hevi_hitl import forecast as fc
from hevi_hitl import social as so
from hevi_hitl.flood import FloodConfig, build_flood
from hevi_hitl.moments import MomentSolver
from hevi_hitl.reduction import aggregate_partition
from hevi_hitl.scaled import FastSolver, ScaledConfig, build_scaled, evaluate


@pytest.mark.parametrize("cfg", [FloodConfig(lead_hours=4), FloodConfig(lead_hours=6, response_reliability=0.6)])
def test_moment_recursion_matches_history_recursion(cfg):
    model = build_flood(cfg)
    history = FastSolver(model)
    value = history.solve()[0]
    moment = MomentSolver(model)
    assert abs(moment.solve()[0] - value) < 1e-12
    assert all(moment.decision(h) == history.memo[h][1] for h in history.memo)
    assert abs(evaluate(model, lambda h, mass: moment.decision(h))["value"] - value) < 1e-12


def test_moment_recursion_social_and_column_blocks():
    model = so.build_social_flood(so.SocialConfig(lead_hours=6, mechanism_strength=0.5))
    value = FastSolver(model).solve()[0]
    for block in (None, 7):
        assert abs(MomentSolver(model, column_block=block).solve()[0] - value) < 1e-12


def test_moment_recursion_on_reduced_model():
    model = build_flood(FloodConfig(lead_hours=6))
    inverse = C.guided_hierarchy(model, [24])[24]
    reduced = aggregate_partition(model, inverse)
    assert abs(MomentSolver(reduced).solve()[0] - FastSolver(reduced).solve()[0]) < 1e-12


def test_two_means_split_is_nested_and_valid():
    model = build_flood(FloodConfig(lead_hours=6))
    refiner = C.GuidedRefiner(model, split="2means")
    coarse = refiner.refine_to(12)
    fine = refiner.refine_to(48)
    assert coarse.max() + 1 == 12 and fine.max() + 1 == 48
    # nested: every fine class lies inside one coarse class
    for c in range(48):
        assert len(np.unique(coarse[fine == c])) == 1
    median = C.GuidedRefiner(model, split="median").refine_to(48)
    assert median.max() + 1 == 48


def test_access_common_structure_reproduces_social_model():
    cfg = so.SocialConfig(lead_hours=4, mechanism_strength=1.0)
    published = so.build_social_flood(cfg)
    common = ac.build_access(cfg, "common")
    assert np.array_equal(common.loss2d, published.loss[:, 0, :])
    assert np.array_equal(common.likelihood, published.likelihood)
    assert np.array_equal(common.group, published.group_features)


def test_access_district_structure_with_equal_access():
    cfg = so.SocialConfig(lead_hours=4, mechanism_strength=0.5)
    published = so.build_social_flood(cfg)
    district = ac.build_access(cfg, "district")
    equal = (district.indices[:, 6] == district.indices[:, 7]) & (district.indices[:, 7] == district.indices[:, 8])
    key = {tuple(r[:6]) + (r[6], r[7]): i for i, r in enumerate(published.indices)}
    rows = [key[tuple(r[:6]) + (r[6], r[9])] for r in district.indices[equal]]
    assert np.allclose(district.loss2d[equal], published.loss[rows, 0, :], rtol=0, atol=1e-15)
    assert abs(district.prior.sum() - 1) < 1e-12


def test_scaled_vulnerability_levels_default_reproduces():
    a = build_scaled(ScaledConfig(3, 2, 4, 0, False))
    b = build_scaled(ScaledConfig(3, 2, 4, 0, False, vulnerability_levels=2))
    assert np.array_equal(a.loss2d, b.loss2d) and np.array_equal(a.likelihood, b.likelihood)
    c = build_scaled(ScaledConfig(3, 2, 4, 0, False, vulnerability_levels=3))
    assert len(c.prior) == ScaledConfig(3, 2, 4, 0, False, vulnerability_levels=3).states
    assert abs(c.prior.sum() - 1) < 1e-12


def test_timed_model_with_published_schedule_reproduces_social_value():
    cfg = so.SocialConfig(lead_hours=4, mechanism_strength=0.5)
    model = fc.build_timed(cfg, [0.9])
    value = fc.TimedSolver(model).solve()[0]
    assert abs(value - FastSolver(so.build_social_flood(cfg)).solve()[0]) < 1e-12


def test_class_constants_hold_for_every_forecast():
    """The class-prior mean of the class constants is the pre-solve certificate, for every severity forecast."""
    from dataclasses import replace
    model = build_flood(FloodConfig(lead_hours=6))
    inverse = C.marginal_partition(model.indices, (1,))     # the footprint is averaged out
    reduced = aggregate_partition(model, inverse)
    U, _ = C.undominated_endpoints(model)
    U_t, _ = C.undominated_endpoints(reduced)
    chi = C.class_constants(model, reduced, inverse, U, U_t)

    def regret_of(m):
        solver = MomentSolver(replace(aggregate_partition(m, inverse), feasible=U_t))
        surrogate = solver.solve()[0]
        value = evaluate(m, lambda h, mass: solver.decision(h))["value"]
        return value - MomentSolver(m).solve()[0], value, surrogate

    regret, value, surrogate = regret_of(model)
    cert = C.refined_certificate(model, reduced, inverse, value, surrogate, U, U_t)
    assert abs(float(reduced.prior @ chi) - cert["refined_uniform"]) < 1e-10
    assert -1e-10 <= regret <= float(reduced.prior @ chi) + 1e-10

    severity = model.indices[:, 0]
    p_old = np.array([model.prior[severity == s].sum() for s in range(3)])
    for p_new in ([0.6, 0.3, 0.1], [0.1, 0.2, 0.7]):
        shifted = replace(model, prior=model.prior * (np.array(p_new) / p_old)[severity])
        reduced_p = aggregate_partition(shifted, inverse)
        assert np.allclose(C.class_constants(shifted, reduced_p, inverse, U, U_t), chi, atol=1e-12)
        assert -1e-10 <= regret_of(shifted)[0] <= float(reduced_p.prior @ chi) + 1e-10


def test_regret_of_separable_coordinate_equals_value_of_its_query():
    """Averaging out an independent coordinate observed only through its own query costs the value of that query."""
    from dataclasses import replace
    model = build_flood(FloodConfig(lead_hours=6, response_reliability=0.9))
    optimum = MomentSolver(model).solve()[0]
    for axis, query in ((2, 1), (3, 2), (4, 3)):            # occupancy, physical vulnerability, restoration
        inverse = C.marginal_partition(model.indices, (axis,))
        solver = MomentSolver(aggregate_partition(model, inverse))
        solver.solve()
        regret = evaluate(model, lambda h, mass: solver.decision(h))["value"] - optimum
        keep = [q for q in range(len(model.queries)) if q != query]
        fewer = replace(model, likelihood=model.likelihood[:, keep], durations=model.durations[keep],
                        costs=model.costs[keep], queries=tuple(model.queries[q] for q in keep))
        assert abs(regret - (MomentSolver(fewer).solve()[0] - optimum)) < 1e-12


def test_query_relaxation_bound_on_random_models():
    """The query relaxation bounds the optimum from below."""
    from types import SimpleNamespace

    from hevi_hitl.model import Portfolio, PortfolioSolver
    rng = np.random.default_rng(2041)
    slack = []
    for trial in range(240):
        n, q, a = int(rng.integers(2, 9)), int(rng.integers(1, 4)), int(rng.integers(2, 5))
        deadline = int(rng.integers(2, 6))
        durations = rng.integers(1, 3, size=q)
        completion = rng.integers(0, deadline + 1, size=a)
        completion[0] = 0
        feasible = np.arange(deadline + 1)[:, None] + completion[None, :] <= deadline
        classes = int(rng.integers(1, n + 1))
        inverse = np.concatenate((np.arange(classes), rng.integers(0, classes, size=n - classes)))
        rng.shuffle(inverse)
        lik = rng.uniform(0, 1, size=(n, q))
        shared = rng.random(q) < 0.4                      # some queries answer alike within classes
        lik[:, shared] = rng.uniform(0, 1, size=(classes, q))[inverse][:, shared]
        lik = np.where(rng.random(lik.shape) < 0.25, np.round(lik), lik)   # zero-probability answers
        if shared.any():                                  # rounding must not separate class members
            first = np.array([np.flatnonzero(inverse == c)[0] for c in range(classes)])
            lik[:, shared] = lik[first][inverse][:, shared]
        if trial % 2:                                     # scores that change with the commitment time
            loss = rng.uniform(0, 1, size=(n, deadline + 1, a))
        else:
            loss = np.repeat(rng.uniform(0, 1, size=(n, 1, a)), deadline + 1, axis=1)
        model = Portfolio(rng.dirichlet(np.ones(n)), lik, loss, np.zeros(loss.shape + (4,)), feasible, durations,
                          rng.uniform(0, 0.05, size=q) * durations, np.arange(n)[:, None], tuple(range(a)),
                          tuple(range(q)), SimpleNamespace(deadline=deadline))
        optimum = PortfolioSolver(model).solve()[0]
        bound = C.query_relaxation_bound(model, inverse)
        assert C.perfect_information_bound(model) - 1e-10 <= bound <= optimum + 1e-10
        slack.append(optimum - bound)
        if not C.differing_queries(lik, inverse):         # no query separates class members: the merge is exact
            assert abs(bound - optimum) < 1e-10
    assert max(slack) > 1e-3                              # the bound is not trivially equal to the optimum


def test_relevance_index_bounds_regret_and_gives_free_answer_value():
    """If the merged policy asks none of a coordinate's queries, the relaxation term is the value of their free answers."""
    from dataclasses import replace
    model = build_flood(FloodConfig(lead_hours=6, response_reliability=0.9))
    optimum = MomentSolver(model).solve()[0]
    for axis in range(6):
        inverse = C.marginal_partition(model.indices, (axis,))
        own = C.differing_queries(model.likelihood, inverse)
        solver = MomentSolver(aggregate_partition(model, inverse))
        solver.solve()
        value = evaluate(model, lambda h, mass: solver.decision(h))["value"]
        bound = C.query_relaxation_bound(model, inverse)
        assert bound <= optimum + 1e-12 and value - bound >= value - optimum - 1e-12
        if axis in (2, 3, 4):                             # occupancy, physical vulnerability, restoration
            keep = [q for q in range(len(model.queries)) if q not in own]
            fewer = replace(model, likelihood=model.likelihood[:, keep], durations=model.durations[keep],
                            costs=model.costs[keep], queries=tuple(model.queries[q] for q in keep))
            without = MomentSolver(fewer).solve()[0]
            assert abs(value - without) < 1e-12           # the merged policy is optimal without the own query
            free = 0.0                                    # value without the query after its free answer
            for y in (0, 1):
                law = model.likelihood[:, own[0]] if y else 1 - model.likelihood[:, own[0]]
                free += MomentSolver(replace(fewer, prior=model.prior * law)).solve()[0]
            assert abs(bound - free) < 1e-12


def test_published_damage_function_is_an_option():
    """The JRC curve replaces the benchmark damage curves only when requested."""
    from hevi_hitl import flood as fl

    knots, structure, contents = fl.damage_curves("jrc_europe")
    assert np.allclose(np.interp([0.5, 1.0, 2.5], knots, structure), [0.25, 0.40, 0.675]) and structure is contents
    assert fl.damage_curves()[1] is fl.STRUCTURE_DAMAGE
    with pytest.raises(ValueError):
        fl.damage_curves("unknown")
    base = build_scaled(ScaledConfig(lead_hours=6, mechanism_strength=0.5))
    jrc = build_scaled(ScaledConfig(lead_hours=6, mechanism_strength=0.5, damage_function="jrc_europe"))
    social = so.build_social_flood(so.SocialConfig(lead_hours=6, mechanism_strength=0.5))
    assert np.allclose(base.loss2d, social.loss[:, 0, :], rtol=0, atol=1e-12)      # the default is the benchmark
    assert np.allclose(base.likelihood, jrc.likelihood) and not np.allclose(base.loss2d, jrc.loss2d)


def test_exposure_cap():
    """Experiment 35: the capped policy satisfies the cap and loses value only where the cap binds."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "exposure_cap", Path(__file__).resolve().parents[1] / "experiments" / "35_exposure_cap.py")
    exp35 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp35)
    cfg = exp35.social_configs()[10]
    for variant in ("benchmark", "shelter 60%", "overtopping 1.4 m"):
        row = exp35.instance((variant, 10, cfg))
        assert row["exposed_capped"] <= row["exposed_immediate"] + 1e-8
        assert -1e-12 <= row["voi_capped"] <= row["voi_capped_mixed"] + 1e-12 <= row["voi_joint"] + 2e-12
        assert row["cap_binds"] == (row["exposed_joint"] > row["exposed_immediate"] + 1e-9)
        if not row["cap_binds"]:
            assert abs(row["voi_capped"] - row["voi_joint"]) < 1e-12 and row["weight"] == 0.0
    benchmark = exp35.instance(("benchmark", 10, cfg))
    assert abs(benchmark["voi_joint"] - 3.4926e-3) < 5e-7 and abs(benchmark["voi_capped"]) < 1e-12

