"""Certificates, guided refinement and the scaled instances."""
import hashlib

import numpy as np
import pytest

from hevi_hitl import certificates as C
from hevi_hitl.flood import FloodConfig, build_flood
from hevi_hitl.model import PortfolioSolver, certificates as prop1_certificates
from hevi_hitl.reduction import aggregate_partition, signature_hierarchy
from hevi_hitl.scaled import (FastSolver, ScaledConfig, aggregate_scaled, build_scaled, evaluate_solver)
from hevi_hitl.social import SocialConfig, build_social_flood


@pytest.fixture(scope="module")
def flood_small():
    model = build_flood(FloodConfig(lead_hours=3, response_reliability=0.6))
    U, _ = C.undominated_endpoints(model)
    return model, U


class RandomPolicy:
    """Deterministic pseudo-random history policy committing only within ``S``."""

    def __init__(self, model, S, seed):
        self.m, self.S, self.seed = model, S, seed
        self.root = (-1,) * len(model.queries)

    def solve(self, h):
        digest = hashlib.sha256(f"{self.seed}:{h}".encode()).digest()
        u = int.from_bytes(digest[:8], "little") / 2 ** 64
        r = int.from_bytes(digest[8:16], "little")
        t = sum(int(d) for d, y in zip(self.m.durations, h) if y != -1)
        free = [q for q, y in enumerate(h) if y == -1 and t + self.m.durations[q] <= self.m.config.deadline]
        if free and u < 0.65:
            return None, ("query", free[r % len(free)])
        acts = np.flatnonzero(self.S[t])
        return None, ("act", int(acts[r % len(acts)]))


def interval(model, reduced, inverse, S):
    """Class-level and per-configuration discrepancy intervals for policies in S."""
    sets, cmax = C.feasible_query_sets(model)
    t = C.bound_terms(C.endpoint_columns(model, S), C.endpoint_columns(reduced, S), inverse, cmax)
    ups = C.divergence(np.abs(model.likelihood - reduced.likelihood[inverse]), sets)
    tv_z, tv_c = C.answer_vector_tv(model.likelihood, model.prior, inverse, reduced.likelihood)
    T = np.minimum(ups, tv_z)
    nc = len(reduced.prior)
    tv_star = np.minimum(tv_c, np.bincount(inverse, weights=model.prior * T, minlength=nc) / reduced.prior)
    ct = C.class_terms(t, model.prior, inverse, T, nc)
    div = reduced.prior @ (ct.span * tv_star)
    E = model.prior @ (t.eps + t.span[inverse] * ups)
    return ct.lower.sum() - div, ct.upper.sum() + div, model.prior @ t.mid, E


def test_divergence_matches_recursion():
    for model in (build_flood(FloodConfig(lead_hours=4)), build_social_flood(SocialConfig(lead_hours=4))):
        sets, _ = C.feasible_query_sets(model)
        rng = np.random.default_rng(0)
        tv = rng.uniform(0, 0.4, size=model.likelihood.shape)
        assert np.allclose(C.divergence(tv, sets), C.divergence_recursion(model, tv), atol=1e-14)


def test_answer_vector_tv_streaming_consistent():
    model = build_flood(FloodConfig())
    inverse = signature_hierarchy(model, [24])[24]
    reduced = aggregate_partition(model, inverse)
    a = C.answer_vector_tv(model.likelihood, model.prior, inverse, reduced.likelihood, chunk=7)
    b = C.answer_vector_tv(model.likelihood, model.prior, inverse, reduced.likelihood, chunk=10_000)
    assert np.allclose(a[0], b[0], atol=1e-14) and np.allclose(a[1], b[1], atol=1e-14)


def test_U_covers_every_belief(flood_small):
    model, U = flood_small
    flat = C.loss2d(model)
    rng = np.random.default_rng(1)
    mask = C.endpoint_mask(model)
    for t in np.flatnonzero(mask.any(axis=1)):
        for _ in range(200):
            b = rng.dirichlet(np.full(len(model.prior), 0.05))
            vals = b @ flat
            assert vals[U[t]].min() <= vals[mask[t]].min() + 1e-14


def test_random_policies_within_intervals(flood_small):
    model, U = flood_small
    for k, inverse in signature_hierarchy(model, [1, 12, 48]).items():
        reduced = aggregate_partition(model, inverse)
        lower, upper, offset, E = interval(model, reduced, inverse, U)
        for seed in range(25):
            policy = RandomPolicy(model, U, seed)
            delta0 = PortfolioSolver(model).evaluate(policy)[0] - PortfolioSolver(reduced).evaluate(policy)[0]
            assert lower - 1e-12 <= delta0 <= upper + 1e-12
            assert abs(delta0 - offset) <= E + 1e-12


def test_bounds_dominate_regret_and_refined_tighter():
    for cfg in (FloodConfig(), FloodConfig(lead_hours=4, response_reliability=0.6, query_cost=0.002)):
        model = build_flood(cfg)
        optimum = PortfolioSolver(model)
        best = optimum.solve()[0]
        U, _ = C.undominated_endpoints(model)
        lower = C.perfect_information_bound(model)
        for method in ("signature", "guided"):
            if method == "signature":
                hierarchy = signature_hierarchy(model, [1, 12, 48, 96])
            else:
                hierarchy = C.guided_hierarchy(model, [1, 12, 48, 96], U=U)
            for k, inverse in hierarchy.items():
                reduced = aggregate_partition(model, inverse)
                policy = PortfolioSolver(reduced)
                policy.solve()
                value = PortfolioSolver(model).evaluate(policy)[0]
                surrogate = PortfolioSolver(reduced).evaluate(policy)[0]
                U_t, _ = C.undominated_endpoints(reduced)
                assert not (C.policy_endpoints(model, policy) & ~U_t).any()
                cert = C.refined_certificate(model, reduced, inverse, value, surrogate, U, U_t, lower_bound=lower)
                prop1 = prop1_certificates(model, reduced, inverse, policy)
                regret = value - best
                for key in ("prop1_uniform", "prop1_postsolve", "U_uniform", "U_postsolve", "U_lb",
                            "refined_uniform", "refined_postsolve", "refined_lb"):
                    assert cert[key] >= regret - 1e-10, key
                assert abs(cert["prop1_postsolve"] - prop1["postsolve_regret_bound"]) < 1e-12
                assert cert["U_postsolve"] <= cert["prop1_postsolve"] + 1e-12
                assert cert["refined_postsolve"] <= cert["U_postsolve"] + 1e-12
                assert cert["refined_lb"] <= cert["refined_postsolve"] + 1e-12
                assert cert["refined_uniform"] <= cert["U_uniform"] + 1e-12 <= cert["prop1_uniform"] + 2e-12


def test_U_contains_optimal_endpoints(flood_small):
    """Every endpoint of the DP-optimal policy is in U or ties with a U endpoint."""
    for model, U in (flood_small, (build_flood(FloodConfig()), None)):
        if U is None:
            U, _ = C.undominated_endpoints(model)
        solver = PortfolioSolver(model)
        solver.solve()
        flat = C.loss2d(model)
        for h in solver.memo:
            kind, j = solver.solve(h)[1]
            if kind == "act":
                t = solver.time(h)
                vals = solver.mass(h) @ flat
                assert U[t, j] or vals[j] <= vals[U[t]].min() + 1e-15


def test_scaled_nests_social_model():
    scaled = build_scaled(ScaledConfig(), with_features=True)
    social = build_social_flood(SocialConfig())
    assert np.array_equal(scaled.prior, social.prior)
    assert np.array_equal(scaled.likelihood, social.likelihood)
    assert np.array_equal(scaled.loss2d, social.loss[:, 0, :])
    assert np.array_equal(scaled.features[:, 0], social.features[:, 0])
    assert np.array_equal(scaled.feasible, social.feasible)
    assert np.array_equal(scaled.indices, social.indices)
    assert scaled.actions == social.actions and scaled.queries == social.queries
    assert np.array_equal(scaled.costs, social.costs) and np.array_equal(scaled.durations, social.durations)


def test_fast_solver_matches_portfolio_solver():
    for model in (build_flood(FloodConfig(lead_hours=4)), build_social_flood(SocialConfig())):
        for method in ("optimal", "one_step"):
            reference = PortfolioSolver(model, method=method)
            ref_value = PortfolioSolver(model).evaluate(reference)[0]
            fast = FastSolver(model, method=method)
            assert abs(evaluate_solver(model, fast)["value"] - ref_value) < 1e-12
    model = build_scaled(ScaledConfig(5, 2, 4, 1, False, lead_hours=4), with_features=True)
    inverse = signature_hierarchy(model, [64])[64]
    reduced = aggregate_scaled(model, inverse)
    reference = aggregate_partition(model, inverse)
    assert np.allclose(reduced.loss2d, reference.loss[:, 0], atol=1e-14)
    assert np.allclose(reduced.likelihood, reference.likelihood, atol=1e-14)
    assert abs(FastSolver(reduced).solve()[0] - PortfolioSolver(reference).solve()[0]) < 1e-12
