"""Identities of the value-of-information analysis."""
from hevi_hitl.flood import FloodConfig, build_flood
from hevi_hitl.insight import (decision_value, misspecification, per_state_cost, plan, revealed_value,
                               stress_policy, physical_axes, preference_axis)
from hevi_hitl.model import PortfolioSolver


def small_model():
    return build_flood(FloodConfig(lead_hours=3, response_reliability=0.9, query_cost=0.00025))


def test_voi_ordering_and_pi_lower_bound():
    m = small_model()
    out = decision_value(m)
    v = {k: p["score"] for k, p in out["policies"].items()}
    for name in ("immediate", "physical_only", "preference_only", "one_step"):
        assert v["joint"] <= v[name] + 1e-12
    pi = out["perfect_information"]["score"]
    assert pi <= v["joint"] + 1e-12
    d = out["decomposition"]
    assert d["voi_joint"] >= -1e-12 and d["evpi"] >= d["voi_joint"] - 1e-12
    assert abs(d["interaction"] - (d["voi_joint"] - d["voi_physical"] - d["voi_preference"])) < 1e-15
    # Revealing more axes can only help; revealing everything equals PI.
    assert revealed_value(m, (preference_axis(m),)) >= pi - 1e-12
    assert revealed_value(m, physical_axes(m)) >= pi - 1e-12
    assert revealed_value(m, ()) >= revealed_value(m, physical_axes(m)) - 1e-12


def test_frozen_policy_cost_matches_solver_value():
    m = small_model()
    solver = PortfolioSolver(m)
    value = solver.solve()[0]
    assert abs(float(m.prior @ per_state_cost(m, plan(m, "joint"))) - value) < 1e-12


def test_stress_tilt_identity():
    m = small_model()
    for lam in (0.0, 1.0, 8.0):
        _, audit = stress_policy(m, lam)
        assert audit["identity_error_solver"] < 1e-12
        assert audit["identity_error_policy"] < 1e-12
    # lambda = 0 reproduces the ordinary optimum.
    _, audit0 = stress_policy(m, 0.0)
    assert abs(audit0["tilted_value"] - PortfolioSolver(m).solve()[0]) < 1e-14


def test_correct_specification_has_zero_regret():
    m = small_model()
    out = misspecification(m, m)
    assert abs(out["regret"]) < 1e-12
    assert out["versus_immediate"] <= 1e-12
