"""Decision display of experiment 07."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hevi_hitl import operational as op
from hevi_hitl.model import PortfolioSolver

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def display():
    spec = importlib.util.spec_from_file_location("realtime_display", ROOT / "experiments" / "07_realtime_display.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model, _, policy, evaluation, _, immediate = module.solve()
    return module, module.compute(), model, policy, evaluation, immediate


def test_posterior_weights_and_preference_answers(display):
    module, results, model, *_ = display
    physical = model.indices[:, [0, 1, 2, 3, 4, 6, 7]]          # every coordinate except preference type
    keys = np.unique(physical, axis=0, return_inverse=True)[1].ravel()
    for path in results["paths"].values():
        answers = []
        for seg, ev in zip(path["segments"], path["events"] + [None]):
            b = module.posterior(model, answers)
            assert np.all(b >= 0) and abs(b.sum() - 1) < 1e-12 and abs(seg["belief_sum"] - 1) < 1e-12
            if ev is None:
                break
            q = model.queries.index(ev["query"])
            after = module.posterior(model, answers + [(q, ev["answer"])])
            if ev["kind"] == "preference":
                # A comparison answer leaves the physical/social marginal, hence P(high), unchanged.
                assert np.allclose(np.bincount(keys, after), np.bincount(keys, b), rtol=0, atol=1e-12)
                assert abs(ev["p_high_after"] - ev["p_high_before"]) < 1e-12
            answers.append((q, ev["answer"]))
    assert module.weighted_quantile(np.array([3.0, 1.0, 2.0]), np.array([0.2, 0.5, 0.3]), 0.5) == 1.0
    assert module.weighted_quantile(np.array([3.0, 1.0, 2.0]), np.array([0.2, 0.5, 0.3]), 0.9) == 3.0


def test_adaptive_equals_immediate_commitment_at_time_zero(display):
    _, results, model, _, _, immediate = display
    assert PortfolioSolver(model, method="immediate").solve()[1] == ("act", immediate)
    for path in results["paths"].values():
        first = path["segments"][0]["options"]
        assert first["adaptive"]["portfolio_index"] == first["immediate"]["portfolio_index"] == immediate
        for metric, values in first["adaptive"].items():
            if isinstance(values, dict):
                assert values == first["immediate"][metric]


def test_committed_means_match_forward_evaluation(display):
    _, results, model, policy, _, _ = display
    nodes = {n["node_id"]: n for n in op.evaluate(model, policy, include_trace=True)["trace"]}
    for path in results["paths"].values():
        leaf = nodes[path["leaf_node_id"]]
        committed = path["segments"][-1]
        assert committed["committed"] and committed["start_hours"] == leaf["time_hours"]
        adaptive = committed["options"]["adaptive"]
        assert adaptive["portfolio_index"] == leaf["index"]
        damage, displacement, exposed, spend = leaf["physical_features"]
        assert abs(adaptive["damage_plus_expenditure_meur"]["mean"] - (damage + spend) / 1e6) < 1e-9
        assert abs(adaptive["exposed_people"]["mean"] - exposed) < 1e-9
        assert abs(adaptive["displacement_thousand_person_days"]["mean"] - displacement / 1e3) < 1e-9
        assert abs(adaptive["constrained_exposed_percent"]["mean"] - 100 * leaf["group_features"][1][3]) < 1e-9
        for option in committed["options"].values():
            for values in option.values():
                if isinstance(values, dict):
                    assert values["p10"] <= values["p90"]
