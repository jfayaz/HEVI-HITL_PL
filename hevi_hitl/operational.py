"""Frozen policies and their forward evaluation."""

from dataclasses import dataclass
import numpy as np
from . import flood as f
from .social import ASSISTANCE_CAPACITY

DISTRICTS = ("Riverside", "Central", "Upstream")
FEATURES = (
    "property_damage_eur",
    "post_event_displacement_person_days",
    "exposed_people",
    "intervention_eur",
)


@dataclass(frozen=True)
class FrozenPolicy:
    """Immutable map from reachable histories to ('query', q) or ('act', a) decisions."""

    decisions: dict
    root: tuple


def freeze(solver):
    """Freeze the decisions of a solved policy on its reachable histories."""
    pending = [solver.root]
    decisions = {}
    while pending:
        h = pending.pop()
        if h in decisions:
            continue
        if h not in solver.memo:
            raise AssertionError("Policy must be completely solved before freezing")
        kind, j = solver.memo[h][1]
        decisions[h] = (kind, int(j))
        if kind == "query":
            for y in (0, 1):
                hh = list(h)
                hh[j] = y
                pending.append(tuple(hh))
    return FrozenPolicy(decisions, solver.root)


def no_intervention(model):
    """Policy that commits immediately to the no-intervention portfolio."""
    a = model.actions.index(((0, 0),) * 3)
    root = (-1,) * len(model.queries)
    return FrozenPolicy({root: ("act", a)}, root)


def portfolio_info(model, j):
    """Human-readable description, resources and implementation time of action ``j``."""
    a = np.asarray(model.actions[j])
    evac = a[:, 1]
    social = hasattr(model, "group_features")
    completion = int(
        max(f.IMPLEMENTATION_TICKS[a[:, 0]].max(), np.array([0, f.EVACUATION_TICKS, 4])[evac].max())
    )
    shelter = float((getattr(model.config, "shelter_share", 0.9) * 1.2 * f.BASE_OCCUPANTS) @ (evac > 0))
    assistance = float((0.3 * 1.2 * f.BASE_OCCUPANTS) @ (evac == 2))
    expense = float(
        (f.PROPERTY_COST_EUR[a[:, 0]] + f.EVACUATION_COST_EUR * (evac > 0) + 15000 * (evac == 2))
        @ f.DISTRICT_COST_MULTIPLIER
    )
    assert expense <= f.BUDGET_EUR + 1e-9
    assert np.sum(a[:, 0] == 1) <= 1 and np.sum(a[:, 0] == 2) <= 2
    assert shelter <= f.SHELTER_CAPACITY + 1e-9
    assert np.sum(evac == 2) <= 1 and assistance <= ASSISTANCE_CAPACITY + 1e-9
    if not social:
        assert not np.any(evac == 2)
    return dict(
        portfolio=[
            dict(
                district=DISTRICTS[k],
                property=f.PROPERTY_LABELS[p],
                evacuation=("none", "standard", "assisted")[e],
            )
            for k, (p, e) in enumerate(a)
        ],
        implementation_hours=completion / 2,
        resources=dict(
            intervention_eur=expense,
            barrier_crews=int(np.sum(a[:, 0] == 1)),
            contents_crews=int(np.sum(a[:, 0] == 2)),
            shelter_places=shelter,
            assistance_teams=int(np.sum(evac == 2)),
            assistance_places=assistance,
        ),
    )


def evaluate(model, policy, prior=None, include_trace=False):
    """Expected score, consequences and queries of a frozen policy (optionally with its decision tree)."""
    weights = model.prior.copy() if prior is None else np.asarray(prior, dtype=float).copy()
    assert np.min(weights) >= 0 and abs(weights.sum() - 1) < 1e-12
    social = hasattr(model, "group_features")
    loss = 0.0
    query_cost = 0.0
    commitment = 0.0
    completion = 0.0
    terminal_mass = 0.0
    counts = np.zeros(len(model.queries))
    features = np.zeros(4)
    groups = np.zeros((2, 4))
    action_prob = {}
    trace = []
    leaves = []
    stack = [(policy.root, weights, "root", [])]
    while stack:
        h, mass, node_id, path = stack.pop()
        kind, j = policy.decisions[h]
        t = int(sum(model.durations[q] for q, y in enumerate(h) if y != -1))
        prob = float(mass.sum())
        node = dict(
            node_id=node_id,
            history=list(h),
            path=path,
            probability=prob,
            time_hours=t / 2,
            kind=kind,
            index=j,
        )
        if kind == "query":
            assert h[j] == -1 and t + model.durations[j] <= model.config.deadline
            c = prob * float(model.costs[j])
            loss += c
            query_cost += c
            counts[j] += prob
            node.update(
                query_name=model.queries[j],
                duration_hours=float(model.durations[j]) / 2,
                acquisition_cost=float(model.costs[j]),
                children=[],
            )
            for y in (0, 1):
                hh = list(h)
                hh[j] = y
                factor = model.likelihood[:, j] if y else 1 - model.likelihood[:, j]
                child_mass = mass * factor
                child_prob = float(child_mass.sum())
                child_id = ("" if node_id == "root" else node_id + "/") + f"q{j}={y}"
                node["children"].append(
                    dict(
                        answer=y,
                        node_id=child_id,
                        conditional_probability=child_prob / prob if prob else None,
                        probability=child_prob,
                    )
                )
                stack.append((tuple(hh), child_mass, child_id, path + [[j, y]]))
            assert abs(sum(x["probability"] for x in node["children"]) - prob) < 1e-12
        else:
            assert kind == "act" and model.feasible[t, j]
            info = portfolio_info(model, j)
            finish = t / 2 + info["implementation_hours"]
            assert finish <= model.config.lead_hours
            assert np.allclose(model.features[:, t, j, 3], info["resources"]["intervention_eur"], atol=1e-8)
            loss += float(mass @ model.loss[:, t, j])
            features += mass @ model.features[:, t, j]
            if social:
                groups += np.einsum("z,zgk->gk", mass, model.group_features[:, j])
            commitment += prob * t / 2
            completion += prob * finish
            terminal_mass += prob
            action_prob[str(j)] = action_prob.get(str(j), 0.0) + prob
            node.update(
                info,
                completion_hours=finish,
                physical_features=(mass @ model.features[:, t, j] / prob).tolist() if prob else None,
            )
            if social:
                node["group_features"] = (
                    (np.einsum("z,zgk->gk", mass, model.group_features[:, j]) / prob).tolist()
                    if prob
                    else None
                )
            leaves.append((h, j, t, path))
        trace.append(node)
    assert abs(terminal_mass - 1) < 1e-12
    assert abs(commitment - float(counts @ model.durations) / 2) < 1e-12
    # group outcomes refer to commitment at the warning
    decay = getattr(model.config, "evacuation_decay", 0.0)
    if social and decay == 0:
        assert np.allclose(groups[:, :2].sum(axis=0), features[1:3], atol=1e-8)
    out = dict(
        expected_loss=loss,
        expected_query_cost=query_cost,
        physical_features=features.tolist(),
        expected_query_counts=counts.tolist(),
        expected_commitment_hours=commitment,
        expected_completion_hours=completion,
        terminal_histories=len(leaves),
        action_probabilities=action_prob,
    )
    if social:
        out["group_features"] = groups.tolist()
    if include_trace:
        out["trace"] = trace
    return out


def condition(model, severity, footprint):
    """Prior conditioned on a severity/footprint scenario, and the scenario probability."""
    mask = (model.indices[:, 0] == severity) & (model.indices[:, 1] == footprint)
    probability = float(model.prior[mask].sum())
    return np.where(mask, model.prior / probability, 0.0), probability
