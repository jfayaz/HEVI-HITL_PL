"""Experiment 08: agreed priorities, flag rates, policy paths and other numbers quoted in the text.

Sections 5.2-5.8 and 6; Appendix C.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from hevi_hitl import flood as fl
from hevi_hitl import social as so
from hevi_hitl.insight import (WORKED_SOCIAL_ID, query_sets, save_json, social_configs,
                               tilted_prior, severe_indicator)
from hevi_hitl.model import PortfolioSolver

RESULTS = ROOT / "results"
TOL = 1e-10
TOLERANCE = 0.005
EXPOSED, DISPLACEMENT = 2, 1          # columns of model.features


# Part A: model-based recomputation
def agreed_priorities() -> dict:
    """Value of knowing the preference type before the warning (risk queries only)."""
    rows = []
    for cid, cfg in enumerate(social_configs()):
        m = so.build_social_flood(cfg)
        risk = query_sets(m)["risk"]
        joint = PortfolioSolver(m).solve()[0]
        risk_only = PortfolioSolver(m, allowed=risk).solve()[0]
        immediate = PortfolioSolver(m, method="immediate").solve()[0]
        pref = m.indices[:, 5]
        agreed = 0.0
        for k in np.unique(pref):
            w = m.prior * (pref == k)
            if w.sum() > 0:
                agreed += w.sum() * PortfolioSolver(replace(m, prior=w / w.sum()), allowed=risk).solve()[0]
        rows.append(dict(instance=cid, voi_joint=immediate - joint, voi_risk=immediate - risk_only,
                         agreed_vs_immediate=immediate - agreed, agreed_vs_risk_only=risk_only - agreed,
                         agreed_vs_joint=joint - agreed))
    df = pd.DataFrame(rows)
    # Lower bound: agreed type versus risk-only planning >= V_J - V_R = V_theta + V_Rtheta.
    margin = df.agreed_vs_risk_only - (df.voi_joint - df.voi_risk)
    assert (margin > -TOL).all()
    w = df[df.instance == WORKED_SOCIAL_ID].iloc[0]
    return dict(
        instance_10=dict(voi_joint=w.voi_joint, voi_risk=w.voi_risk, agreed_vs_immediate=w.agreed_vs_immediate,
                         agreed_vs_risk_only=w.agreed_vs_risk_only, agreed_vs_joint=w.agreed_vs_joint),
        social_grid=dict(agreed_vs_joint_mean=df.agreed_vs_joint.mean(), agreed_vs_joint_min=df.agreed_vs_joint.min(),
                         agreed_vs_joint_max=df.agreed_vs_joint.max(),
                         agreed_vs_risk_only_mean=df.agreed_vs_risk_only.mean(),
                         agreed_vs_risk_only_max=df.agreed_vs_risk_only.max(),
                         min_margin_over_lower_bound=float(margin.min())),
        per_instance=rows)


def stress_planning_probability(model) -> dict:
    s = severe_indicator(model)
    out = dict(prior_high=float(model.prior @ s))
    for lam in (0.3, 0.35, 0.4):
        out[f"lambda_{lam:g}"] = float(tilted_prior(model, lam) @ s)
    return out


def depth_flag_rates() -> dict:
    """P(flag | severity) of the depth diagnostic (query 0), range over the other coordinates."""
    out = {}
    for rho in (0.6, 0.75, 0.9):
        m = fl.build_flood(fl.FloodConfig(response_reliability=rho))
        p, sev = m.likelihood[:, 0], m.indices[:, 0]
        out[f"rho_{rho:g}"] = {name: [float(p[sev == k].min()), float(p[sev == k].max())]
                               for k, name in enumerate(("low", "medium", "high"))}
    return out


def instance10_paths(m) -> dict:
    """Leaves of the instance-10 policy, the probability of selecting B and B - A after no flag."""
    solver = PortfolioSolver(m)
    solver.solve()
    immediate = PortfolioSolver(m, method="immediate").solve()[1][1]
    leaves, pending = [], [solver.root]
    while pending:
        h = pending.pop()
        kind, j = solver.solve(h)[1]
        if kind == "query":
            pending.extend(solver.children(h, j))
        else:
            leaves.append(dict(history=h, action=int(j), time=solver.time(h), probability=float(solver.mass(h).sum())))
    others = sorted({leaf["action"] for leaf in leaves} - {immediate})
    assert len(others) == 1
    b_index = others[0]
    no_flag = list(solver.root)
    no_flag[0] = 0
    post = solver.mass(tuple(no_flag))
    post = post / post.sum()
    flag = list(solver.root)
    flag[0] = 1
    post_flag = solver.mass(tuple(flag))
    post_flag = post_flag / post_flag.sum()
    high = severe_indicator(m)
    t = max(leaf["time"] for leaf in leaves if leaf["action"] == b_index)
    diff = m.features[:, t, b_index] - m.features[:, t, immediate]
    p_b = sum(leaf["probability"] for leaf in leaves if leaf["action"] == b_index)
    p_b_high = sum(float(solver.mass(leaf["history"]) @ high) for leaf in leaves if leaf["action"] == b_index)
    joint_minus_immediate_exposed = sum(
        leaf["probability"] * (solver.mass(leaf["history"]) / leaf["probability"])
        @ (m.features[:, leaf["time"], leaf["action"], EXPOSED] - m.features[:, leaf["time"], immediate, EXPOSED])
        for leaf in leaves)
    return dict(immediate_index=int(immediate), portfolio_b_index=int(b_index),
                leaf_probabilities=sorted((round(leaf["probability"], 6) for leaf in leaves), reverse=True),
                probability_select_b=p_b, probability_b_and_highest_severity=p_b_high,
                p_high=dict(prior=float(m.prior @ high), flag=float(post_flag @ high), no_flag=float(post @ high)),
                b_minus_a_no_flag=dict(exposed_people=float(post @ diff[:, EXPOSED]),
                                       displacement_person_days=float(post @ diff[:, DISPLACEMENT])),
                joint_minus_immediate_exposed_people=float(joint_minus_immediate_exposed))


# Part B: quantities derived from saved results
def certified_share() -> dict:
    """Bracket S - L(pi~) <= V_J <= S - L(pi~) + certificate at the accepted class count."""
    cert = pd.read_csv(RESULTS / "certificates.csv")
    out = {}
    for domain in ("flood", "social"):
        dv = pd.read_csv(RESULTS / f"decision_value_{domain}.csv").set_index("configuration_id")
        shares, ratios = [], []
        for cid, g in cert[(cert.domain == domain) & (cert.method == "guided")].groupby("configuration_id"):
            ok = g[g.refined_lb <= TOLERANCE].sort_values("classes")
            row = ok.iloc[0]
            vj = float(dv.loc[cid, "voi_joint"])
            if vj <= TOL:
                continue
            achieved = vj - float(row.regret)            # S - L(pi~), exact
            shares.append(achieved / (achieved + float(row.refined_lb)))
            ratios.append(float(row.refined_lb) / vj)
        vj_all = pd.read_csv(RESULTS / f"decision_value_{domain}.csv").voi_joint
        out[domain] = dict(querying_instances=len(shares), guaranteed_share_min=min(shares),
                           guaranteed_share_max=max(shares), certificate_exceeds_voi_joint=int(sum(r > 1 for r in ratios)),
                           certificate_over_voi_joint_max=max(ratios), instances=int(len(vj_all)),
                           voi_joint_at_most_tolerance=int((vj_all <= TOLERANCE).sum()),
                           voi_joint_zero=int((vj_all <= TOL).sum()), voi_joint_mean=float(vj_all.mean()))
    scales = fl.SCALES
    out["units"] = dict(per_1e3_score=(1e-3 * scales).tolist(), tolerance_0005=(TOLERANCE * scales).tolist(),
                        order="EUR, person-days, exposed people, at unit weight on that attribute")
    return out


def preference_and_access() -> dict:
    flood = pd.read_csv(RESULTS / "decision_value_flood.csv")
    social = pd.read_csv(RESULTS / "decision_value_social.csv")
    rnd = pd.read_csv(RESULTS / "random_instances.csv")
    used = rnd.preference_queries > TOL
    cert = pd.read_csv(RESULTS / "certificates.csv")
    accepted = {}
    for cid, g in cert[(cert.domain == "social") & (cert.method == "guided")].groupby("configuration_id"):
        accepted[int(cid)] = int(g[g.refined_lb <= TOLERANCE].classes.min())
    cfgs = social_configs()
    pairs = [(i, j) for i in range(len(cfgs)) for j in range(i + 1, len(cfgs))
             if replace(cfgs[i], social_query_reliability=0) == replace(cfgs[j], social_query_reliability=0)]
    return dict(
        flood_voi_preference_positive=flood.loc[flood.voi_preference > TOL, "configuration_id"].tolist(),
        flood_voi_preference_max=float(flood.voi_preference.max()),
        social_voi_preference_max=float(social.voi_preference.max()),
        social_evppi_preference_max=float(social.evppi_preference.max()),
        random_voi_preference_positive=int((rnd.voi_preference > TOL).sum()),
        random_evppi_preference_positive=int((rnd.evppi_preference > TOL).sum()),
        random_preference_used=int(used.sum()),
        random_preference_used_with_voi_preference_positive=int((used & (rnd.voi_preference > TOL)).sum()),
        social_access_check_pairs=len(pairs),
        social_pairs_with_different_accepted_classes=int(sum(accepted[i] != accepted[j] for i, j in pairs)))


def misspecification_counts() -> dict:
    mis = pd.read_csv(RESULTS / "misspecification.csv")
    dv = pd.read_csv(RESULTS / "decision_value_social.csv").set_index("configuration_id")
    haz = mis[(mis.family == "hazard_prior") & ~mis.correctly_specified]
    pp = mis[(mis.family == "preference_prior") & ~mis.correctly_specified]
    worst = pp.loc[pp.regret.idxmax()]
    mis = mis[~mis.correctly_specified]

    def distinct(g: pd.DataFrame) -> int:
        """Cases with identical (family, planned, true) and identical outcomes count once (access-check duplicates)."""
        r = g.assign(r=g.regret.round(12), v=g.versus_immediate.round(12), w=g.versus_planned_immediate.round(12))
        return int(len(r.drop_duplicates(["family", "planned", "true", "r", "v", "w"])))

    rel = mis[mis.family == "reliability"]
    over = rel[(rel.true.astype(float) == 0.6) & (rel.planned.astype(float) >= 0.75)]
    other = mis[mis.family.isin(["preference_slope", "preference_prior", "mechanism_strength"])]
    better = haz[haz.versus_planned_immediate < -TOL]
    distinct_counts = dict(
        overconfident_true_0p6=dict(cases=int(len(over)), distinct=distinct(over),
                                    worse_than_immediate=int((over.versus_immediate > TOL).sum())),
        preference_and_mechanism=dict(cases=int(len(other)), distinct=distinct(other),
                                      worse_than_immediate=int((other.versus_immediate > TOL).sum())),
        hazard=dict(cases=int(len(haz)), distinct=distinct(haz), strictly_better=int(len(better)),
                    strictly_better_distinct=distinct(better)))
    return dict(distinct_counts=distinct_counts,
                hazard_cases=int(len(haz)), hazard_strictly_better_than_planned_immediate=int((haz.versus_planned_immediate < -TOL).sum()),
                hazard_equal_to_planned_immediate=int((haz.versus_planned_immediate.abs() <= TOL).sum()),
                preference_prior_max_regret=float(worst.regret), preference_prior_worst_planned=worst.planned,
                preference_prior_worst_true=worst.true,
                interaction_of_true_instance=float(dv.loc[int(worst.configuration_id), "interaction"]))


def randomized_extremes() -> dict:
    rnd = pd.read_csv(RESULTS / "random_instances.csv")
    q = rnd.any_query > 0
    slack = rnd.lead_hours >= 4
    used = rnd.preference_queries > TOL
    top = rnd[[f"preference_prior_{k}" for k in range(4)]].max(axis=1)
    band = slack & rnd.p_high_severity.between(0.2, 0.4)
    above = slack & (rnd.p_high_severity > 0.55)
    first = np.where(rnd.root_kind == "act", "commit", np.where(rnd.root_index == 0, "depth", "comparison"))
    s = rnd.assign(first=first)[slack]
    ph = s.p_high_severity

    def share(mask, what="any"):
        g = s[mask]
        return float((g.any_query > 0).mean()) if what == "any" else float((g["first"] == what).mean())

    low = ph <= 0.2
    return dict(max_p_high_with_query=float(rnd.loc[q, "p_high_severity"].max()),
                slack_instances_p_high_0p2_0p4=int(band.sum()), share_query_p_high_0p2_0p4=float(rnd.loc[band, "any_query"].mean()),
                slack_instances_above_0p55=int(above.sum()),
                max_top_preference_probability_with_comparison=float(top[used].max()),
                instances_top_preference_above_that=int((top > top[used].max()).sum()),
                instances_per_lead={int(k): int(v) for k, v in rnd.lead_hours.value_counts().sort_index().items()},
                sequencing_lead_ge_4=dict(
                    instances=int(len(s)),
                    query_share_p_high_0p2_0p3=share((ph > 0.2) & (ph <= 0.3)),
                    depth_first_share_p_high_0p2_0p3=share((ph > 0.2) & (ph <= 0.3), "depth"),
                    comparison_first_share_of_querying_p_high_le_0p2=float(
                        (s[low]["first"] == "comparison").sum() / max(1, (s[low].any_query > 0).sum())),
                    query_share_p_high_0p5_0p55=share((ph > 0.5) & (ph <= 0.55)),
                    query_share_p_high_0p6_0p65=share((ph > 0.6) & (ph <= 0.65)),
                    instances_above_0p65=int((ph > 0.65).sum()),
                    querying_above_0p65=int(((ph > 0.65) & (s.any_query > 0)).sum()),
                    query_share_reliability_below_0p6=float(s[s.original_reliability <= 0.6].any_query.mean()),
                    query_share_reliability_above_0p9=float(s[s.original_reliability > 0.9].any_query.mean())),
                comparison_first=int((first == "comparison").sum()),
                comparison_first_without_risk_query=int(((first == "comparison") & (rnd.risk_queries <= TOL)).sum()),
                comparison_use_top_preference_below_0p6=float(used[top <= 0.6].mean()),
                comparison_use_top_preference_above_0p8=float(used[top > 0.8].mean()),
                access_check_used=int((rnd.social_queries > TOL).sum()),
                lead_3_querying=int(((rnd.lead_hours == 3) & q).sum()))


def answers_and_flags(m) -> dict:
    """Comparison answer probabilities by type, value of an exposed person by type, flag rates (instance 10)."""
    types = m.indices[:, 5]
    comparisons = {m.queries[j]: [float(m.likelihood[types == k, j].mean()) for k in range(int(types.max()) + 1)]
                   for j in range(len(m.queries)) if "_vs_" in m.queries[j]}
    w = fl.PREFERENCE_WEIGHTS
    value_exposed = (w[:, 2] / w[:, 0] * fl.SCALES[0] / fl.SCALES[2]).tolist()
    high = severe_indicator(m)
    flag = m.likelihood[:, 0]
    return dict(prefer_lower_exposure_profile=comparisons, eur_per_exposed_person=value_exposed,
                flag_rate_highest=float((m.prior * high) @ flag / (m.prior @ high)),
                flag_rate_not_highest=float((m.prior * (1 - high)) @ flag / (m.prior @ (1 - high))))


def main() -> None:
    m10 = so.build_social_flood(social_configs()[WORKED_SOCIAL_ID])
    stress = pd.read_csv(RESULTS / "stress_weighting.csv")
    s10 = stress[stress.configuration_id == WORKED_SOCIAL_ID].set_index("lam")
    sacrifice = float(s10.loc[0.3, "original__score"] - s10.loc[0.0, "original__score"])
    out = dict(
        agreed_priorities=agreed_priorities(),
        stress_weight=dict(planning_probability_high=stress_planning_probability(m10),
                           score_sacrifice_lambda_0p3=sacrifice,
                           score_sacrifice_lambda_0p3_eur_at_unit_weight=sacrifice * float(fl.SCALES[0])),
        depth_flag_probability=depth_flag_rates(),
        instance_10=instance10_paths(m10),
        certified_share=certified_share(),
        preference_and_access=preference_and_access(),
        misspecification=misspecification_counts(),
        randomized=randomized_extremes(),
        answers_and_flags=answers_and_flags(m10),
    )
    save_json(RESULTS / "practice_checks.json", out)
    a = out["agreed_priorities"]
    print("agreed, instance 10:", {k: round(v, 6) for k, v in a["instance_10"].items()})
    print("agreed, grid:", {k: round(v, 6) for k, v in a["social_grid"].items()})
    for key in ("stress_weight", "depth_flag_probability", "instance_10", "certified_share",
                "preference_and_access", "misspecification", "randomized", "answers_and_flags"):
        print(key, out[key])


if __name__ == "__main__":
    main()
