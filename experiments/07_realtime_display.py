"""Experiment 07: decision display for social instance 10.

Supplementary Section S14, Fig. S4.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from hevi_hitl import flood as fl
from hevi_hitl import operational as op
from hevi_hitl import social as so
from hevi_hitl.insight import save_json
from hevi_hitl.model import PortfolioSolver
from hevi_hitl.plotting import FULL_WIDTH_IN, PALETTE, apply_style, save

RESULTS = ROOT / "results"
CONFIG = so.SocialConfig(lead_hours=6, mechanism_strength=0.5, social_query_reliability=0.9,
                         preference_prior="diffuse")
QUANTILES = (0.1, 0.9)
HIGH = 2                       # severity index of the 1.65 m scenario
SEVERITY_M = (0.35, 0.85, 1.65)
# answer paths of the policy, identified by their leaf; the figure shows DISPLAY_PATH
PATHS = (
    dict(key="no_flag_lower_displacement", leaf="q0=0/q6=0", heading="No flag, lower displacement preferred"),
    dict(key="flag", leaf="q0=1", heading="Flag raised"),
)
DISPLAY_PATH = "no_flag_lower_displacement"
FLAG_LEAF, B_LEAF = "q0=1", "q0=0/q6=0"
# Plotted metrics: key -> (label, unit factor, extractor(model, t, j) of per-configuration values).
METRICS = {
    "damage_plus_expenditure_meur": (
        "Damage and\nexpenditure\n(million EUR)",
        lambda m, t, j: (m.features[:, t, j, 0] + m.features[:, t, j, 3]) / 1e6),
    "exposed_people": (
        "Exposed\npeople\n(persons)",
        lambda m, t, j: m.features[:, t, j, 2]),
    "displacement_thousand_person_days": (
        "Displacement\n(thousand\nperson-days)",
        lambda m, t, j: m.features[:, t, j, 1] / 1e3),
    "constrained_exposed_percent": (
        "Constrained group\nexposed (%)",
        lambda m, t, j: 100.0 * m.group_features[:, j, 1, 3]),
}
PLOTTED = ("damage_plus_expenditure_meur", "exposed_people", "displacement_thousand_person_days")
CRITERIA = ("economic cost", "displacement", "exposure")
COMPARISON_PAIR = {"economic_vs_exposure": 0, "displacement_vs_exposure": 1}   # rows of PREFERENCE_PAIRS


# Computation
def weighted_quantile(values: np.ndarray, weights: np.ndarray, p: float) -> float:
    """Lower inverse CDF of a discrete distribution: smallest x with F(x) >= p."""
    order = np.argsort(values, kind="stable")
    cdf = np.cumsum(weights[order])
    k = int(np.searchsorted(cdf, p * cdf[-1] - 1e-12, side="left"))
    return float(values[order][min(k, len(values) - 1)])


def posterior(model, answers) -> np.ndarray:
    """Normalised belief after the (query, answer) pairs in ``answers`` (Eq. update)."""
    b = model.prior.copy()
    for q, y in answers:
        b = b * (model.likelihood[:, q] if y else 1 - model.likelihood[:, q])
    return b / b.sum()


def summary(values: np.ndarray, b: np.ndarray) -> dict:
    lo, hi = (weighted_quantile(values, b, p) for p in QUANTILES)
    return dict(mean=float(b @ values), p10=lo, p90=hi, band_width=hi - lo)


def answer_label(model, q: int, y: int) -> str:
    """Plain-language answer label derived from the query definition, not hand-written."""
    name = model.queries[q]
    if name == "depth_update":
        return "flag" if y else "no flag"
    pair = fl.PREFERENCE_PAIRS[COMPARISON_PAIR[name]]
    chosen, other = pair[y], pair[1 - y]
    lower = [CRITERIA[k] for k in range(3) if chosen[k] < other[k]]
    assert len(lower) == 1
    return f"prefers lower {lower[0]}"


def check_semantics(model) -> dict:
    """Verify that answer 1 means 'flag'/'prefers profile B' as the labels assume."""
    pref = model.indices[:, 5]
    sev = model.indices[:, 0]
    flag = [float(model.likelihood[sev == s, 0].mean()) for s in range(3)]
    assert flag[0] < flag[1] < flag[2], "depth flag must become more likely with severity"
    lik = {name: [float(model.likelihood[pref == k, model.queries.index(name)].mean()) for k in range(4)]
           for name in COMPARISON_PAIR}
    # Preference types: 0 economic, 1 displacement, 2 exposure, 3 equal weights.
    assert lik["economic_vs_exposure"][0] < 0.5 < lik["economic_vs_exposure"][2]
    assert lik["displacement_vs_exposure"][1] < 0.5 < lik["displacement_vs_exposure"][2]
    for q in (5, 6):   # preference answers carry no information about physical/social axes
        for k in range(4):
            assert np.ptp(model.likelihood[pref == k, q]) < 1e-14
    return dict(p_flag_by_severity=flag, p_answer1_by_preference_type=lik,
                preference_types=["economic", "displacement", "exposure", "equal"],
                answer_1_meaning={"depth_update": "flag (deeper Riverside water indicated)",
                                  "economic_vs_exposure": answer_label(model, 5, 1),
                                  "displacement_vs_exposure": answer_label(model, 6, 1)},
                answer_0_meaning={"depth_update": "no flag",
                                  "economic_vs_exposure": answer_label(model, 5, 0),
                                  "displacement_vs_exposure": answer_label(model, 6, 0)})


def solve(cfg=CONFIG):
    """Model, frozen optimal policy, its trace and the immediate-commitment portfolio."""
    model = so.build_social_flood(cfg)
    solver = PortfolioSolver(model)
    value = solver.solve()[0]
    policy = op.freeze(solver)
    evaluation = op.evaluate(model, policy, include_trace=True)
    immediate_solver = PortfolioSolver(model, method="immediate")
    immediate_value, (kind, immediate) = immediate_solver.solve()
    assert kind == "act"
    return model, value, policy, evaluation, immediate_value, int(immediate)


def feasibility(model) -> dict:
    """Latest commitment times (h) after which barrier / any portfolio cannot be completed."""
    barrier = np.array([any(p == 1 for p, _ in a) for a in model.actions])
    active = np.array([any(p > 0 or e > 0 for p, e in a) for a in model.actions])
    ticks = np.arange(model.config.deadline + 1)
    last_barrier = int(ticks[model.feasible[:, barrier].any(axis=1)].max())
    last_any = int(ticks[model.feasible[:, active].any(axis=1)].max())
    return dict(lead_hours=model.config.lead_hours,
                barrier_last_commit_hours=last_barrier / 2,
                any_portfolio_last_commit_hours=last_any / 2,
                feasible_portfolios_by_half_hour=model.feasible.sum(axis=1).tolist())


def path_display(model, evaluation, leaf_id: str, immediate: int, labels: dict, fixed: dict) -> dict:
    """Belief, recommendation and option summaries on each segment between evidence arrivals."""
    nodes = {n["node_id"]: n for n in evaluation["trace"]}
    leaf = nodes[leaf_id]
    assert leaf["kind"] == "act"
    answers = [tuple(x) for x in leaf["path"]]
    ends, t = [], 0
    for q, _ in answers:
        t += int(model.durations[q])
        ends.append(t)
    commit_tick = int(round(2 * leaf["time_hours"]))
    assert commit_tick == (ends[-1] if ends else 0)
    starts = [0] + ends
    stops = ends + [model.config.deadline]
    committed = int(leaf["index"])
    segments, events = [], []
    node_id = "root"
    for k, (s, e) in enumerate(zip(starts, stops)):
        b = posterior(model, answers[:k])
        scores = b @ model.loss[:, s]
        feas = np.flatnonzero(model.feasible[s])
        ranked = feas[np.argsort(scores[feas], kind="stable")]
        recommended = int(ranked[0])
        is_committed = s >= commit_tick
        decision = nodes[node_id]
        assert decision["time_hours"] * 2 == s
        if is_committed:
            assert decision["kind"] == "act" and recommended == committed
        adaptive = committed if is_committed else recommended
        options = dict(no_intervention=(0, s), immediate=(immediate, 0), adaptive=(adaptive, s))
        seg = dict(
            start_hours=s / 2, end_hours=e / 2,
            answers_received=[[model.queries[q], int(y)] for q, y in answers[:k]],
            belief_sum=float(b.sum()),
            p_severity={f"{SEVERITY_M[v]}m": float(b[model.indices[:, 0] == v].sum()) for v in range(3)},
            p_high=float(b[model.indices[:, 0] == HIGH].sum()),
            policy_decision=(f"query {model.queries[decision['index']]}" if decision["kind"] == "query"
                             else f"commit portfolio {labels.get(committed, committed)}"),
            recommended_index=recommended, recommended_label=labels.get(recommended, f"#{recommended}"),
            recommended_posterior_score=float(scores[recommended]),
            runner_up_index=int(ranked[1]), runner_up_posterior_score=float(scores[ranked[1]]),
            committed=bool(is_committed),
            options={name: dict(portfolio_index=int(j), portfolio_label=labels.get(int(j), f"#{j}"),
                                **{key: summary(f(model, tt, j), b) for key, (_, f) in METRICS.items()})
                     for name, (j, tt) in options.items()},
            # Every displayed option (no action, A, B) under the same belief, as the group would see it at s.
            fixed_portfolios={name: dict(portfolio_index=int(j), feasible=bool(model.feasible[s, j]),
                                         posterior_score=float(scores[j]),
                                         **{key: summary(f(model, s, j), b) for key, (_, f) in METRICS.items()})
                              for name, j in fixed.items()},
        )
        segments.append(seg)
        if k < len(answers):
            q, y = answers[k]
            child = decision["children"][y]
            assert decision["kind"] == "query" and decision["index"] == q
            b_next = posterior(model, answers[:k + 1])
            events.append(dict(
                query=model.queries[q], kind="risk" if q < 5 else "preference",
                start_hours=s / 2, answer_hours=e / 2, answer=int(y), answer_label=answer_label(model, q, y),
                conditional_probability=child["conditional_probability"],
                p_high_before=seg["p_high"], p_high_after=float(b_next[model.indices[:, 0] == HIGH].sum())))
            node_id = child["node_id"]
    # The plotted committed means must equal the forward evaluator's leaf consequences.
    last = segments[-1]["options"]["adaptive"]
    phys = leaf["physical_features"]
    check = dict(
        damage_plus_expenditure_meur=abs(last["damage_plus_expenditure_meur"]["mean"] - (phys[0] + phys[3]) / 1e6),
        exposed_people=abs(last["exposed_people"]["mean"] - phys[2]),
        displacement_thousand_person_days=abs(last["displacement_thousand_person_days"]["mean"] - phys[1] / 1e3),
        constrained_exposed_percent=abs(last["constrained_exposed_percent"]["mean"]
                                        - 100 * leaf["group_features"][1][3]),
    )
    assert max(check.values()) < 1e-9, check
    widths = {
        moment: {key: segments[idx]["options"]["adaptive"][key]["band_width"]
                 for key in ("damage_plus_expenditure_meur", "exposed_people")}
        for moment, idx in (("t0", 0), ("commitment", -1))
    }
    return dict(
        leaf_node_id=leaf_id, path_probability=leaf["probability"],
        commitment_hours=commit_tick / 2, completion_hours=leaf["completion_hours"],
        committed_index=committed, committed_label=labels.get(committed, f"#{committed}"),
        events=events, segments=segments,
        adaptive_band_width=widths,
        p_high_along_path=[seg["p_high"] for seg in segments],
        leaf_audit_absolute_error=check,
    )


def compute() -> dict:
    model, value, policy, evaluation, immediate_value, immediate = solve()
    semantics = check_semantics(model)
    nodes = {n["node_id"]: n for n in evaluation["trace"]}
    a_index = int(nodes[FLAG_LEAF]["index"])
    b_index = int(nodes[B_LEAF]["index"])
    assert a_index == immediate, "portfolio A is the immediate-commitment portfolio"
    assert a_index != b_index
    labels = {a_index: "A", b_index: "B", 0: "none"}
    fixed = {"none": 0, "A": a_index, "B": b_index}
    portfolios = {}
    for name, j in (("A", a_index), ("B", b_index)):
        info = op.portfolio_info(model, j)
        barrier = [p["district"] for p in info["portfolio"] if p["property"] == "temporary_barrier"]
        assisted = [p["district"] for p in info["portfolio"] if p["evacuation"] == "assisted"]
        portfolios[name] = dict(index=j, barrier_district=barrier[0], assisted_evacuation_district=assisted[0],
                                **info)
    assert portfolios["A"]["barrier_district"] == "Upstream" and portfolios["B"]["barrier_district"] == "Riverside"
    paths = {}
    for spec in PATHS:
        out = path_display(model, evaluation, spec["leaf"], immediate, labels, fixed)
        # At t = 0 the adaptive recommendation is the immediate-commitment portfolio.
        assert out["segments"][0]["recommended_index"] == immediate
        paths[spec["key"]] = dict(heading=spec["heading"], **out)
    first, second = paths["flag"], paths[DISPLAY_PATH]
    # Riverside ground depth at high severity
    riverside_high = [round(SEVERITY_M[HIGH] * 1.0 + d, 2) for d in (-0.12, 0.12)]
    return dict(
        configuration=dict(lead_hours=CONFIG.lead_hours, mechanism_strength=CONFIG.mechanism_strength,
                           social_query_reliability=CONFIG.social_query_reliability,
                           preference_prior=CONFIG.preference_prior, hazard_prior=CONFIG.hazard_prior,
                           states=len(model.prior), portfolios=len(model.actions), queries=list(model.queries),
                           query_duration_hours=(model.durations / 2).tolist()),
        units=dict(damage_plus_expenditure_meur="million EUR (property damage plus intervention expenditure)",
                   exposed_people="persons", displacement_thousand_person_days="thousand post-event person-days",
                   constrained_exposed_percent="percent of the assistance-constrained group exposed",
                   time="hours since warning; flood arrival at lead_hours"),
        plotted_metrics=list(PLOTTED),
        quantiles=list(QUANTILES),
        answer_semantics=semantics,
        policy=dict(optimal_value=value, immediate_value=immediate_value,
                    voi_joint=immediate_value - value, immediate_index=immediate,
                    immediate_label="A", portfolios=portfolios),
        feasibility=feasibility(model),
        severe_scenario=dict(severity_m=SEVERITY_M[HIGH], riverside_ground_depth_m=riverside_high,
                             barrier_overtopping_limit_m=1.2),
        paths=paths,
        text_numbers=dict(
            adaptive_band_width_t0=first["adaptive_band_width"]["t0"],
            adaptive_band_width_commitment={k: p["adaptive_band_width"]["commitment"] for k, p in paths.items()},
            p_high_along_path={k: p["p_high_along_path"] for k, p in paths.items()},
            committed_vs_immediate_at_commitment={
                k: {m: p["segments"][-1]["options"]["adaptive"][m]["mean"]
                    - p["segments"][-1]["options"]["immediate"][m]["mean"] for m in METRICS}
                for k, p in paths.items()},
            recommendation_sequence={k: [s["recommended_label"] for s in p["segments"]] for k, p in paths.items()},
        ),
        checks=dict(belief_sums_minus_one_max=max(abs(s["belief_sum"] - 1) for p in paths.values()
                                                 for s in p["segments"]),
                    leaf_audit_absolute_error_max=max(max(p["leaf_audit_absolute_error"].values())
                                                      for p in paths.values()),
                    displayed_path_commits=second["committed_label"]),
    )


# Figure: what the response group sees at each decision point of one answer path
BLUE = PALETTE["joint"]                  # option recommended at that time
RISK = PALETTE["physical_only"]          # risk question (orange)
PREF = PALETTE["preference_only"]        # preference comparison (green)
COMMIT = PALETTE["homogeneous"]          # commitment and implementation (vermilion)
NONE_C, OTHER_C, GHOST_C = "#BDBDBD", "#7A7A7A", "#9E9E9E"
SHADE_BARRIER, SHADE_NONE = "0.91", "0.80"
OPTIONS = (("none", "None"), ("A", "A"), ("B", "B"))
ROWS = (
    ("damage_plus_expenditure_meur", "Damage plus\nexpenditure\n(million EUR)", 34, [0, 10, 20, 30]),
    ("displacement_thousand_person_days", "Displacement\n(thousand\nperson-days)", 32, [0, 10, 20, 30]),
    ("exposed_people", "Exposed\npeople", 1000, [0, 250, 500, 750]),
)


def _number(ax, x, y, k, transform, ms=11.5):
    """Black disc with a white panel letter; links timeline marks to the snapshot panels."""
    ax.plot([x], [y], marker="o", ms=ms, color="black", transform=transform, clip_on=False, zorder=8)
    ax.text(x, y, "abc"[k - 1], color="white", fontsize=8, fontweight="bold", ha="center", va="center_baseline",
            transform=transform, zorder=9, clip_on=False)


def _headers(path) -> list[list[tuple[str, dict]]]:
    """Plain-language header lines for each snapshot, derived from the path's events."""
    segs, events = path["segments"], path["events"]
    out = []
    for k, seg in enumerate(segs):
        if k == 0:
            what, detail, risk = "warning issued", "forecast only", f"P(highest severity) {seg['p_high']:.2f}"
        else:
            ev = events[k - 1]
            if ev["kind"] == "risk":
                what = "depth diagnostic"
                detail = "flag raised" if ev["answer"] else "no flag"
                risk = f"P(highest severity) {ev['p_high_after']:.2f}"
            else:
                what = "comparison"
                detail = "prefers " + ev["answer_label"].replace("prefers ", "")
                risk = f"P(highest severity) {seg['p_high']:.2f}"
        if seg["committed"]:
            action = [(f"Commit portfolio {path['committed_label']}", dict(color=COMMIT, fontweight="bold"))]
        else:
            # The policy asks a query here; the blue portfolio is what committing now would select.
            nxt = events[k]
            ask = "depth diagnostic" if nxt["kind"] == "risk" else "comparison"
            action = [(f"Next: {ask}", dict(color=RISK if nxt["kind"] == "risk" else PREF, fontweight="bold")),
                      (f"Best portfolio now: {seg['recommended_label']}", dict(color=BLUE))]
        out.append([(f"{seg['start_hours']:g} h: {what}", dict(fontweight="bold")),
                    (detail, dict(color="0.25")), (risk, dict(color="0.25"))] + action)
    return out


def _timeline(ax, path, results):
    feas = results["feasibility"]
    lead = feas["lead_hours"]
    barrier, anyp = feas["barrier_last_commit_hours"], feas["any_portfolio_last_commit_hours"]
    ax.axvspan(barrier, anyp, color=SHADE_BARRIER, lw=0, zorder=0)
    ax.axvspan(anyp, lead, color=SHADE_NONE, lw=0, zorder=0)
    ax.text(0.5 * (barrier + anyp), 0.77, "Too late to commit\na barrier", fontsize=8, ha="center", va="center",
            color="0.25", linespacing=1.05)
    ax.text(0.5 * (anyp + lead), 0.77, "Too late to commit\nany portfolio", fontsize=8, ha="center", va="center",
            color="0.25", linespacing=1.05)
    lanes = {"risk": 0.86, "preference": 0.62, "commit": 0.33}
    h = 0.19
    names = {"risk": "Depth diagnostic", "preference": "Preference comparison"}
    for ev in path["events"]:
        y = lanes[ev["kind"]]
        ax.add_patch(Rectangle((ev["start_hours"], y - h / 2), ev["answer_hours"] - ev["start_hours"], h,
                               facecolor=RISK if ev["kind"] == "risk" else PREF, lw=0, zorder=3))
        ax.text(ev["answer_hours"] + 0.07, y, names[ev["kind"]], fontsize=8, ha="left", va="center", zorder=4)
    commit, done = path["commitment_hours"], path["completion_hours"]
    y = lanes["commit"]
    ax.add_patch(Rectangle((commit, y - h / 2), done - commit, h, facecolor=COMMIT, alpha=0.22, lw=0, zorder=3))
    ax.add_patch(Rectangle((commit, y - h / 2), done - commit, h, facecolor="none", edgecolor=COMMIT, lw=0.8,
                           zorder=3))
    ax.plot([commit], [y], marker="D", ms=5, color=COMMIT, mec="white", mew=0.6, zorder=5)
    ax.text(0.5 * (commit + done), y, f"Implement portfolio {path['committed_label']}", fontsize=8, ha="center",
            va="center", zorder=4)
    for k, seg in enumerate(path["segments"], start=1):
        _number(ax, seg["start_hours"], 0.0, k, ax.transData)
    ax.set_xlim(0, lead)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(0, lead + 1))
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel(f"Time since warning (h); the flood arrives at {lead} h", labelpad=3)
    ax.set_yticks([])
    ax.grid(False)
    for side in ("left", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel("Timeline", rotation=0, ha="right", va="center", labelpad=8)


def _snapshot(ax, seg, prev, key, xmax, xticks, first_col):
    """Ranges of every option under the belief held at this decision point."""
    rec = seg["recommended_label"]
    for row, (name, label) in enumerate(OPTIONS):
        y = len(OPTIONS) - 1 - row
        s = seg["fixed_portfolios"][name][key]
        color = BLUE if name == rec else (NONE_C if name == "none" else OTHER_C)
        if prev is not None:
            g = prev["fixed_portfolios"][name][key]
            ax.plot([g["p10"], g["p90"]], [y + 0.36] * 2, color=GHOST_C, lw=1.0, solid_capstyle="butt", zorder=2)
        ax.plot([s["p10"], s["p90"]], [y, y], color=color, lw=6.5, solid_capstyle="butt", zorder=3)
        ax.plot([s["mean"]], [y], marker="o", ms=4.2, mfc="white", mec=color if name == rec else "0.2",
                mew=1.0, zorder=4)
    ax.set_ylim(-0.6, len(OPTIONS) - 0.35)
    ax.set_yticks(range(len(OPTIONS)))
    ax.set_yticklabels([label for _, label in reversed(OPTIONS)] if first_col else [])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, xmax)
    ax.set_xticks(xticks)
    ax.grid(axis="y", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figure(results: dict) -> None:
    apply_style()
    path = results["paths"][DISPLAY_PATH]
    segs = path["segments"]
    W, H = FULL_WIDTH_IN, 4.32
    left, right, gap = 1.12, FULL_WIDTH_IN - 0.06, 0.14
    colw = (right - left - gap * (len(segs) - 1)) / len(segs)
    fig = plt.figure(figsize=(W, H))

    def box(x0, y0, w, h):
        return [x0 / W, y0 / H, w / W, h / H]

    tl = fig.add_axes(box(left, 3.43, right - left, 0.55))
    _timeline(tl, path, results)
    header_top, line = 3.13, 0.140
    row_bottoms, row_h = (2.00, 1.26, 0.52), 0.48
    for c, (seg, lines) in enumerate(zip(segs, _headers(path))):
        x0 = left + c * (colw + gap)
        _number(fig.axes[0], (x0 + 0.08) / W, header_top / H, c + 1, fig.transFigure)
        for i, (text, kw) in enumerate(lines):
            fig.text((x0 + (0.2 if i == 0 else 0.0)) / W, (header_top - i * line) / H, text, fontsize=8,
                     ha="left", va="center_baseline", **kw)
        prev = segs[c - 1] if c > 0 else None
        for r, (key, label, xmax, xticks) in enumerate(ROWS):
            ax = fig.add_axes(box(x0, row_bottoms[r], colw, row_h))
            _snapshot(ax, seg, prev, key, xmax, xticks, first_col=(c == 0))
            if c == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=6, fontsize=8,
                              linespacing=1.05)
    handles = [
        (Line2D([], [], color=OTHER_C, lw=6.5, solid_capstyle="butt"),
         Line2D([], [], marker="o", ms=4.2, mfc="white", mec="0.2", ls="none")),
        Line2D([], [], color=BLUE, lw=6.5, solid_capstyle="butt"),
        Line2D([], [], color=GHOST_C, lw=1.0),
    ]
    labels = ["10–90% range and expected value", "Portfolio selected if committing now", "Previous range"]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3,
               handlelength=1.8, columnspacing=1.0, handletextpad=0.4)
    save(fig, "realtime_display")


def main() -> None:
    if "--figure-only" in sys.argv:
        import json
        figure(json.loads((RESULTS / "realtime_display.json").read_text(encoding="utf-8")))
        return
    results = compute()
    save_json(RESULTS / "realtime_display.json", results)
    figure(results)
    t = results["text_numbers"]
    print("Recommendations:", t["recommendation_sequence"])
    print("P(high):", t["p_high_along_path"])
    print("Adaptive band width t0:", t["adaptive_band_width_t0"])
    print("Adaptive band width at commitment:", t["adaptive_band_width_commitment"])


if __name__ == "__main__":
    main()
