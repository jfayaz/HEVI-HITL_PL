"""Experiment 02: 10,000 randomized instances of the social model.

Section 5.3, Fig. 3; Supplementary Section S1.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hevi_hitl import social as so
from hevi_hitl.insight import (per_state_cost, perfect_information, plan, policy_profile, revealed_value,
                               preference_axis, save_json)
from hevi_hitl.plotting import FULL_WIDTH_IN, PALETTE, apply_style, panel_label, save

RESULTS = ROOT / "results"
SEED = 2026
BOOT_SEED = 2027
FACTORS = {
    "lead_hours": "Lead time (h)",
    "original_reliability": "Response reliability",
    "social_query_reliability": "Access-check reliability",
    "log10_query_cost": "log$_{10}$ query cost",
    "p_high_severity": "P(highest severity)",
    "preference_concentration": "Preference-prior concentration",
    "mechanism_strength": "Social-mechanism strength",
    "log10_preference_slope": "log$_{10}$ preference slope",
}
RESPONSES = {"voi_joint": "Joint VOI", "voi_preference_used": "VOI from preference comparisons",
             "any_query": "P(any query)"}
P_HIGH_EDGES = np.round(np.arange(0.0, 1.0001, 0.05), 2)   # bins of Fig. 3(d)


def sample_parameters(n: int) -> list[dict]:
    """Draw all instances up front from one generator so the set is independent of N's use."""
    rng = np.random.default_rng(SEED)
    out = []
    for i in range(n):
        out.append(dict(
            instance=i,
            lead_hours=int(rng.integers(3, 9)),
            original_reliability=float(rng.uniform(0.5, 1.0)),
            social_query_reliability=float(rng.uniform(0.3, 1.0)),
            query_cost=float(10 ** rng.uniform(-4, np.log10(4e-3))),
            hazard_prior=tuple(float(x) for x in rng.dirichlet(np.ones(3))),
            preference_prior=tuple(float(x) for x in rng.dirichlet(np.ones(4))),
            mechanism_strength=float(rng.uniform(0, 1)),
            preference_slope=float(10 ** rng.uniform(1, np.log10(80))),
        ))
    return out


def solve_instance(p: dict) -> dict:
    t0 = time.perf_counter()
    cfg = so.SocialConfig(lead_hours=p["lead_hours"], mechanism_strength=p["mechanism_strength"],
                          social_query_reliability=p["social_query_reliability"],
                          preference_prior=p["preference_prior"], query_cost=p["query_cost"],
                          hazard_prior=p["hazard_prior"], original_reliability=p["original_reliability"],
                          preference_slope=p["preference_slope"])
    model = so.build_social_flood(cfg)
    joint = plan(model, "joint")
    prof = policy_profile(model, joint)
    values = {"joint": prof["score"]}
    for name in ("immediate", "physical_only", "preference_only", "one_step"):
        values[name] = float(model.prior @ per_state_cost(model, plan(model, name)))
    pi = perfect_information(model)["score"]
    eps = 1e-10
    assert all(values["joint"] <= v + eps for v in values.values()) and values["joint"] >= pi - eps
    pref = np.asarray(p["preference_prior"])
    entropy = -float(np.sum(pref * np.log(np.where(pref > 0, pref, 1))))
    row = dict(p)
    row.pop("hazard_prior")
    row.pop("preference_prior")
    row.update({f"hazard_prior_{k}": v for k, v in enumerate(p["hazard_prior"])})
    row.update({f"preference_prior_{k}": v for k, v in enumerate(p["preference_prior"])})
    row.update(
        log10_query_cost=np.log10(p["query_cost"]), log10_preference_slope=np.log10(p["preference_slope"]),
        p_high_severity=p["hazard_prior"][2], preference_concentration=1 - entropy / np.log(4),
        **{f"value_{k}": v for k, v in values.items()}, perfect_information=pi,
        evppi_preference_value=revealed_value(model, (preference_axis(model),)),
        root_kind=prof["root_decision"][0], root_index=prof["root_decision"][1],
        risk_queries=prof["risk_queries"], social_queries=prof["social_queries"],
        preference_queries=prof["preference_queries"], commitment_hours=prof["commitment_hours"],
        commit_without_query=float(not prof["any_query"]), any_query=float(prof["any_query"]),
        barrier_probability=prof["barrier_probability"],
        barrier_ever=float(prof["barrier_probability"] > eps),
    )
    row["voi_joint"] = values["immediate"] - values["joint"]
    row["voi_physical"] = values["immediate"] - values["physical_only"]
    row["voi_preference"] = values["immediate"] - values["preference_only"]
    # Value attributable to preference queries inside the joint plan (>=0).
    row["voi_preference_used"] = values["physical_only"] - values["joint"]
    row["interaction"] = row["voi_joint"] - row["voi_physical"] - row["voi_preference"]
    row["evpi"] = values["immediate"] - pi
    row["evppi_preference"] = values["immediate"] - row["evppi_preference_value"]
    row["captured_share"] = row["voi_joint"] / row["evpi"] if row["evpi"] > 0 else np.nan
    row["one_step_regret"] = values["one_step"] - values["joint"]
    row["solve_seconds"] = time.perf_counter() - t0
    return row


# Analysis
def standardized_regression(df: pd.DataFrame, y: str, rng: np.random.Generator, boots: int = 1000) -> dict:
    X = df[list(FACTORS)].to_numpy(float)
    Y = df[y].to_numpy(float)

    def fit(idx):
        x = X[idx]
        yy = Y[idx]
        xs = (x - x.mean(0)) / x.std(0)
        ys = (yy - yy.mean()) / yy.std() if yy.std() > 0 else yy * 0
        A = np.column_stack((np.ones(len(xs)), xs))
        beta = np.linalg.lstsq(A, ys, rcond=None)[0][1:]
        resid = ys - A @ np.linalg.lstsq(A, ys, rcond=None)[0]
        r2 = 1 - resid.var() / ys.var() if ys.var() > 0 else np.nan
        return beta, r2

    beta, r2 = fit(np.arange(len(Y)))
    boot = np.array([fit(rng.integers(0, len(Y), len(Y)))[0] for _ in range(boots)])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return dict(r2=r2, coefficients={f: dict(estimate=beta[k], ci_low=lo[k], ci_high=hi[k])
                                     for k, f in enumerate(FACTORS)})


def first_order_index(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Var(E[Y | binned X]) / Var(Y) with quantile bins (discrete factors use their values)."""
    if len(np.unique(x)) <= bins:
        labels = np.unique(x, return_inverse=True)[1]
    else:
        edges = np.quantile(x, np.linspace(0, 1, bins + 1))
        labels = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)
    if y.var() == 0:
        return 0.0
    means = np.array([y[labels == k].mean() for k in np.unique(labels)])
    weights = np.array([(labels == k).mean() for k in np.unique(labels)])
    return float(weights @ (means - y.mean()) ** 2 / y.var())


def sensitivity(df: pd.DataFrame, y: str, rng: np.random.Generator, boots: int = 1000) -> dict:
    Y = df[y].to_numpy(float)
    out = {}
    for f in FACTORS:
        x = df[f].to_numpy(float)
        est = first_order_index(x, Y)
        bs = [first_order_index(x[i], Y[i]) for i in (rng.integers(0, len(Y), len(Y)) for _ in range(boots))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[f] = dict(estimate=est, ci_low=float(lo), ci_high=float(hi))
    return out


def permutation_floor(df: pd.DataFrame, y: str, rng: np.random.Generator, perms: int = 200) -> dict:
    """95th percentile of each factor's first-order index when the response is randomly permuted."""
    Y = df[y].to_numpy(float)
    return {f: float(np.percentile([first_order_index(df[f].to_numpy(float), rng.permutation(Y))
                                    for _ in range(perms)], 95)) for f in FACTORS}


def first_decision(df: pd.DataFrame) -> pd.Series:
    """Root decision: commit, depth diagnostic or preference comparison (query indices of social.py)."""
    names = np.where(df.root_kind == "act", "commit",
                     np.where(df.root_index == 0, "depth", np.where(df.root_index.isin([5, 6]), "comparison", "other")))
    return pd.Series(names, index=df.index)


def sequencing(df: pd.DataFrame) -> dict:
    """First decision by P(highest severity) for instances with slack (lead of at least four hours)."""
    eps = 1e-10
    d = df.assign(first=first_decision(df))
    s = d[d.lead_hours >= 4]
    cut = pd.cut(s.p_high_severity, P_HIGH_EDGES.tolist(), include_lowest=True)
    table = s.groupby(cut, observed=False).agg(
        n=("instance", "size"), any_query=("any_query", "mean"),
        depth_first=("first", lambda x: float((x == "depth").mean())),
        comparison_first=("first", lambda x: float((x == "comparison").mean())))
    pref = d[d.preference_queries > eps]
    return dict(
        first_decision_counts=d["first"].value_counts().to_dict(),
        by_p_high_lead_ge_4={str(k): {c: float(v) for c, v in row.items()} for k, row in table.iterrows()},
        max_p_high_with_query_lead_ge_4=float(s.loc[s.any_query > 0, "p_high_severity"].max()),
        comparison_first=dict(n=int((d["first"] == "comparison").sum()),
                              p_high_median=float(d.loc[d["first"] == "comparison", "p_high_severity"].median()),
                              p_high_at_most_0_2=int(((d["first"] == "comparison") & (d.p_high_severity <= 0.2)).sum()),
                              without_risk_query=int(((d["first"] == "comparison") & (d.risk_queries <= eps)).sum())),
        interaction_sign_by_first_decision_when_comparisons_used={
            k: dict(n=int(len(g)), positive=int((g.interaction > eps).sum()), negative=int((g.interaction < -eps).sum()))
            for k, g in pref.assign(first=first_decision(pref)).groupby("first")},
        query_share_lead_ge_4_p_high_0_2_to_0_4=float(s.loc[(s.p_high_severity > 0.2) & (s.p_high_severity <= 0.4),
                                                            "any_query"].mean()),
        querying_above_p_high={f"{t:.2f}": dict(n=int((s.p_high_severity > t).sum()),
                                                querying=int(((s.p_high_severity > t) & (s.any_query > 0)).sum()))
                               for t in (0.5, 0.55, 0.6, 0.65, 0.7)},
    )


def reliability_bins(df: pd.DataFrame) -> pd.Series:
    return pd.cut(df["original_reliability"], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], include_lowest=True)


def heatmaps(df: pd.DataFrame) -> dict:
    d = df.assign(rel_bin=reliability_bins(df))
    out = {}
    for col in ("captured_share", "any_query", "preference_queries"):
        table = d.pivot_table(index="lead_hours", columns="rel_bin", values=col, aggfunc="mean", observed=False)
        out[col] = dict(index=table.index.tolist(), columns=[str(c) for c in table.columns],
                        values=table.to_numpy().tolist())
    counts = d.pivot_table(index="lead_hours", columns="rel_bin", values="instance", aggfunc="count", observed=False)
    out["count"] = dict(index=counts.index.tolist(), columns=[str(c) for c in counts.columns],
                        values=counts.to_numpy().tolist())
    return out


def rules(df: pd.DataFrame) -> dict:
    """Numerical support for decision rules (thresholds read off the sample)."""
    eps = 1e-10
    out = {}
    pref_used = df["preference_queries"] > eps
    out["n"] = len(df)
    out["share_any_query"] = df["any_query"].mean()
    out["share_preference_used"] = pref_used.mean()
    out["share_social_check_used"] = (df["social_queries"] > eps).mean()
    out["share_voi_preference_alone_positive"] = (df["voi_preference"] > eps).mean()
    out["share_interaction_positive"] = (df["interaction"] > eps).mean()
    out["share_interaction_negative"] = (df["interaction"] < -eps).mean()
    out["preference_used_by_lead"] = df.groupby("lead_hours").apply(
        lambda g: float((g["preference_queries"] > eps).mean()), include_groups=False).to_dict()
    out["any_query_by_lead"] = df.groupby("lead_hours")["any_query"].mean().to_dict()
    out["captured_share_by_lead"] = df.groupby("lead_hours")["captured_share"].mean().to_dict()
    out["min_lead_with_preference_query"] = int(df.loc[pref_used, "lead_hours"].min()) if pref_used.any() else None
    out["min_lead_with_any_query"] = int(df.loc[df.any_query > 0, "lead_hours"].min()) if (df.any_query > 0).any() else None
    # cost thresholds
    cost_bins = pd.cut(df["log10_query_cost"], np.linspace(-4, np.log10(4e-3), 6), include_lowest=True)
    out["any_query_by_cost_bin"] = {str(k): float(v) for k, v in df.groupby(cost_bins, observed=False)["any_query"].mean().items()}
    out["preference_used_by_cost_bin"] = {str(k): float(v) for k, v in df.groupby(cost_bins, observed=False).apply(
        lambda g: float((g["preference_queries"] > eps).mean()), include_groups=False).items()}
    out["max_cost_with_any_query"] = float(df.loc[df.any_query > 0, "query_cost"].max())
    out["max_cost_with_preference_query"] = float(df.loc[pref_used, "query_cost"].max()) if pref_used.any() else None
    # preference concentration
    conc_bins = pd.qcut(df["preference_concentration"], 4)
    out["preference_used_by_concentration_quartile"] = {str(k): float(v) for k, v in df.groupby(conc_bins, observed=False).apply(
        lambda g: float((g["preference_queries"] > eps).mean()), include_groups=False).items()}
    out["max_concentration_with_preference_query"] = float(df.loc[pref_used, "preference_concentration"].max()) if pref_used.any() else None
    out["max_top_preference_probability_with_preference_query"] = float(
        df.loc[pref_used, [f"preference_prior_{k}" for k in range(4)]].max(axis=1).max()) if pref_used.any() else None
    # one-step regret
    r = df["one_step_regret"]
    out["one_step_regret"] = dict(share_positive=float((r > eps).mean()), mean=float(r.mean()),
                                  p90=float(r.quantile(0.9)), max=float(r.max()),
                                  mean_relative_to_voi=float((r / df["voi_joint"].where(df.voi_joint > eps)).mean()),
                                  share_positive_by_lead=df.assign(pos=(r > eps)).groupby("lead_hours")["pos"].mean().to_dict(),
                                  share_positive_when_preference_used=float((r[pref_used] > eps).mean()) if pref_used.any() else None,
                                  share_positive_when_preference_not_used=float((r[~pref_used] > eps).mean()))
    out["captured_share"] = dict(mean=float(df.captured_share.mean()), median=float(df.captured_share.median()),
                                 max=float(df.captured_share.max()),
                                 mean_when_querying=float(df.loc[df.any_query > 0, "captured_share"].mean()))
    out["voi_preference_alone_positive_count"] = int((df["voi_preference"] > eps).sum())
    top = df[[f"preference_prior_{k}" for k in range(4)]].max(axis=1)
    out["preference_used_by_top_preference_probability"] = {
        str(k): dict(n=int(len(g)), share=float((g["preference_queries"] > eps).mean()))
        for k, g in df.groupby(pd.cut(top, [0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]), observed=False)}
    sev_bins = pd.cut(df["p_high_severity"], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    slack = df[df.lead_hours >= 4]
    out["any_query_by_p_high_bin_lead_ge_4"] = {str(k): float(v) for k, v in slack.groupby(
        pd.cut(slack["p_high_severity"], [0, 0.2, 0.4, 0.6, 0.8, 1.0]), observed=False)["any_query"].mean().items()}
    out["instances_by_p_high_bin"] = {str(k): int(v) for k, v in df.groupby(sev_bins, observed=False).size().items()}
    out["max_p_high_severity_with_any_query"] = float(df.loc[df.any_query > 0, "p_high_severity"].max())
    out["instances_lead_3"] = int((df.lead_hours == 3).sum())
    out["preference_used_without_standalone_preference_value"] = int(
        ((df["preference_queries"] > eps) & (df["voi_preference"] <= eps)).sum())
    out["preference_used_by_slope_quartile"] = {str(k): float(v) for k, v in df.groupby(
        pd.qcut(df["log10_preference_slope"], 4), observed=False).apply(
        lambda g: float((g["preference_queries"] > eps).mean()), include_groups=False).items()}
    out["barrier_ever_share"] = df["barrier_ever"].mean()
    out["commitment_hours_mean"] = df["commitment_hours"].mean()
    out["reliability_any_query"] = {str(k): float(v) for k, v in df.groupby(reliability_bins(df), observed=False)["any_query"].mean().items()}
    return out


def figure(df: pd.DataFrame, sens: dict, floor: dict) -> None:
    apply_style()
    W, H = FULL_WIDTH_IN, 3.86
    fig = plt.figure(figsize=(W, H))

    def box(left, bottom, width, height):
        """Axes rectangle in inches, so that every label stays inside the printed width."""
        return (left / W, bottom / H, width / W, height / H)

    d = df.assign(rel_bin=reliability_bins(df))
    leads = sorted(df.lead_hours.unique())
    bins = d.rel_bin.cat.categories
    cmap = plt.get_cmap("viridis")
    heat = (("captured_share", "VOI captured\n(% of EVPI)", None, 0.85),
            ("any_query", "Instances querying\nbefore committing (%)", 100, 3.62))
    for k, (col, label, vmax, left) in enumerate(heat):
        ax = fig.add_axes(box(left, 2.36, 1.72, 1.27))
        t = d.pivot_table(index="lead_hours", columns="rel_bin", values=col, aggfunc="mean", observed=False)
        t = t.reindex(index=leads, columns=bins) * 100
        im = ax.imshow(t.to_numpy().T, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        vals = t.to_numpy().T + 0.0
        hi = np.nanmax(vals) if vmax is None else vmax
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                if np.isfinite(vals[i, j]):
                    shown = 0.0 if abs(vals[i, j]) < 0.5 else vals[i, j]
                    ax.text(j, i, f"{shown:.0f}", ha="center", va="center", fontsize=7.5,
                            color="white" if vals[i, j] < 0.6 * hi else "black")
        ax.set_xticks(range(len(leads)), [str(x) for x in leads])
        ax.set_yticks(range(len(bins)), [f"{b.left:.1f}–{b.right:.1f}" for b in bins] if k == 0 else [])
        ax.set_xlabel("Lead time (h)")
        if k == 0:
            ax.set_ylabel("Response reliability")
        ax.grid(False)
        cax = fig.add_axes(box(left + 1.78, 2.36, 0.09, 1.27))
        cb = fig.colorbar(im, cax=cax)
        cb.set_ticks([0, 2, 4, 6, 8, 10] if vmax is None else [0, 20, 40, 60, 80, 100])
        cb.set_label(label, fontsize=8.5, linespacing=1.0)
        panel_label(ax, "ab"[k], x=-0.45 if k == 0 else -0.08, y=1.03)
    # (c) first-order sensitivity indices with 95% bootstrap intervals
    names = list(FACTORS)
    ypos = np.arange(len(names))
    styles = {"voi_joint": (PALETTE["joint"], "o", r"$V_{\mathrm{J}}$"),
              "voi_preference_used": (PALETTE["preference_only"], "^", r"$V_{\mathrm{J}}-V_{\mathrm{R}}$"),
              "any_query": (PALETTE["immediate"], "s", "Querying")}
    ax = fig.add_axes(box(1.86, 0.43, 1.42, 1.36))
    for m, (resp, (color, marker, label)) in enumerate(styles.items()):
        coef = sens[resp]
        est = np.array([coef[f]["estimate"] for f in names])
        lo = np.array([coef[f]["ci_low"] for f in names])
        hi = np.array([coef[f]["ci_high"] for f in names])
        off = (m - 1) * 0.25
        ax.errorbar(est, ypos + off, xerr=[np.maximum(est - lo, 0), np.maximum(hi - est, 0)], fmt=marker,
                    color=color, ms=3.5, elinewidth=0.9, capsize=1.5, label=label)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(ypos, [FACTORS[f] for f in names])
    ax.invert_yaxis()
    ax.set_xlabel("First-order index")
    ax.set_xlim(0, 0.27)
    ax.set_xticks([0, 0.1, 0.2])
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(s[2]) for s in styles.values()]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower right", handletextpad=0.3,
              borderaxespad=0.3, labelspacing=0.25)
    panel_label(ax, "c", x=-1.27, y=1.03)
    # (d) first decision by P(highest severity), instances with slack
    ax = fig.add_axes(box(3.97, 0.43, 2.23, 1.36))
    s = df.assign(first=first_decision(df))
    s = s[s.lead_hours >= 4]
    edges = P_HIGH_EDGES.tolist()
    cut = pd.cut(s.p_high_severity, edges, include_lowest=True)
    depth = s.groupby(cut, observed=False)["first"].apply(lambda x: (x == "depth").mean() * 100).to_numpy()
    comp = s.groupby(cut, observed=False)["first"].apply(lambda x: (x == "comparison").mean() * 100).to_numpy()
    left = np.array(edges[:-1])
    width = np.diff(edges)
    ax.bar(left, depth, width=width * 0.92, align="edge", color=PALETTE["physical_only"], label="Depth diagnostic")
    ax.bar(left, comp, width=width * 0.92, align="edge", bottom=depth, color=PALETTE["preference_only"],
           label="Comparison")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "0.2", "0.4", "0.6", "0.8", "1"])
    ax.set_ylim(0, 100)
    ax.set_xlabel("P(highest severity)")
    ax.set_ylabel("Instances (%)")
    ax.legend(loc="upper right", frameon=False, handlelength=1.2, title="First query", title_fontsize=8,
              alignment="left")
    panel_label(ax, "d", x=-0.2, y=1.03)
    save(fig, "when_information_pays")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--analysis-only", action="store_true", help="re-analyse the saved instances")
    parser.add_argument("--figure-only", action="store_true", help="redraw the figure from saved results")
    args = parser.parse_args()
    csv = RESULTS / "random_instances.csv"
    out = RESULTS / "random_instances.json"
    if args.figure_only:
        a = json.loads(out.read_text())
        figure(pd.read_csv(csv), a["first_order_sensitivity"], a["permutation_floor_95"])
        return
    if not args.analysis_only:
        params = sample_parameters(args.n)
        start = time.perf_counter()
        rows = []
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for k, row in enumerate(pool.map(solve_instance, params, chunksize=8)):
                    rows.append(row)
                    if (k + 1) % 500 == 0:
                        print(f"{k + 1} instances: {time.perf_counter() - start:.1f} s", flush=True)
        else:
            for k, p in enumerate(params):
                rows.append(solve_instance(p))
                if (k + 1) % 50 == 0:
                    print(f"{k + 1} instances: {time.perf_counter() - start:.1f} s", flush=True)
        pd.DataFrame(rows).to_csv(csv, index=False)
    df = pd.read_csv(csv)
    rng = np.random.default_rng(BOOT_SEED)
    reg = {y: standardized_regression(df, y, rng) for y in RESPONSES}
    sens = {y: sensitivity(df, y, rng) for y in RESPONSES}
    floor = {y: permutation_floor(df, y, rng) for y in RESPONSES}
    querying = df[df.any_query > 0]
    sens_given_query = sensitivity(querying, "voi_joint", rng, boots=300)
    floor_given_query = permutation_floor(querying, "voi_joint", rng)
    r = rules(df)
    r["sequencing"] = sequencing(df)
    save_json(out, dict(n=len(df), heatmaps=heatmaps(df), regression=reg, first_order_sensitivity=sens,
                        permutation_floor_95=floor,
                        voi_joint_given_query=dict(n=int(len(querying)), first_order_sensitivity=sens_given_query,
                                                   permutation_floor_95=floor_given_query),
                        rules=r))
    figure(df, sens, floor)


if __name__ == "__main__":
    main()
