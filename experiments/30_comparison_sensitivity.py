"""Experiment 30: when comparisons pay in the randomized instances.

Section 5.3; Supplementary Section S1, Fig. S1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hevi_hitl import plotting as P

RESULTS = ROOT / "results"
EPS = 1e-10


def binned(df, column, edges, labels=None):
    cut = pd.cut(df[column], edges, include_lowest=True, right=True)
    g = df.groupby(cut, observed=True)
    out = pd.DataFrame(dict(
        n=g.size(),
        uses=g.uses.mean(), standalone=g.standalone.mean(), interaction=g.positive_interaction.mean(),
        added=g.added.mean()))
    out["centre"] = [0.5 * (iv.left + iv.right) for iv in out.index]
    return out.reset_index(drop=True)


def main():
    df = pd.read_csv(RESULTS / "random_instances.csv")
    df = df[df.lead_hours >= 4].copy()
    df["uses"] = df.preference_queries > EPS
    df["standalone"] = df.voi_preference > EPS
    df["positive_interaction"] = df.interaction > EPS
    df["added"] = (df.voi_joint - df.voi_physical) * 1e3
    df["top_type"] = df[[f"preference_prior_{k}" for k in range(4)]].max(axis=1)
    panels = [
        ("original_reliability", np.linspace(0.5, 1.0, 6), "Response reliability $\\rho$", None, None),
        ("log10_query_cost", np.linspace(-4, np.log10(4e-3), 6), "Query cost per half hour ($\\times10^{-3}$)",
         "log", [0.1, 0.3, 1, 3]),
        ("lead_hours", np.arange(3.5, 8.6, 1.0), "Lead time (h)", None, None),
        ("top_type", np.array([0.25, 0.4, 0.55, 0.7, 0.85, 1.0]), "Prior of most likely type", None, None),
        ("log10_preference_slope", np.linspace(1, np.log10(80), 6), "Response slope $\\zeta$", "log",
         [10, 20, 40, 80]),
        ("p_high_severity", np.linspace(0, 1, 11), "Prior of highest severity", None, None),
    ]
    P.apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(P.FULL_WIDTH_IN, 4.1), sharey=True, constrained_layout=True)
    summary = {}
    series = (("uses", "Comparison asked", P.PALETTE["joint"], "o"),
              ("interaction", "$V_{\\mathrm{R}\\theta}>0$", P.PALETTE["interaction"], "s"),
              ("standalone", "$V_\\theta>0$", P.PALETTE["preference_only"], "^"))
    for k, (ax, (col, edges, label, scale, ticks)) in enumerate(zip(axes.ravel(), panels)):
        b = binned(df, col, edges)
        x = 10 ** b.centre if scale == "log" else b.centre
        shown = x * 1e3 if col == "log10_query_cost" else x
        for key, name, colour, marker in series:
            ax.plot(shown, 100 * b[key], color=colour, marker=marker, label=name)
        if scale == "log":
            ax.set_xscale("log")
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}" for t in ticks])
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel(label)
        ax.set_ylim(0, 40)
        P.panel_label(ax, "abcdef"[k], x=-0.2 if k % 3 == 0 else -0.08)
        summary[col] = b.assign(centre=x).to_dict(orient="list")
    for ax in axes[:, 0]:
        ax.set_ylabel("Instances (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3)
    P.save(fig, "comparison_sensitivity")
    summary["instances"] = int(len(df))
    summary["overall"] = dict(uses=float(df.uses.mean()), standalone=float(df.standalone.mean()),
                              positive_interaction=float(df.positive_interaction.mean()),
                              mean_added=float(df.added.mean()))
    (RESULTS / "comparison_sensitivity.json").write_text(json.dumps(summary, indent=1, default=float),
                                                         encoding="utf-8")
    print(json.dumps(summary["overall"], indent=1))
    for col, b in summary.items():
        if isinstance(b, dict) and "centre" in b:
            print(col, [f"{c:.3g}:{100 * u:.0f}%/{100 * i:.0f}%/{100 * s:.0f}% (n={n})"
                        for c, u, i, s, n in zip(b["centre"], b["uses"], b["interaction"], b["standalone"], b["n"])])


if __name__ == "__main__":
    main()
