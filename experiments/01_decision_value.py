"""Experiment 01: value of information in the 32 flood and 16 social instances.

Section 5.2, Fig. 2.
"""
from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hevi_hitl import flood as fl
from hevi_hitl import social as so
from hevi_hitl.insight import WORKED_SOCIAL_ID, decision_value, flood_configs, save_json, social_configs
from hevi_hitl.plotting import FULL_WIDTH_IN, MARKERS, PALETTE, apply_style, panel_label, save

RESULTS = ROOT / "results"
COMPARED = ("joint", "physical_only", "preference_only", "one_step", "homogeneous")


def flatten(cid: int, grid: str, cfg, out: dict) -> dict:
    """One CSV row per configuration."""
    row = dict(grid=grid, configuration_id=cid, **{f"cfg_{k}": v for k, v in asdict(cfg).items()})
    row.update(out["decomposition"])
    row["perfect_information"] = out["perfect_information"]["score"]
    for name, prof in out["policies"].items():
        for key, value in prof.items():
            if key != "root_decision":
                row[f"{name}__{key}"] = value
    for name, diff in out["minus_immediate"].items():
        for key, value in diff.items():
            row[f"{name}_minus_immediate__{key}"] = value
    return row


def run_grid(grid: str) -> pd.DataFrame:
    rows = []
    configs = flood_configs() if grid == "flood" else social_configs()
    for cid, cfg in enumerate(configs):
        if grid == "flood":
            out = decision_value(fl.build_flood(cfg))
        else:
            out = decision_value(so.build_social_flood(cfg), so.build_social_flood(cfg, homogeneous=True))
        rows.append(flatten(cid, grid, cfg, out))
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, grid: str) -> dict:
    eps = 1e-10
    s = dict(
        configurations=len(df),
        joint_uses_information=int((df["joint__any_query"]).sum()),
        joint_uses_preference_queries=int((df["joint__preference_queries"] > eps).sum()),
        voi_joint_mean=df["voi_joint"].mean(), voi_joint_max=df["voi_joint"].max(),
        voi_joint_positive=int((df["voi_joint"] > eps).sum()),
        voi_physical_mean=df["voi_physical"].mean(), voi_preference_mean=df["voi_preference"].mean(),
        voi_preference_positive=int((df["voi_preference"] > eps).sum()),
        interaction_mean=df["interaction"].mean(), interaction_min=df["interaction"].min(),
        interaction_max=df["interaction"].max(),
        interaction_positive=int((df["interaction"] > eps).sum()),
        interaction_negative=int((df["interaction"] < -eps).sum()),
        evpi_mean=df["evpi"].mean(), evpi_min=df["evpi"].min(), evpi_max=df["evpi"].max(),
        evppi_preference_mean=df["evppi_preference"].mean(),
        evppi_physical_mean=df["evppi_physical"].mean(),
        captured_share_mean=df["captured_share"].mean(), captured_share_max=df["captured_share"].max(),
        one_step_regret_max=df["one_step_regret"].max(),
        one_step_regret_positive=int((df["one_step_regret"] > eps).sum()),
        joint_minus_immediate_damage_plus_intervention_eur_mean=df["joint_minus_immediate__damage_plus_intervention_eur"].mean(),
        joint_minus_immediate_damage_plus_intervention_eur_min=df["joint_minus_immediate__damage_plus_intervention_eur"].min(),
        joint_minus_immediate_damage_plus_intervention_eur_max=df["joint_minus_immediate__damage_plus_intervention_eur"].max(),
        joint_minus_immediate_person_days_mean=df["joint_minus_immediate__displacement_person_days"].mean(),
        joint_minus_immediate_exposed_people_mean=df["joint_minus_immediate__exposed_people"].mean(),
        joint_minus_immediate_exposed_people_min=df["joint_minus_immediate__exposed_people"].min(),
        joint_minus_immediate_exposed_people_max=df["joint_minus_immediate__exposed_people"].max(),
        cases_joint_raises_exposed_people=int((df["joint_minus_immediate__exposed_people"] > 1e-6).sum()),
        cases_joint_raises_damage_plus_intervention=int((df["joint_minus_immediate__damage_plus_intervention_eur"] > 1e-6).sum()),
    )
    if grid == "social":
        s.update(
            homogeneous_excess_mean=(df["homogeneous__score"] - df["joint__score"]).mean(),
            homogeneous_excess_max=(df["homogeneous__score"] - df["joint__score"]).max(),
            homogeneous_minus_joint_constrained_exposed_fraction_mean=(df["homogeneous__constrained_exposed_fraction"] - df["joint__constrained_exposed_fraction"]).mean(),
            homogeneous_minus_joint_constrained_exposed_fraction_max=(df["homogeneous__constrained_exposed_fraction"] - df["joint__constrained_exposed_fraction"]).max(),
            homogeneous_minus_joint_exposed_people_mean=(df["homogeneous__exposed_people"] - df["joint__exposed_people"]).mean(),
            homogeneous_minus_joint_damage_plus_intervention_eur_mean=(df["homogeneous__damage_plus_intervention_eur"] - df["joint__damage_plus_intervention_eur"]).mean(),
            joint_uses_social_check=int((df["joint__social_queries"] > eps).sum()),
            worked_case=df.loc[df.configuration_id == WORKED_SOCIAL_ID].iloc[0].to_dict(),
        )
    return s


def figure(flood: pd.DataFrame, social: pd.DataFrame) -> None:
    apply_style()
    fig = plt.figure(figsize=(FULL_WIDTH_IN, 4.85))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.3, 1], height_ratios=[1.15, 1],
                          hspace=0.66, wspace=0.42, left=0.115, right=0.985, top=0.855, bottom=0.12)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    sub = gs[1, :].subgridspec(1, 3, wspace=0.3)
    ax_c, ax_d, ax_e = (fig.add_subplot(sub[0, k]) for k in range(3))
    components = (("voi_physical", "physical_only", r"Risk only ($V_{\mathrm{R}}$)"),
                  ("voi_preference", "preference_only", r"Preference only ($V_\theta$)"),
                  ("interaction", "interaction", r"Interaction ($V_{\mathrm{R}\theta}$)"),
                  ("voi_joint", "joint", r"Joint ($V_{\mathrm{J}}$)"))
    marker = {"voi_physical": "s", "voi_preference": "^", "interaction": "D", "voi_joint": "o"}
    x = social.configuration_id.to_numpy()
    offsets = np.linspace(-0.27, 0.27, len(components))
    for (col, key, label), dx in zip(components, offsets):
        ax_a.scatter(x + dx, 1e3 * social[col], color=PALETTE[key], marker=marker[col], s=14,
                     label=label, zorder=3, edgecolor="none" if marker[col] != "x" else None)
    ax_a.axhline(0, color="black", lw=0.6)
    ax_a.set_xticks(x)
    ax_a.set_xlabel("Social instance")
    ax_a.set_ylabel("VOI\n(score units $\\times 10^{3}$)")
    ax_a.axvspan(WORKED_SOCIAL_ID - 0.45, WORKED_SOCIAL_ID + 0.45, color="0.9", zorder=0)
    ax_a.legend(ncol=2, loc="lower left", bbox_to_anchor=(0.04, 1.01), handletextpad=0.2, columnspacing=1.2)
    panel_label(ax_a, "a", x=-0.11, y=1.2)
    # (b) captured share of EVPI, both grids
    for k, (df, name) in enumerate(((flood, "Flood (n = 32)"), (social, "Social (n = 16)"))):
        rng = np.random.default_rng(1)
        jitter = rng.uniform(-0.15, 0.15, len(df))
        ax_b.scatter(k + jitter, 100 * df["captured_share"], color=PALETTE["joint"], marker="o",
                     s=10, alpha=0.75, edgecolor="none")
        ax_b.hlines(100 * df["captured_share"].mean(), k - 0.28, k + 0.28, color="black", lw=1.2)
    ax_b.set_xticks([0, 1], ["Flood\n(n = 32)", "Social\n(n = 16)"])
    ax_b.set_xlim(-0.6, 1.6)
    ax_b.set_ylim(bottom=0)
    ax_b.set_yticks([0, 4, 8, 12])
    ax_b.set_ylabel("Joint VOI captured\n(% of EVPI)")
    panel_label(ax_b, "b", x=-0.42, y=1.2)
    # (c-e) consequence differences, joint minus immediate (social grid)
    specs = ((ax_c, "damage_plus_intervention_eur", 1e-6, "Damage plus expenditure\n(million EUR)", "c"),
             (ax_d, "displacement_person_days", 1e-3, "Displacement\n(thousand person-days)", "d"),
             (ax_e, "exposed_people", 1.0, "Exposed people\n(persons)", "e"))
    for ax, col, scale, label, letter in specs:
        for name in ("joint", "homogeneous"):
            vals = scale * (social[f"{name}__{col}"] - social[f"immediate__{col}"])
            ax.scatter(vals, x, color=PALETTE[name], marker=MARKERS[name], s=13, zorder=3,
                       label={"joint": "Joint planning", "homogeneous": "Homogeneous-access planner"}[name])
        ax.axvline(0, color="black", lw=0.6)
        ax.axhspan(WORKED_SOCIAL_ID - 0.45, WORKED_SOCIAL_ID + 0.45, color="0.9", zorder=0)
        ax.set_yticks(x[::3])
        if letter != "c":
            ax.tick_params(labelleft=False)
        ax.invert_yaxis()
        ax.set_xlabel(label)
        panel_label(ax, letter, x=-0.24 if letter == "c" else -0.1, y=1.04)
    ax_c.set_ylabel("Social instance")
    ax_c.legend(loc="lower left", bbox_to_anchor=(0, 1.1), ncol=2, handletextpad=0.2,
                title="Difference from immediate commitment:", title_fontsize=8, alignment="left")
    save(fig, "decision_value")


def main() -> None:
    if "--figure-only" in sys.argv:
        figure(pd.read_csv(RESULTS / "decision_value_flood.csv"), pd.read_csv(RESULTS / "decision_value_social.csv"))
        return
    start = time.perf_counter()
    flood_df = run_grid("flood")
    social_df = run_grid("social")
    RESULTS.mkdir(exist_ok=True)
    flood_df.to_csv(RESULTS / "decision_value_flood.csv", index=False)
    social_df.to_csv(RESULTS / "decision_value_social.csv", index=False)
    save_json(RESULTS / "decision_value.json",
              dict(flood=summarise(flood_df, "flood"), social=summarise(social_df, "social")))
    figure(flood_df, social_df)
    print(f"finished in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
