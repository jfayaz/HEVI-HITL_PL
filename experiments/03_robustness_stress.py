"""Experiment 03: misspecified response laws and priors, and stress-weighted planning.

Sections 5.4 and 5.5; Supplementary Section S14, Fig. S3.
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from hevi_hitl import social as so
from hevi_hitl.insight import (CONSEQUENCES, WORKED_SOCIAL_ID, misspecification, plan, save_json, social_configs,
                               stress_policy, stress_profile)
from hevi_hitl.plotting import FULL_WIDTH_IN, MARKERS, PALETTE, apply_style, panel_label, save

RESULTS = ROOT / "results"
RELIABILITIES = (0.6, 0.75, 0.9, 1.0)
SLOPES = (10.0, 20.0, 40.0, 80.0)
LAMBDAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
FINE_LAMBDAS = tuple(np.round(np.r_[np.arange(0, 1.0001, 0.05), 1.5, 2, 4, 8, 16], 4))
EPS = 1e-10

_models: dict = {}


def build(cfg: so.SocialConfig):
    if cfg not in _models:
        _models[cfg] = so.build_social_flood(cfg)
    return _models[cfg]


def misspecification_cases(cfg: so.SocialConfig) -> list[tuple[str, str, str, so.SocialConfig, so.SocialConfig]]:
    """(family, planned label, true label, plan cfg, true cfg)."""
    cases = []
    for rp in RELIABILITIES:
        for rt in RELIABILITIES:
            cases.append(("reliability", f"{rp:g}", f"{rt:g}", replace(cfg, original_reliability=rp),
                          replace(cfg, original_reliability=rt)))
    for st in SLOPES:
        cases.append(("preference_slope", "40", f"{st:g}", replace(cfg, preference_slope=40.0),
                      replace(cfg, preference_slope=st)))
    for a, b in (("diffuse", "economic_concentrated"), ("economic_concentrated", "diffuse")):
        cases.append(("preference_prior", a, b, replace(cfg, preference_prior=a), replace(cfg, preference_prior=b)))
    for a, b in (("mild", "elevated"), ("elevated", "mild")):
        cases.append(("hazard_prior", a, b, replace(cfg, hazard_prior=a), replace(cfg, hazard_prior=b)))
    for a, b in ((0.5, 1.0), (1.0, 0.5)):
        cases.append(("mechanism_strength", f"{a:g}", f"{b:g}", replace(cfg, mechanism_strength=a),
                      replace(cfg, mechanism_strength=b)))
    return cases


def run_misspecification() -> pd.DataFrame:
    rows = []
    caches: dict = {}
    for cid, cfg in enumerate(social_configs()):
        for family, planned, true, pcfg, tcfg in misspecification_cases(cfg):
            out = misspecification(build(tcfg), build(pcfg), caches.setdefault(tcfg, {}))
            rows.append(dict(configuration_id=cid, family=family, planned=planned, true=true,
                             correctly_specified=pcfg == tcfg, **out))
        _models.clear()
    return pd.DataFrame(rows)


def run_stress() -> tuple[pd.DataFrame, list]:
    rows, audits = [], []
    for cid, cfg in enumerate(social_configs()):
        model = so.build_social_flood(cfg)
        immediate = stress_profile(model, plan(model, "immediate"))
        lambdas = FINE_LAMBDAS if cid == WORKED_SOCIAL_ID else LAMBDAS
        for lam in lambdas:
            policy, audit = stress_policy(model, float(lam))
            audits.append(dict(configuration_id=cid, **audit))
            prof = stress_profile(model, policy)
            row = dict(configuration_id=cid, lam=float(lam), in_main_grid=float(lam) in LAMBDAS,
                       identity_error=max(audit["identity_error_solver"], audit["identity_error_policy"]))
            for scope in ("original", "severe_footprint_0", "severe_footprint_1", "severe_any_footprint"):
                for key in ("score",) + CONSEQUENCES:
                    row[f"{scope}__{key}"] = prof[scope][key]
                    row[f"{scope}__{key}__minus_immediate"] = prof[scope][key] - immediate[scope][key]
            row["risk_queries"] = prof["original"]["risk_queries"]
            row["preference_queries"] = prof["original"]["preference_queries"]
            row["root_decision"] = str(prof["original"]["root_decision"])
            rows.append(row)
    df = pd.DataFrame(rows)
    base = df[df.lam == 0].set_index("configuration_id")["original__score"]
    df["score_sacrifice_vs_lambda0"] = df["original__score"] - df.configuration_id.map(base)
    return df, audits


def stress_summary(df: pd.DataFrame) -> dict:
    """Per configuration: smallest lambda removing the severe-scenario disadvantage vs immediate."""
    out = []
    for cid, g in df[df.in_main_grid].groupby("configuration_id"):
        g = g.sort_values("lam")
        l0 = g.iloc[0]
        disadv = lambda r: (r["severe_footprint_0__score__minus_immediate"] > EPS
                            or r["severe_footprint_1__score__minus_immediate"] > EPS)
        disadv_eur = lambda r: (r["severe_footprint_0__damage_plus_intervention_eur__minus_immediate"] > 1e-6
                                or r["severe_footprint_1__damage_plus_intervention_eur__minus_immediate"] > 1e-6)
        ok = g[[not disadv(r) for _, r in g.iterrows()]]
        ok_eur = g[[not disadv_eur(r) for _, r in g.iterrows()]]
        first = ok.iloc[0] if len(ok) else None
        first_eur = ok_eur.iloc[0] if len(ok_eur) else None
        out.append(dict(
            configuration_id=int(cid),
            lambda0_severe_fp0_score_minus_immediate=l0["severe_footprint_0__score__minus_immediate"],
            lambda0_severe_fp1_score_minus_immediate=l0["severe_footprint_1__score__minus_immediate"],
            lambda0_severe_fp0_eur_minus_immediate=l0["severe_footprint_0__damage_plus_intervention_eur__minus_immediate"],
            lambda0_severe_fp1_eur_minus_immediate=l0["severe_footprint_1__damage_plus_intervention_eur__minus_immediate"],
            lambda0_severe_fp0_exposed_minus_immediate=l0["severe_footprint_0__exposed_people__minus_immediate"],
            lambda0_severe_fp1_exposed_minus_immediate=l0["severe_footprint_1__exposed_people__minus_immediate"],
            lambda0_has_severe_score_disadvantage=bool(disadv(l0)),
            lambda0_has_severe_eur_disadvantage=bool(disadv_eur(l0)),
            min_lambda_no_severe_score_disadvantage=None if first is None else first["lam"],
            score_sacrifice_at_that_lambda=None if first is None else first["score_sacrifice_vs_lambda0"],
            score_sacrifice_share_of_voi=None if first is None else
            first["score_sacrifice_vs_lambda0"] / -l0["original__score__minus_immediate"],
            policy_at_that_lambda_is_immediate=None if first is None else
            bool(abs(first["original__score__minus_immediate"]) < EPS),
            min_lambda_no_severe_eur_disadvantage=None if first_eur is None else first_eur["lam"],
            score_sacrifice_at_eur_lambda=None if first_eur is None else first_eur["score_sacrifice_vs_lambda0"],
            lambda0_minus_immediate_original_score=l0["original__score__minus_immediate"],
            stress_policy_equals_immediate_score_at_lambda16=abs(g.iloc[-1]["original__score__minus_immediate"]) < EPS,
        ))
    return dict(per_configuration=out)


# Figure
def figure(mis: pd.DataFrame, stress: pd.DataFrame) -> None:
    apply_style()
    W, H = FULL_WIDTH_IN, 4.52
    fig = plt.figure(figsize=(W, H))

    def box(left, bottom, width, height):
        """Axes rectangle in inches, so that every label stays inside the printed width."""
        return (left / W, bottom / H, width / W, height / H)

    # (a) reliability misspecification: mean (true score of misspecified plan - true immediate);
    # every cell is annotated, so the diverging colour scale needs no colour bar.
    ax = fig.add_axes(box(0.70, 2.70, 1.95, 1.40))
    rel = mis[mis.family == "reliability"]
    t = rel.pivot_table(index="planned", columns="true", values="versus_immediate", aggfunc="mean") * 1e3
    order = [f"{r:g}" for r in RELIABILITIES]
    t = t.reindex(index=order, columns=order)
    vals = t.to_numpy() + 0.0  # avoid printing -0.00
    lim = float(np.nanmax(np.abs(vals)))
    ax.imshow(vals, origin="lower", cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    worse = rel.assign(w=rel.versus_immediate > EPS).pivot_table(index="planned", columns="true", values="w",
                                                                 aggfunc="sum").reindex(index=order, columns=order)
    for i in range(len(order)):
        for j in range(len(order)):
            v = 0.0 if abs(vals[i, j]) < 5e-3 else vals[i, j]
            ax.text(j, i, f"{v:.2f}".replace("-", "\u2212") + f"\n({int(worse.to_numpy()[i, j])}/16)", ha="center",
                    va="center", fontsize=7.5, linespacing=1.1, color="white" if abs(v) > 0.6 * lim else "black")
    ax.set_xticks(range(4), order)
    ax.set_yticks(range(4), order)
    ax.set_xlabel("True reliability")
    ax.set_ylabel("Planned reliability")
    ax.grid(False)
    panel_label(ax, "a", x=-0.33, y=1.03)
    # (b) other misspecification families across the 16 configurations
    ax = fig.add_axes(box(4.86, 2.70, 1.36, 1.40))
    other = mis[(mis.family != "reliability") & ~mis.correctly_specified]
    labels = []
    abbrev = {"economic_concentrated": "economic"}
    short = {"preference_slope": "Preference slope", "preference_prior": "Preference prior",
             "hazard_prior": "Hazard prior", "mechanism_strength": "Mechanism strength"}
    for k, ((fam, p, tr), g) in enumerate(other.groupby(["family", "planned", "true"], sort=False)):
        labels.append(f"{short[fam]} {abbrev.get(p, p)}→{abbrev.get(tr, tr)}")
        ax.scatter(1e3 * g.regret, np.full(len(g), k) - 0.17, color=PALETTE["regret"], marker="o", s=10,
                   label="Regret relative to the true optimum" if k == 0 else None, zorder=3)
        ax.scatter(1e3 * g.versus_planned_immediate, np.full(len(g), k) + 0.17, color=PALETTE["immediate"],
                   marker="x", s=14, label="Score minus immediate commitment (same model)" if k == 0 else None,
                   zorder=3)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xticks([-20, 0, 20, 40])
    ax.set_xlabel("True score difference ($\\times 10^{3}$)", loc="right")
    ax.legend(loc="lower right", bbox_to_anchor=(1.03, 1.0), ncol=1, handletextpad=0.1, borderaxespad=0.1,
              labelspacing=0.25)
    panel_label(ax, "b", x=-1.43, y=1.03)
    # (c, d) stress frontier in configuration 10 (differences from immediate commitment)
    s10 = stress[stress.configuration_id == WORKED_SOCIAL_ID].sort_values("lam")
    first_immediate = s10[np.abs(s10["original__score__minus_immediate"]) < EPS].lam.min()
    for k, (col, scale, ylabel, left) in enumerate((("damage_plus_intervention_eur", 1e-6,
                                                     "Damage plus\nexpenditure\n(million EUR)", 0.86),
                                                    ("exposed_people", 1.0, "Exposed people", 3.96))):
        ax = fig.add_axes(box(left, 0.62, 2.22, 1.45))
        x = 1e3 * s10["original__score__minus_immediate"]
        for fp, ls, mk in ((0, "-", "o"), (1, "--", "s")):
            y = scale * s10[f"severe_footprint_{fp}__{col}__minus_immediate"]
            ax.plot(x, y, ls=ls, marker=mk, color=PALETTE["joint"], ms=3.5,
                    label=f"Severe footprint {fp + 1}")
        r0 = s10.iloc[0]
        # Label the upper of the two λ = 0 points, to its right, where no curve passes.
        top = max(r0[f"severe_footprint_{fp}__{col}__minus_immediate"] for fp in (0, 1))
        ax.annotate("λ=0", (1e3 * r0["original__score__minus_immediate"], scale * top),
                    textcoords="offset points", xytext=(6, 0), va="center", fontsize=8)
        ax.annotate(f"λ≥{first_immediate:g}", (0, 0), textcoords="offset points", xytext=(-7, -4), fontsize=8,
                    ha="right", va="top")
        ax.scatter([0], [0], color=PALETTE["immediate"], marker=MARKERS["immediate"], s=30, zorder=4,
                   label="Immediate commitment")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlabel("Expected score minus immediate\ncommitment ($\\times 10^{3}$)")
        ax.set_ylabel(ylabel, linespacing=1.05)
        ax.set_ylim(bottom=-0.13 * ax.get_ylim()[1])
        panel_label(ax, "cd"[k], x=-0.36 if k == 0 else -0.26, y=1.03)
        if k == 1:
            ax.legend(loc="upper right", handlelength=1.8, labelspacing=0.3, borderaxespad=0.3)
    save(fig, "robustness_stress")


def main() -> None:
    if "--figure-only" in sys.argv:
        figure(pd.read_csv(RESULTS / "misspecification.csv"), pd.read_csv(RESULTS / "stress_weighting.csv"))
        return
    start = time.perf_counter()
    mis = run_misspecification()
    mis.to_csv(RESULTS / "misspecification.csv", index=False)
    stress, audits = run_stress()
    stress.to_csv(RESULTS / "stress_weighting.csv", index=False)
    per = stress_summary(stress)["per_configuration"]
    wrong = mis[~mis.correctly_specified]
    rel = mis[mis.family == "reliability"]
    fam = {}
    for (f, p, tr), g in mis.groupby(["family", "planned", "true"], sort=False):
        fam[f"{f}:{p}->{tr}"] = dict(mean_regret=g.regret.mean(), max_regret=g.regret.max(),
                                     mean_versus_immediate=g.versus_immediate.mean(),
                                     max_versus_immediate=g.versus_immediate.max(),
                                     worse_than_immediate=int((g.versus_immediate > EPS).sum()),
                                     mean_versus_planned_immediate=g.versus_planned_immediate.mean(),
                                     worse_than_planned_immediate=int((g.versus_planned_immediate > EPS).sum()))
    s10 = stress[(stress.configuration_id == WORKED_SOCIAL_ID)].sort_values("lam")
    save_json(RESULTS / "robustness_stress.json", dict(
        misspecification=dict(
            cases=fam,
            misspecified_cases=len(wrong),
            misspecified_worse_than_immediate=int((wrong.versus_immediate > EPS).sum()),
            misspecified_worse_than_assumed_model_immediate=int((wrong.versus_planned_immediate > EPS).sum()),
            max_regret=wrong.regret.max(),
            max_versus_immediate=wrong.versus_immediate.max(),
            reliability_overconfident_worse_than_immediate=int(((rel.planned.astype(float) > rel.true.astype(float))
                                                               & (rel.versus_immediate > EPS)).sum()),
            reliability_underconfident_worse_than_immediate=int(((rel.planned.astype(float) < rel.true.astype(float))
                                                                & (rel.versus_immediate > EPS)).sum())),
        stress=dict(
            max_identity_error=max(max(a["identity_error_solver"], a["identity_error_policy"]) for a in audits),
            configs_with_lambda0_severe_score_disadvantage=sum(r["lambda0_has_severe_score_disadvantage"] for r in per),
            configs_with_lambda0_severe_eur_disadvantage=sum(r["lambda0_has_severe_eur_disadvantage"] for r in per),
            min_lambda_no_severe_score_disadvantage={r["configuration_id"]: r["min_lambda_no_severe_score_disadvantage"] for r in per},
            score_sacrifice={r["configuration_id"]: r["score_sacrifice_at_that_lambda"] for r in per},
            worked_case_frontier=s10[["lam", "original__score", "original__score__minus_immediate",
                                      "severe_footprint_0__damage_plus_intervention_eur__minus_immediate",
                                      "severe_footprint_1__damage_plus_intervention_eur__minus_immediate",
                                      "severe_footprint_0__exposed_people__minus_immediate",
                                      "severe_footprint_1__exposed_people__minus_immediate",
                                      "risk_queries", "preference_queries", "root_decision"]].to_dict(orient="records"))))
    figure(mis, stress)
    print(f"finished in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
