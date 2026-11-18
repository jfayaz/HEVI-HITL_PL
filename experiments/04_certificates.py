"""Experiment 04: regret certificates, guided refinement and decision relevance in the designed instances.

Section 5.6, Fig. 5; Supplementary Section S6.
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
from dataclasses import asdict
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from hevi_hitl import certificates as C
from hevi_hitl import flood as fl
from hevi_hitl import social as so
from hevi_hitl.model import PortfolioSolver, certificates as prop1_certificates
from hevi_hitl.reduction import aggregate_partition, signature_hierarchy

RESULTS = ROOT / "results"
TOLERANCES = (0.001, 0.0025, 0.005, 0.01, 0.025)
# class counts: flood in steps of 12 (plus one class), social in steps of 48
CLASS_COUNTS = {"flood": (1,) + tuple(range(12, 193, 12)), "social": tuple(range(48, 769, 48))}
# order of panel (d): hazard, exposure, vulnerability, impact, preferences
HEVI_ORDER = ("severity", "footprint", "occupancy", "physical_vulnerability", "evacuation_access",
              "recovery_access", "restoration", "preference_type")
VOI_FRACTIONS = (0.1, 0.25, 0.5)
COORDINATES = {
    "flood": ("severity", "footprint", "occupancy", "physical_vulnerability", "restoration",
              "preference_type"),
    "social": ("severity", "footprint", "occupancy", "physical_vulnerability", "restoration",
               "preference_type", "evacuation_access", "recovery_access"),
}
BOUNDS = ("prop1_uniform", "prop1_postsolve", "U_uniform", "U_postsolve", "U_lb", "refined_uniform",
          "refined_postsolve", "refined_lb")
FEATURES = ("property_and_intervention_eur", "person_days", "exposed_people", "intervention_eur")


def flood_configs():
    grid = fl.GRID
    return [fl.FloodConfig(**dict(zip(grid, values))) for values in product(*grid.values())]


def social_configs():
    grid = so.GRID
    return [so.SocialConfig(**dict(zip(grid, values))) for values in product(*grid.values())]


def build(domain, cfg):
    return fl.build_flood(cfg) if domain == "flood" else so.build_social_flood(cfg)


def physical(features) -> list:
    """Damage+intervention EUR, person-days, exposed people, intervention EUR."""
    f = np.asarray(features, dtype=float)
    return [float(f[0] + f[3]), float(f[1]), float(f[2]), float(f[3])]


def candidate(model, inverse, U, sets, cmax, lower, optimum, J_all, J_U, with_tilde=True):
    """Aggregate, solve, evaluate and certify one partition."""
    tick = time.perf_counter()
    reduced = aggregate_partition(model, inverse)
    policy = PortfolioSolver(reduced)
    policy.solve()
    solve_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    value, counts, features = PortfolioSolver(model).evaluate(policy)
    surrogate = PortfolioSolver(reduced).evaluate(policy)[0]
    U_tilde = None
    tilde_ok = None
    if with_tilde:
        U_tilde, _ = C.undominated_endpoints(reduced)
        tilde_ok = not bool((C.policy_endpoints(model, policy) & ~U_tilde).any())
    cert = C.refined_certificate(model, reduced, inverse, value, surrogate, U, U_tilde if tilde_ok else None,
                                 lower_bound=lower, sets=sets, cmax=cmax, J_all=J_all, J_U=J_U)
    cert_seconds = time.perf_counter() - tick
    prop1 = prop1_certificates(model, reduced, inverse, policy)
    assert abs(prop1["uniform_regret_bound"] - cert["prop1_uniform"]) < 1e-10
    assert abs(prop1["postsolve_regret_bound"] - cert["prop1_postsolve"]) < 1e-10
    regret = float(value - optimum["value"])
    cert.update(classes=int(inverse.max() + 1), regret=regret,
                U_tilde_endpoints=None if U_tilde is None else int(U_tilde.sum()),
                U_tilde_actions=None if U_tilde is None else int(U_tilde.any(axis=0).sum()),
                policy_within_U_tilde=tilde_ok, solve_seconds=solve_seconds, certify_seconds=cert_seconds,
                expected_queries=float(np.sum(counts)),
                consequence_change=[a - b for a, b in zip(physical(features), optimum["physical"])])
    for name in BOUNDS:
        if name in cert:
            assert cert[name] >= regret - 1e-10, (name, cert[name], regret)
    return cert


def analyse(task):
    """All certificate computations for one configuration (runs in a worker)."""
    domain, cid, cfg = task
    start = time.perf_counter()
    model = build(domain, cfg)
    full = PortfolioSolver(model)
    value = full.solve()[0]
    _, _, features = full.evaluate(full)
    optimum = dict(value=float(value), physical=physical(features))
    immediate_value = float(PortfolioSolver(model, method="immediate").solve()[0])
    tick = time.perf_counter()
    U, ustats = C.undominated_endpoints(model)
    u_seconds = time.perf_counter() - tick
    opt_endpoints = C.policy_endpoints(model, full)
    outside = opt_endpoints & ~U
    # Tie-aware check: an optimal-policy endpoint outside U must tie with a U endpoint.
    sets, cmax = C.feasible_query_sets(model)
    E = C.endpoint_mask(model)
    J_all = C.endpoint_columns(model, E)
    J_U = C.endpoint_columns(model, U)
    lower = C.perfect_information_bound(model)
    meta = dict(domain=domain, configuration_id=cid, config=asdict(cfg), states=len(model.prior),
                optimal_value=optimum["value"], optimal_physical=optimum["physical"],
                immediate_value=immediate_value, voi_joint=immediate_value - optimum["value"],
                perfect_information_bound=lower, E_endpoints=int(E.sum()), E_actions=int(E.any(axis=0).sum()),
                U_endpoints=int(U.sum()), U_actions=int(U.any(axis=0).sum()), U_stats=ustats,
                U_seconds=u_seconds, optimal_endpoints_outside_U=int(outside.sum()),
                query_sets=sets.astype(int).tolist(), cmax=cmax)
    rows = []
    hierarchies = {}
    tick = time.perf_counter()
    hierarchies["signature"] = signature_hierarchy(model, CLASS_COUNTS[domain])
    meta["signature_seconds"] = time.perf_counter() - tick
    tick = time.perf_counter()
    hierarchies["guided"] = C.guided_hierarchy(model, CLASS_COUNTS[domain], J_U=J_U, sets=sets, cmax=cmax)
    meta["guided_seconds"] = time.perf_counter() - tick
    for method, hierarchy in hierarchies.items():
        for k in CLASS_COUNTS[domain]:
            cert = candidate(model, hierarchy[min(k, len(model.prior))], U, sets, cmax, lower, optimum, J_all, J_U)
            rows.append(dict(domain=domain, configuration_id=cid, method=method, **cert))
    relevance = []
    names = COORDINATES[domain]
    for subset in C.coordinate_subsets(len(names), pairs=True):
        inverse = C.marginal_partition(model.indices, subset)
        # DRI guard: the answers of the queries that observe the averaged coordinates, revealed at the warning
        guard = max(lower, C.query_relaxation_bound(model, inverse))
        cert = candidate(model, inverse, U, sets, cmax, guard, optimum, J_all, J_U, with_tilde=False)
        cert["class_level_guarded"] = float(min(cert["refined_postsolve"], cert["original_policy_value"] - lower))
        relevance.append(dict(domain=domain, configuration_id=cid, coordinates="+".join(names[j] for j in subset),
                              order=len(subset), **cert))
    meta["seconds"] = time.perf_counter() - start
    return dict(meta=meta, rows=rows, relevance=relevance)


def smallest(rows, bound, tol):
    ok = [r["classes"] for r in rows if r[bound] <= tol]
    return min(ok) if ok else None


def summarise(metas, rows, relevance):
    df = pd.DataFrame(rows)
    rel = pd.DataFrame(relevance)
    summary = dict(tolerances=list(TOLERANCES), class_counts={k: list(v) for k, v in CLASS_COUNTS.items()})
    slack = {b: float((df[b] - df["regret"]).min()) for b in BOUNDS}
    slack["relevance_refined_postsolve"] = float((rel["refined_postsolve"] - rel["regret"]).min())
    slack["relevance_refined_lb"] = float((rel["refined_lb"] - rel["regret"]).min())
    summary["minimum_slack"] = slack
    summary["all_bounds_valid"] = bool(min(slack.values()) >= -1e-10)
    summary["U"] = {d: dict(
        E_actions=sorted({m["E_actions"] for m in metas if m["domain"] == d}),
        U_actions_min=min(m["U_actions"] for m in metas if m["domain"] == d),
        U_actions_median=float(np.median([m["U_actions"] for m in metas if m["domain"] == d])),
        U_actions_max=max(m["U_actions"] for m in metas if m["domain"] == d),
        U_endpoints_median=float(np.median([m["U_endpoints"] for m in metas if m["domain"] == d])),
        E_endpoints_median=float(np.median([m["E_endpoints"] for m in metas if m["domain"] == d])),
        optimal_endpoints_outside_U=sum(m["optimal_endpoints_outside_U"] for m in metas if m["domain"] == d),
        median_U_seconds=float(np.median([m["U_seconds"] for m in metas if m["domain"] == d])))
        for d in CLASS_COUNTS}
    nontrivial = df[df["classes"] < df["domain"].map({"flood": 192, "social": 768})]
    ratio = {}
    for d in CLASS_COUNTS:
        sub = nontrivial[(nontrivial["domain"] == d) & (nontrivial["prop1_postsolve"] > 1e-12)]
        ratio[d] = dict(
            U_over_prop1_postsolve_median=float((sub["U_postsolve"] / sub["prop1_postsolve"]).median()),
            U_over_prop1_postsolve_min=float((sub["U_postsolve"] / sub["prop1_postsolve"]).min()),
            U_error_over_prop1_error_median=float((sub["U_error"] / sub["prop1_error"]).median()),
            U_uniform_over_prop1_uniform_median=float((sub["U_uniform"] / sub["prop1_uniform"]).median()),
            refined_over_prop1_postsolve_median=float((sub["refined_postsolve"] / sub["prop1_postsolve"]).median()),
            refined_over_prop1_postsolve_min=float((sub["refined_postsolve"] / sub["prop1_postsolve"]).min()),
            refined_over_prop1_postsolve_max=float((sub["refined_postsolve"] / sub["prop1_postsolve"]).max()),
            refined_lb_over_prop1_postsolve_median=float((sub["refined_lb"] / sub["prop1_postsolve"]).median()),
            refined_lb_over_prop1_postsolve_min=float((sub["refined_lb"] / sub["prop1_postsolve"]).min()),
            refined_uniform_over_prop1_uniform_median=float((sub["refined_uniform"] / sub["prop1_uniform"]).median()),
            refined_lb_over_refined_postsolve_median=float((sub["refined_lb"] / sub["refined_postsolve"]).median()),
            lb_binds_U_cases=int(sub["lb_binds_U"].sum()),
            lb_binds_cases=int(sub["lb_binds"].sum()), cases=int(len(sub)),
            policy_within_U_tilde_all=bool(df[df["domain"] == d]["policy_within_U_tilde"].all()))
    summary["tightening"] = ratio
    by_class = []
    for (d, method, k), g in df.groupby(["domain", "method", "classes"]):
        rec = dict(domain=d, method=method, classes=int(k), configurations=int(len(g)),
                   median_regret=float(g["regret"].median()), max_regret=float(g["regret"].max()))
        for b in BOUNDS:
            rec[f"median_{b}"] = float(g[b].median())
            rec[f"max_{b}"] = float(g[b].max())
        rec["lb_binds"] = int(g["lb_binds"].sum())
        rec["median_U_tilde_actions"] = float(g["U_tilde_actions"].median())
        by_class.append(rec)
    summary["by_class_count"] = by_class
    acceptance = []
    for d in CLASS_COUNTS:
        full = CLASS_COUNTS[d][-1]
        for method in ("signature", "guided"):
            for b in BOUNDS:
                for tol in TOLERANCES:
                    picks = []
                    for cid in sorted(df[df["domain"] == d]["configuration_id"].unique()):
                        rr = df[(df["domain"] == d) & (df["method"] == method) & (df["configuration_id"] == cid)]
                        picks.append(smallest(rr.to_dict("records"), b, tol))
                    acceptance.append(dict(
                        domain=d, method=method, bound=b, tolerance=tol, configurations=len(picks),
                        compressed_certified=int(sum(p is not None and p < full for p in picks)),
                        median_smallest_classes=float(np.median([p if p is not None else np.nan for p in picks])),
                        smallest_classes=picks))
    summary["acceptance"] = acceptance
    rel_summary = []
    for (d, coords), g in rel.groupby(["domain", "coordinates"]):
        rel_summary.append(dict(
            domain=d, coordinates=coords, order=int(g["order"].iloc[0]), classes=int(g["classes"].iloc[0]),
            mean_regret=float(g["regret"].mean()), max_regret=float(g["regret"].max()),
            mean_refined_postsolve=float(g["refined_postsolve"].mean()),
            max_refined_postsolve=float(g["refined_postsolve"].max()),
            mean_refined_lb=float(g["refined_lb"].mean()), max_refined_lb=float(g["refined_lb"].max()),
            mean_prop1_postsolve=float(g["prop1_postsolve"].mean()),
            certified_at={str(t): int((g["refined_lb"] <= t).sum()) for t in TOLERANCES},
            mean_abs_consequence_change=np.abs(np.array(g["consequence_change"].tolist())).mean(axis=0).tolist(),
            max_abs_consequence_change=np.abs(np.array(g["consequence_change"].tolist())).max(axis=0).tolist()))
    summary["decision_relevance"] = rel_summary
    for d in CLASS_COUNTS:
        singles = sorted((r for r in rel_summary if r["domain"] == d and r["order"] == 1),
                         key=lambda r: r["mean_refined_lb"])
        summary[f"relevance_ranking_{d}"] = [r["coordinates"] for r in singles]
    summary["calibration"] = calibration(metas, df, rel)
    summary["runtime_seconds_per_configuration"] = {
        d: float(np.median([m["seconds"] for m in metas if m["domain"] == d])) for d in CLASS_COUNTS}
    return summary


def calibration(metas, df, rel, cert="refined_lb"):
    """VOI-relative acceptance, certificate-to-regret ratios, hindsight class counts and coordinate merges."""
    voi = {(m["domain"], m["configuration_id"]): m["voi_joint"] for m in metas}
    out = {}
    guided = df[df["method"] == "guided"]
    for d in CLASS_COUNTS:
        full = CLASS_COUNTS[d][-1]
        g = guided[guided["domain"] == d]
        rd = rel[rel["domain"] == d]
        rec = {}
        querying = sorted(cid for (dd, cid), v in voi.items() if dd == d and v > 1e-10)
        rec["querying_instances"] = len(querying)
        for f in VOI_FRACTIONS:
            picks, merges = [], []
            for cid in querying:
                tol = f * voi[(d, cid)]
                ok = g[(g["configuration_id"] == cid) & (g[cert] <= tol)]["classes"]
                picks.append(int(ok.min()) if len(ok) else full)
                ok = rd[(rd["configuration_id"] == cid) & (rd[cert] <= tol)]["classes"]
                merges.append(int(ok.min()) if len(ok) else full)
            rec[f"voi_fraction_{f:g}"] = dict(
                guided_compressed=int(sum(p < full for p in picks)), guided_classes=picks,
                merge_compressed=int(sum(p < full for p in merges)), merge_classes=merges,
                either_compressed=int(sum(min(p, q) < full for p, q in zip(picks, merges))))
        pos = g[g["regret"] > 1e-6]
        ratio = pos[cert] / pos["regret"]
        rec["certificate_over_regret"] = dict(cases=int(len(pos)),
                                              median=float(ratio.median()) if len(pos) else None,
                                              minimum=float(ratio.min()) if len(pos) else None)
        hind, certified, merge = [], [], []
        for cid in sorted(g["configuration_id"].unique()):
            rows = g[g["configuration_id"] == cid]
            hind.append(int(rows[rows["regret"] <= 0.005]["classes"].min()))
            ok = rows[rows[cert] <= 0.005]["classes"]
            certified.append(int(ok.min()) if len(ok) else full)
            ok = rd[(rd["configuration_id"] == cid) & (rd[cert] <= 0.005)]["classes"]
            merge.append(int(ok.min()) if len(ok) else full)
        rec["tolerance_0.005"] = dict(hindsight_classes=hind, certified_classes=certified, merge_classes=merge,
                                      merge_compressed=int(sum(m < full for m in merge)),
                                      merge_fewer_than_guided=int(sum(m < c for m, c in zip(merge, certified))))
        out[d] = rec
    return out


def to_json(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="run only the first instances of each grid")
    parser.add_argument("--figure-only", action="store_true", help="redraw the figure from saved results")
    args = parser.parse_args()
    if args.figure_only:
        make_figure()
        return
    tasks = [("flood", cid, cfg) for cid, cfg in enumerate(flood_configs())]
    tasks += [("social", cid, cfg) for cid, cfg in enumerate(social_configs())]
    if args.limit:
        tasks = [t for t in tasks if t[1] < args.limit]
    began = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = []
        for out in pool.map(analyse, tasks):
            m = out["meta"]
            print(f"{m['domain']} {m['configuration_id']}: {m['seconds']:.1f}s, |U|={m['U_actions']}", flush=True)
            results.append(out)
    metas = [r["meta"] for r in results]
    rows = [row for r in results for row in r["rows"]]
    relevance = [row for r in results for row in r["relevance"]]
    summary = summarise(metas, rows, relevance)
    summary["wall_seconds"] = time.perf_counter() - began
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).drop(columns=["consequence_change"]).to_csv(RESULTS / "certificates.csv", index=False)
    rel = pd.DataFrame(relevance)
    change = pd.DataFrame(rel.pop("consequence_change").tolist(), columns=[f"change_{f}" for f in FEATURES])
    pd.concat([rel, change], axis=1).to_csv(RESULTS / "certificates_relevance.csv", index=False)
    (RESULTS / "certificates.json").write_text(json.dumps(summary, indent=1, default=to_json, allow_nan=True),
                                               encoding="utf-8")
    print(json.dumps(dict(minimum_slack=summary["minimum_slack"], tightening=summary["tightening"],
                          wall=summary["wall_seconds"]), indent=1, default=to_json))
    make_figure()


LABELS = {"prop1_uniform": "Uniform bound", "prop1_postsolve": "Per-configuration bound",
          "refined_postsolve": "Class-level certificate", "regret": "Observed regret"}
STYLE_KEY = {"prop1_uniform": "uniform", "prop1_postsolve": "postsolve", "refined_postsolve": "refined",
             "regret": "regret"}
REFINED = "refined_postsolve"
# panel (d): the DRI is the class-level certificate capped by the query relaxation (Eq. 19)
DRI = "refined_lb"
GRID_TITLES = {"flood": "Flood grid (n = 32)", "social": "Social grid (n = 16)"}
LINTHRESH = 1e-5


def make_figure():
    """Figure 'certificates': bounds by class count, acceptance, decision relevance."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from hevi_hitl.plotting import FULL_WIDTH_IN, MARKERS, PALETTE, apply_style, panel_label, save

    apply_style()
    df = pd.read_csv(RESULTS / "certificates.csv")
    rel = pd.read_csv(RESULTS / "certificates_relevance.csv")
    summary = json.loads((RESULTS / "certificates.json").read_text(encoding="utf-8"))
    W, H = FULL_WIDTH_IN, 4.64
    fig = plt.figure(figsize=(W, H))

    def box(left, bottom, width, height):
        """Axes rectangle in inches, so that every label stays inside the printed width."""
        return (left / W, bottom / H, width / W, height / H)

    top_a = fig.add_axes(box(0.80, 2.71, 2.55, 1.36))
    top_b = fig.add_axes(box(3.60, 2.71, 2.55, 1.36), sharey=top_a)
    axes = np.array([[top_a, top_b], [fig.add_axes(box(0.80, 0.84, 2.12, 1.36)),
                                      fig.add_axes(box(4.27, 0.84, 1.88, 1.36))]])
    for ax, domain, letter in ((axes[0, 0], "flood", "a"), (axes[0, 1], "social", "b")):
        n_cfg = df[df.domain == domain].configuration_id.nunique()
        for method, ls in (("signature", "--"), ("guided", "-")):
            sub = df[(df.domain == domain) & (df.method == method)]
            g = sub.groupby("classes")
            for key in ("prop1_uniform", "prop1_postsolve", REFINED, "regret"):
                if key == "prop1_uniform" and method == "guided":
                    continue
                y = g[key].max().clip(lower=0)
                ax.plot(y.index, y.values, ls=ls, color=PALETTE[STYLE_KEY[key]],
                        marker=MARKERS[STYLE_KEY[key]], markersize=3.5, markerfacecolor="white" if ls == "--" else None)
        ax.axhline(0.005, color="black", lw=0.6, ls=":")
        ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=0.5)
        ax.set_ylim(-1e-6, 1.0)
        ax.set_xlabel("Number of classes")
        if domain == "flood":
            ax.set_ylabel("Maximum regret\n(symlog scale)")
        else:
            ax.tick_params(labelleft=False)
        assert GRID_TITLES[domain].endswith(f"(n = {n_cfg})")
        ax.set_title(GRID_TITLES[domain], fontsize=9, pad=4)
        panel_label(ax, letter, x=-0.29 if domain == "flood" else -0.08, y=1.04)
    handles = [Line2D([], [], color=PALETTE[STYLE_KEY[k]], marker=MARKERS[STYLE_KEY[k]], label=LABELS[k])
               for k in ("prop1_uniform", "prop1_postsolve", REFINED, "regret")]
    handles += [Line2D([], [], color="black", ls="-", label="Guided refinement"),
                Line2D([], [], color="black", ls="--", label="Signature bisection"),
                Line2D([], [], color="black", ls=":", lw=0.6, label="Tolerance 0.005")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), columnspacing=1.2,
               labelspacing=0.3, handlelength=1.8)
    # (c) acceptance
    ax = axes[1, 0]
    acc = pd.DataFrame(summary["acceptance"])
    combos = [("prop1_postsolve", "signature"), ("prop1_postsolve", "guided"),
              (REFINED, "signature"), (REFINED, "guided")]
    width = 0.2
    x = np.arange(len(TOLERANCES))
    full = {d: CLASS_COUNTS[d][-1] for d in CLASS_COUNTS}
    for i, (bound, method) in enumerate(combos):
        sel = acc[(acc.bound == bound) & (acc.method == method)]
        med, lo, hi = [], [], []
        for t in TOLERANCES:
            # Accepted class count as a percentage of configurations, pooled over the 48 instances
            # (an uncertified instance counts as 100 %).
            pct = [100 * (k if k is not None else full[d]) / full[d]
                   for d, ks in zip(sel[sel.tolerance == t].domain, sel[sel.tolerance == t].smallest_classes) for k in ks]
            med.append(np.median(pct))
            lo.append(min(pct))
            hi.append(max(pct))
        med, lo, hi = np.array(med), np.array(lo), np.array(hi)
        xx = x + (i - 1.5) * width
        ax.bar(xx, med, width, color=PALETTE[STYLE_KEY[bound]],
               hatch="////" if method == "signature" else None, edgecolor="white" if method == "guided" else "black",
               linewidth=0.4)
        ax.errorbar(xx, med, yerr=[med - lo, hi - med], fmt="none", ecolor="0.25", elinewidth=0.7, capsize=1.2)
    ax.set_xticks(x, [f"{t:g}" for t in TOLERANCES])
    ax.set_xlabel("Regret tolerance (score units)")
    ax.set_ylabel("Accepted classes\n(% of configurations)")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    # Colours follow the legend above; this legend below the panel explains the hatching.
    ax.legend(handles=[Patch(facecolor="white", edgecolor="black", hatch="////", label="Signature bisection"),
                       Patch(facecolor="0.55", label="Guided refinement")], loc="upper center",
              bbox_to_anchor=(0.45, -0.31), ncol=1, handlelength=1.4, labelspacing=0.3)
    panel_label(ax, "c", x=-0.38, y=1.04)
    # (d) decision relevance
    ax = axes[1, 1]
    singles = rel[rel.order == 1]
    order = list(HEVI_ORDER)
    assert set(order) == set(COORDINATES["social"])
    y = np.arange(len(order))[::-1]          # HEVI module order from the top; not a ranking
    # mean DRI (filled), maximum over instances (whisker), mean observed regret (open)
    for domain, dy, mk in (("social", 0.18, "o"), ("flood", -0.18, "s")):
        means, maxes, regs = [], [], []
        for n in order:
            g = singles[(singles.domain == domain) & (singles.coordinates == n)]
            means.append(g[DRI].mean() if len(g) else np.nan)
            maxes.append(g[DRI].max() if len(g) else np.nan)
            regs.append(g["regret"].clip(lower=0).mean() if len(g) else np.nan)
        means, maxes = np.array(means), np.array(maxes)
        ax.hlines(y + dy, means, maxes, color=PALETTE["refined"], lw=0.9)
        ax.plot(maxes, y + dy, "|", color=PALETTE["refined"], markersize=5, mew=0.9)
        ax.plot(means, y + dy, mk, color=PALETTE["refined"], markersize=4)
        ax.plot(regs, y + dy, mk, color=PALETTE["regret"], markersize=5.5, markerfacecolor="none", mew=0.9)
    ax.set_yticks(y, [n.replace("_", " ").capitalize() for n in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xscale("symlog", linthresh=LINTHRESH, linscale=0.5)
    ax.set_xlim(-2e-6, 0.1)
    ax.axvline(0.005, color="black", lw=0.6, ls=":")
    ax.set_xlabel("Regret from ignoring the\ncoordinate (symlog scale)")
    ax.legend(handles=[Line2D([], [], color=PALETTE["refined"], marker="o", ls="-", label="DRI, social"),
                       Line2D([], [], color=PALETTE["refined"], marker="s", ls="-", label="DRI, flood"),
                       Line2D([], [], color=PALETTE["regret"], marker="o", ls="", markerfacecolor="none",
                              markersize=5.5, label="Observed regret, social"),
                       Line2D([], [], color=PALETTE["regret"], marker="s", ls="", markerfacecolor="none",
                              markersize=5.5, label="Observed regret, flood")], loc="upper center", bbox_to_anchor=(0.18, -0.38),
              ncol=2, handlelength=1.6, columnspacing=1.0, labelspacing=0.3)
    panel_label(ax, "d", x=-0.66, y=1.04)
    save(fig, "certificates")


if __name__ == "__main__":
    main()
