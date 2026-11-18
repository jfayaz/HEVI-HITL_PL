"""Experiment 06: framework diagram and optimal policy of social instance 10.

Figs. 1 and 4.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

from hevi_hitl import flood as fl
from hevi_hitl import operational as o
from hevi_hitl import social as so
from hevi_hitl.model import PortfolioSolver
from hevi_hitl.plotting import apply_style, save

# boxes sit at fixed inch positions on a 7.0 in canvas, printed at 90%
DESIGN_WIDTH_IN = 7.0

# colours: risk information, preferences, commitment
BLUE, GREEN, VERMILION = "#E69F00", "#009E73", "#D55E00"
INK, GREY, MIDGREY, LIGHTGREY = "#222222", "#555555", "#8A8A8A", "#D4D4D4"
TINT = {BLUE: "#FDF3E1", GREEN: "#E5F5EF", VERMILION: "#FCEDE4", GREY: "#F3F3F3", INK: "#F3F3F3"}
TEXT_SHADE = {BLUE: "#9A6400"}
BODY, TITLE, HEAD, LETTER = 8.5, 9.0, 9.5, 11.0  # font sizes (pt); about 7.7, 8.1, 8.6 and 9.9 pt in print
LW = 0.9  # box and arrow line width (pt)
MARGIN = 0.08  # in; save() adds a further 0.08 in pad
EPS = "ϵ"

WORKED_CONFIG = so.SocialConfig(lead_hours=6, mechanism_strength=0.5, social_query_reliability=0.9,
                                preference_prior="diffuse")


def canvas(width: float, height: float):
    """Axes in inch data units covering [0, width] x [0, height] minus an outer strip."""
    crop = MARGIN - 0.01
    fig = plt.figure(figsize=(width - 2 * crop, height - 2 * crop))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(crop, width - crop)
    ax.set_ylim(crop, height - crop)
    ax.axis("off")
    return fig, ax


def write(ax, x, y, s, size=BODY, color=INK, weight="normal", ha="center", va="center", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=5,
                   clip_on=False, **kw)


def T(s, color=INK, size=TITLE):
    """Bold title line spec for ``stack``."""
    return s, dict(size=size, color=TEXT_SHADE.get(color, color), weight="bold")


def L(s, color=INK, size=BODY):
    """Body line spec for ``stack``."""
    return s, dict(size=size, color=color)


def stack(ax, x, cy, lines, ha="center", spacing=1.27):
    """Write lines vertically centred on ``cy``; line pitch = spacing * font size."""
    pitch = [spacing * kw["size"] / 72 for _, kw in lines]
    y = cy + sum(pitch) / 2
    for (s, kw), p in zip(lines, pitch):
        write(ax, x, y - p / 2, s, ha=ha, **kw)
        y -= p


def rbox(ax, x0, y0, w, h, edge, face=None, lw=LW, ls="-", radius=0.05, z=1):
    patch = FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                           lw=lw, ec=edge, fc=face if face is not None else "white", ls=ls, zorder=z)
    ax.add_patch(patch)
    return patch


def node(ax, x0, y0, w, h, lines, edge, face=None, lw=LW, ha="center"):
    """Rounded box with a centred stack of text lines."""
    rbox(ax, x0, y0, w, h, edge, TINT.get(edge, "white") if face is None else face, lw=lw)
    stack(ax, x0 + w / 2 if ha == "center" else x0 + 0.07, y0 + h / 2, lines, ha=ha)


def arrow(ax, p0, p1, color=INK, lw=LW, ms=7.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=3, joinstyle="miter", capstyle="butt"))


def path(ax, pts, color=INK, lw=LW, head=True):
    """Orthogonal connector through ``pts``; arrow head on the last segment."""
    if len(pts) > 2:
        xs, ys = zip(*pts[:-1])
        ax.add_line(Line2D(xs, ys, lw=lw, color=color, solid_joinstyle="miter", solid_capstyle="butt",
                           zorder=3))
    if head:
        arrow(ax, pts[-2], pts[-1], color=color, lw=lw)
    else:
        ax.add_line(Line2D(*zip(*pts[-2:]), lw=lw, color=color, zorder=3))


def heading(ax, x, y, letter, text):
    """Bold panel letter and a short heading on one baseline."""
    write(ax, x, y, f"({letter})", size=LETTER, weight="bold", ha="left", va="baseline")
    write(ax, x + 0.29, y, text, size=HEAD, weight="bold", ha="left", va="baseline")


def framework():
    """Three-panel concept figure: consequence model, ask-or-commit loop, certified reduction."""
    W, H = DESIGN_WIDTH_IN, 3.05
    fig, ax = canvas(W, H)
    RISK, PREF, COMMIT = BLUE, GREEN, VERMILION   # BLUE holds the risk orange (see colour roles)
    xa, xb, xc = (MARGIN, 2.03), (2.30, 5.04), (5.22, W - MARGIN)
    hy = H - MARGIN - 0.13
    heading(ax, xa[0], hy, "a", "HEVI consequences")
    heading(ax, xb[0] - 0.11, hy, "b", "Adaptive acquisition")
    heading(ax, xc[0], hy, "c", "Certified reduction")
    for xs in (2.11, 5.13):
        ax.add_line(Line2D([xs, xs], [MARGIN, hy + 0.12], lw=0.6, color=LIGHTGREY, zorder=0))
    r1, r2, r3 = (1.84, 2.54), (0.96, 1.64), (MARGIN + 0.05, 0.70)   # three rows shared by all panels

    # (a) HEVI consequence model ---------------------------------------------------------------
    half = (xa[1] - xa[0] - 0.10) / 2
    risk_box = (xa[0], r1[0], half, r1[1] - r1[0])
    act_box = (xa[0] + half + 0.10, r1[0], half, r1[1] - r1[0])
    node(ax, *risk_box, [T(r"Risk $\eta$", RISK), L("hazard,"), L("exposure,"), L("vulnerability")], RISK)
    node(ax, *act_box, [T(r"Portfolio $a$", COMMIT), L("committed at"), L(r"time $s$")], COMMIT)
    cons = (xa[0], r2[0], xa[1] - xa[0], r2[1] - r2[0])
    node(ax, *cons, [T("Consequences by group"), L("damage plus expenditure,"), L("displacement, exposed people")],
         GREY)
    for bx in (risk_box, act_box):
        cx = bx[0] + bx[2] / 2
        arrow(ax, (cx, r1[0]), (cx, r2[1]))
    score = (xa[0], r3[0], xa[1] - xa[0], r3[1] - r3[0])
    node(ax, *score, [T(r"Terminal score $J(z,a,s)$")], INK, face="white")
    mid = (xa[0] + xa[1]) / 2
    arrow(ax, (mid, r2[0]), (mid, r3[1]), color=PREF)
    write(ax, mid + 0.07, (r2[0] + r3[1]) / 2, r"priorities $\theta$", color=PREF, ha="left")

    # (b) Adaptive acquisition -----------------------------------------------------------------
    belief = (xb[0], r1[0], 0.95, r1[1] - r1[0])
    decide = (3.39, r1[0] + 0.10, 0.62, r1[1] - r1[0] - 0.20)
    commit = (4.15, r1[0], xb[1] - 4.15, r1[1] - r1[0])
    node(ax, *belief, [T(r"Belief $b$"), L(r"over risk $\eta$"), L(r"and priorities $\theta$")], INK,
         face="white")
    node(ax, *decide, [T("Ask or"), T("commit?")], INK, face=TINT[GREY])
    node(ax, *commit, [T("Commit", COMMIT), L(r"portfolio $a$,"), L("completed"), L(r"by $s_{\max}$")], COMMIT)
    ym = r1[0] + (r1[1] - r1[0]) / 2
    arrow(ax, (belief[0] + belief[2], ym), (decide[0], ym))
    arrow(ax, (decide[0] + decide[2], ym), (commit[0], ym), color=COMMIT)
    rw, pw = 1.0, xb[1] - xb[0] - 1.0 - 0.08
    rq = (xb[0], r2[0], rw, r2[1] - r2[0])
    pq = (xb[1] - pw, r2[0], pw, r2[1] - r2[0])
    node(ax, *rq, [T("Risk query", RISK), L("e.g. depth"), L("diagnostic or"), L("access check")], RISK)
    node(ax, *pq, [T("Preference comparison", PREF, size=BODY), L("the authority chooses"),
                   L("between two profiles")], PREF)
    dcx = decide[0] + decide[2] / 2
    rcx, pcx = rq[0] + rw / 2, pq[0] + pw / 2
    yf = (decide[1] + r2[1]) / 2
    ax.add_line(Line2D([dcx, dcx], [decide[1], yf], lw=LW, color=INK, zorder=3))
    ax.add_line(Line2D([rcx, pcx], [yf, yf], lw=LW, color=INK, zorder=3))
    arrow(ax, (rcx, yf), (rcx, r2[1]), color=RISK)
    arrow(ax, (pcx, yf), (pcx, r2[1]), color=PREF)
    yr, xr = r2[0] - 0.12, xb[0] - 0.08
    ax.add_line(Line2D([rcx, rcx], [r2[0], yr], lw=LW, color=RISK, zorder=3))
    ax.add_line(Line2D([pcx, pcx], [r2[0], yr], lw=LW, color=PREF, zorder=3))
    path(ax, [(pcx, yr), (xr, yr), (xr, ym), (belief[0], ym)], color=INK)
    write(ax, (xb[0] + xb[1]) / 2, yr - 0.04, r"answer updates $b$; time passes", va="top")

    # timeline: 6 h warning, two 0.5 h queries, commitment at 1 h, 3 h implementation; portfolios
    # start to expire once the longest implementation (3 h) no longer fits before the flood
    t0, t1, ty, th = xb[0], xb[1], 0.33, 0.08
    lead, expire = 6.0, 3.0

    def tx(hours):
        return t0 + (t1 - t0) * hours / lead

    up, down = ty + th / 2, ty - th / 2
    ax.add_patch(Rectangle((t0, down), t1 - t0, th, lw=0.6, ec=MIDGREY, fc="white", zorder=2))
    for h0, h1, colour in ((0.0, 0.5, RISK), (0.5, 1.0, PREF)):
        ax.add_patch(Rectangle((tx(h0), down), tx(h1) - tx(h0), th, lw=0, fc=colour, zorder=3))
    ax.add_patch(Rectangle((tx(1.0), down), tx(4.0) - tx(1.0), th, lw=0.6, ec=COMMIT, fc="white",
                           hatch="//////", zorder=3))
    ax.add_line(Line2D([tx(1.0)] * 2, [down - 0.03, up + 0.03], lw=1.4, color=COMMIT, zorder=4))
    ax.add_line(Line2D([t1] * 2, [down - 0.03, up + 0.03], lw=1.4, color=INK, zorder=4))
    strip, steps = 0.04, 40
    for k in range(steps):
        a0 = tx(expire + (lead - expire) * k / steps)
        a1 = tx(expire + (lead - expire) * (k + 1) / steps)
        ax.add_patch(Rectangle((a0, down - strip), a1 - a0, strip, lw=0, fc=INK, alpha=0.06 + 0.6 * k / steps,
                               zorder=1))
    write(ax, t0, up + 0.03, "Queries", ha="left", va="bottom")
    write(ax, tx(2.5), up + 0.03, "Implementation", color=COMMIT, va="bottom")
    write(ax, t1, up + 0.03, "Flood", ha="right", va="bottom")
    write(ax, t0, down - strip - 0.02, "0 h", ha="left", va="top")
    write(ax, tx(1.0) + 0.04, down - strip - 0.02, "Commitment", color=COMMIT, ha="left", va="top")
    write(ax, t1, down - strip - 0.02, "Options expire", color=GREY, ha="right", va="top")

    # (c) Certified reduction ------------------------------------------------------------------
    # scores of four configurations at portfolios A and B: pairs with equal contrasts form one class
    gz = (xc[0], r1[0], 0.62, 0.42)
    gc = (xc[1] - 0.78, r1[0], 0.78, 0.42)
    write(ax, gz[0] + gz[2] / 2, gz[1] + gz[3] + 0.04, "Configurations", va="bottom")
    write(ax, gc[0] + gc[2] / 2, gc[1] + gc[3] + 0.04, "Classes", va="bottom")
    rbox(ax, *gz, LIGHTGREY, "white", lw=0.6)
    xA, xB, ybase = gz[0] + 0.14, gz[0] + gz[2] - 0.14, gz[1] + 0.04
    pairs = ((INK, ((0.03, 0.10), (0.10, 0.17))), (MIDGREY, ((0.27, 0.22), (0.33, 0.28))))
    for colour, lines in pairs:
        for a, b in lines:
            ax.add_line(Line2D([xA, xB], [ybase + a, ybase + b], lw=0.9, color=colour, zorder=4))
            ax.scatter([xA, xB], [ybase + a, ybase + b], s=5, c=colour, lw=0, zorder=5)
    write(ax, xA, gz[1] - 0.015, "A", va="top")
    write(ax, xB, gz[1] - 0.015, "B", va="top")
    for (colour, lines), ey in zip(pairs, (0.11, 0.30)):
        ex = gc[0] + gc[2] / 2
        ax.add_patch(Ellipse((ex, gc[1] + ey), 0.50, 0.17, lw=0.6, ec=colour, fc=TINT[GREY], zorder=2))
        mean_a = np.mean([p[0] for p in lines])
        mean_b = np.mean([p[1] for p in lines])
        slope = (mean_b - mean_a) * 0.45
        ax.add_line(Line2D([ex - 0.13, ex + 0.13], [gc[1] + ey - slope / 2, gc[1] + ey + slope / 2], lw=0.9,
                           color=colour, zorder=4))
    yg = gz[1] + gz[3] / 2
    arrow(ax, (gz[0] + gz[2] + 0.04, yg), (gc[0] - 0.04, yg))
    write(ax, (gz[0] + gz[2] + gc[0]) / 2, yg + 0.05, r"$g$", va="bottom", size=9)
    red = (xc[0], r2[0], xc[1] - xc[0], r2[1] - r2[0])
    node(ax, *red, [T("Reduced model"), L("solved exactly, then"), L("checked in the original")], GREY)
    ccx = gc[0] + gc[2] / 2
    arrow(ax, (ccx, gc[1]), (ccx, r2[1]))
    cert = (xc[0], r3[0], xc[1] - xc[0], r3[1] - r3[0])
    scx = xc[0] + (xc[1] - xc[0]) / 2
    arrow(ax, (scx, r2[0]), (scx, r3[1]))
    rbox(ax, *cert, INK, "white")
    stack(ax, xc[0] + 0.40, cert[1] + cert[3] / 2, [T("Certificate"), L(r"regret $\leq\bar{" + EPS + "}$")])
    # schematic ranking of the coordinates against the tolerance
    gx0, gy0, unit = xc[0] + 0.86, cert[1] + 0.07, 0.55
    for k, (length, colour) in enumerate(zip((0.82, 0.58, 0.22, 0.08), (PREF, RISK, RISK, RISK))):
        ax.add_patch(Rectangle((gx0, gy0 + 0.26 - k * 0.08), length * unit, 0.05, lw=0, fc=colour, zorder=3))
    tol = gx0 + 0.30 * unit
    ax.add_line(Line2D([tol, tol], [gy0 - 0.01, gy0 + 0.34], lw=0.7, color=INK, ls=(0, (1, 1.5)), zorder=4))
    write(ax, tol, gy0 + 0.36, r"$\bar{" + EPS + "}$", va="bottom")
    write(ax, gx0 + 0.58 * unit + 0.05, gy0 + 0.14, "DRI", ha="left")
    save(fig, "framework")


QUERY_TITLES = {
    "depth_update": ("Depth diagnostic",),
    "displacement_vs_exposure": ("Displacement", "vs exposure"),
    "economic_vs_exposure": ("Economic cost", "vs exposure"),
}
ATTRIBUTES = ("cost", "displacement", "exposure")  # column order of PREFERENCE_PAIRS profiles
PROPERTY = {"none": "No property measure", "temporary_barrier": "Barrier",
            "contents_protection": "Contents protection"}
EVACUATION = {"none": "No evacuation", "standard": "Standard evacuation", "assisted": "Assisted evacuation"}


def check_answer_semantics(model):
    """Confirm what answer 1 means for each channel before it is labelled."""
    q = model.queries.index("depth_update")
    by_severity = [model.likelihood[model.indices[:, 0] == k, q].mean() for k in range(3)]
    assert np.all(np.diff(by_severity) > 0), "depth flag (answer 1) must indicate deeper water"
    weights = fl.PREFERENCE_WEIGHTS[model.indices[:, 5]]
    for name in fl.QUERIES[4:]:
        pair = fl.PREFERENCE_PAIRS[fl.QUERIES.index(name) - 4]
        prefers_second = weights @ ((pair[0] - pair[1]) / fl.SCALES) > 0
        assert np.array_equal(model.likelihood[:, model.queries.index(name)] > 0.5, prefers_second), (
            "answer 1 must mean that the second profile is preferred")


def answer_label(name, y):
    if name in fl.QUERIES[4:]:
        pair = fl.PREFERENCE_PAIRS[fl.QUERIES.index(name) - 4]
        lower = np.flatnonzero(pair[y] < pair[1 - y])
        assert len(lower) == 1
        return f"Lower {ATTRIBUTES[lower[0]]}"
    return "Flag raised" if y else "No flag"


def query_title(name):
    return QUERY_TITLES.get(name, (name.replace("_", " ").capitalize(),))


def worked_policy():
    model = so.build_social_flood(WORKED_CONFIG)
    check_answer_semantics(model)
    solver = PortfolioSolver(model)
    solver.solve()
    report = o.evaluate(model, o.freeze(solver), include_trace=True)
    kind, immediate = PortfolioSolver(model, method="immediate").solve()[1]
    assert kind == "act"
    return model, report, immediate


def policy_tree():
    model, report, immediate = worked_policy()
    nodes = {n["node_id"]: n for n in report["trace"]}

    def children(n):  # leaves before subtrees, then by answer: gives a staircase layout
        return sorted(n["children"], key=lambda c: (nodes[c["node_id"]]["kind"] == "query", c["answer"]))

    row, depth, leaves = {}, {}, []

    def place(nid, d):
        n = nodes[nid]
        depth[nid] = d
        if n["kind"] == "act":
            row[nid] = len(leaves)
            leaves.append(nid)
        else:
            row[nid] = min(place(c["node_id"], d + 1) for c in children(n))
        return row[nid]

    place("root", 0)
    letters = {}
    for nid in leaves:
        letters.setdefault(nodes[nid]["index"], "ABCDEFGH"[len(letters)])
    if immediate not in letters:
        letters[immediate] = "ABCDEFGH"[len(letters)]

    W, H = DESIGN_WIDTH_IN, 3.47
    fig, ax = canvas(W, H)
    x0, x1 = MARGIN, W - MARGIN
    qw, qh, qstep, elbow = 1.32, 0.44, 1.27, 0.22  # query boxes, column pitch, elbow offset
    lx0, lw_, lh = 5.00, 0.98, 0.50  # leaf column
    pitch, y_top = 0.60, 3.13

    def ycen(nid):
        return y_top - row[nid] * pitch

    for nid in leaves:
        n = nodes[nid]
        y = ycen(nid)
        node(ax, lx0, y - lh / 2, lw_, lh,
             [T(f"Portfolio {letters[n['index']]}", VERMILION),
              L(f"{n['time_hours']:.1f} h → {n['completion_hours']:.1f} h"),
              L(f"Pr = {n['probability']:.3f}")], VERMILION, face="white", lw=1.1)
        damage, _, exposed, _ = n["physical_features"]  # conditional means given the leaf
        stack(ax, lx0 + lw_ + 0.08, y, [L(f"EUR {damage / 1e6:.1f} million", GREY), L(f"{exposed:.0f} exposed", GREY)],
              ha="left")

    # query nodes and edges
    for nid, n in nodes.items():
        if n["kind"] != "query":
            continue
        qx0 = x0 + depth[nid] * qstep
        y = ycen(nid)
        colour = GREEN if n["index"] in so.PREFERENCE_QUERIES else BLUE
        start, end = n["time_hours"], n["time_hours"] + n["duration_hours"]
        node(ax, qx0, y - qh / 2, qw, qh,
             [T(t, colour) for t in query_title(n["query_name"])] + [L(f"{start:.1f}–{end:.1f} h")], colour)
        for c in children(n):
            child = nodes[c["node_id"]]
            cy = ycen(c["node_id"])
            cx_left = lx0 if child["kind"] == "act" else x0 + depth[c["node_id"]] * qstep
            if abs(cy - y) < 1e-9:
                pts = [(qx0 + qw, y), (cx_left, y)]
            else:
                pts = [(qx0 + elbow, y - qh / 2), (qx0 + elbow, cy), (cx_left, cy)]
            path(ax, pts, color=INK)
            lx = pts[-2][0] + 0.07
            write(ax, lx, cy + 0.035, answer_label(n["query_name"], c["answer"]), ha="left", va="bottom")
            write(ax, lx, cy - 0.035, f"{c['conditional_probability']:.3f}", ha="left", va="top", color=GREY)

    # colour key for node types, in the empty lower-left corner
    ky = y_top - 2.4 * pitch
    for k, (label, colour) in enumerate((("Risk query", BLUE), ("Preference comparison", GREEN),
                                         ("Commitment", VERMILION))):
        yy = ky - k * 0.19
        rbox(ax, x0, yy - 0.06, 0.2, 0.12, colour, TINT[colour] if colour != VERMILION else "white",
             lw=1.0, radius=0.025)
        write(ax, x0 + 0.28, yy, label, ha="left")
    write(ax, x0, ky - 3 * 0.19, "Grey: expected damage and exposed people", ha="left", color=GREY)

    # portfolio key: district allocations from the action inventory, and immediate commitment
    sep = y_top - 3 * pitch - lh / 2 - 0.10
    ax.add_line(Line2D([x0, x1], [sep, sep], lw=0.6, color=LIGHTGREY))
    cols = (x0, 0.85, 2.35, 3.85, 5.35)
    hy = sep - 0.12
    for c, label in zip(cols, ("Portfolio",) + o.DISTRICTS + ("Immediate commitment",)):
        write(ax, c, hy, label, weight="bold", ha="left")
    info = o.portfolio_info(model, immediate)
    for k, (j, letter) in enumerate(sorted(letters.items(), key=lambda kv: kv[1])):
        yy = hy - 0.29 - k * 0.35
        write(ax, cols[0], yy, letter, weight="bold", color=VERMILION, ha="left", size=TITLE)
        for c, item in zip(cols[1:4], o.portfolio_info(model, j)["portfolio"]):
            stack(ax, c, yy, [L(PROPERTY[item["property"]]), L(EVACUATION[item["evacuation"]])], ha="left",
                  spacing=1.2)
        chosen = f"Selected, 0.0 h → {info['implementation_hours']:.1f} h" if j == immediate else "–"
        write(ax, cols[4], yy, chosen, ha="left", color=INK if j == immediate else GREY)
    save(fig, "policy_tree")
    return model, report, immediate, letters


def main():
    apply_style()
    framework()
    model, report, immediate, letters = policy_tree()
    print("immediate commitment action", immediate, "-> portfolio", letters[immediate])
    for j, letter in sorted(letters.items(), key=lambda kv: kv[1]):
        print(letter, j, o.portfolio_info(model, j)["portfolio"])
    print("expected features (damage, person-days, exposed, intervention):", report["physical_features"])


if __name__ == "__main__":
    main()
