"""Figure style and export."""
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FULL_WIDTH_IN = 6.3      # text width of the manuscript
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

# Okabe-Ito palette
PALETTE = {
    "joint": "#0072B2",
    "physical_only": "#E69F00",
    "preference_only": "#009E73",
    "one_step": "#CC79A7",
    "immediate": "#555555",
    "oracle": "#000000",
    "homogeneous": "#D55E00",
    "interaction": "#56B4E9",
    "uniform": "#999999",
    "postsolve": "#E69F00",
    "refined": "#0072B2",
    "regret": "#D55E00",
    "signature": "#999999",
    "guided": "#0072B2",
}
MARKERS = {"joint": "o", "physical_only": "s", "preference_only": "^", "one_step": "D",
           "immediate": "x", "homogeneous": "v", "uniform": "s", "postsolve": "^",
           "refined": "o", "regret": "x", "signature": "s", "guided": "o"}


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "savefig.facecolor": "white",
    })


def panel_label(ax, letter, x=-0.14, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    """Write figures/<name>.pdf."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
