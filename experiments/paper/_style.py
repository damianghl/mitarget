"""Shared matplotlib style for the paper figures (single column, vector PDF)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order: a method keeps its colour in every figure.
COLORS = {
    "baseline": "#2a78d6",
    "freq": "#eb6834",
    "npmle": "#1baf7a",
    "fixed_m": "#eda100",
    "glmm": "#e87ba4",
    "k2": "#008300",
    "ref": "#6b6b6b",
}

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e4e4e4",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})
