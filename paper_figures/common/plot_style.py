"""Single source of truth for TNSE paper colors and Matplotlib styling."""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl


METHOD_ORDER = [
    "Random",
    "Top-SNR",
    "Local Greedy",
    "Spatial DL",
    "FPLinQ",
    "Proposed-GAT",
]

# Colorblind-conscious, stable across every paper figure.
METHOD_COLORS = {
    "Random": "#D4D4D4",
    "Top-SNR": "#B4C0E4",
    "Local Greedy": "#7884B4",
    "Spatial DL": "#CDB8D6",
    "FPLinQ": "#484878",
    "Proposed-GAT": "#C65A78",
    "GAT-Assisted MarginalMP": "#C65A78",
    "Direct-Link MarginalMP": "#7884B4",
}

# Stable network-size colors for convergence, data-efficiency, and
# generalization panels.  Keep this mapping separate from method colors so a
# given hue never changes meaning inside one figure.
NODE_COLORS = {
    20: "#4F76B8",
    30: "#65A9A6",
    40: "#C98B4A",
    50: "#C65A78",
}

# Palette candidates for author review.  The method-to-color mapping stays
# fixed across panels; only the family changes.  ``colorblind`` is the current
# high-contrast version, while the two restrained families treat Proposed-GAT
# as the visual hero and keep centralized/heuristic references quieter.
PALETTE_PRESETS = {
    "colorblind": {
        "Random": "#9A9A9A",
        "Top-SNR": "#8BC8E8",
        "LocalMax": "#56B4E9",
        "Local Greedy": "#0072B2",
        "Spatial DL": "#CC79A7",
        "FPLinQ": "#009E73",
        "Proposed-GAT": "#D55E00",
    },
    "nature_blue": {
        "Random": "#C9C9C9",
        "Top-SNR": "#C3D7E5",
        "LocalMax": "#A9C4D8",
        "Local Greedy": "#6E94B5",
        "Spatial DL": "#A995C7",
        "FPLinQ": "#4D4D4D",
        "Proposed-GAT": "#0F4D92",
    },
    "nmi_pastel": {
        "Random": "#D4D4D4",
        "Top-SNR": "#B4C0E4",
        "LocalMax": "#B4C0E4",
        "Local Greedy": "#7884B4",
        "Spatial DL": "#CDB8D6",
        "FPLinQ": "#484878",
        "Proposed-GAT": "#C65A78",
    },
    "nmi_bright_rose": {
        "Random": "#D8D8DE",
        "Top-SNR": "#BFD4F6",
        "LocalMax": "#A8C9F0",
        "Local Greedy": "#6F8ED8",
        "Spatial DL": "#C7A6E8",
        "FPLinQ": "#5E58A6",
        "Proposed-GAT": "#F06C91",
    },
    "nmi_sky_coral": {
        "Random": "#D9DEE7",
        "Top-SNR": "#B1DBF6",
        "LocalMax": "#90CDF4",
        "Local Greedy": "#4F86C6",
        "Spatial DL": "#C4A7E7",
        "FPLinQ": "#5C5AA7",
        "Proposed-GAT": "#FF7A78",
    },
    "nmi_lilac_orange": {
        "Random": "#D9D9D9",
        "Top-SNR": "#B7E2E5",
        "LocalMax": "#A8DADC",
        "Local Greedy": "#5DA9B5",
        "Spatial DL": "#C5B0E3",
        "FPLinQ": "#7161A8",
        "Proposed-GAT": "#F28E6B",
    },
    "nmi_candy": {
        "Random": "#D3D8E0",
        "Top-SNR": "#B3D6F5",
        "LocalMax": "#9BC7F0",
        "Local Greedy": "#6F8FD8",
        "Spatial DL": "#D0AFE6",
        "FPLinQ": "#7067A8",
        "Proposed-GAT": "#EC6FAD",
    },
}

# Kept for backwards compatibility with older figure scripts. New figures use
# solid fills so that the legend remains legible at IEEE single-column width.
METHOD_HATCHES = {method: "" for method in METHOD_ORDER}

RC_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial", "Helvetica"],
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.5,
    "legend.framealpha": 1.0,
    "legend.facecolor": "white",
    "legend.edgecolor": "#B0B0B0",
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "patch.linewidth": 0.65,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#C0C0C0",
    "grid.alpha": 1.0,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "axes.grid.which": "both",
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "ps.fonttype": 42,
    "pdf.fonttype": 42,
}


@contextmanager
def paper_style():
    """Apply and then restore the shared paper style."""
    with mpl.rc_context(RC_PARAMS):
        yield
