import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Shared figure palette.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID, NEUTRAL = "#e4e3df", "#b9b8b2"
SURFACE = "#ffffff"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

COL_W, FULL_W = 3.5, 7.16  # Single-column and full-width figures (inches).

def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "axes.titlelocation": "left", "axes.titleweight": "normal",
    })

def panel(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left", fontsize=8, color=INK, pad=4)

def hgrid(ax):
    ax.yaxis.grid(True); ax.set_axisbelow(True)

def vgrid(ax):
    ax.xaxis.grid(True); ax.set_axisbelow(True)

def strip(ax, values, pos, horizontal=True, seed=0, color=NEUTRAL, size=6, spread=0.28):
    rng = np.random.default_rng(seed)
    j = rng.uniform(-spread, spread, len(values))
    if horizontal:
        ax.scatter(values, pos + j, s=size, color=color, alpha=0.75, linewidths=0, zorder=2)
    else:
        ax.scatter(pos + j, values, s=size, color=color, alpha=0.75, linewidths=0, zorder=2)

def mean_ci(ax, m, lo, hi, pos, horizontal=True, color=INK, marker="D", ms=4):
    if horizontal:
        ax.plot([lo, hi], [pos, pos], color=color, lw=1.2, solid_capstyle="butt", zorder=4)
        ax.plot([m], [pos], marker=marker, ms=ms, color=color, mec=SURFACE, mew=0.8, zorder=5)
    else:
        ax.plot([pos, pos], [lo, hi], color=color, lw=1.2, solid_capstyle="butt", zorder=4)
        ax.plot([pos], [m], marker=marker, ms=ms, color=color, mec=SURFACE, mew=0.8, zorder=5)

def tci(v):
    from scipy import stats
    v = np.asarray(v, float); m = v.mean()
    if v.std() == 0: return m, m, m
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return m, m - h, m + h

def zero_line(ax, horizontal=True):
    (ax.axvline if horizontal else ax.axhline)(0, color=INK2, lw=0.6, zorder=1)
