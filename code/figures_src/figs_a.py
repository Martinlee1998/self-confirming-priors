import sys; sys.path.insert(0, ".")
from style import *
from data import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
setup()
from pathlib import Path
OUT = str(Path(__file__).resolve().parents[1] / "generated_figures") + "/"
Path(OUT).mkdir(parents=True, exist_ok=True)

# ---------------- Study 1 (column) ----------------
def study1():
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 2.7), gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.62})
    names = ["AB", "BC", "CD", "DE"]
    for i, k in enumerate(names):
        y = 3 - i
        v = STUDY1["swaps"][k]
        strip(a, v, y, seed=i)
        m, lo, hi = tci(v)
        mean_ci(a, m, lo, hi, y, color=BLUE)
        a.text(0.665, y, f"{m:.3f}", va="center", ha="left", color=INK, fontsize=7)
    a.set_yticks([3, 2, 1, 0], [f"Swap-{k}" for k in names])
    a.set_xlim(-0.08, 0.70); a.set_ylim(-0.6, 3.6)
    zero_line(a); vgrid(a)
    a.set_xlabel(r"World effect $R(\mathrm{Canonical})-R(\mathrm{swap})$")
    panel(a, "a", "Effect of each adjacent swap")
    a.text(0.665, 3.55, "Mean", ha="left", va="bottom", color=INK2, fontsize=7)
    v = STUDY1["composite"]
    strip(b, v, 0, seed=7, spread=0.3)
    m, lo, hi = tci(v)
    mean_ci(b, m, lo, hi, 0, color=BLUE)
    b.text(0.33, 0, f"{m:.3f} [{lo:.3f}, {hi:.3f}]", va="center", ha="left", color=INK, fontsize=7)
    b.set_yticks([]); b.spines["left"].set_visible(False)
    b.set_xlim(-0.02, 0.50); b.set_ylim(-0.6, 0.6)
    zero_line(b); vgrid(b)
    b.set_xlabel(r"World effect $R(\mathrm{Canonical})$ $-$ mean of four swaps")
    panel(b, "b", "Composite effect (100 worlds)")
    h = [Line2D([], [], marker="o", ls="", color=NEUTRAL, ms=3, label="World effect"),
         Line2D([], [], marker="D", color=BLUE, ms=4, lw=1.2, label="Mean and 95% interval")]
    a.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 1.08), ncol=2, handlelength=1.4, columnspacing=1.0)
    fig.savefig(OUT + "fig3_order_sensitivity.pdf")

# ---------------- Study 2 (full width) ----------------
def run_axis(ax, rotate=False):
    ax.set_xticks(range(8), RUNS, rotation=45 if rotate else 0, ha="right" if rotate else "center")

def study2():
    fig, ax = plt.subplots(1, 4, figsize=(FULL_W, 2.05), gridspec_kw={"width_ratios": [1.15, 1, 1, 0.8], "wspace": 0.55})
    a, b, c, d = ax
    rep = [o >= 9 for o in P1_OBS]
    cols = [BLUE if r else NEUTRAL for r in rep]
    a.barh(range(8)[::-1], P1_OBS, color=cols, height=0.6)
    a.set_yticks(range(8)[::-1], RUNS)
    a.axvline(9, color=INK, lw=0.8)
    a.text(10.5, 4.5, "threshold\n(9 observations)", fontsize=7, color=INK, va="center")
    for i, o in enumerate(P1_OBS):
        a.text(o + 0.8, 7 - i, str(o), va="center", fontsize=7, color=INK2)
    a.set_xlim(0, 48); a.set_xlabel(r"$A$-before-$B$ observations"); vgrid(a)
    panel(a, "a", "P1 evidence per run")
    a.legend(handles=[Patch(color=BLUE, label="Repaired"), Patch(color=NEUTRAL, label="Not repaired")],
             loc="lower right", bbox_to_anchor=(1.02, 0.0), handlelength=1.0)
    # (b) recovery vs evidence
    for o, r, rp in zip(P1_OBS, P1_REC, rep):
        b.plot(o, r, "o", ms=5, color=BLUE if rp else NEUTRAL, mec=SURFACE, mew=0.8, zorder=3)
    b.axvline(9, color=INK, lw=0.8)
    b.set_xlim(-2, 40); b.set_ylim(-0.03, 0.31); hgrid(b)
    b.set_xlabel(r"$A$-before-$B$ observations"); b.set_ylabel("Provider recovery $T_c$")
    panel(b, "b", "Recovery against evidence")
    b.text(32, 0.282, "4 runs", fontsize=7, color=INK2, ha="center"); b.text(12, 0.0, "4 runs", fontsize=7, color=INK2, va="center")
    # (c) run effects
    for ci in range(4):
        for s in range(2):
            i = 2 * ci + s
            c.plot(ci + (-0.16 if s == 0 else 0.16), P1_EFF[i], "o" if s == 0 else "s", ms=4.5,
                   color=BLUE if rep[i] else NEUTRAL, mec=SURFACE, mew=0.8, zorder=3)
        c.plot(ci, P1_CTRL[ci], "D", ms=5, color=INK, mec=SURFACE, mew=0.8, zorder=4)
    c.set_xticks(range(4), ["C1", "C2", "C3", "C4"]); c.set_xlim(-0.5, 3.5)
    c.set_ylim(-0.02, 0.31); hgrid(c); c.set_ylabel(r"Run effect $\delta_c$ ($\mathrm{N}-\mathrm{F}$)")
    panel(c, "c", "Run effects")
    c.legend(handles=[Line2D([], [], marker="o", ls="", color=MUTED, ms=4, label="Replicate 0"),
                      Line2D([], [], marker="s", ls="", color=MUTED, ms=4, label="Replicate 1"),
                      Line2D([], [], marker="D", ls="", color=INK, ms=4, label="Controller mean")],
             loc="center", bbox_to_anchor=(0.5, 0.42), handletextpad=0.2, fontsize=6.5)
    # (d) world effects
    for x, v, nm in [(0, P1_WORLD, "P1"), (1, P2_WORLD, "P2")]:
        strip(d, v, x, horizontal=False, seed=x, spread=0.22)
        m, lo, hi = tci(v)
        mean_ci(d, m, lo, hi, x, horizontal=False, color=BLUE)
        if x == 0: d.text(x + 0.3, m, f"{m:.3f}", fontsize=7, color=INK, va="center")
        else: d.text(x, m + 0.025, f"{m:.3f}", fontsize=7, color=INK, ha="center")
    d.set_xticks([0, 1], ["P1", "P2"]); d.set_xlim(-0.5, 1.7); d.set_ylim(-0.06, 0.31)
    zero_line(d, horizontal=False); hgrid(d); d.set_ylabel(r"World effect $\mathrm{N}-\mathrm{F}$")
    panel(d, "d", "100 evaluation worlds")
    fig.savefig(OUT + "fig4_natural_recovery.pdf")

if __name__ == "__main__":
    study1(); study2()
