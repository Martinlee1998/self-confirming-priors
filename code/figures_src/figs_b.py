import sys; sys.path.insert(0, ".")
from style import *
from data import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
setup()
from pathlib import Path
OUT = str(Path(__file__).resolve().parents[1] / "generated_figures") + "/"
Path(OUT).mkdir(parents=True, exist_ok=True)
CTRL = ["C1", "C2", "C3", "C4"]

def run_effect_panel(ax, eff, ctrl, color_fn, ylim):
    for ci in range(4):
        for s in range(2):
            i = 2 * ci + s
            ax.plot(ci + (-0.16 if s == 0 else 0.16), eff[i], "o" if s == 0 else "s", ms=4.5,
                    color=color_fn(i), mec=SURFACE, mew=0.8, zorder=3)
        ax.plot(ci, ctrl[ci], "D", ms=5, color=INK, mec=SURFACE, mew=0.8, zorder=4)
    ax.set_xticks(range(4), CTRL); ax.set_xlim(-0.5, 3.5); ax.set_ylim(*ylim); hgrid(ax)

REP_LEG = [Line2D([], [], marker="o", ls="", color=MUTED, ms=4, label="Replicate 0"),
           Line2D([], [], marker="s", ls="", color=MUTED, ms=4, label="Replicate 1"),
           Line2D([], [], marker="D", ls="", color=INK, ms=4, label="Controller mean")]

# ---------------- Study 3 ----------------
def study3():
    fig, ax = plt.subplots(2, 3, figsize=(FULL_W, 3.45), gridspec_kw={"hspace": 0.62, "wspace": 0.42})
    a, b, c, d, e, f = ax.ravel()
    for i, v in enumerate(X_OBS):
        a.plot([0, 1], [0, v], color=GRID, lw=0.8, zorder=1)
    a.scatter([0] * 8, [0] * 8, s=18, color=NEUTRAL, zorder=3, edgecolors=SURFACE, linewidths=0.6)
    a.scatter([1] * 8, X_OBS, s=18, color=BLUE, zorder=3, edgecolors=SURFACE, linewidths=0.6)
    a.text(1.08, np.mean(X_OBS), f"mean {np.mean(X_OBS):.1f}", va="center", fontsize=7, color=INK)
    a.set_xticks([0, 1], ["N", "X"]); a.set_xlim(-0.3, 1.7); a.set_ylim(-1.5, 32); hgrid(a)
    a.set_ylabel(r"$B$-before-$C$ observations"); panel(a, "a", "Evidence per run")
    # (b) beliefs
    for x, pbc, pcb in [(0, [0.1] * 8, [0.9] * 8), (1, X_PBC, X_PCB)]:
        b.scatter([x - 0.08] * 8, pbc, s=16, color=BLUE, marker="o", edgecolors=SURFACE, linewidths=0.6, zorder=3)
        b.scatter([x + 0.08] * 8, pcb, s=16, color=ORANGE, marker="s", edgecolors=SURFACE, linewidths=0.6, zorder=3)
    b.axhline(0.5, color=INK, lw=0.8)
    b.text(1.45, 0.52, "active if\n$p>0.5$", fontsize=6.5, color=INK2, va="bottom", ha="center")
    b.set_xticks([0, 1], ["N", "X"]); b.set_xlim(-0.4, 1.8); b.set_ylim(0, 1); hgrid(b)
    b.set_ylabel("Frozen edge belief"); panel(b, "b", "Beliefs on the reversed pair")
    b.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, ms=4, label=r"$p_{BC}$ (correct)"),
                      Line2D([], [], marker="s", ls="", color=ORANGE, ms=4, label=r"$p_{CB}$ (wrong)")],
             loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.2, columnspacing=0.8, fontsize=6.5)
    b.set_ylim(0, 1.2); b.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    # (c) provider distance
    w = 0.32
    for k, (lab, vals, col) in enumerate([("All 31 masks", [0.129, 0.0], BLUE), ("5 reachable masks", [0.200, 0.0], ORANGE)]):
        xs = np.arange(2) + (k - 0.5) * (w + 0.04)
        c.bar(xs, vals, width=w, color=col, label=lab)
        for x, v in zip(xs, vals):
            c.text(x, v + 0.006, f"{v:.3f}" if v else "0", ha="center", fontsize=6.5, color=INK2)
    c.set_xticks([0, 1], ["N", "X"]); c.set_ylim(0, 0.32); c.set_yticks([0, 0.1, 0.2]); hgrid(c)
    c.set_ylabel(r"Mean TV distance to $G_{\mathrm{O}}$"); panel(c, "c", "Provider distance")
    c.set_xlim(-0.6, 2.0)
    c.legend(loc="upper right", bbox_to_anchor=(1.02, 1.0), handlelength=1.0, fontsize=6.5)
    # (d) world effects
    strip(d, X_WORLD, 0, horizontal=False, seed=3, spread=0.25)
    m, lo, hi = tci(X_WORLD)
    mean_ci(d, m, lo, hi, 0, horizontal=False, color=BLUE)
    d.text(0.38, m, f"{m:.3f}\n[{lo:.3f}, {hi:.3f}]", va="center", fontsize=7, color=INK)
    d.set_xticks([0], ["100 worlds"]); d.set_xlim(-0.5, 1.4); d.set_ylim(0, 0.5); hgrid(d)
    d.set_ylabel(r"World effect $\mathrm{X}-\mathrm{N}$"); panel(d, "d", "Behavioral effect")
    # (e) run effects
    run_effect_panel(e, X_EFF, X_CTRL, lambda i: BLUE, (0, 0.45))
    e.set_ylabel(r"Run effect $\delta_c$ ($\mathrm{X}-\mathrm{N}$)"); panel(e, "e", "Run effects")
    e.legend(handles=REP_LEG, loc="lower right", ncol=1, handletextpad=0.2, fontsize=6.5)
    # (f) depth distribution
    labels = ["O and X", "F and N"]
    for r, lab in enumerate(labels):
        left = 0
        for k, v in enumerate(DEPTH[lab]):
            if v > 0:
                f.barh(r, v, left=left, color=SEQ[k + 1], height=0.55, edgecolor=SURFACE, linewidth=0.8)
                if v > 0.1:
                    f.text(left + v / 2, r, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                           color=SURFACE if k >= 3 else INK)
            left += v
    f.set_yticks([0, 1], labels); f.invert_yaxis(); f.set_xlim(0, 1); vgrid(f)
    f.set_xlabel("Fraction of 800 rows"); panel(f, "f", "Final prefix depth")
    hs = [Patch(color="none", label="Depth:")] + [Patch(color=SEQ[k + 1], label=f"{k}") for k in range(5)]
    f.legend(handles=hs, loc="center", bbox_to_anchor=(0.5, 0.5), ncol=6, handlelength=0.8, handletextpad=0.3,
             columnspacing=0.5, fontsize=6.5)
    fig.savefig(OUT + "fig5_evidence_pathway.pdf")

# ---------------- Uncertainty levels ----------------
def uncertainty():
    fig, ax = plt.subplots(figsize=(FULL_W, 1.9))
    styles = [("World-conditional $t$ interval (100 worlds)", BLUE, "o"),
              ("Run-by-world bootstrap", ORANGE, "^"),
              ("Controller-by-world bootstrap", AQUA, "s")]
    for r, (lab, cond, rb, cb, cm) in enumerate(UNC):
        y0 = 2 - r
        ivs = [cond, (cond[0],) + tuple(rb), (cond[0],) + tuple(cb)]
        for k, ((m, lo, hi), (_, col, mk)) in enumerate(zip(ivs, styles)):
            y = y0 + 0.22 - k * 0.15
            ax.plot([lo, hi], [y, y], color=col, lw=1.2, solid_capstyle="butt")
            ax.plot([m], [y], marker=mk, color=col, ms=4, mec=SURFACE, mew=0.6)
        y = y0 - 0.26
        ax.scatter(cm, [y] * 4, marker="D", s=14, facecolors="none", edgecolors=INK, linewidths=0.8)
        ax.axhspan(y0 - 0.42, y0 + 0.38, color="#f4f3f0" if r % 2 == 0 else SURFACE, zorder=0)
    ax.set_yticks([2, 1, 0], ["Study 2, P1: $\\mathrm{N}-\\mathrm{F}$", "Study 2, P2: $\\mathrm{N}-\\mathrm{F}$",
                              "Study 3: $\\mathrm{X}-\\mathrm{N}$"])
    ax.set_ylim(-0.45, 2.42); ax.set_xlim(-0.02, 0.37); zero_line(ax); vgrid(ax)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(r"Behavioral effect $\Delta R$")
    h = [Line2D([], [], color=c, marker=m, lw=1.2, ms=4, label=l) for l, c, m in styles]
    h.append(Line2D([], [], marker="D", ls="", mfc="none", mec=INK, ms=4, label="Controller means"))
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, columnspacing=1.0, handlelength=1.6)
    fig.savefig(OUT + "fig6_uncertainty_hierarchy.pdf")

# ---------------- Study 4 ----------------
def study4():
    fig, ax = plt.subplots(1, 3, figsize=(FULL_W, 2.1), gridspec_kw={"width_ratios": [1.2, 0.9, 1.1], "wspace": 0.5})
    a, b, c = ax
    part = [o >= 8 for o in ES_P2_OBS]
    a.bar(range(8), ES_P2_OBS, color=[ORANGE if p else NEUTRAL for p in part], width=0.6)
    a.axhline(8, color=INK, lw=0.8)
    a.text(7.4, 8.3, "wrong edge inactive (8)", ha="right", fontsize=6.5, color=INK)
    a.set_xticks(range(8), RUNS, rotation=45, ha="right"); a.set_ylim(0, 12); a.set_yticks([0, 2, 4, 6, 8, 10]); hgrid(a)
    a.set_ylabel(r"Challenge observations ($B$ before $C$)"); panel(a, "a", "P2 evidence under ES")
    a.legend(handles=[Patch(color=ORANGE, label="Partly repaired"), Patch(color=NEUTRAL, label="Not repaired")],
             loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=2, handlelength=1.0, fontsize=6.5, columnspacing=0.8)
    w = 0.34
    for k, (cond, col) in enumerate([("N", NEUTRAL), ("ES", BLUE)]):
        xs = np.arange(2) + (k - 0.5) * (w + 0.04)
        vals = [REPAIRED["P2"][cond], REPAIRED["P1"][cond]]
        b.bar(xs, vals, width=w, color=col if cond == "ES" else MUTED, label=cond)
        for x, v in zip(xs, vals):
            b.text(x, v + 0.12, f"{v}/8", ha="center", fontsize=6.5, color=INK2)
    b.set_xticks([0, 1], ["P2\n(partial repair)", "P1\n(full repair)"]); b.set_ylim(0, 8); hgrid(b)
    b.set_ylabel("Repaired runs"); panel(b, "b", "Runs that change class")
    b.legend(loc="upper left", handlelength=1.0, fontsize=6.5)
    for x, (rel, eff) in enumerate([("P2", ES_P2_EFF), ("P1", ES_P1_EFF)]):
        rng = np.random.default_rng(x)
        jit = rng.uniform(-0.12, 0.12, 8)
        cols = [ORANGE if (rel == "P2" and part[i]) else (BLUE if v < 0 else NEUTRAL) for i, v in enumerate(eff)]
        c.scatter(np.array([x] * 8) - 0.15 + jit, eff, s=16, c=cols, edgecolors=SURFACE, linewidths=0.6, zorder=3)
        m, lo, hi = ES_MEANS[rel]
        mean_ci(c, m, lo, hi, x + 0.2, horizontal=False, color=INK)
        c.text(x + 0.3, m, f"{m:+.3f}", fontsize=7, va="center", color=INK)
    c.set_xticks([0, 1], ["P2", "P1"]); c.set_xlim(-0.5, 1.8); c.set_ylim(-0.5, 0.25)
    zero_line(c, horizontal=False); hgrid(c)
    c.set_ylabel(r"$\mathrm{ES}-\mathrm{N}$"); panel(c, "c", "Behavioral effect")
    c.legend(handles=[Line2D([], [], marker="o", ls="", color=MUTED, ms=4, label="Run effect"),
                      Line2D([], [], marker="D", color=INK, lw=1.2, ms=4, label="World mean, 95% interval")],
             loc="lower left", fontsize=6.5, handlelength=1.3)
    fig.savefig(OUT + "fig7_evidence_seeking.pdf")

# ---------------- Comparators ----------------
def comparators():
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.25), gridspec_kw={"wspace": 1.15})
    for ax, rel in zip(axes, ["P1", "P2"]):
        rows = [r for r in COMP if r[1] == rel]
        ys = [5.2, 4.2, 2.6, 1.6, 0.6]
        for y, (grp, _, name, m, lo, hi, cls) in zip(ys, rows):
            ax.plot([lo, hi], [y, y], color=INK, lw=1.2)
            ax.plot([m], [y], "D", color=INK, ms=4, mec=SURFACE, mew=0.6)
            for k, ch in enumerate(cls):
                x0 = 1.05 + k * 0.04
                fc = {"r": BLUE, "p": SURFACE, "u": NEUTRAL}[ch]
                rect = Rectangle((x0, y - 0.28), 0.03, 0.56, transform=ax.get_yaxis_transform(), clip_on=False,
                                 facecolor=fc, edgecolor=ORANGE if ch == "p" else "none", hatch="////" if ch == "p" else None,
                                 linewidth=0.8)
                ax.add_patch(rect)
        ax.set_yticks(ys, [r[2] for r in rows]); ax.tick_params(axis="y", length=0)
        ax.axhline(3.4, color=MUTED, lw=0.6)
        ax.text(-0.02, 5.95, "Study 4 seed sets", fontsize=6.5, color=INK2, transform=ax.get_yaxis_transform(), ha="right", va="top")
        ax.text(-0.02, 3.3, "Comparator seed sets", fontsize=6.5, color=INK2, transform=ax.get_yaxis_transform(), ha="right", va="top")
        ax.set_xlim(-0.30, 0.10); ax.set_ylim(0.0, 6.0); zero_line(ax); vgrid(ax)
        ax.set_xlabel(r"$R(\mathrm{rule})-R(\mathrm{N})$, same seed set")
        panel(ax, "a" if rel == "P1" else "b", f"{rel} reversal")
        ax.text(1.05 + 0.16, 6.05, "Frozen provider\nof 8 runs", fontsize=6.5, color=INK2, transform=ax.get_yaxis_transform(), ha="center", va="bottom")
    h = [Patch(color=BLUE, label="Equivalent to $G_{\\mathrm{O}}$"),
         Patch(facecolor=SURFACE, edgecolor=ORANGE, hatch="////", label="Partly repaired"),
         Patch(color=NEUTRAL, label="Not repaired"),
         Line2D([], [], marker="D", color=INK, lw=1.2, ms=4, label="Mean and 95% interval")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, handlelength=1.2, columnspacing=1.2)
    fig.savefig(OUT + "fig8_baseline_comparison.pdf")

# ---------------- Selector audit ----------------
def selectors():
    fig, ax = plt.subplots(2, 3, figsize=(FULL_W, 3.15), gridspec_kw={"width_ratios": [1.35, 1, 0.8], "hspace": 0.62, "wspace": 0.7})
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f6f9fe"] + SEQ)
    names = ["Random", "Uncertainty", "DI"]
    for r, rel in enumerate(["P1", "P2"]):
        S = SEL[rel]; a, b, c = ax[r]
        M = np.array(S["M"])
        a.imshow(M, cmap=cmap, vmin=0, vmax=0.55, aspect="auto")
        for i in range(3):
            for j in range(6):
                a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color=SURFACE if M[i, j] > 0.3 else INK)
        a.set_xticks(range(6), S["pairs"]); a.set_yticks(range(3), names)
        a.tick_params(length=0); [s.set_visible(False) for s in a.spines.values()]
        panel(a, "abcdef"[3 * r], f"{rel}: share of challenges per pair")
        labs = [d[0].replace(" vs ", " vs\n") for d in S["div"]]; vals = [d[1] for d in S["div"]]; ns = [d[2] for d in S["div"]]
        b.barh(range(3)[::-1], vals, color=BLUE, height=0.5)
        for y, v, n in zip(range(3)[::-1], vals, ns):
            b.text(v + 0.02, y, f"{v:.0%} (n={n})", va="center", fontsize=6.5, color=INK2)
        b.set_yticks(range(3)[::-1], labs, fontsize=6.5); b.set_xlim(0, 1); vgrid(b); b.tick_params(axis="y", length=0)
        b.set_xlabel("Fraction choosing a different pair")
        panel(b, "abcdef"[3 * r + 1], f"{rel}: rule disagreement")
        c.bar(range(3), S["obs"], color=BLUE, width=0.55)
        for x, v, rp in zip(range(3), S["obs"], S["rep"]):
            c.text(x, v + 0.25, f"{v} obs.\n{rp}/8 rep.", ha="center", fontsize=6, color=INK2)
        c.set_xticks(range(3), ["Random", "Uncertainty", "DI"], fontsize=6.5, rotation=30, ha="right"); c.set_ylim(0, 9); hgrid(c)
        c.set_ylabel("Challenge observations")
        panel(c, "abcdef"[3 * r + 2], f"{rel}: evidence")
    fig.savefig(OUT + "fig9_selector_audit.pdf")

# ---------------- Reachable-mask decisions (new) ----------------
def decisions():
    import itertools
    def chain():
        P = np.full((5, 5), 0.1)
        for i in range(5):
            for j in range(i + 1, 5): P[i, j] = 0.9
        return P
    def dist(P, mask):
        U = [j for j in range(5) if not mask[j]]
        unb = [j for j in U if not any(P[i, j] > 0.5 for i in U if i != j)]
        if unb: return {j: 1 / len(unb) for j in unb}
        o = {j: sum(P[i, j] for i in U if i != j) for j in U}; mn = min(o.values())
        a = [j for j in U if abs(o[j] - mn) < 1e-12]; return {j: 1 / len(a) for j in a}
    O = chain(); F1 = chain(); F1[0, 1], F1[1, 0] = 0.1, 0.9
    F2 = chain(); F2[1, 2], F2[2, 1] = 0.1, 0.9
    ES = chain(); ES[1, 2] = ES[2, 1] = 0.5
    q = np.array([[0, .96, .01, .96, .01], [.01, 0, .9767, .2733, .01], [.01, .01, 0, .96, .01],
                  [.01, .01, .01, 0, .9767], [.01] * 5])
    graphs = [(r"$G_{\mathrm{O}}$", O), (r"$G_{\mathrm{F}}$, P1", F1), (r"$G_{\mathrm{F}}$, P2", F2),
              (r"$\mathrm{ES}$, P2 (8 obs.)", ES), (r"$\mathrm{L}_0$ (LLM prior)", q)]
    masks = [(0, 0, 0, 0, 0), (1, 0, 0, 0, 0), (1, 1, 0, 0, 0), (1, 1, 1, 0, 0), (1, 1, 1, 1, 0)]
    L = "ABCDE"
    fig, ax = plt.subplots(figsize=(COL_W, 1.75))
    for r, (name, P) in enumerate(graphs):
        for c, m in enumerate(masks):
            d = dist(P, m); do = dist(O, m)
            same = d == do
            txt = ", ".join(L[j] if abs(p - 1) < 1e-9 else f"{L[j]}:{p:.1f}" for j, p in sorted(d.items()))
            ax.add_patch(Rectangle((c, r), 0.96, 0.9, facecolor="#f2f1ee" if same else "#fbd9c9", edgecolor="none"))
            ax.text(c + 0.48, r + 0.45, txt, ha="center", va="center", fontsize=6.5, color=INK)
    ax.set_xlim(0, 5); ax.set_ylim(5, 0)
    ax.set_xticks(np.arange(5) + 0.48, ["".join(map(str, m)) for m in masks])
    ax.set_yticks(np.arange(5) + 0.45, [g[0] for g in graphs])
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.xaxis.tick_top(); ax.set_xlabel("Reachable mask (achieved $A$..$E$)", labelpad=4); ax.xaxis.set_label_position("top")
    ax.legend(handles=[Patch(color="#f2f1ee", label="Same target as $G_{\\mathrm{O}}$"),
                       Patch(color="#fbd9c9", label="Different target")],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, handlelength=1.0)
    fig.savefig(OUT + "fig10_reachable_decisions.pdf")

if __name__ == "__main__":
    for fn in sys.argv[1:] or ["study3", "uncertainty", "study4", "comparators", "selectors", "decisions"]:
        globals()[fn]()
