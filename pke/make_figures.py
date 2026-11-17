"""Publication figures for the PKE paper. Run: python make_figs.py (reads numbers from figdata.py)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D
from figdata import *

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
LBLUE, LORANGE, LGRAY = "#e8f0fb", "#fdeee7", "#f4f3ef"
INK, INK2, GRID = "#1a1a1a", "#55544f", "#e4e3de"
W = 5.45  # inches, text width of the journal template
plt.rcParams.update({
    "font.family": "Liberation Sans", "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.8, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42,
    "savefig.dpi": 300, "figure.dpi": 150})
DS = ["Norman", "Wessels", "TianIPSC", "TianNeuron"]
DSL = {"Norman": "Norman", "Wessels": "Wessels", "TianIPSC": "Tian\niPSC", "TianNeuron": "Tian\nneuron"}
DSF = 8
SHORT = ["Norman", "Wessels", "Tian iPSC", "Tian neuron"]


def tag(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left", color=INK)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(name + ".pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(name + ".png", bbox_inches="tight", pad_inches=0.03, dpi=200)
    plt.close(fig)


# =========================================================================== Figure 1: overview
def rbox(ax, x0, y0, w, h, fc, ec, lw=1.0, r=1.2):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def harrow(ax, x0, x1, y, col=INK2):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, mutation_scale=11,
                                                                 shrinkA=0, shrinkB=0))


def varrow(ax, x, y0, y1, col=INK2):
    ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, mutation_scale=11,
                                                                 shrinkA=0, shrinkB=0))


def fig1():
    H = 3.95
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100 * H / W); ax.axis("off")
    TT, MT, ST = 9.5, 10.5, 8.5          # title, math, small text sizes
    top, h = 50.0, 20.0
    xs = [0.5 + k * 25.0 for k in range(4)]; w = 24.0 - 3.0
    xs = [0.5, 26.5, 52.5, 78.5]; w = 21.0
    cy = top + h / 2
    titles = ["Single\nperturbations", "Gene\nkernels", "Pair\nkernels", "Kernel ridge\nregression"]
    fills = [LGRAY, LGRAY, "white", LGRAY]
    for x0, t, fc in zip(xs, titles, fills):
        rbox(ax, x0, top, w, h, fc, INK2)
        ax.text(x0 + w / 2, top + h - 4.6, t, ha="center", va="center", fontsize=TT, fontweight="bold", linespacing=1.1)
    ax.text(xs[0] + w / 2, top + 4.6, r"$\delta_g = x_g - x_0$", ha="center", va="center", fontsize=MT)
    ax.text(xs[1] + w / 2, top + 4.6, r"$k(a,b)$", ha="center", va="center", fontsize=MT)
    rbox(ax, xs[2] + 1.5, top + 6.2, w - 3.0, 4.6, LBLUE, BLUE, r=0.9)
    ax.text(xs[2] + w / 2, top + 8.5, r"$K_{+}$ main", ha="center", va="center", fontsize=ST + 0.5)
    rbox(ax, xs[2] + 1.5, top + 0.9, w - 3.0, 4.6, LORANGE, ORANGE, r=0.9)
    ax.text(xs[2] + w / 2, top + 3.2, r"$K_{\times}$ interaction", ha="center", va="center", fontsize=ST + 0.5)
    ax.text(xs[3] + w / 2, top + 4.6, r"centered, on $r_{ab}$", ha="center", va="center", fontsize=ST, color=INK2)
    for k in range(3):
        harrow(ax, xs[k] + w + 0.5, xs[k + 1] - 0.5, cy)
    # row 2: prediction
    pt, ph = 26.0, 18.5
    rbox(ax, 0.5, pt, 99, ph, "white", INK2)
    ax.text(50, pt + ph - 3.6, r"Prediction for any pair $u=\{a,b\}$", ha="center", va="center", fontsize=TT, fontweight="bold")
    ey = pt + 7.6
    ax.text(26.6, ey, r"$\hat{f}(u)\;=\;\bar{y}\;+$", ha="left", va="center", fontsize=MT + 1)
    rbox(ax, 42.0, ey - 3.4, 19.0, 6.8, LBLUE, BLUE, r=0.9)
    ax.text(51.5, ey, r"$g(a)+g(b)$", ha="center", va="center", fontsize=MT + 1)
    ax.text(63.6, ey, "+", ha="center", va="center", fontsize=MT + 1)
    rbox(ax, 66.2, ey - 3.4, 15.0, 6.8, LORANGE, ORANGE, r=0.9)
    ax.text(73.7, ey, r"$f_{\mathrm{int}}(u)$", ha="center", va="center", fontsize=MT + 1)
    ax.text(51.5, pt + 1.9, "gene level", ha="center", va="center", fontsize=ST, color=BLUE)
    ax.text(73.7, pt + 1.9, "pair specific", ha="center", va="center", fontsize=ST, color=ORANGE)
    varrow(ax, xs[3] + w / 2, top - 0.5, pt + ph + 0.5)
    # row 3: explanations
    et, eh = 0.5, 20.0
    rbox(ax, 0.5, et, 99, eh, LGRAY, INK2)
    ax.text(50, et + eh - 3.6, "Exact explanations", ha="center", va="center", fontsize=TT, fontweight="bold")
    chips = [("Gene propensity", r"$\|g(a)\|$"), ("Main vs. interaction", "share of each prediction"), ("Representer shares", r"$s_i(u)$ over doubles")]
    for k, (t1, t2) in enumerate(chips):
        cx0 = 2.5 + k * 32.5
        rbox(ax, cx0, et + 1.8, 30.5, 10.6, "white", GRAY, lw=0.8, r=0.9)
        ax.text(cx0 + 15.25, et + 9.4, t1, ha="center", va="center", fontsize=ST + 0.5)
        ax.text(cx0 + 15.25, et + 4.7, t2, ha="center", va="center", fontsize=ST + 0.5, color=INK2)
    varrow(ax, 50, pt - 0.5, et + eh + 0.5)
    save(fig, "fig1_overview")


# =========================================================================== Figure 2: benchmark
def fig2():
    fig = plt.figure(figsize=(W, 5.2))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.5, left=0.1, right=0.99, top=0.92, bottom=0.11, width_ratios=[1.2, 1])
    # a: pooled gain over ridge with CIs, 3 gene sets
    ax = fig.add_subplot(gs[0, 0]); tag(ax, "a")
    mk = {"de": "o", "s50": "s", "hvg": "D"}
    for i, ds in enumerate(DS):
        for j, sc in enumerate(["random", "genegrp"]):
            col = BLUE if sc == "random" else AQUA
            for k, g in enumerate(["de", "s50", "hvg"]):
                d, lo, hi = POOLED[(ds, sc)][g]; x = i + (j - .5) * .42 + (k - 1) * .11
                ax.plot([x, x], [lo, hi], color=col, lw=1.1, solid_capstyle="round")
                ax.plot(x, d, mk[g], ms=4.0, color=col, mec="white", mew=0.5, zorder=3)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels([DSL[d] for d in DS], fontsize=DSF); ax.set_xlim(-.6, 3.6); ax.set_ylim(-.01, .22)
    ax.set_ylabel(r"$\Delta R^2$ versus ridge"); grid(ax)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="Random pairs"),
                       Line2D([], [], marker="o", ls="", color=AQUA, label="Held-out genes"),
                       Line2D([], [], marker="o", ls="", color=INK2, label="Most changed"),
                       Line2D([], [], marker="s", ls="", color=INK2, label="Singles ranked"),
                       Line2D([], [], marker="D", ls="", color=INK2, label="All genes")],
              loc="upper right", ncol=1, handletextpad=.2, borderaxespad=0.1, fontsize=7.5, labelspacing=.25)
    # b: ablations
    ax = fig.add_subplot(gs[0, 1]); tag(ax, "b", x=-0.48)
    names = list(ABL); y = np.arange(len(names))[::-1]
    r = np.array([ABL[n][0] for n in names]); g = np.array([ABL[n][1] for n in names])
    ax.barh(y + .2, r, height=.38, color=BLUE, label="Random pairs")
    ax.barh(y - .2, g, height=.38, color=AQUA, label="Held-out genes")
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlim(.2, .5); ax.set_xticks([.2, .3, .4, .5]); ax.set_xlabel(r"Held-out $R^2$ (Norman)"); grid(ax, "x")
    ax.legend(loc="lower center", bbox_to_anchor=(.4, 1.0), ncol=2, handlelength=1.2, columnspacing=1.2)
    # c: other predictors, random split
    ax = fig.add_subplot(gs[1, 0]); tag(ax, "c")
    meths = list(OTHERS); cols = [GRAY, "#c9c8c2", "#8e6bbf", "#d4a72c", "#e07ba0", "#4fb3c9", ORANGE, BLUE]
    cols = cols[-len(meths):] if len(meths) <= len(cols) else cols
    pal = dict(zip(meths, [GRAY, "#c9c8c2", "#8e6bbf", "#d4a72c", "#e07ba0", "#4fb3c9", "#6c8f3a", ORANGE, BLUE][-len(meths):]))
    pal["Ridge"], pal["PKE"] = ORANGE, BLUE
    wbar = .8 / len(meths)
    for i, ds in enumerate(DS):
        for k, m in enumerate(meths):
            v = OTHERS[m][ds]; x = i - .4 + (k + .5) * wbar
            ax.bar(x, max(v, -.2), width=wbar * .92, color=pal[m], label=m if i == 0 else None)
            if v < -.2:
                ax.annotate("", xy=(x, -.235), xytext=(x, -.13), arrowprops=dict(arrowstyle="-|>", color=INK, lw=.8, mutation_scale=6))
    ax.axhline(0, color=INK2, lw=0.8); ax.set_xticks(range(4)); ax.set_xticklabels([DSL[d] for d in DS], fontsize=DSF)
    ax.set_xlim(-.55, 3.55); ax.set_ylim(-.25, .8); ax.set_ylabel(r"Held-out $R^2$"); grid(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=4, handlelength=.9, columnspacing=.7, fontsize=7.5, handletextpad=.3)
    # d: GI modes
    ax = fig.add_subplot(gs[1, 1]); tag(ax, "d", x=-0.2)
    keys = [("mag", "Mag.\nρ"), ("dom", "Dom.\nρ"), ("syn", "Syn.\nAUC"), ("buf", "Buf.\nAUC"), ("neoauc", "Neo.\nAUC")]
    x = np.arange(len(keys))
    for m, c, off, lab in [("pke", BLUE, -.27, "PKE"), ("ridge", ORANGE, 0, "Ridge"), ("additive", GRAY, .27, "Additive")]:
        ax.bar(x + off, [MODES[m][kk] for kk, _ in keys], width=.25, color=c, label=lab)
    ax.axhline(0, color=INK2, lw=0.8); ax.set_xticks(x); ax.set_xticklabels([l for _, l in keys])
    ax.set_ylim(-.1, 1); ax.set_ylabel("Score"); grid(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=3, handlelength=1.2, columnspacing=1.2)
    save(fig, "fig2_benchmark")


# =========================================================================== Figure 3: interpretation
def fig3():
    fig = plt.figure(figsize=(W, 6.0))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.75, left=0.15, right=0.99, top=0.95, bottom=0.08)
    ax = fig.add_subplot(gs[0, 0]); tag(ax, "a", x=-0.36)
    P = PROP_TOP[:15][::-1]; y = np.arange(len(P))
    ax.barh(y, [v for _, v in P], color=[BLUE if g.startswith("CEBP") else GRAY for g, _ in P], height=.68)
    ax.set_yticks(y); ax.set_yticklabels([g for g, _ in P], style="italic"); ax.set_ylim(-.6, len(P) - .4)
    ax.set_xlabel(r"Interaction propensity $\|g(x)\|$"); grid(ax, "x")
    ax = fig.add_subplot(gs[0, 1]); tag(ax, "b", x=-0.62)
    S = SCREEN_TOP[:12][::-1]; y = np.arange(len(S))
    ax.barh(y, [s[2] for s in S], color=BLUE, height=.68, label="Gene level")
    ax.barh(y, [s[3] for s in S], left=[s[2] + .03 for s in S], color=ORANGE, height=.68, label="Pair specific")
    ax.set_yticks(y); ax.set_yticklabels([s[0] for s in S], style="italic"); ax.set_ylim(-.6, len(S) - .4)
    ax.set_xlabel("Norm of predicted part"); grid(ax, "x")
    ax.legend(loc="lower center", bbox_to_anchor=(.35, 1.0), ncol=2, handlelength=1.2, columnspacing=1.0)
    ax = fig.add_subplot(gs[1, 0]); tag(ax, "c", x=-0.36)
    yy, ticks, labs, heads = 0, [], [], []
    for pair, e in list(EXPLAIN.items())[::-1]:
        col = ORANGE if e["inter"] > .5 else BLUE
        for p, s in e["sup"][:4][::-1]:
            ax.barh(yy, s, color=col, height=.68); ticks.append(yy); labs.append(p); yy += 1
        heads.append((yy - .05, pair, col)); yy += 1.9
    for hy, pair, col in heads:
        ax.text(0, hy, pair, fontsize=10, fontweight="bold", style="italic", color=col, va="bottom")
    ax.set_yticks(ticks); ax.set_yticklabels(labs, style="italic", fontsize=9); ax.set_ylim(-.6, yy - .2)
    ax.set_xlabel(r"Representer share $s_i(u)$"); grid(ax, "x")
    ax = fig.add_subplot(gs[1, 1]); tag(ax, "d", x=-0.62)
    labs = ["PKE strength", "PKE gene part", "PKE pair part", "ANOVA pair part", "PKE propensity", "Ridge strength"]
    y = np.arange(len(EXT2))[::-1]
    ax.barh(y + .19, [e[1] for e in EXT2], height=.36, color=BLUE, label=r"$\rho$")
    ax.barh(y - .19, [e[2] for e in EXT2], height=.36, color=AQUA, label=r"Partial $\rho$ (hubness)")
    ax.axvline(0, color=INK2, lw=0.8); ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=8.5)
    ax.set_xlim(-.03, .15); ax.set_xticks([0, .05, .1, .15]); ax.set_xlabel("Association with |growth GI|"); grid(ax, "x")
    ax.legend(loc="lower center", bbox_to_anchor=(.35, 1.0), ncol=2, handlelength=1.0, columnspacing=.8, fontsize=8)
    save(fig, "fig3_interpret")


# =========================================================================== Figure 4: design and robustness
def fig4():
    fig = plt.figure(figsize=(W, 5.0))
    outer = fig.add_gridspec(2, 1, hspace=0.62, left=0.1, right=0.99, top=0.9, bottom=0.14, height_ratios=[1, 1.05])
    top = outer[0].subgridspec(1, 4, wspace=0.35)
    n = np.arange(20, 81, 10); sty = [("random", GRAY, "Random"), ("variance", BLUE, "Variance"), ("strength", ORANGE, "Strength"),
                                      ("var_x_strength", AQUA, "Variance x strength")]
    for k, ds in enumerate(DS):
        ax = fig.add_subplot(top[0, k])
        if k == 0: tag(ax, "a", x=-0.55)
        for key, c, lab in sty:
            ax.plot(n, AL[ds][key], "-o", color=c, ms=2.6, lw=1.3, label=lab)
        ax.set_title(DSL[ds].replace("\n", " "), fontsize=8.5, pad=3); ax.set_xticks([20, 50, 80]); ax.tick_params(labelsize=7.5)
        ax.set_xlabel("Doubles measured", fontsize=8); grid(ax)
        if k == 0: ax.set_ylabel(r"$R^2$, unmeasured", fontsize=8.5)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(.55, 1.0), ncol=4, handlelength=1.4, columnspacing=1.0, fontsize=7.5)
    bot = outer[1].subgridspec(1, 3, wspace=0.6, width_ratios=[.8, 1.3, 1])
    ax = fig.add_subplot(bot[0, 0]); tag(ax, "b", x=-0.55, y=1.2); x = np.arange(4)
    ax.bar(x, [GPRHO[d] for d in DS], width=.6, color=BLUE)
    ax.axhline(0, color=INK2, lw=0.8); ax.set_xticks(x); ax.set_xticklabels(SHORT, fontsize=7.5, rotation=35, ha="right")
    ax.set_ylabel(r"$\rho$(variance, error)"); ax.set_ylim(-.1, .6); grid(ax)
    ax = fig.add_subplot(bot[0, 1]); tag(ax, "c", x=-0.32, y=1.2)
    ms = ["Ridge", "PKE", "PKE+STRING"]; cs = [ORANGE, BLUE, "#8e6bbf"]
    for k, (m, c) in enumerate(zip(ms, cs)):
        xx = x + (k - 1) * .27
        ax.bar(xx, [NOISY[d][m][1] for d in DS], width=.25, color=c, label=m)
        for xi, d in zip(xx, DS):
            ax.plot([xi - .12, xi + .12], [NOISY[d][m][0]] * 2, color=INK, lw=1.3)
    ax.set_xticks(x); ax.set_xticklabels(SHORT, fontsize=7.5, rotation=35, ha="right"); ax.set_ylim(0, .8); ax.set_ylabel(r"Held-out $R^2$"); grid(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=3, handlelength=1.0, columnspacing=.7, fontsize=7.5)
    ax = fig.add_subplot(bot[0, 2]); tag(ax, "d", x=-0.4, y=1.2)
    ax.bar(x - .19, [SIM[d][1] for d in DS], width=.36, color=BLUE, label="More similar")
    ax.bar(x + .19, [SIM[d][2] for d in DS], width=.36, color=GRAY, label="Less similar")
    ax.set_xticks(x); ax.set_xticklabels(SHORT, fontsize=7.5, rotation=35, ha="right"); ax.set_ylim(0, 1.05)
    ax.set_ylabel("Relative error"); grid(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=2, handlelength=1.0, columnspacing=.7, fontsize=7.5)
    save(fig, "fig4_design")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
