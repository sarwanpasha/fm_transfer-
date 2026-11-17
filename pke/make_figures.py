import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from figdata import *

BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "axes.titlesize": 8, "axes.titleweight": "bold", "axes.titlelocation": "left", "pdf.fonttype": 42})
DS = ["Norman", "Wessels", "TianIPSC", "TianNeuron"]; DSL = {"Norman": "Norman", "Wessels": "Wessels", "TianIPSC": "Tian\niPSC", "TianNeuron": "Tian\nneuron"}
def tag(ax, s): ax.text(-0.13, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
def grid(ax, axis="y"): ax.grid(axis=axis, color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)

# ------------------------------------------------------------------ Figure 1: overview
def box(ax, x, y, w, h, txt, fc="#f3f2ee", ec=INK2, fs=7.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else None, linespacing=1.3)
def arr(ax, a, b): ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, lw=0.8, color=INK2))
fig = plt.figure(figsize=(7.4, 3.4)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
box(ax, .01, .60, .17, .30, "Single\nperturbations\n$\\delta_g$", fs=9.5)
box(ax, .01, .12, .17, .30, "Measured\ndoubles\n$r_{ab}$", fs=9.5)
box(ax, .23, .60, .19, .30, "Gene kernels\n$k(a,b)$\n(9 RBF, 4 erf)", fs=9.5)
box(ax, .47, .72, .22, .20, "$K_+$ (main effect)", fc="#e3edfa", ec=BLUE, fs=9.5)
box(ax, .47, .46, .22, .20, "$K_\\times$ (interaction)", fc="#fde8df", ec=ORANGE, fs=9.5)
box(ax, .47, .12, .22, .24, "Centered kernel\nridge regression", fs=9.5)
box(ax, .75, .58, .24, .32, "$\\hat f(u)=\\bar y+g(a)+g(b)$\n$+\\,f_{\\mathrm{int}}(u)$", fs=10)
box(ax, .75, .08, .24, .36, "Propensity $\\|g(a)\\|$\nMain/interaction split\nShares $s_i(u)$", fs=9.5)
for a_, b_ in [((.18, .75), (.23, .75)), ((.42, .80), (.47, .82)), ((.42, .70), (.47, .56)), ((.18, .27), (.47, .24)),
             ((.58, .72), (.58, .66)), ((.58, .46), (.58, .36)), ((.69, .26), (.75, .70)), ((.87, .58), (.87, .44))]:
    arr(ax, a_, b_)
fig.savefig("fig1_overview.pdf"); plt.close(fig)

# ------------------------------------------------------------------ Figure 2: benchmark
fig, axs = plt.subplots(2, 2, figsize=(7.6, 6.4), gridspec_kw=dict(hspace=0.62, wspace=0.62))
ax = axs[0, 0]; tag(ax, "a")
for i, ds in enumerate(DS):
    for j, sc in enumerate(["random", "genegrp"]):
        b = BENCH[(ds, sc)]; d = np.array(b["pke"]) - np.array(b["ridge"]); x = i + (j - .5) * .34
        col = BLUE if sc == "random" else AQUA
        for k in range(5):
            xx = x + (k - 2) * .045; ax.plot([xx, xx], [b["lo"][k], b["hi"][k]], color=col, lw=0.8, alpha=.45)
            ax.plot(xx, d[k], "o", ms=4, color=col, mec="white", mew=0.5)
        ax.plot([x - .14, x + .14], [d.mean()] * 2, color=INK, lw=1.1)
ax.axhline(0, color=INK2, lw=0.6); ax.set_xticks(range(4)); ax.set_xticklabels([DSL[d] for d in DS], fontsize=8.5)
ax.set_ylabel("$\\Delta R^2$ vs. ridge"); grid(ax)
ax.plot([], [], "o", color=BLUE, label="random pairs held out"); ax.plot([], [], "o", color=AQUA, label="held-out genes")
ax.legend(loc="upper right", fontsize=8.5, handletextpad=0.2)

ax = axs[0, 1]; tag(ax, "b")
names = list(ABL); y = np.arange(len(names))[::-1]
r = np.array([ABL[n][0] for n in names]); g = np.array([ABL[n][1] for n in names])
ax.barh(y + .19, r, height=.36, color=BLUE, label="random"); ax.barh(y - .19, g, height=.36, color=AQUA, label="held-out genes")
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.5); ax.set_xlim(.2, .5); ax.set_xlabel("held-out $R^2$ (Norman)")
grid(ax, "x"); ax.legend(loc="upper center", bbox_to_anchor=(.35, -.2), ncol=2, fontsize=8.5)
ax = axs[1, 0]; tag(ax, "c")
for i, ds in enumerate(DS):
    for j, sc in enumerate(["random", "genegrp"]):
        v = np.mean(BENCH[(ds, sc)]["gears_gi"]); x = i + (j - .5) * .34
        ax.bar(x, v, width=.3, color=ORANGE if sc == "random" else "#f4a582")
ax.axhline(0, color=INK2, lw=0.6); ax.set_xticks(range(4)); ax.set_xticklabels([DSL[d] for d in DS], fontsize=8.5)
ax.set_ylabel("GEARS $\\Delta R^2$ vs. ridge"); ax.set_ylim(-5.2, .3); grid(ax)
ax.legend(handles=[Patch(color=ORANGE, label="random"), Patch(color="#f4a582", label="held-out genes")], loc="lower left", fontsize=8.5)

ax = axs[1, 1]; tag(ax, "d")
keys = [("mag", "magnitude ρ"), ("dom", "dominance ρ"), ("syn", "synergy AUC"), ("buf", "buffering AUC"), ("neoauc", "neomorphic AUC")]
x = np.arange(len(keys))
for k, (m, c, off) in enumerate([("pke", BLUE, -.27), ("ridge", ORANGE, 0), ("additive", GRAY, .27)]):
    ax.bar(x + off, [MODES[m][kk] for kk, _ in keys], width=.25, color=c, label={"pke": "PKE", "ridge": "ridge", "additive": "additive"}[m])
ax.axhline(0, color=INK2, lw=0.6); ax.set_xticks(x); ax.set_xticklabels([l for _, l in keys], fontsize=8.5, rotation=30, ha="right")
ax.set_ylim(-.15, 1); grid(ax); ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.0), fontsize=8.5, ncol=3)

fig.savefig("fig2_benchmark.pdf", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Figure 3: interpretability + validation
fig, axs = plt.subplots(2, 2, figsize=(7.6, 7.6), gridspec_kw=dict(hspace=0.62, wspace=0.6))
ax = axs[0, 0]; tag(ax, "a")
genes = [g for g, _ in PROP_TOP][::-1]; vals = [v for _, v in PROP_TOP][::-1]
cols = [BLUE if g.startswith("CEBP") else GRAY for g in genes]
ax.barh(range(len(genes)), vals, color=cols, height=.7); ax.set_yticks(range(len(genes)))
ax.set_yticklabels(genes, fontsize=8.5, style="italic"); ax.set_xlabel("interaction propensity $\\|g(a)\\|$"); grid(ax, "x")

ax = axs[0, 1]; tag(ax, "b")
S = SCREEN_TOP[:15][::-1]; y = np.arange(len(S))
ax.barh(y, [s[2] for s in S], color=BLUE, height=.7, label="gene-level part")
ax.barh(y, [s[3] for s in S], left=[s[2] + .03 for s in S], color=ORANGE, height=.7, label="pair-specific part")
ax.set_yticks(y); ax.set_yticklabels([s[0] for s in S], fontsize=8.5, style="italic"); ax.set_xlabel("norm of predicted part")
grid(ax, "x"); ax.legend(loc="upper center", bbox_to_anchor=(.35, -.2), ncol=2, fontsize=8.5)
ax = axs[1, 0]; tag(ax, "c")
yy = 0; ticks = []; labs = []
for pair, e in EXPLAIN.items():
    sup = e["sup"][:5][::-1]
    for (p, s) in sup:
        ax.barh(yy, s, color=ORANGE if e["inter"] > .5 else BLUE, height=.7); ticks.append(yy); labs.append(p); yy += 1
    ax.text(0.0, yy + .2, pair, fontsize=9, fontweight="bold", style="italic")
    yy += 1.6
ax.set_yticks(ticks); ax.set_yticklabels(labs, fontsize=8.5, style="italic"); ax.set_xlabel("representer share $s_i(u)$")
ax.set_ylim(-.8, yy); grid(ax, "x"); 
ax = axs[1, 1]; tag(ax, "d")
names = list(EXT); x = np.arange(len(names))
ax.bar(x - .2, [EXT[n]["rho"] for n in names], width=.38, color=BLUE, label="Spearman ρ")
ax.bar(x + .2, [EXT[n]["partial"] or 0 for n in names], width=.38, color=AQUA, label="partial ρ (vs single effect)")
ax.axhline(0, color=INK2, lw=0.6); ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5, rotation=30, ha="right")
ax.set_ylim(-.07, .14); ax.set_ylabel("association with |growth GI|"); grid(ax); ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=1, fontsize=8.5)

fig.savefig("fig3_interpret.pdf", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Figure 4: transfer + similarity bound
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw=dict(wspace=0.35))
ax = axs[0]; tag(ax, "a"); rng = np.random.default_rng(0)
groups = [("observed", [t[0] for t in T_I2N], GRAY), ("in\n(neuron)", [t[1] for t in T_I2N], BLUE),
          ("cross\n(neuron)", [t[2] for t in T_I2N], ORANGE), ("in\n(iPSC)", [t[1] for t in T_N2I], BLUE),
          ("cross\n(iPSC)", [t[2] for t in T_N2I], ORANGE)]
for i, (lab, v, c) in enumerate(groups):
    ax.scatter(i + rng.uniform(-.17, .17, len(v)), v, s=9, color=c, alpha=.7, lw=0)
    ax.plot([i - .25, i + .25], [np.median(v)] * 2, color=INK, lw=1.2)
ax.axhline(0, color=INK2, lw=0.6); ax.set_xticks(range(5)); ax.set_xticklabels([g[0] for g in groups], fontsize=8.5)
ax.set_ylabel("cosine with observed residual"); grid(ax)
f1 = np.mean([t[1] > t[2] for t in T_I2N]); f2 = np.mean([t[1] > t[2] for t in T_N2I])

ax = axs[1]; tag(ax, "b"); x = np.arange(4)
ax.bar(x - .2, [SIM[d][1] for d in DS], width=.38, color=BLUE, label="more similar half")
ax.bar(x + .2, [SIM[d][2] for d in DS], width=.38, color=GRAY, label="less similar half")
ax.set_xticks(x); ax.set_xticklabels([DSL[d] for d in DS], fontsize=8.5); ax.set_ylim(0, 1.12)
ax.set_ylabel("relative error (held-out genes)"); grid(ax); ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=2, fontsize=8.5)

fig.savefig("fig4_transfer.pdf", bbox_inches="tight"); plt.close(fig)
print("medians", np.median([t[1] for t in T_I2N]), np.median([t[2] for t in T_I2N]), np.median([t[1] for t in T_N2I]), np.median([t[2] for t in T_N2I]), np.median([t[0] for t in T_I2N]), f1, f2)
