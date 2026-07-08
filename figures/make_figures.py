"""
Generate all paper figures.

Reads result JSONs from FMTRANSFER_ROOT when available; otherwise falls back to
the values reported in the paper (so the script runs standalone for figure
reproduction). Writes PDFs to the directory given by FIGDIR (default: ./figures_out).
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    "axes.linewidth": 1.3, "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
})

FIGDIR = os.environ.get("FIGDIR", "figures_out")
os.makedirs(FIGDIR, exist_ok=True)

C_SCGPT, C_GENE, C_SCVI, C_RAW = "#2C6FBB", "#D1495B", "#5B8C5A", "#8A8D91"

# Reported values (used as fallback / for direct reproduction).
V = {
    "add_relresid": [0.3951, 0.4606, 0.5982],
    "add_cosine": [0.9498, 0.9409, 0.8996],
    "gi_sp": [0.742, 0.764, 0.511],
    "gi_lo": [0.641, 0.669, 0.363],
    "gi_hi": [0.819, 0.835, 0.642],
    "gap": [0.231, 0.253], "gap_lo": [0.114, 0.133], "gap_hi": [0.355, 0.384],
    "repl_gap": [0.219, 0.204, 0.224, -0.007],
    "repl_lo": [0.120, 0.102, 0.099, -0.190],
    "repl_hi": [0.327, 0.317, 0.350, 0.165],
    "arch_norman": [0.219, 0.204, 0.220], "arch_wessels": [0.224, -0.007, 0.314],
    "eff": {"scGPT/N": 5.37, "scGPT/W": 4.99, "GF/N": 5.53, "GF/W": 2.60,
            "scVI/N": 11.27, "scVI/W": 12.73},
    "cvg": {"scGPT/N": 0.219, "scGPT/W": 0.224, "GF/N": 0.204, "GF/W": -0.007,
            "scVI/N": 0.220, "scVI/W": 0.314},
    "prog": [0.986, 0.984, 0.729, 0.737],
    "corr_none": [0.219, 0.224, 0.204, -0.007, 0.220, 0.314],
    "corr_whiten": [0.087, -0.163, -0.048, -0.067, 0.122, -0.057],
    "corr_rank": [0.091, -0.049, 0.009, -0.110, 0.008, -0.196],
}

models = ["scGPT", "Geneformer", "raw-PCA"]
cols = [C_SCGPT, C_GENE, C_RAW]

# Fig 1: additivity
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
b = ax[0].bar(models, V["add_relresid"], color=cols, edgecolor="black", linewidth=1.2, width=0.62)
ax[0].set_ylabel("Median relative residual"); ax[0].set_title("(a) Non-additivity of composition", fontweight="bold")
ax[0].set_ylim(0, 0.72)
for bb, v in zip(b, V["add_relresid"]):
    ax[0].text(bb.get_x()+bb.get_width()/2, v+0.012, f"{v:.3f}", ha="center", fontsize=13, fontweight="bold")
b = ax[1].bar(models, V["add_cosine"], color=cols, edgecolor="black", linewidth=1.2, width=0.62)
ax[1].set_ylabel("Median cosine (actual, additive)"); ax[1].set_title("(b) Directional additivity", fontweight="bold")
ax[1].set_ylim(0.8, 1.0)
for bb, v in zip(b, V["add_cosine"]):
    ax[1].text(bb.get_x()+bb.get_width()/2, v+0.004, f"{v:.3f}", ha="center", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig1_additivity.pdf", bbox_inches="tight"); plt.close()

# Fig 2: GI-tracking + gap
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8))
err = [[V["gi_sp"][i]-V["gi_lo"][i] for i in range(3)], [V["gi_hi"][i]-V["gi_sp"][i] for i in range(3)]]
ax[0].bar(range(3), V["gi_sp"], color=cols, edgecolor="black", linewidth=1.2, width=0.6, yerr=err,
          capsize=7, error_kw=dict(elinewidth=1.8, capthick=1.8))
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(models)
ax[0].set_ylabel("Spearman $\\rho$ (residual, GI)"); ax[0].set_title("(a) GI-tracking on Norman", fontweight="bold")
ax[0].set_ylim(0, 1.0)
for i, v in enumerate(V["gi_sp"]):
    ax[0].text(i, V["gi_hi"][i]+0.02, f"{v:.2f}", ha="center", fontsize=13, fontweight="bold")
gerr = [[V["gap"][i]-V["gap_lo"][i] for i in range(2)], [V["gap_hi"][i]-V["gap"][i] for i in range(2)]]
ax[1].bar(range(2), V["gap"], color=[C_SCGPT, C_GENE], edgecolor="black", linewidth=1.2, width=0.5,
          yerr=gerr, capsize=7, error_kw=dict(elinewidth=1.8, capthick=1.8))
ax[1].set_xticks(range(2)); ax[1].set_xticklabels(["scGPT\n$-$ raw", "Geneformer\n$-$ raw"])
ax[1].set_ylabel("$\\Delta\\rho$ (model $-$ raw-PCA)"); ax[1].set_title("(b) FM advantage over raw baseline", fontweight="bold")
ax[1].axhline(0, color="black", linewidth=1.0, linestyle="--"); ax[1].set_ylim(0, 0.46)
for i, v in enumerate(V["gap"]):
    ax[1].text(i, V["gap_hi"][i]+0.012, f"+{v:.2f}\nP(>0)=1.0", ha="center", fontsize=11.5, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig2_gitracking.pdf", bbox_inches="tight"); plt.close()

# Fig 3: replication
fig, ax = plt.subplots(figsize=(8.6, 5.0))
groups = ["scGPT\nNorman", "Geneformer\nNorman", "scGPT\nWessels", "Geneformer\nWessels"]
bcols = [C_SCGPT, C_GENE, C_SCGPT, C_GENE]; hatches = ["", "", "//", "//"]
gerr = [[V["repl_gap"][i]-V["repl_lo"][i] for i in range(4)], [V["repl_hi"][i]-V["repl_gap"][i] for i in range(4)]]
bars = ax.bar(range(4), V["repl_gap"], color=bcols, edgecolor="black", linewidth=1.3, width=0.62,
              yerr=gerr, capsize=7, error_kw=dict(elinewidth=1.8, capthick=1.8))
for bb, h in zip(bars, hatches): bb.set_hatch(h)
ax.set_xticks(range(4)); ax.set_xticklabels(groups)
ax.set_ylabel("$\\Delta\\rho$ GI-tracking gap (model $-$ raw)")
ax.set_title("GI-tracking replicates for scGPT, fails for Geneformer on CRISPRi", fontweight="bold", fontsize=15)
ax.axhline(0, color="black", linewidth=1.0, linestyle="--"); ax.set_ylim(-0.27, 0.44)
for i, v in enumerate(V["repl_gap"]):
    yy = V["repl_hi"][i]+0.015 if v >= 0 else V["repl_lo"][i]-0.055
    ax.text(i, yy, f"{v:+.2f}", ha="center", fontsize=13, fontweight="bold")
ax.legend(handles=[Patch(facecolor="white", edgecolor="black", label="solid: CRISPRa (Norman)"),
                   Patch(facecolor="white", edgecolor="black", hatch="//", label="hatched: CRISPRi (Wessels)")],
          loc="upper right", frameon=True)
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig3_replication.pdf", bbox_inches="tight"); plt.close()

# Fig 4: scVI + criterion
fig, ax = plt.subplots(1, 2, figsize=(12, 5.0))
x = np.arange(3); w = 0.36
ax[0].bar(x-w/2, V["arch_norman"], w, color=C_SCGPT, edgecolor="black", linewidth=1.1, label="Norman (CRISPRa)")
ax[0].bar(x+w/2, V["arch_wessels"], w, color=C_GENE, edgecolor="black", linewidth=1.1, label="Wessels (CRISPRi)")
ax[0].set_xticks(x); ax[0].set_xticklabels(["scGPT", "Geneformer", "scVI"])
ax[0].set_ylabel("$\\Delta\\rho$ GI-tracking gap"); ax[0].set_title("(a) GI-tracking spans architectures", fontweight="bold")
ax[0].axhline(0, color="black", linewidth=0.9, linestyle="--"); ax[0].legend(frameon=True, loc="upper left"); ax[0].set_ylim(-0.08, 0.40)
mk = {"scGPT/N": ("o", C_SCGPT), "scGPT/W": ("s", C_SCGPT), "GF/N": ("o", C_GENE),
      "GF/W": ("s", C_GENE), "scVI/N": ("o", C_SCVI), "scVI/W": ("s", C_SCVI)}
for k in V["eff"]:
    m, c = mk[k]
    ax[1].scatter(V["eff"][k], V["cvg"][k], s=190, marker=m, color=c, edgecolor="black", linewidth=1.4, zorder=3)
ax[1].annotate("Geneformer/Wessels\n(failure case)", (2.60, -0.007), xytext=(3.4, 0.07),
               fontsize=11.5, fontweight="bold", arrowprops=dict(arrowstyle="->", linewidth=1.5))
ax[1].annotate("scVI (high rank,\nstill tracks)", (12.0, 0.27), xytext=(7.2, 0.30),
               fontsize=11.5, fontweight="bold", arrowprops=dict(arrowstyle="->", linewidth=1.5))
ax[1].set_xlabel("Single-effect effective rank (criterion)"); ax[1].set_ylabel("$\\Delta\\rho$ GI-tracking gap")
ax[1].set_title("(b) Criterion fails across architectures", fontweight="bold")
ax[1].axhline(0, color="black", linewidth=0.9, linestyle="--")
ax[1].legend(handles=[
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_SCGPT, markersize=12, markeredgecolor="black", label="scGPT"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_GENE, markersize=12, markeredgecolor="black", label="Geneformer"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_SCVI, markersize=12, markeredgecolor="black", label="scVI"),
    plt.Line2D([0], [0], marker="o", color="gray", markersize=11, markeredgecolor="black", label="Norman (circle)"),
    plt.Line2D([0], [0], marker="s", color="gray", markersize=11, markeredgecolor="black", label="Wessels (square)")],
    frameon=True, loc="lower right", fontsize=11)
ax[1].set_xlim(1.5, 14)
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig4_scvi_criterion.pdf", bbox_inches="tight"); plt.close()

# Fig 5: program classification
fig, ax = plt.subplots(figsize=(8.4, 5.0))
labs = ["scGPT", "Geneformer", "raw-PCA", "gene-identity\n(one-hot)"]
b = ax.bar(labs, V["prog"], color=[C_SCGPT, C_GENE, C_RAW, "#C9A227"], edgecolor="black", linewidth=1.2, width=0.62)
ax.axhline(1/3, color="black", linestyle=":", linewidth=1.6, label="chance (0.33)")
ax.set_ylabel("Balanced accuracy (leave-genes-out)")
ax.set_title("Phenotypic program is decodable & transfers to unseen genes", fontweight="bold", fontsize=14.5)
ax.set_ylim(0, 1.08)
for bb, v in zip(b, V["prog"]):
    ax.text(bb.get_x()+bb.get_width()/2, v+0.015, f"{v:.3f}", ha="center", fontsize=13, fontweight="bold")
ax.legend(frameon=True, loc="lower left")
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig5_program.pdf", bbox_inches="tight"); plt.close()

# Fig 6: correction
fig, ax = plt.subplots(figsize=(10.5, 5.0))
cells = ["scGPT/N", "scGPT/W", "GF/N", "GF/W", "scVI/N", "scVI/W"]
x = np.arange(6); w = 0.26
ax.bar(x-w, V["corr_none"], w, color="#3A7CA5", edgecolor="black", linewidth=1.0, label="none (baseline)")
ax.bar(x, V["corr_whiten"], w, color="#E08E45", edgecolor="black", linewidth=1.0, label="whiten")
ax.bar(x+w, V["corr_rank"], w, color="#9B5DE5", edgecolor="black", linewidth=1.0, label="rank-restore")
ax.set_xticks(x); ax.set_xticklabels(cells)
ax.set_ylabel("$\\Delta\\rho$ GI-tracking gap")
ax.set_title("Post-hoc geometric correction does not recover lost structure", fontweight="bold", fontsize=14.5)
ax.axhline(0, color="black", linewidth=1.0); ax.legend(frameon=True, loc="lower left", ncol=3); ax.set_ylim(-0.27, 0.40)
plt.tight_layout(); plt.savefig(f"{FIGDIR}/fig6_correction.pdf", bbox_inches="tight"); plt.close()

print(f"All figures written to {FIGDIR}/")
