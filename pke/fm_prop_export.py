"""Per-gene table for the propensity figure: Norman propensity (PKE on all doubles, subsample s0), single-perturbation magnitude,
observed transcriptional hubness and growth hubness from the growth GI map. Output: results/propensity_genes.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_propensity import full_props
W, gi_path = sys.argv[1], sys.argv[2]
import openpyxl
ws = openpyxl.load_workbook(gi_path, read_only=True).worksheets[0]; GI = {}
for i, r in enumerate(ws.iter_rows(values_only=True)):
    if i < 2 or r[0] is None or r[1] is None or r[0] == r[1] or r[4] is None: continue
    GI[frozenset((str(r[0]).upper(), str(r[1]).upper()))] = abs(float(r[4]))
genes = sorted({g for k in GI for g in k}); ghub = {g: float(np.mean([v for k, v in GI.items() if g in k])) for g in genes}
p, h, m = full_props(os.path.join(W, "results", "Norman_s0_mc500")); up = {g.upper(): g for g in p}
rows = [dict(gene=up[g], propensity=float(p[up[g]]), magnitude=m[up[g]], growth_hub=ghub[g],
             obs_hub=(h[up[g]][0] if up[g] in h else None), n_doubles=(h[up[g]][1] if up[g] in h else 0)) for g in genes if g in up]
json.dump(rows, open(os.path.join(W, "results", "propensity_genes.json"), "w"), indent=1); print(len(rows), "EXPORT_DONE")
