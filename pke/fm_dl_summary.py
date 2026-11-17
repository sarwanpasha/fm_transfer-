"""Summarize fm_dl.py outputs: R2 pooled over folds and subsamples per dataset, model and split."""
import glob, json, os, sys
import numpy as np
W = sys.argv[1] if len(sys.argv) > 1 else "."; out = {}
for ds in ["Norman", "Wessels", "TianIPSC", "TianNeuron"]:
    for model in ["gperturb", "cpa"]:
        for sc in ["random", "genegrp"]:
            fs = sorted(glob.glob(f"{W}/results/{ds}_s*_mc500/dl_{model}_{sc}_f*.npz"))
            if not fs: continue
            acc = {}
            for f in fs:
                Z = np.load(f)
                for k in Z.files: acc.setdefault(k, np.zeros(2)); acc[k] += Z[k].sum(1)
            r = {k: round(float(1 - v[0] / v[1]), 3) for k, v in acc.items()}; r["n_files"] = len(fs)
            out[f"{ds}|{model}|{sc}"] = r; print(ds, model, sc, json.dumps(r))
json.dump(out, open(f"{W}/results/dl_summary.json", "w"), indent=1); print("DL_SUMMARY_DONE")
