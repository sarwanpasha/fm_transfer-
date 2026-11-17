#!/usr/bin/env bash
# End-to-end pipeline. Set FMTRANSFER_ROOT to a directory containing data/ (scPerturb .h5ad files and,
# for the external validation, data/external/<growth GI table>.xlsx). Results are written to $FMTRANSFER_ROOT/results.
set -euo pipefail
: "${FMTRANSFER_ROOT:?set FMTRANSFER_ROOT}"
cd "$(dirname "$0")"
DATASETS="Norman Wessels TianIPSC TianNeuron"
for ds in $DATASETS; do for s in 0 1 2 3 4; do python prepare.py --dataset $ds --seed $s; done; done
for r in "$FMTRANSFER_ROOT"/results/*_mc500; do
  python fm_pke.py --run "$r"            # PKE, baselines, ablations
  python fm_bio.py --run "$r"            # GI modes, screen, propensity, explanations
  python fm_ceiling.py --run "$r"        # split-half reliability of the residuals
done
python fm_bio2.py --wd "$FMTRANSFER_ROOT"  # iPSC/neuron transfer and similarity analysis
python fm_ext.py --wd "$FMTRANSFER_ROOT" --gi "${GROWTH_GI_XLSX:-data/external/aax4438_tables5.xlsx}" \
  --format norman --out external_norman.json --datasets Norman
# Optional: GEARS baseline (requires cell-gears) and foundation-model kernels (requires embeddings in results/emb/)
#   python fm_gears.py --run <run> ; python fm_gears_merge.py --run <run>
#   python fm_pkefm.py --run <run>
python fm_final_table.py "$FMTRANSFER_ROOT"
python fm_figdata.py
