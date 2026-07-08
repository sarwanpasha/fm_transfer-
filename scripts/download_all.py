"""
Download all datasets and model checkpoints.

Run on a machine with internet access (e.g. a cluster login node).
Requires FMTRANSFER_ROOT to be set (see config/paths.py).

Downloads:
  1. Norman 2019 combinatorial Perturb-seq (via GEARS PertData)
  2. scGPT whole-human checkpoint (via gdown, from the public scGPT repo)
  3. Geneformer weights (via HuggingFace snapshot)
  4. Wessels/CaRPool-seq (via Zenodo/scPerturb; see download_wessels.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import ROOT

os.environ.setdefault("GEARS_DATA", ROOT)
os.makedirs(ROOT, exist_ok=True)

# --- 1. GEARS perturbation data: Norman 2019 ---
from gears import PertData

pert_data = PertData(ROOT)
try:
    print("[GEARS] loading norman ...", flush=True)
    pert_data.load(data_name="norman")
    print("[GEARS] norman OK", flush=True)
except Exception as e:
    print(f"[GEARS] norman FAILED: {e}", flush=True)

# --- 2. scGPT whole-human checkpoint (public Google Drive folder) ---
import gdown

scgpt_dir = os.path.join(ROOT, "scGPT_human")
os.makedirs(scgpt_dir, exist_ok=True)
# Public whole-human checkpoint folder id from the scGPT repository.
WHOLE_HUMAN_FOLDER = "1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
try:
    gdown.download_folder(id=WHOLE_HUMAN_FOLDER, output=scgpt_dir,
                          quiet=False, use_cookies=False)
    print("[scGPT] download OK", flush=True)
except Exception as e:
    print(f"[scGPT] folder download FAILED: {e}", flush=True)
    print("        Fallback: download the whole-human checkpoint manually from "
          "the scGPT GitHub 'Pretrained scGPT checkpoints' table and place "
          "best_model.pt, args.json, vocab.json into", scgpt_dir, flush=True)

# --- 3. Geneformer (HuggingFace) ---
from huggingface_hub import snapshot_download

try:
    p = snapshot_download(repo_id="ctheodoris/Geneformer",
                          cache_dir=os.path.join(ROOT, "hf_cache"))
    print(f"[Geneformer] OK at {p}", flush=True)
except Exception as e:
    print(f"[Geneformer] FAILED: {e}", flush=True)

print("DONE. For the Wessels dataset, run scripts/download_wessels.py", flush=True)
