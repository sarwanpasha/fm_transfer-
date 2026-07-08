"""
Download the Wessels 2022/2023 CaRPool-seq dataset (THP-1, combinatorial
CRISPR-Cas13) from the scPerturb harmonized repository on Zenodo.

Run on a machine with internet access. Requires FMTRANSFER_ROOT.
"""
import os
import sys
import hashlib
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import ROOT

DEST = os.path.join(ROOT, "wessels")
os.makedirs(DEST, exist_ok=True)
URL = "https://zenodo.org/records/13350497/files/WesselsSatija2023.h5ad?download=1"
MD5 = "6897bfdcda928a678208fecf4eeb282e"
path = os.path.join(DEST, "WesselsSatija2023.h5ad")

if not os.path.exists(path):
    print(f"[wessels] downloading {URL}", flush=True)
    urllib.request.urlretrieve(URL, path)
else:
    print("[wessels] already downloaded", flush=True)

h = hashlib.md5()
with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
print(f"[wessels] md5 {'OK' if h.hexdigest() == MD5 else 'MISMATCH: ' + h.hexdigest()}",
      flush=True)
print("[wessels] DONE", flush=True)
