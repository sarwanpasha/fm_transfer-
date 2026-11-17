"""Collect the per-run results needed for the figures into results/figdata.json."""
import csv, glob, json, os

W = os.path.join(os.environ.get("FMTRANSFER_ROOT", "."), "results")
r4 = lambda x: round(float(x), 4)


def tsv(path):
    return list(csv.DictReader(open(path), delimiter="\t"))


def main():
    out = {"bench": {}}
    for ds in ("Norman", "Wessels", "TianIPSC", "TianNeuron"):
        out["bench"][ds] = {}
        for sc in ("random", "genegrp"):
            runs = []
            for f in sorted(glob.glob(f"{W}/{ds}_s*_mc500/pke.json")):
                j = json.load(open(f))[sc]
                rec = {k: dict(r2=r4(v["r2"]), d=r4(v["d"]), lo=r4(v["lo"]), hi=r4(v["hi"])) for k, v in j.items()}
                g = f.replace("pke.json", "gears.json")
                if os.path.exists(g) and json.load(open(g)).get(sc):
                    G = json.load(open(g))[sc]
                    rec["gears"] = dict(d=r4(G["d"]), d_gi=r4(G["d_gi"]))
                runs.append(rec)
            out["bench"][ds][sc] = runs
    N = f"{W}/Norman_s0_mc500/bio"
    out["prop"] = [[r["gene"], r4(r["propensity"]), r["lgo_pred_dev"], r["lgo_obs_dev"], r["n_obs_doubles"]] for r in tsv(N + "/propensity.tsv")]
    out["screen"] = [[r["gene_a"], r["gene_b"], r4(r["pred_dev_norm"]), r4(r["main_part_norm"]), r4(r["int_part_norm"])] for r in tsv(N + "/screen.tsv")]
    out["explain"] = json.load(open(N + "/explain.json"))["observed"]
    b = json.load(open(f"{W}/bio2.json"))
    out["transfer"] = {k: v["rows"] for k, v in b["transfer"].items()}
    out["sim"] = b["similarity"]
    out["modes"] = json.load(open(N + "/modes_cv.json"))["modes"]
    out["ext"] = json.load(open(f"{W}/external_norman.json"))
    out["pkefm"] = {ds: [json.load(open(f)) for f in sorted(glob.glob(f"{W}/{ds}_s*_mc500/pkefm.json"))]
                    for ds in ("Norman", "Wessels", "TianIPSC", "TianNeuron")}
    json.dump(out, open(f"{W}/figdata.json", "w"), separators=(",", ":"))
    print("wrote", f"{W}/figdata.json")


if __name__ == "__main__":
    main()
