"""Deep-learning baselines on the same folds as fm_pke.py (fold seed 0), trained on cells (CPU):
 cpa      : Compositional Perturbation Autoencoder (cpa-tools; hyperparameters of the official Norman tutorial), counts, NB loss
 gperturb : GPerturb Gaussian model on log-normalized expression, two-hot perturbation vectors for doubles
Both models are fit on control, all single and the training double cells, restricted to the evaluation gene universe (union of
the top-50 DE genes and the singles-ranked top-50 genes of all doubles, filled to 2,000 with the most variable genes).
Predicted double and single profiles are means over predictions for 300 control cells; the residual is scored on each double's
top-50 DE genes against observed singles (obs) and against the model's own single predictions (own).
Output: <run>/dl_<model>_<scheme>_f<fold>.npz.  Usage: fm_dl.py --run R --model cpa|gperturb --scheme random|genegrp --fold k"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FILES = {"Norman": "NormanWeissman2019_filtered.h5ad", "Wessels": "WesselsSatija2023.h5ad",
         "TianIPSC": "TianKampmann2019_iPSC.h5ad", "TianNeuron": "TianKampmann2019_day7neuron.h5ad"}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--model", required=True)
    p.add_argument("--scheme", required=True); p.add_argument("--fold", type=int, required=True); p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--wd", default=os.environ.get("FMTRANSFER_ROOT", ".")); a = p.parse_args()
    import fmcore as F, fmpredict as FP, anndata as ad, scanpy as sc, scipy.sparse as sp
    from fm_gira import gene_feats, ridge_pairs
    t0 = time.time(); run = a.run.rstrip("/")
    S = json.load(open(os.path.join(run, "summary.json"))); Z = np.load(os.path.join(run, "pb_hvg.npz"), allow_pickle=True)
    conds = list(Z["conds"]); expr = {c: Z["pb"][i].astype(np.float64) for i, c in enumerate(conds)}; hn = np.asarray(Z["genes"])
    singles, doubles, _ = F.parse_conditions(conds); R = np.asarray(F.residual_exact(expr, "control", doubles), np.float64)
    DE = FP.de_sets(expr, "control", doubles, hn, 50); pos = {g: i for i, g in enumerate(hn)}
    uni = set(np.concatenate(DE).tolist())
    for d, x, y in doubles:
        s = np.maximum(np.abs(expr[x] - expr["control"]), np.abs(expr[y] - expr["control"])).astype(float)
        for g in (x, y):
            if g in pos: s[pos[g]] = -np.inf
        uni |= set(np.argpartition(-s, 50)[:50].tolist())
    var = np.stack([expr[c] for c in conds]).var(0)
    for i in np.argsort(-var):
        if len(uni) >= 2000: break
        uni.add(int(i))
    gsel = np.array(sorted(uni)); gmap = {g: k for k, g in enumerate(gsel)}
    split = list(FP.splits(doubles, a.scheme, 5, 0))[a.fold]; tr, te = split
    idx = np.load(os.path.join(run, "subsample.npz"))["idx"]
    H = ad.read_h5ad(os.path.join(a.wd, "data", FILES[S["dataset"]]), backed="r"); X = H[idx].to_memory(); H.file.close(); X.var_names_make_unique()
    lab = X.obs["perturbation"].astype(str).values
    col = {g: i for i, g in enumerate(X.var_names)}; cols = np.array([col[g] for g in hn[gsel]])
    Craw = sp.csr_matrix(X.X if sp.issparse(X.X) else sp.csr_matrix(X.X), dtype=np.float32); Craw.data = np.round(Craw.data)
    lib = np.asarray(Craw.sum(1)).ravel(); C = Craw[:, cols].tocsr()
    L = C.multiply(1e4 / np.maximum(lib, 1)[:, None]).tocsr(); L.data = np.log1p(L.data)
    trd = {doubles[i][0] for i in tr}; ted = [doubles[i] for i in te]
    keep = np.array([(l == "control") or (l in singles) or (l in trd) for l in lab])
    ctrl_ix = np.flatnonzero(lab == "control"); rng = np.random.default_rng(0); cf = rng.choice(ctrl_ix, min(300, len(ctrl_ix)), replace=False)
    print(f"{os.path.basename(run)} {a.model} {a.scheme} f{a.fold}: train cells={keep.sum()} genes={len(gsel)} test doubles={len(ted)}", flush=True)
    targets = [d for d, _, _ in ted] + list(singles)
    pred = {}
    if a.model == "cpa":
        import cpa
        import scipy.sparse as sps
        lab_cpa = np.array(["ctrl" if l == "control" else "+".join(l.split("_")) for l in lab])
        # counterfactual rows: control cells relabelled with every target condition, registered with the training data so that
        # all perturbation categories are known, but held out of training and validation (split "ood")
        tnames = ["+".join(t.split("_")) for t in targets] + ["ctrl"]
        Xa = sps.vstack([C[keep]] + [C[cf]] * len(tnames)).tocsr()
        conds_a = np.concatenate([lab_cpa[keep]] + [np.array([nm] * len(cf)) for nm in tnames])
        split_a = np.concatenate([np.where(rng.random(int(keep.sum())) < 0.85, "train", "valid"), np.array(["ood"] * (len(cf) * len(tnames)))])
        A = ad.AnnData(X=Xa, obs={"cond_harm": conds_a, "cell_type": ["cell"] * len(conds_a), "split": split_a})
        A.obs["dose_value"] = ["+".join(["1.0"] * len(c.split("+"))) for c in A.obs["cond_harm"]]
        A.var_names = list(hn[gsel]); A.obs["smiles"] = A.obs["cond_harm"].astype(str)
        cpa.CPA.setup_anndata(A, perturbation_key="cond_harm", control_group="ctrl", dosage_key="dose_value",
                              categorical_covariate_keys=["cell_type"], is_count_data=True, max_comb_len=2, smiles_key="smiles")
        mp = dict(n_latent=32, recon_loss="nb", doser_type="linear", n_hidden_encoder=256, n_layers_encoder=4, n_hidden_decoder=256,
                  n_layers_decoder=2, use_batch_norm_encoder=True, use_layer_norm_encoder=False, use_batch_norm_decoder=False,
                  use_layer_norm_decoder=False, dropout_rate_encoder=0.2, dropout_rate_decoder=0.0, variational=False, seed=8206)
        tp = dict(n_epochs_kl_warmup=None, n_epochs_adv_warmup=50, n_epochs_mixup_warmup=10, n_epochs_pretrain_ae=10, mixup_alpha=0.1, lr=1e-4,
                  wd=3.2170178270865573e-06, adv_steps=3, reg_adv=10.0, pen_adv=20.0, adv_lr=1e-4, adv_wd=7.051355554517135e-06, n_layers_adv=2,
                  n_hidden_adv=128, use_batch_norm_adv=True, use_layer_norm_adv=False, dropout_rate_adv=0.3, step_size_lr=25, do_clip_grad=False,
                  adv_loss="cce", gradient_clip_value=5.0)
        model = cpa.CPA(adata=A, split_key="split", train_split="train", valid_split="valid", test_split="ood", **mp)
        model.train(max_epochs=a.epochs, use_gpu=False, batch_size=1024, plan_kwargs=tp, early_stopping_patience=5, check_val_every_n_epoch=5)
        B = A[A.obs["split"] == "ood"].copy()
        model.predict(B, batch_size=1024); Pall = np.asarray(B.obsm["CPA_pred"], np.float64); cb = B.obs["cond_harm"].astype(str).values
        libc = np.maximum(lib[cf][:, None], 1)
        # the control is predicted by CPA as well (control cells passed through the model), so that any per-gene bias of the
        # reconstruction cancels in the model's own residual and shift, as for GPerturb
        for t, nm in zip(targets + ["control"], tnames):
            Pm = np.log1p(Pall[cb == nm] * 1e4 / libc)
            pred[t] = Pm.mean(0)
    else:
        import torch
        import glob, importlib.util
        src = [f for f in glob.glob(os.path.join(a.wd, "fm_gperturb", "**", "*.py"), recursive=True) if "def Gaussian_estimates" in open(f).read()][0]
        sys.path.insert(0, os.path.dirname(src)); sys.path.insert(0, os.path.dirname(os.path.dirname(src)))
        spec = importlib.util.spec_from_file_location("gpm", src); gpm = importlib.util.module_from_spec(spec); spec.loader.exec_module(gpm)
        GPerturb_Gaussian, Gaussian_estimates = gpm.GPerturb_Gaussian, gpm.Gaussian_estimates
        print("GPerturb module", src, flush=True)
        torch.manual_seed(0); torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
        sing = sorted(singles); si = {s: k for k, s in enumerate(sing)}
        def cvec(l):
            v = np.zeros(len(sing), np.float32)
            if l != "control":
                for g in l.split("_"): v[si[g]] = 1.0
            return v
        kk = np.flatnonzero(keep); obs = torch.tensor(L[kk].toarray(), dtype=torch.float32)
        cond = torch.tensor(np.stack([cvec(lab[i]) for i in kk]), dtype=torch.float32)
        info = torch.tensor(np.stack([np.log1p(lib[kk]), np.asarray(C[kk].sum(1)).ravel() / np.maximum(lib[kk], 1)], 1), dtype=torch.float32)
        model = GPerturb_Gaussian(conditioner_dim=len(sing), output_dim=obs.shape[1], base_dim=2, data_size=obs.shape[0],
                                  hidden_node=256, hidden_layer_1=2, hidden_layer_2=2, tau=torch.tensor(1.))
        model.GPerturb_train(epoch=a.epochs, observation=obs, cell_info=info, perturbation=cond, lr=1e-3, device="cpu")
        model.eval()
        cinfo = torch.tensor(np.stack([np.log1p(lib[cf]), np.asarray(C[cf].sum(1)).ravel() / np.maximum(lib[cf], 1)], 1), dtype=torch.float32)
        with torch.no_grad():
            for t in targets + ["control"]:
                cc = torch.tensor(np.repeat(cvec(t)[None], len(cf), 0)); est = Gaussian_estimates(model, torch.zeros(len(cf), obs.shape[1]), cc, cinfo)
                pred[t] = np.asarray(est["pert_mean"], np.float64).mean(0)
    f20 = gene_feats({s: expr[s] - expr["control"] for s in singles}, 20)
    Fr = np.stack([FP.pair_features(f20[x], f20[y]) for _, x, y in doubles]); Pr = ridge_pairs(Fr[tr], Fr[te], R[tr])
    acc = {m: np.zeros((2, len(doubles))) for m in ("dl_obs", "dl_own", "ridge_raw", "dl_shift", "dl_shift_own", "additive_shift")}
    for j, i in enumerate(te):
        d, x, y = doubles[i]; ix = [gmap[g] for g in DE[i] if g in gmap]; gi = np.array([gsel[k] for k in ix]); ix = np.array(ix)
        t = R[i, gi]
        p_obs = pred[d][ix] - expr[x][gi] - expr[y][gi] + expr["control"][gi]
        p_own = pred[d][ix] - pred[x][ix] - pred[y][ix] + pred["control"][ix]
        ts = (expr[d] - expr["control"])[gi]
        for m, pp, tt in (("dl_obs", p_obs, t), ("dl_own", p_own, t), ("ridge_raw", Pr[j, gi], t),
                          ("dl_shift", pred[d][ix] - expr["control"][gi], ts), ("dl_shift_own", pred[d][ix] - pred["control"][ix], ts), ("additive_shift", (expr[x] + expr[y] - 2 * expr["control"])[gi], ts)):
            e = tt - pp; acc[m][:, i] += (e @ e, tt @ tt)
    pd20 = np.full((4, len(doubles)), np.nan)
    for i in te:
        d, x, y = doubles[i]; sh = (expr[d] - expr["control"])[gsel]; top = np.argsort(-np.abs(sh))[:20]
        pm = pred[d][top] - pred["control"][top]; pa = (expr[x] + expr[y] - 2 * expr["control"])[gsel][top]
        pd20[:, i] = (np.corrcoef(pm, sh[top])[0, 1], np.corrcoef(pa, sh[top])[0, 1], np.mean((pm - sh[top]) ** 2), np.mean((pa - sh[top]) ** 2))
    np.save(os.path.join(run, f"dl_{a.model}_{a.scheme}_f{a.fold}_pd20.npy"), pd20)
    np.savez(os.path.join(run, f"dl_{a.model}_{a.scheme}_f{a.fold}.npz"), **{m: v for m, v in acc.items()})
    r2 = {m: 1 - v[0].sum() / v[1].sum() for m, v in acc.items() if v[1].sum() > 0}
    print("DL", a.model, a.scheme, a.fold, json.dumps({k: round(v, 3) for k, v in r2.items()}), f"{time.time() - t0:.0f}s", flush=True); print("DL_DONE")


if __name__ == "__main__":
    main()
