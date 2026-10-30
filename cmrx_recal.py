"""In-domain recalibration on real CMRxMotion: subject-level cal/test splits, risk by protocol and by IQA grade."""
import numpy as np
import conformal as cf
from analysis import load_pred, crc_threshold, QS
from analysis import subject as subject_of

BANK = np.load("/home/claude/data/banks/cmrx_real.npz", allow_pickle=True)


def cmrx_recal(tag, n_bins=3, alpha=0.1, k=1, predictor="auto", n_splits=50, frac=0.5, seed=0):
    p = load_pred(tag, "cmrx_real")
    assert len(p["vol"]) == len(BANK["vol"]) and (p["vol"] == BANK["vol"]).all()
    use_mot = predictor == "mot" or (predictor == "auto" and np.isfinite(p["mot"][:, 0]).all())
    q = p["mot"][:, 0] if use_mot else p["ent"]
    L = np.maximum(p["miss"][:, k], p["incl"][:, k])
    ok = np.isfinite(L[:, 0]) & (p["gt_area"][:, k] > 0)
    subj = np.array([subject_of(str(v)) for v in p["vol"]])
    grade = BANK["iqa"].astype(int)
    prot = p["protocol"]
    S = np.unique(subj[ok])
    rng = np.random.default_rng(seed)
    out = dict(risk=[], width=[], rg={g: [] for g in (1, 2, 3)}, wg={g: [] for g in (1, 2, 3)},
               rp={x: [] for x in ("BH", "HBH", "FB", "IB")}, wp={x: [] for x in ("BH", "HBH", "FB", "IB")})
    for _ in range(n_splits):
        cs = set(rng.choice(S, int(frac * len(S)), replace=False))
        c = ok & np.isin(subj, list(cs)); t = ok & ~np.isin(subj, list(cs))
        edges = cf.bin_edges(q[c], n_bins)
        bc = np.digitize(q[c], edges)
        th = np.array([crc_threshold(L[c][bc == b], alpha) for b in range(max(n_bins, 1))])
        tt = th[np.digitize(q[t], edges)]
        r = L[t][np.arange(t.sum()), np.searchsorted(QS, tt)]
        out["risk"].append(r.mean()); out["width"].append(2 * tt.mean())
        for g in (1, 2, 3):
            m = grade[t] == g
            out["rg"][g].append(r[m].mean()); out["wg"][g].append(2 * tt[m].mean())
        for x in out["rp"]:
            m = prot[t] == x
            out["rp"][x].append(r[m].mean()); out["wp"][x].append(2 * tt[m].mean())
    f = lambda a: float(np.nanmean(a))
    return dict(risk=f(out["risk"]), width=f(out["width"]),
                risk_by_grade={g: f(v) for g, v in out["rg"].items()}, width_by_grade={g: f(v) for g, v in out["wg"].items()},
                risk_by_prot={g: f(v) for g, v in out["rp"].items()}, width_by_prot={g: f(v) for g, v in out["wp"].items()},
                worst_grade=float(max(f(v) for v in out["rg"].values())),
                p_viol_g3=float(np.mean(np.array(out["rg"][3]) > alpha)))


if __name__ == "__main__":
    for tag in ["respire_s0", "unet_s0", "attunet_s0"]:
        for B in (1, 3):
            r = cmrx_recal(tag, B)
            print(tag, B, round(r["risk"], 3), round(r["width"], 2), {g: round(v, 3) for g, v in r["risk_by_grade"].items()},
                  {g: round(v, 2) for g, v in r["width_by_grade"].items()}, "worstG", round(r["worst_grade"], 3), "Pviol3", r["p_viol_g3"])


def ef_recal(tag, n_bins=3, alpha=0.1, predictor="auto", q0=2.0, scale=True, n_splits=50, frac=0.5, seed=0):
    """Study-level LVEF intervals recalibrated on real CMRxMotion with subject-level splits."""
    import pandas as pd
    from analysis import study_units
    rows = []
    for u in study_units(tag, "cmrx_real", predictor):
        edm, edp = np.interp(-q0, cf.TGRID, u["ed_curve"]), np.interp(q0, cf.TGRID, u["ed_curve"])
        esm, esp = np.interp(-q0, cf.TGRID, u["es_curve"]), np.interp(q0, cf.TGRID, u["es_curve"])
        w = 100 * ((1 - esm / max(edp, 1e-6)) - (1 - esp / max(edm, 1e-6))) if scale else 1.0
        rows.append(dict(subject=u["subject"], protocol=u["protocol"], q=u["q"], err=abs(u["ef_pred"] - u["ef_true"]),
                         w=max(w, 1e-3)))
    d = pd.DataFrame(rows)
    d["s"] = d["err"] / d["w"]
    S = d["subject"].unique()
    rng = np.random.default_rng(seed)
    cov, wid, n_test = [], [], []
    for _ in range(n_splits):
        cs = rng.choice(S, int(frac * len(S)), replace=False)
        c, t = d[d.subject.isin(cs)], d[~d.subject.isin(cs)]
        edges, th = cf.calibrate(c["s"].values, c["q"].values, alpha, n_bins)
        half = cf.thresholds_for(t["q"].values, edges, th) * t["w"].values
        cov.append(float((t["err"].values <= half).mean())); wid.append(float(np.median(2 * half))); n_test.append(len(t))
    return dict(coverage=float(np.mean(cov)), width=float(np.mean(wid)), n_units=len(d), n_test=float(np.mean(n_test)))


def ef_crossfit(tag, n_bins=3, alpha=0.1, predictor="auto", q0=2.0, seed=0):
    """Two-fold cross-fitted LVEF intervals on CMRxMotion: each half of the volunteers is calibrated on the other."""
    import pandas as pd
    from analysis import study_units
    rows = []
    for u in study_units(tag, "cmrx_real", predictor):
        edm, edp = np.interp(-q0, cf.TGRID, u["ed_curve"]), np.interp(q0, cf.TGRID, u["ed_curve"])
        esm, esp = np.interp(-q0, cf.TGRID, u["es_curve"]), np.interp(q0, cf.TGRID, u["es_curve"])
        w = 100 * ((1 - esm / max(edp, 1e-6)) - (1 - esp / max(edm, 1e-6)))
        rows.append(dict(subject=u["subject"], protocol=u["protocol"], q=u["q"], ef_pred=u["ef_pred"],
                         ef_true=u["ef_true"], w=max(w, 1e-3)))
    d = pd.DataFrame(rows)
    d["s"] = (d["ef_pred"] - d["ef_true"]).abs() / d["w"]
    S = d["subject"].unique()
    A_ = set(np.random.default_rng(seed).choice(S, len(S) // 2, replace=False))
    fold = d["subject"].map(lambda x: x in A_).values
    out = []
    for f in (True, False):
        c, t = d[fold == f], d[fold != f].copy()
        edges, th = cf.calibrate(c["s"].values, c["q"].values, alpha, n_bins)
        half = cf.thresholds_for(t["q"].values, edges, th) * t["w"].values
        t["lo"], t["hi"], t["width"] = t["ef_pred"] - half, t["ef_pred"] + half, 2 * half
        t["cov"] = (t["ef_true"] >= t["lo"]) & (t["ef_true"] <= t["hi"])
        out.append(t)
    return pd.concat(out)
