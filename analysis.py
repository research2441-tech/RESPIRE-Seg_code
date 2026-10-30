"""Aggregate predictions into the tables used in the manuscript.

Every number in the paper is produced here from runs/<tag>/vol_*.json and pred_*.npz.
"""
import json
import re
import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

import conformal as cf

RUNS = os.environ.get("RESPIRE_RUNS", "/home/claude/runs")
OUT = os.environ.get("RESPIRE_RESULTS", "/home/claude/results")
os.makedirs(OUT, exist_ok=True)


def subject(vol):
    """Statistical unit: ACDC / M&Ms-2 patient, or CMRxMotion volunteer (all four scans of a volunteer pooled)."""
    v = vol.rsplit("_", 1)[0]
    if re.fullmatch(r"P\d+-\d", v):
        v = v.split("-")[0]
    return v


def load_vol(tag, bank):
    f = os.path.join(RUNS, tag, f"vol_{bank}.json")
    if not os.path.exists(f):
        return None
    d = pd.DataFrame(json.load(open(f)))
    d["subject"] = d["vol"].map(subject)
    d["phase"] = d["vol"].map(lambda v: v.rsplit("_", 1)[1])
    d["tag"] = tag
    return d


def seeds(prefix):
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(RUNS, prefix + "_s*")))


def seg_table(method_prefixes, bank, metrics=("dice_LV", "dice_MYO", "dice_RV", "hd95_LV", "hd95_MYO", "hd95_RV")):
    """Subject-level means (averaged over phases and seeds) per protocol."""
    rows = []
    for name, pre in method_prefixes.items():
        ds = [load_vol(t, bank) for t in seeds(pre)]
        ds = [d for d in ds if d is not None]
        if not ds:
            continue
        d = pd.concat(ds)
        d = d[d["has_gt"]]
        g = d.groupby(["protocol", "subject"])[list(metrics)].mean().reset_index()
        g["method"] = name
        g["n_seeds"] = len(ds)
        rows.append(g)
    return pd.concat(rows) if rows else pd.DataFrame()


def ef_frame(tag, bank):
    d = load_vol(tag, bank)
    if d is None:
        return None
    rows = []
    for (s, dr, _pr), g in d.groupby(["subject", "draw", "protocol"]):
        if set(g["phase"]) != {"ed", "es"}:
            continue
        ed, es = g[g.phase == "ed"].iloc[0], g[g.phase == "es"].iloc[0]
        r = dict(subject=s, draw=dr, protocol=ed["protocol"], tag=tag,
                 sev=0.5 * (ed["sev"] + es["sev"]), pred_sev=0.5 * (ed["pred_sev"] + es["pred_sev"]),
                 ent=0.5 * (ed["ent"] + es["ent"]))
        for k in ("LV", "RV"):
            r[f"EF_pred_{k}"] = 100 * (ed[f"Vpred_{k}"] - es[f"Vpred_{k}"]) / max(ed[f"Vpred_{k}"], 1e-6)
            r[f"EF_gt_{k}"] = 100 * (ed[f"Vgt_{k}"] - es[f"Vgt_{k}"]) / max(ed[f"Vgt_{k}"], 1e-6)
        r["ed_idx"], r["es_idx"] = ed["idx"], es["idx"]
        r["has_gt"] = bool(ed["has_gt"] and es["has_gt"])
        rows.append(r)
    return pd.DataFrame(rows)


def bland_altman(diff):
    diff = np.asarray(diff)
    b, sd = diff.mean(), diff.std(ddof=1)
    return b, b - 1.96 * sd, b + 1.96 * sd


def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = (x[:, None] > y[None]).mean()
    lt = (x[:, None] < y[None]).mean()
    return gt - lt


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    run = 0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1, run)
    return adj


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def load_pred(tag, bank, suffix=""):
    f = os.path.join(RUNS, tag, f"pred_{bank}{suffix}.npz")
    return dict(np.load(f, allow_pickle=True)) if os.path.exists(f) else None


def conformal_eval(tag, alpha=0.1, n_bins=3, predictor="auto", cal="acdc_cal", test="acdc_test", suffix=""):
    """Slice-level contour-band coverage per structure, overall and per protocol."""
    pc, pt = load_pred(tag, cal, suffix), load_pred(tag, test, suffix)
    use_mot = predictor == "mot" or (predictor == "auto" and np.isfinite(pc["mot"][:, 0]).all())
    qc = pc["mot"][:, 0] if use_mot else pc["ent"]
    qt = pt["mot"][:, 0] if use_mot else pt["ent"]
    out = {}
    for k, name in enumerate(["LV", "EPI", "RV"]):
        ok_c = np.isfinite(pc["scores"][:, k])
        edges, th = cf.calibrate(pc["scores"][ok_c, k], qc[ok_c], alpha, n_bins)
        t = cf.thresholds_for(qt, edges, th)
        ok = np.isfinite(pt["scores"][:, k])
        cov = (pt["scores"][ok, k] <= t[ok])
        prot = pt["protocol"][ok]
        band = np.array([cf.area_at(pt["area_curve"][i, k], t[i]) - cf.area_at(pt["area_curve"][i, k], -t[i])
                         for i in np.where(ok)[0]])
        out[name] = dict(coverage=float(cov.mean()), t_mean=float(np.mean(np.clip(t[ok], -15, 15))),
                         band_rel=float(np.median(band / np.maximum(pt["gt_area"][ok, k], 1))),
                         by_protocol={p: float(cov[prot == p].mean()) for p in np.unique(prot)},
                         t_by_protocol={p: float(np.mean(np.clip(t[ok][prot == p], -15, 15))) for p in np.unique(prot)},
                         worst_protocol=float(min(cov[prot == p].mean() for p in np.unique(prot))))
    return out


def ef_conformal(tag, alpha=0.1, n_bins=3, cal="acdc_cal", test="acdc_test", suffix="", predictor="auto"):
    """Study-level LVEF intervals calibrated on subject-draw units."""
    res = {}
    frames = {}
    for bank in (cal, test):
        p = load_pred(tag, bank, suffix)
        e = ef_frame(tag, bank) if not suffix else None
        if e is None:
            d = pd.DataFrame(json.load(open(os.path.join(RUNS, tag, f"vol_{bank}{suffix}.json"))))
            d["subject"] = d["vol"].map(subject)
            d["phase"] = d["vol"].map(lambda v: v.rsplit("_", 1)[1])
            e = ef_frame_from(d)
        use_mot = predictor == "mot" or (predictor == "auto" and np.isfinite(p["mot"][:, 0]).all())
        q = p["mot"][:, 0] if use_mot else p["ent"]
        rows = []
        for _, r in e[e.has_gt].iterrows():
            idx = list(r["ed_idx"]) + list(r["es_idx"])
            s = np.nanmax(p["scores"][idx, 0])
            rows.append(dict(subject=r["subject"], protocol=r["protocol"], score=s, q=float(np.mean(q[idx])),
                             ed=[p["area_curve"][i, 0] for i in r["ed_idx"]],
                             es=[p["area_curve"][i, 0] for i in r["es_idx"]], ef_gt=r["EF_gt_LV"],
                             ef_pred=r["EF_pred_LV"]))
        frames[bank] = rows
    sc = np.array([r["score"] for r in frames[cal]])
    qc = np.array([r["q"] for r in frames[cal]])
    edges, th = cf.calibrate(sc, qc, alpha, n_bins)
    cov, wid, prots = [], [], []
    for r in frames[test]:
        t = float(cf.thresholds_for(np.array([r["q"]]), edges, th)[0])
        lo, hi = cf.ef_interval(r["ed"], r["es"], t, 10.0)
        cov.append(lo <= r["ef_gt"] <= hi)
        wid.append(hi - lo)
        prots.append(r["protocol"])
    cov, wid, prots = np.array(cov), np.array(wid), np.array(prots)
    res["coverage"] = float(cov.mean())
    res["width_median"] = float(np.median(wid))
    res["by_protocol"] = {p: (float(cov[prots == p].mean()), float(np.median(wid[prots == p]))) for p in np.unique(prots)}
    return res


def ef_frame_from(d):
    rows = []
    for (s, dr, _pr), g in d.groupby(["subject", "draw", "protocol"]):
        if set(g["phase"]) != {"ed", "es"}:
            continue
        ed, es = g[g.phase == "ed"].iloc[0], g[g.phase == "es"].iloc[0]
        r = dict(subject=s, draw=dr, protocol=ed["protocol"], has_gt=bool(ed["has_gt"] and es["has_gt"]),
                 ed_idx=ed["idx"], es_idx=es["idx"])
        for k in ("LV", "RV"):
            r[f"EF_pred_{k}"] = 100 * (ed[f"Vpred_{k}"] - es[f"Vpred_{k}"]) / max(ed[f"Vpred_{k}"], 1e-6)
            r[f"EF_gt_{k}"] = 100 * (ed[f"Vgt_{k}"] - es[f"Vgt_{k}"]) / max(ed[f"Vgt_{k}"], 1e-6)
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- revised scores
def band_eval(tag, alpha=0.1, n_bins=3, predictor="auto", cal="acdc_cal", test="acdc_test", score="scores95"):
    """Contour-band conformal evaluation with the boundary-percentile score.

    The band is the annulus |phi_hat| <= q around the predicted contour; coverage means that at
    least 95% of the true contour lies inside it.
    """
    pc, pt = load_pred(tag, cal), load_pred(tag, test)
    if pc is None or pt is None:
        raise FileNotFoundError(f"{tag}: {cal}/{test}")
    use_mot = predictor == "mot" or (predictor == "auto" and np.isfinite(pc["mot"][:, 0]).all())
    qc = pc["mot"][:, 0] if use_mot else pc["ent"]
    qt = pt["mot"][:, 0] if use_mot else pt["ent"]
    out = {}
    for k, name in enumerate(["LV", "EPI", "RV"]):
        okc = np.isfinite(pc[score][:, k]) & (pc["gt_area"][:, k] > 0)
        edges, th = cf.calibrate(pc[score][okc, k], qc[okc], alpha, n_bins)
        t = cf.thresholds_for(qt, edges, th)
        ok = np.isfinite(pt[score][:, k]) & (pt["gt_area"][:, k] > 0)
        cov = pt[score][ok, k] <= t[ok]
        prot = pt["protocol"][ok]
        tt = np.clip(t[ok], 0, 15)
        out[name] = dict(coverage=float(cov.mean()), width=float(np.mean(2 * tt)), t=tt, cov=cov, prot=prot,
                         by_protocol={str(p): float(cov[prot == p].mean()) for p in np.unique(prot)},
                         width_by_protocol={str(p): float(np.mean(2 * tt[prot == p])) for p in np.unique(prot)},
                         worst=float(min(cov[prot == p].mean() for p in np.unique(prot))))
    return out


def study_units(tag, bank, predictor="auto"):
    """Per subject-draw units with ED/ES area curves for volume-bracketing EF calibration."""
    p = load_pred(tag, bank)
    d = pd.DataFrame(json.load(open(os.path.join(RUNS, tag, f"vol_{bank}.json"))))
    d["subject"] = d["vol"].map(subject)
    d["phase"] = d["vol"].map(lambda v: v.rsplit("_", 1)[1])
    use_mot = predictor == "mot" or (predictor == "auto" and np.isfinite(p["mot"][:, 0]).all())
    q = p["mot"][:, 0] if use_mot else p["ent"]
    units = []
    for (s, dr, _pr), g in d.groupby(["subject", "draw", "protocol"]):
        if set(g["phase"]) != {"ed", "es"}:
            continue
        ed, es = g[g.phase == "ed"].iloc[0], g[g.phase == "es"].iloc[0]
        if not (ed["has_gt"] and es["has_gt"]):
            continue
        u = dict(subject=s, draw=int(dr), protocol=ed["protocol"], q=float(np.mean(q[ed["idx"] + es["idx"]])))
        for ph, row in (("ed", ed), ("es", es)):
            u[ph + "_curve"] = p["area_curve"][row["idx"], 0].sum(0) * 10.0 / 1000      # LV cavity volume vs threshold (mL)
            u[ph + "_true"] = float(p["gt_area"][row["idx"], 0].sum() * 10.0 / 1000)
            u[ph + "_pred"] = float(p["pred_area"][row["idx"], 0].sum() * 10.0 / 1000)
        # smallest q such that V^-(q) <= V <= V^+(q) for both phases
        sc = []
        for ph in ("ed", "es"):
            c, v = u[ph + "_curve"], u[ph + "_true"]
            lo = np.interp(-cf.TGRID[::-1], cf.TGRID, c)[::-1]  # V^-(q) on grid q=TGRID
            okq = [(np.interp(-qq, cf.TGRID, c) <= v <= np.interp(qq, cf.TGRID, c)) for qq in cf.TGRID[cf.TGRID >= 0]]
            qs = cf.TGRID[cf.TGRID >= 0]
            idx = np.argmax(okq) if any(okq) else None
            sc.append(qs[idx] if idx is not None else np.inf)
        u["score"] = max(sc)
        u["ef_true"] = 100 * (u["ed_true"] - u["es_true"]) / u["ed_true"]
        u["ef_pred"] = 100 * (u["ed_pred"] - u["es_pred"]) / max(u["ed_pred"], 1e-6)
        units.append(u)
    return units


def ef_eval(tag, alpha=0.1, n_bins=3, cal="acdc_cal", test="acdc_test", predictor="auto"):
    uc, ut = study_units(tag, cal, predictor), study_units(tag, test, predictor)
    sc = np.array([u["score"] for u in uc]); qc = np.array([u["q"] for u in uc])
    edges, th = cf.calibrate(sc, qc, alpha, n_bins)
    rows = []
    for u in ut:
        t = float(min(cf.thresholds_for(np.array([u["q"]]), edges, th)[0], 15))
        edm, edp = np.interp(-t, cf.TGRID, u["ed_curve"]), np.interp(t, cf.TGRID, u["ed_curve"])
        esm, esp = np.interp(-t, cf.TGRID, u["es_curve"]), np.interp(t, cf.TGRID, u["es_curve"])
        lo, hi = 100 * (1 - esp / max(edm, 1e-6)), 100 * (1 - esm / max(edp, 1e-6))
        rows.append(dict(subject=u["subject"], protocol=u["protocol"], lo=lo, hi=hi, ef_true=u["ef_true"],
                         ef_pred=u["ef_pred"], cov=lo <= u["ef_true"] <= hi, width=hi - lo, t=t))
    return pd.DataFrame(rows)


# ------------------------------------------------------- conformal risk control
QS = cf.TGRID[cf.TGRID >= 0]


def crc_threshold(L, alpha):
    """Smallest q with (n/(n+1)) mean L(q) + 1/(n+1) <= alpha; L is (n, len(QS)), nonincreasing in q."""
    n = len(L)
    if n == 0:
        return QS[-1]
    r = (n / (n + 1)) * L.mean(0) + 1 / (n + 1)
    ok = np.where(r <= alpha)[0]
    return QS[ok[0]] if len(ok) else QS[-1]


def crc_eval(tag, alpha=0.1, n_bins=3, predictor="auto", cal="acdc_cal", test="acdc_test", k=1):
    pc, pt = load_pred(tag, cal), load_pred(tag, test)
    if pc is None or pt is None or "miss" not in pc or "miss" not in pt:
        raise FileNotFoundError(f"{tag}: {cal}/{test}")
    use_mot = predictor == "mot" or (predictor == "auto" and np.isfinite(pc["mot"][:, 0]).all())
    qc = pc["mot"][:, 0] if use_mot else pc["ent"]
    qt = pt["mot"][:, 0] if use_mot else pt["ent"]
    Lc = np.maximum(pc["miss"][:, k], pc["incl"][:, k])
    Lt = np.maximum(pt["miss"][:, k], pt["incl"][:, k])
    okc = np.isfinite(Lc[:, 0]) & (pc["gt_area"][:, k] > 0)
    okt = np.isfinite(Lt[:, 0]) & (pt["gt_area"][:, k] > 0)
    edges = cf.bin_edges(qc[okc], n_bins)
    bc = np.digitize(qc[okc], edges)
    th = np.array([crc_threshold(Lc[okc][bc == b], alpha) for b in range(max(n_bins, 1))])
    t = th[np.digitize(qt[okt], edges)]
    qi = np.searchsorted(QS, t)
    risk = Lt[okt][np.arange(okt.sum()), qi]
    prot = pt["protocol"][okt]
    P = [p for p in ["BH", "HBH", "FB", "IB"] if p in set(prot)]
    return dict(risk=float(risk.mean()), width=float(np.mean(2 * t)), th=th.tolist(),
                risk_by=dict((p, float(risk[prot == p].mean())) for p in P),
                width_by=dict((p, float(np.mean(2 * t[prot == p]))) for p in P),
                worst=float(max(risk[prot == p].mean() for p in P)), t=t, prot=prot)


def ef_norm_eval(tag, alpha=0.1, n_bins=3, cal="acdc_cal", test="acdc_test", predictor="auto", q0=2.0, scale=True):
    """Study-level LVEF intervals: |EF_hat - EF| normalised by the contour-band EF range at q0, Mondrian by severity."""
    def frame(bank):
        rows = []
        for u in study_units(tag, bank, predictor):
            edm, edp = np.interp(-q0, cf.TGRID, u["ed_curve"]), np.interp(q0, cf.TGRID, u["ed_curve"])
            esm, esp = np.interp(-q0, cf.TGRID, u["es_curve"]), np.interp(q0, cf.TGRID, u["es_curve"])
            w = 100 * ((1 - esm / max(edp, 1e-6)) - (1 - esp / max(edm, 1e-6))) if scale else 1.0
            rows.append(dict(subject=u["subject"], protocol=u["protocol"], q=u["q"], ef_pred=u["ef_pred"],
                             ef_true=u["ef_true"], w=max(w, 1e-3)))
        return pd.DataFrame(rows)
    c, t = frame(cal), frame(test)
    sc = (c["ef_pred"] - c["ef_true"]).abs() / c["w"]
    edges, th = cf.calibrate(sc.values, c["q"].values, alpha, n_bins)
    qh = cf.thresholds_for(t["q"].values, edges, th)
    half = qh * t["w"].values
    t["lo"], t["hi"] = t["ef_pred"] - half, t["ef_pred"] + half
    t["cov"] = (t["ef_true"] >= t["lo"]) & (t["ef_true"] <= t["hi"])
    t["width"] = 2 * half
    return t
