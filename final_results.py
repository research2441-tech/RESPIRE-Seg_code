"""Compute every number, table and result figure used in the manuscript.

Writes /home/claude/results/results.json and figure PNGs to /home/claude/figs.
"""
import glob
import json
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

import analysis as A
import conformal as cf
import plotstyle

plotstyle.apply()
import matplotlib.pyplot as plt

RUNS = "/home/claude/runs"
FIGS = "/home/claude/figs"
OUT = "/home/claude/results"
os.makedirs(FIGS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
NAMES = {"respire": "RESPIRE-Seg", "unet": "U-Net", "attunet": "Attention U-Net", "sharpunet": "Sharp U-Net",
         "nnunet2d": "nnU-Net-style 2-D", "transunet": "CNN–Transformer", "multitask": "Multi-task U-Net"}
MAIN = list(NAMES)
PROTS = ["BH", "HBH", "FB", "IB"]
METRICS = ["dice_LV", "dice_MYO", "dice_RV", "hd95_LV", "hd95_MYO", "hd95_RV"]
R = {}


def tags(prefix):
    return sorted(os.path.basename(p) for p in glob.glob(f"{RUNS}/{prefix}_s[0-9]") if
                  os.path.exists(f"{p}/train.json"))


def vol_frame(prefix, bank):
    fr = []
    for t in tags(prefix):
        f = f"{RUNS}/{t}/vol_{bank}.json"
        if os.path.exists(f):
            d = pd.DataFrame(json.load(open(f)))
            d["seed"] = t
            fr.append(d)
    if not fr:
        return None
    d = pd.concat(fr)
    d["subject"] = d["vol"].map(A.subject)
    return d


def subj(prefix, bank, prots=None):
    """Subject x protocol means (over phases and seeds), labelled volumes only."""
    d = vol_frame(prefix, bank)
    if d is None:
        return None
    d = d[d["has_gt"]]
    if prots:
        d = d[d["protocol"].isin(prots)]
    return d.groupby(["subject", "protocol"])[METRICS].mean().reset_index()


def msd(x, nd=3):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return f"{x.mean():.{nd}f}±{x.std(ddof=1):.{nd}f}" if len(x) > 1 else "—"


# ------------------------------------------------------------------ segmentation
def segmentation():
    rows, cache = [], {}
    for m in MAIN:
        row = [NAMES[m]]
        for bank, prots in (("cmrx_real", None), ("acdc_test", ["FB", "IB"]), ("mnms2_test", ["FB", "IB"])):
            s = subj(m, bank, prots)
            cache[(m, bank)] = s
            if s is None:
                row += ["—"] * 3
                continue
            g = s.groupby("subject")[METRICS].mean()
            row += [msd(g["dice_MYO"]), msd(g["dice_RV"]), msd(g["hd95_RV"], 2)]
        rows.append(row)
    R["tab_main"] = rows
    # per-protocol tables for the degradation figure and text
    deg = {}
    for m in MAIN:
        for bank in ("acdc_test", "mnms2_test", "cmrx_real"):
            s = cache.get((m, bank))
            if s is None:
                continue
            deg[f"{m}|{bank}"] = s.groupby("protocol")[METRICS].mean().reindex(PROTS).to_dict()
    R["deg"] = deg
    # statistics vs strongest baseline per endpoint
    st = []
    for bank, prots, metric, better in (("cmrx_real", None, "hd95_RV", "lower"), ("cmrx_real", None, "dice_MYO", "higher"),
                                        ("acdc_test", ["FB", "IB"], "hd95_RV", "lower"), ("acdc_test", ["FB", "IB"], "dice_MYO", "higher"),
                                        ("mnms2_test", ["FB", "IB"], "hd95_RV", "lower"), ("mnms2_test", ["FB", "IB"], "dice_RV", "higher")):
        r0 = cache.get(("respire", bank))
        if r0 is None:
            continue
        a = r0.groupby("subject")[metric].mean()
        ps, recs = [], []
        for m in MAIN[1:]:
            s = cache.get((m, bank))
            if s is None:
                continue
            b = s.groupby("subject")[metric].mean()
            idx = a.index.intersection(b.index)
            dlt = (a[idx] - b[idx]).values
            p = stats.wilcoxon(a[idx], b[idx]).pvalue if np.any(dlt != 0) else 1.0
            ci = A.boot_ci(dlt)
            recs.append(dict(bank=bank, metric=metric, base=NAMES[m], diff=float(dlt.mean()), ci=[float(ci[0]), float(ci[1])],
                             p=float(p), cliff=float(A.cliffs_delta(a[idx].values, b[idx].values)), n=len(idx)))
            ps.append(p)
        adj = A.holm(ps)
        for rr, pa in zip(recs, adj):
            rr["p_holm"] = float(pa)
        st += recs
    R["stats"] = st
    return cache


def mixed_model(cache):
    import statsmodels.formula.api as smf
    out = {}
    for bank in ("acdc_test", "cmrx_real"):
        fr = []
        for m in ("respire", "unet", "attunet"):
            s = subj(m, bank)
            if s is not None:
                s = s.copy(); s["method"] = NAMES[m]; fr.append(s)
        d = pd.concat(fr)
        d = d[np.isfinite(d["hd95_RV"])]
        md = smf.mixedlm("hd95_RV ~ C(method, Treatment('U-Net')) * C(protocol, Treatment('BH'))", d, groups=d["subject"]).fit()
        out[bank] = {k: [float(v), float(md.pvalues[k])] for k, v in md.params.items() if "method" in k or "protocol" in k}
    R["lmm"] = out


def fig_degradation(cache):
    fig, axs = plt.subplots(2, 3, figsize=(7.16, 3.9), sharex=True)
    titles = {"acdc_test": "ACDC (simulated)", "mnms2_test": "M&Ms-2 (simulated)", "cmrx_real": "CMRxMotion (real)"}
    for j, bank in enumerate(("acdc_test", "mnms2_test", "cmrx_real")):
        for i, met in enumerate(("dice_MYO", "hd95_RV")):
            ax = axs[i, j]
            for k, m in enumerate(MAIN):
                s = subj(m, bank)
                if s is None:
                    continue
                y = s.groupby("protocol")[met].mean().reindex(PROTS)
                lw = 1.8 if m == "respire" else 0.9
                ax.plot(range(4), y.values, color=plotstyle.GRAYS[min(k, 4)] if m != "respire" else "black",
                        ls=plotstyle.LINES[k % len(plotstyle.LINES)], marker=plotstyle.MARK[k], ms=3.5, lw=lw,
                        label=NAMES[m])
            ax.set_xticks(range(4)); ax.set_xticklabels(PROTS)
            if j == 0:
                ax.set_ylabel("Myocardium Dice" if met == "dice_MYO" else "RV HD$_{95}$ (mm)")
            if i == 0:
                ax.set_title(titles[bank])
            ax.grid(axis="y", lw=0.3, color="#bbbbbb")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), columnspacing=1.2, handlelength=2.6)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(f"{FIGS}/fig_degradation.png", dpi=500)
    plt.close(fig)


# ------------------------------------------------------------------ training data / engine fidelity
def paradigm():
    rows = []
    for key, name in (("unetclean", "Clean only"), ("unetimgwarp", "Image-domain warp"), ("unetgeneric", "Generic rigid k-space"),
                      ("unetsine", "Engine, sinusoidal breathing"), ("unet1coil", "Engine, single coil"),
                      ("unetnotp", "Engine, no through-plane"), ("unet", "Full engine (proposed)")):
        row = [name]
        for bank, prots in (("cmrx_real", ["FB", "IB"]), ("cmrx_real", ["BH", "HBH"]), ("acdc_test", ["FB", "IB"])):
            s = subj(key, bank, prots)
            if s is None:
                row += ["—", "—"]; continue
            g = s.groupby("subject")[METRICS].mean()
            row += [msd(g["dice_MYO"]), msd(g["hd95_RV"], 2)]
        rows.append(row)
    R["tab_paradigm"] = rows
    # paired test: full engine vs image-domain warp on real FB/IB
    a = subj("unet", "cmrx_real", ["FB", "IB"])
    for key in ("unetimgwarp", "unetgeneric", "unetclean"):
        b = subj(key, "cmrx_real", ["FB", "IB"])
        if a is None or b is None:
            continue
        for met, suf in (("hd95_RV", ""), ("dice_MYO", "_myo")):
            x, y = a.groupby("subject")[met].mean(), b.groupby("subject")[met].mean()
            idx = x.index.intersection(y.index)
            R[f"paradigm_p_{key}{suf}"] = [float((x[idx] - y[idx]).mean()), float(stats.wilcoxon(x[idx], y[idx]).pvalue)]


# ------------------------------------------------------------------ ablation and sensitivity
def ablation():
    rows = []
    for key, name in (("respire", "RESPIRE-Seg (final)"), ("ablnoA", "− motion state (A)"), ("ablnoG", "− saliency gates (G)"),
                      ("ablnoB", "− SDF + topology (B)"), ("ablbase", "− A, G, B"), ("ablplusD", "+ paired consistency (D)")):
        t = tags(key)
        if not t:
            continue
        val = np.mean([json.load(open(f"{RUNS}/{x}/train.json"))["best"] for x in t])
        row = [name, f"{val:.3f}"]
        for bank, prots in (("acdc_test", ["FB", "IB"]), ("cmrx_real", None)):
            s = subj(key, bank, prots)
            g = s.groupby("subject")[METRICS].mean()
            row += [msd(g["dice_MYO"]), msd(g["hd95_RV"], 2)]
        fl = flag_scores(t[0])
        row += [f"{fl['rho']:.2f}" if fl else "—"]
        try:
            bev = A.crc_eval(t[0], 0.1, 3, "auto" if key not in ("ablnoA", "ablbase") else "ent", "acdc_cal", "acdc_test", k=1)
            row += [f"{bev['worst']:.3f}", f"{bev['width']:.2f}"]
        except Exception:
            row += ["—", "—"]
        rows.append(row)
    R["tab_ablation"] = rows


def sensitivity():
    """Sensitivity of the EF interval to the reference band width q0 (RESPIRE-Seg, severity bins, alpha = 0.1)."""
    rows = []
    for q0 in (1.0, 2.0, 3.0, 4.0):
        row = [f"{q0:.0f}"]
        for bank in ("acdc_test", "cmrx_real"):
            fr = [A.ef_norm_eval(t, 0.1, 3, "acdc_cal", bank, "auto", q0=q0) for t in tags("respire")]
            row += [f"{np.mean([f['cov'].mean() for f in fr]):.3f}", f"{np.mean([f['width'].median() for f in fr]):.1f}"]
        rows.append(row)
    R["tab_q0"] = rows


# ------------------------------------------------------------------ quality flagging
def iqa_series():
    b = np.load("/home/claude/data/banks/cmrx_real.npz", allow_pickle=True)
    return pd.Series(b["iqa"], index=b["vol"]).groupby(level=0).first()


def flag_scores(tag, col=None):
    f = f"{RUNS}/{tag}/vol_cmrx_real.json"
    if not os.path.exists(f):
        return None
    d = pd.DataFrame(json.load(open(f))).set_index("vol")
    iqa = iqa_series()[d.index]
    col = col or ("pred_sev" if d["pred_sev"].notna().all() else "ent")
    x = d[col]
    if not np.all(np.isfinite(x.values)):
        return None
    return dict(auc=float(roc_auc_score((iqa == 3).astype(int), x)), rho=float(stats.spearmanr(iqa, x).correlation),
                x=x.values, y=iqa.values, prot=d["protocol"].values)


def flagging():
    rows = []
    for m in MAIN:
        for col, lab in (("ent", "softmax entropy"), ("pred_sev", "predicted severity $\\hat{\\mathcal{S}}$")):
            aucs, rhos = [], []
            for t in tags(m):
                fs = flag_scores(t, col)
                if fs is None or not np.all(np.isfinite(fs["x"])):
                    continue
                aucs.append(fs["auc"]); rhos.append(fs["rho"])
            if aucs:
                rows.append([NAMES[m], lab, f"{np.mean(aucs):.3f}", f"{np.mean(rhos):.3f}"])
    R["tab_flag"] = rows
    # figure: severity by grade and ROC
    t0 = tags("respire")[0]
    fs = flag_scores(t0, "pred_sev")
    fu = flag_scores(tags("unet")[0], "ent")
    fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.9))
    data = [fs["x"][fs["y"] == g] for g in (1, 2, 3)]
    bp = ax[0].boxplot(data, widths=0.55, patch_artist=True, medianprops=dict(color="black"), flierprops=dict(ms=2))
    for p, c in zip(bp["boxes"], ("#ffffff", "#bfbfbf", "#7f7f7f")):
        p.set_facecolor(c)
    ax[0].set_xticks([1, 2, 3]); ax[0].set_xticklabels(["1", "2", "3"])
    ax[0].set_xlabel("Radiologist grade"); ax[0].set_ylabel("$\\hat{\\mathcal{S}}$ (mm)")
    for f, lab, ls in ((fs, "$\\hat{\\mathcal{S}}$", "-"), (fu, "Entropy (U-Net)", "--")):
        fpr, tpr, _ = roc_curve((f["y"] == 3).astype(int), f["x"])
        ax[1].plot(fpr, tpr, color="black", ls=ls, label=f"{lab}")
    ax[1].plot([0, 1], [0, 1], color="#999999", lw=0.5)
    ax[1].set_xlabel("False-positive rate"); ax[1].set_ylabel("True-positive rate")
    ax[1].legend(loc="lower right", fontsize=6)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(f"{FIGS}/fig_flag.png", dpi=500)
    plt.close(fig)


# ------------------------------------------------------------------ conformal
def conformal():
    alpha = 0.1
    rows, cov_fig = [], {}
    configs = [("respire", 3, "auto", "RESPIRE-Seg, severity bins"), ("respire", 1, "auto", "RESPIRE-Seg, marginal"),
               ("unet", 3, "ent", "U-Net, entropy bins"), ("unet", 1, "ent", "U-Net, marginal"),
               ("attunet", 3, "ent", "Attention U-Net, entropy bins"), ("respire", 3, "ent", "RESPIRE-Seg, entropy bins")]
    for m, nb, pr, name in configs:
        row = [name]
        for cal, test in (("acdc_cal", "acdc_test"), ("acdc_cal", "mnms2_test"), ("acdc_cal", "cmrx_real")):
            vals = []
            for t in tags(m):
                try:
                    vals.append(A.crc_eval(t, alpha, nb, pr, cal, test, k=1))
                except Exception:
                    pass
            if not vals:
                row += ["—"] * 3; continue
            row += [f"{np.mean([v['risk'] for v in vals]):.3f}", f"{np.mean([v['worst'] for v in vals]):.3f}",
                    f"{np.mean([v['width'] for v in vals]):.2f}"]
            cov_fig[(name, test)] = ({p: np.mean([v["risk_by"].get(p, np.nan) for v in vals]) for p in PROTS},
                                     {p: np.mean([v["width_by"].get(p, np.nan) for v in vals]) for p in PROTS})
        rows.append(row)
    R["tab_conformal"] = rows
    R["cov_fig"] = {f"{k[0]}|{k[1]}": v for k, v in cov_fig.items()}
    rec = []
    for t in tags("respire"):
        try:
            rec.append(A.crc_eval(t, alpha, 3, "auto", "mnms2_cal", "mnms2_test", k=1))
        except Exception:
            pass
    if rec:
        R["mnms_recal"] = {k: float(np.mean([r[k] for r in rec])) for k in ("risk", "worst", "width")}
    # structures and bins
    det = {}
    for k, nm in enumerate(["LV", "EPI", "RV"]):
        v = [A.crc_eval(t, alpha, 3, "auto", "acdc_cal", "acdc_test", k=k) for t in tags("respire")]
        det[nm] = dict(risk=float(np.mean([x["risk"] for x in v])), worst=float(np.mean([x["worst"] for x in v])),
                       width=float(np.mean([x["width"] for x in v])),
                       width_by={p: float(np.mean([x["width_by"][p] for x in v])) for p in PROTS})
    R["band_struct"] = det
    sweep = []
    for nb in (1, 2, 3, 4, 5):
        for a in (0.05, 0.1, 0.2):
            v = [A.crc_eval(t, a, nb, "auto", "acdc_cal", "acdc_test", k=1) for t in tags("respire")]
            sweep.append([nb, a, float(np.mean([x["risk"] for x in v])), float(np.mean([x["worst"] for x in v])),
                          float(np.mean([x["width"] for x in v]))])
    R["bin_sweep"] = sweep
    # in-domain recalibration on real CMRxMotion (subject-level 50/50 splits, 50 repetitions)
    from cmrx_recal import cmrx_recal
    rr, recal_fig = [], {}
    for m, nb, pr, name in (("unet", 1, "ent", "U-Net, marginal"), ("unet", 3, "ent", "U-Net, entropy bins"),
                            ("attunet", 3, "ent", "Attention U-Net, entropy bins"),
                            ("respire", 1, "auto", "RESPIRE-Seg, marginal"), ("respire", 3, "ent", "RESPIRE-Seg, entropy bins"),
                            ("respire", 3, "auto", "RESPIRE-Seg, severity bins")):
        v = [cmrx_recal(t, nb, alpha, 1, pr) for t in tags(m)]
        if not v:
            continue
        g = lambda key, sub: float(np.mean([x[key][sub] for x in v]))
        rr.append([name, f"{np.mean([x['risk'] for x in v]):.3f}", f"{g('risk_by_grade', 1):.3f}", f"{g('risk_by_grade', 2):.3f}",
                   f"{abs(g('risk_by_grade', 2) - g('risk_by_grade', 1)):.3f}", f"{np.mean([x['width'] for x in v]):.2f}",
                   f"{g('width_by_grade', 1):.2f}", f"{g('width_by_grade', 2):.2f}"])
        recal_fig[name] = ({p: g("risk_by_prot", p) for p in PROTS}, {p: g("width_by_prot", p) for p in PROTS})
    R["tab_recal"] = rr
    R["recal_fig"] = recal_fig
    names = ["U-Net, entropy bins", "RESPIRE-Seg, marginal", "RESPIRE-Seg, severity bins"]
    fills, hatches = ["#ffffff", "#bfbfbf", "#404040"], ["////", "", ""]
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.15))
    panels = [(cov_fig, "acdc_test", 0, "(a) ACDC, simulated motion", "Realised band risk"),
              (recal_fig, None, 0, "(b) CMRxMotion, recalibrated", "Realised band risk"),
              (recal_fig, None, 1, "(c) CMRxMotion, recalibrated", "Band width $2\\hat{q}$ (mm)")]
    for ax, (src, test, comp, title, ylab) in zip(axs, panels):
        for k, n in enumerate(names):
            v = src.get((n, test)) if test else src.get(n)
            if v is None:
                continue
            ax.bar(np.arange(4) + (k - 1) * 0.26, [v[comp][p] for p in PROTS], 0.26, color=fills[k], edgecolor="black",
                   lw=0.5, hatch=hatches[k], label=n)
        if comp == 0:
            ax.axhline(alpha, color="black", ls="--", lw=0.7)
            ax.set_ylim(0, 0.16)
        ax.set_xticks(range(4)); ax.set_xticklabels(PROTS)
        ax.set_title(title, fontsize=8); ax.set_ylabel(ylab)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01), columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.6)
    fig.savefig(f"{FIGS}/fig_coverage.png", dpi=500)
    plt.close(fig)


def ef():
    rows, ba = [], {}
    for m in MAIN:
        row = [NAMES[m]]
        for bank, prots in (("acdc_test", ["FB", "IB"]), ("mnms2_test", ["FB", "IB"]), ("cmrx_real", None)):
            fr = []
            for t in tags(m):
                try:
                    u = pd.DataFrame(A.study_units(t, bank, "auto" if m == "respire" else "ent"))
                except Exception:
                    continue
                if prots:
                    u = u[u.protocol.isin(prots)]
                fr.append(u)
            if not fr:
                row += ["—", "—"]; continue
            e = pd.concat(fr).groupby(["subject", "protocol"])[["ef_pred", "ef_true"]].mean()
            b, lo, hi = A.bland_altman(e["ef_pred"] - e["ef_true"])
            row += [f"{b:+.1f}", f"[{lo:.1f}, {hi:.1f}]"]
            ba[f"{m}|{bank}"] = (e["ef_true"].values.tolist(), e["ef_pred"].values.tolist())
        rows.append(row)
    R["tab_ef"] = rows
    R["ba"] = ba
    ivs = []
    for name, m, nb, pr, sc in (("RESPIRE-Seg, contour-scaled, severity bins", "respire", 3, "auto", True),
                                ("RESPIRE-Seg, contour-scaled, marginal", "respire", 1, "auto", True),
                                ("RESPIRE-Seg, unscaled, severity bins", "respire", 3, "auto", False),
                                ("U-Net, unscaled, marginal", "unet", 1, "ent", False),
                                ("U-Net, contour-scaled, entropy bins", "unet", 3, "ent", True)):
        row = [name]
        for bank in ("acdc_test", "mnms2_test", "cmrx_real"):
            fr = []
            for t in tags(m):
                try:
                    fr.append(A.ef_norm_eval(t, 0.1, nb, "acdc_cal", bank, pr, scale=sc))
                except Exception:
                    pass
            if not fr:
                row += ["—", "—", "—"]; continue
            cov = np.mean([f["cov"].mean() for f in fr])
            worst = np.mean([min(f[f.protocol == p]["cov"].mean() for p in f.protocol.unique()) for f in fr])
            wid = np.mean([f["width"].median() for f in fr])
            row += [f"{cov:.3f}", f"{worst:.3f}", f"{wid:.1f}"]
        ivs.append(row)
    R["tab_efint"] = ivs
    from cmrx_recal import ef_recal
    er = {}
    for m, pr, key in (("respire", "auto", "respire"), ("unet", "ent", "unet")):
        v = [ef_recal(t, 3, 0.1, pr) for t in tags(m)]
        er[f"{key}_cov"] = float(np.mean([x["coverage"] for x in v]))
        er[f"{key}_w"] = float(np.mean([x["width"] for x in v]))
        er["n_units"] = v[0]["n_units"]
    R["ef_recal"] = er
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.95))
    for k, m in enumerate(("unet", "respire")):
        key = f"{m}|cmrx_real"
        if key not in ba:
            continue
        tt, pp = map(np.asarray, ba[key])
        mean, dif = (tt + pp) / 2, pp - tt
        axs[0].scatter(mean, dif, s=7, marker=["o", "s"][k], facecolors=["white", "black"][k], edgecolors="black", lw=0.5,
                       label=NAMES[m])
        b, lo, hi = A.bland_altman(dif)
        for y, ls in ((b, "-"), (lo, "--"), (hi, "--")):
            axs[0].axhline(y, color=["#8c8c8c", "black"][k], ls=ls, lw=0.6)
    axs[0].set_xlabel("Mean LVEF (%)"); axs[0].set_ylabel("Pred. − ref. LVEF (%)")
    axs[0].legend(fontsize=5.5, loc="lower left", handletextpad=0.2)
    from cmrx_recal import ef_crossfit
    e = ef_crossfit(tags("respire")[0], 3, 0.1, "auto").sort_values("ef_true").reset_index(drop=True)
    R["ef_crossfit"] = dict(coverage=float(e["cov"].mean()), width=float(e["width"].median()), n=len(e))
    xs = np.arange(len(e))
    axs[1].vlines(xs, e["lo"], e["hi"], color="#8c8c8c", lw=0.9)
    axs[1].scatter(xs[e["cov"].values], e["ef_true"][e["cov"].values], s=5, color="black", zorder=3, label="Reference, covered")
    miss = ~e["cov"].values
    axs[1].scatter(xs[miss], e["ef_true"][miss], s=16, marker="x", color="black", lw=0.8, zorder=4, label="Reference, not covered")
    axs[1].set_xlabel("Study (sorted by LVEF)"); axs[1].set_ylabel("LVEF (%)")
    axs[1].legend(fontsize=5.5, loc="lower right", handletextpad=0.2, frameon=False)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(f"{FIGS}/fig_ef.png", dpi=500)
    plt.close(fig)


def stress():
    rows = []
    for m in MAIN:
        s = subj(m, "acdc_test_stress")
        if s is None:
            continue
        g = s.groupby("subject")[METRICS].mean()
        rows.append([NAMES[m], msd(g["dice_MYO"]), msd(g["dice_RV"]), msd(g["hd95_RV"], 2)])
    R["tab_stress"] = rows


def efficiency():
    import time
    import torch
    import models
    from torch.utils.flop_counter import FlopCounterMode
    torch.set_num_threads(1)
    rows = []
    for m in MAIN:
        net = models.build(m).eval()
        x = torch.randn(1, 1, 96, 96)
        with FlopCounterMode(display=False) as fc:
            net(x)
        fl = fc.get_total_flops() / 2 / 1e6
        with torch.no_grad():
            for _ in range(3):
                net(x)
            t0 = time.time()
            for _ in range(20):
                net(x)
        lat = (time.time() - t0) / 20 * 1000
        tt = [json.load(open(f"{RUNS}/{t}/train.json"))["train_time"] / 60 for t in tags(m)]
        rows.append([NAMES[m], f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f}", f"{fl:.0f}", f"{lat:.1f}",
                     f"{np.mean(tt):.0f}" if tt else "—"])
    R["tab_eff"] = rows


if __name__ == "__main__":
    cache = segmentation()
    fig_degradation(cache)
    for fn in (mixed_model, ):
        try:
            fn(cache)
        except Exception as ex:
            R["lmm_error"] = str(ex)
    for fn in (paradigm, ablation, sensitivity, flagging, conformal, ef, stress, efficiency):
        try:
            fn()
        except Exception as ex:
            import traceback
            traceback.print_exc()
            R[f"error_{fn.__name__}"] = str(ex)
    json.dump(R, open(f"{OUT}/results.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("done", [k for k in R if k.startswith("error")])
