"""Fig.: real CMRxMotion examples with predicted contours and conformal epicardial bands."""
import json
import numpy as np
import torch
from skimage import measure

import plotstyle
plotstyle.apply()
import matplotlib.pyplot as plt

import models
import conformal as cf
import analysis as A
from losses import TAU, signed_distance

RUNS = "/home/claude/runs"
CROP = 96


def load(tag):
    cfg = json.load(open(f"{RUNS}/{tag}/train.json"))["args"]
    net = models.build(cfg["model"])
    net.load_state_dict(torch.load(f"{RUNS}/{tag}/best.pt", map_location="cpu"))
    return net.eval()


def contour(ax, mask, **kw):
    for c in measure.find_contours(mask.astype(float), 0.5):
        ax.plot(c[:, 1], c[:, 0], **kw)


def main(out):
    b = np.load("/home/claude/data/banks/cmrx_real.npz", allow_pickle=True)
    o = (b["X"].shape[-1] - CROP) // 2
    X = b["X"][:, o:o + CROP, o:o + CROP].astype(np.float32)
    Y = b["Y"][:, o:o + CROP, o:o + CROP]
    resp, unet = load("respire_s0"), load("unet_s0")
    cases = []
    for prot, grade in (("FB", 2), ("IB", 2), ("IB", 3)):
        vols = sorted({v for v, p, g in zip(b["vol"], b["protocol"], b["iqa"]) if p == prot and g == grade and v.endswith("_ed")})
        idx = np.where(b["vol"] == vols[0])[0]
        cases.append(idx[len(idx) // 2])
    # risk-controlled epicardial thresholds recalibrated on real CMRxMotion subjects other than those displayed
    from analysis import subject as subject_of
    pc = A.load_pred("respire_s0", "cmrx_real")
    subj = np.array([subject_of(str(v)) for v in pc["vol"]])
    shown = {subj[i] for i in cases}
    L = np.maximum(pc["miss"][:, 1], pc["incl"][:, 1])
    okc = np.isfinite(L[:, 0]) & (pc["gt_area"][:, 1] > 0) & ~np.isin(subj, list(shown))
    edges = cf.bin_edges(pc["mot"][okc, 0], 3)
    bc = np.digitize(pc["mot"][okc, 0], edges)
    th = np.array([A.crc_threshold(L[okc][bc == k], 0.1) for k in range(3)])
    fig, axs = plt.subplots(2, 3, figsize=(3.45, 2.5), gridspec_kw=dict(wspace=0.04, hspace=0.12))
    for j, i in enumerate(cases):
        x = torch.from_numpy(X[i][None, None])
        with torch.no_grad():
            ou, orr = unet(x), resp(x)
        pu = ou["logits"].argmax(1)[0].numpy()
        pr = orr["logits"].argmax(1)[0].numpy()
        sev = float(np.expm1(orr["mot"][0, 0].item()))
        phi = TAU * np.arctanh(np.clip(orr["sdf"][0, 1].numpy(), -0.999, 0.999))
        q = float(th[np.digitize([sev], edges)[0]])
        for r, (p, name) in enumerate(((pu, "U-Net"), (pr, "RESPIRE-Seg"))):
            ax = axs[r, j]
            ax.imshow(X[i], cmap="gray", vmin=0, vmax=1)
            if b["has_gt"][i]:
                contour(ax, (Y[i] == 1) | (Y[i] == 2), color="white", lw=0.9)
            contour(ax, (p == 1) | (p == 2), color="black", lw=0.9, ls="--")
            if r == 1:
                contour(ax, phi <= -q, color="#bfbfbf", lw=0.6, ls=":")
                contour(ax, phi <= q, color="#bfbfbf", lw=0.6, ls=":")
                ax.text(2, 92, f"$\\hat{{S}}$={sev:.2f} mm, $\\hat{{q}}$={q:.1f} mm", color="white", fontsize=5.5,
                        bbox=dict(facecolor="black", edgecolor="none", pad=0.8, alpha=0.7))
            if r == 0:
                ax.set_title(f"{b['protocol'][i]}, grade {b['iqa'][i]}", fontsize=7)
            if j == 0:
                ax.set_ylabel(name, fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(out, dpi=500, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
