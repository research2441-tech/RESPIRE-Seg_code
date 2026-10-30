"""Run a trained model over an evaluation bank and store compact per-slice and per-volume records.

python predict.py --tag respire_s0 --bank acdc_test
"""
import argparse
import json
import os
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, binary_erosion, label as cc_label

import models
from losses import TAU, structures, signed_distance

DATA = os.environ.get("RESPIRE_DATA", "/home/claude/data")
RUNS = os.environ.get("RESPIRE_RUNS", "/home/claude/runs")
PX = 1.5
TGRID = np.arange(-15, 15.01, 0.25)
CROP = int(os.environ.get("CROP", 96))


def crop(a):
    n = a.shape[-1]
    o = (n - CROP) // 2
    return a[..., o:o + CROP, o:o + CROP]


def keep_largest(mask):
    lab, n = cc_label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())[1:]
    return lab == (1 + sizes.argmax())


def postprocess(p):
    out = np.zeros_like(p)
    for c in (1, 2, 3):
        out[keep_largest(p == c)] = c
    return out


def surface(mask):
    return mask & ~binary_erosion(mask)


def hd95_assd(pm, gm, sp):
    """3D 95th-percentile Hausdorff distance and average symmetric surface distance (mm)."""
    if pm.sum() == 0 or gm.sum() == 0:
        return np.nan, np.nan
    dp = distance_transform_edt(~gm, sampling=sp)[surface(pm)]
    dg = distance_transform_edt(~pm, sampling=sp)[surface(gm)]
    d = np.concatenate([dp, dg])
    return float(np.percentile(d, 95)), float(d.mean())


@torch.no_grad()
def run(net, X, bs=64):
    net.eval()
    P, PSI, ENT, MOT = [], [], [], []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs].astype(np.float32))[:, None]
        o = net(x)
        pr = o["logits"].softmax(1)
        P.append(pr.argmax(1).numpy().astype(np.uint8))
        ent = -(pr * pr.clamp_min(1e-8).log()).sum(1)
        heart = (pr[:, 1:].sum(1) > 0.05).float()
        ENT.append(((ent * heart).sum((1, 2)) / (heart.sum((1, 2)) + 1)).numpy())
        PSI.append(o["sdf"].numpy() if "sdf" in o else None)
        MOT.append(np.expm1(o["mot"].numpy()) if ("mot" in o and getattr(net, "use_motion", False)) else None)
    P = np.concatenate(P)
    PSI = None if PSI[0] is None else np.concatenate(PSI)
    MOT = None if MOT[0] is None else np.concatenate(MOT)
    return P, PSI, np.concatenate(ENT), MOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--mask_sdf", type=int, default=0, help="use mask-derived distances even if an SDF head exists")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THREADS", 2)))
    cfg = json.load(open(os.path.join(RUNS, a.tag, "train.json")))["args"]
    net = models.build(cfg["model"])
    net.load_state_dict(torch.load(os.path.join(RUNS, a.tag, "best.pt"), map_location="cpu"))
    b = np.load(os.path.join(DATA, "banks", a.bank + ".npz"), allow_pickle=True)
    X, Y = crop(b["X"]), crop(b["Y"])
    has_gt = b["has_gt"] if "has_gt" in b.files else np.ones(len(X), bool)
    P, PSI, ENT, MOT = run(net, X)
    P = np.stack([postprocess(p) for p in P])
    n = len(X)
    QG = TGRID[TGRID >= 0]
    rec = dict(scores=np.full((n, 3), np.nan, np.float32), scores95=np.full((n, 3), np.nan, np.float32),
               miss=np.full((n, 3, len(QG)), np.nan, np.float32), incl=np.full((n, 3, len(QG)), np.nan, np.float32), area_curve=np.zeros((n, 3, len(TGRID)), np.float32),
               gt_area=np.zeros((n, 3), np.float32), pred_area=np.zeros((n, 3), np.float32),
               ent=ENT.astype(np.float32), mot=(MOT if MOT is not None else np.full((n, 3), np.nan)).astype(np.float32),
               pred=P)
    for i in range(n):
        gts = structures(Y[i])
        prs = structures(P[i])
        for k in range(3):
            if PSI is not None and not a.mask_sdf:
                phi = TAU * np.arctanh(np.clip(PSI[i, k], -0.999, 0.999))
            else:
                phi = signed_distance(prs[k], PX)
            rec["area_curve"][i, k] = [(phi <= t).sum() * PX * PX for t in TGRID]
            rec["pred_area"][i, k] = prs[k].sum() * PX * PX
            rec["gt_area"][i, k] = gts[k].sum() * PX * PX
            if has_gt[i]:
                g = gts[k]
                s_out = phi[g].max() if g.any() else -50.0
                s_in = (-phi[~g]).max() if (~g).any() else -50.0
                rec["scores"][i, k] = max(s_out, s_in)
                gs = max(int(g.sum()), 1)
                for qi, qq in enumerate(QG):
                    inner = phi <= -qq
                    rec["miss"][i, k, qi] = (g & (phi > qq)).sum() / gs
                    rec["incl"][i, k, qi] = (inner & ~g).sum() / max(int(inner.sum()), 1)
                bd = surface(g)
                rec["scores95"][i, k] = np.percentile(np.abs(phi[bd]), 95) if bd.any() else 0.0
    # volume-level metrics: group by (volume id, draw)
    vols = {}
    for i, (v, d) in enumerate(zip(b["vol"], b["draw"])):
        vols.setdefault((str(v), int(d)), []).append(i)
    vrec = []
    for (v, d), idx in vols.items():
        idx = sorted(idx, key=lambda i: b["z"][i])
        dz = float(b["dz"][idx[0]])
        sp = (dz, PX, PX)
        pv, gv = P[idx], Y[idx]
        r = dict(vol=v, draw=d, protocol=str(b["protocol"][idx[0]]), sev=float(np.mean(b["sev"][idx])),
                 has_gt=bool(np.all(has_gt[idx])), n=len(idx),
                 pred_sev=float(np.nanmean(rec["mot"][idx, 0])) if MOT is not None else np.nan,
                 ent=float(np.mean(ENT[idx])))
        for k, name in enumerate(["LV", "EPI", "RV"]):
            r[f"Vpred_{name}"] = float(rec["pred_area"][idx, k].sum() * dz / 1000)
            r[f"Vgt_{name}"] = float(rec["gt_area"][idx, k].sum() * dz / 1000)
        if r["has_gt"]:
            for c, name in [(1, "LV"), (2, "MYO"), (3, "RV")]:
                pm, gm = pv == c, gv == c
                den = pm.sum() + gm.sum()
                r[f"dice_{name}"] = float(2 * (pm & gm).sum() / den) if den else np.nan
                r[f"hd95_{name}"], r[f"assd_{name}"] = hd95_assd(pm, gm, sp)
        r["idx"] = [int(i) for i in idx]
        vrec.append(r)
    np.savez_compressed(os.path.join(RUNS, a.tag, f"pred_{a.bank}{'_masksdf' if a.mask_sdf else ''}.npz"),
                        **{k: v for k, v in rec.items() if k != "pred"},
                        vol=b["vol"], draw=b["draw"], z=b["z"], protocol=b["protocol"], sev=b["sev"], has_gt=has_gt)
    json.dump(vrec, open(os.path.join(RUNS, a.tag, f"vol_{a.bank}{'_masksdf' if a.mask_sdf else ''}.json"), "w"))
    ok = [r for r in vrec if r["has_gt"]]
    print(a.tag, a.bank, "dice", {c: round(float(np.nanmean([r[f'dice_{c}'] for r in ok])), 4) for c in ("LV", "MYO", "RV")},
          "hd95", {c: round(float(np.nanmean([r[f'hd95_{c}'] for r in ok])), 2) for c in ("LV", "MYO", "RV")}, flush=True)


if __name__ == "__main__":
    main()
