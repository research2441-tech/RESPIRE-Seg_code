"""Build the real-motion external test bank from the CMRxMotion training release.

Each volunteer was scanned four times in one session: (1) breath-hold as
instructed, (2) half breath-hold, (3) free breathing, (4) intensive breathing.
Volumes rated non-diagnostic (IQA label 3) carry no segmentation; for those
the heart-centred crop reuses the centre found on the same volunteer's
labelled acquisition of the same phase.
"""
import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import map_coordinates

import kspace_sim as ks

SRC = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/Downloads/CMRxMotion Training Dataset"
OUT = "/home/claude/data/banks/cmrx_real.npz"
N, PX = 128, 1.5
PROT = {1: "BH", 2: "HBH", 3: "FB", 4: "IB"}


def load(path):
    img = nib.load(path)
    a = np.asarray(img.dataobj).astype(np.float32)
    if a.ndim == 4:
        a = a[..., 0]
    return a, img.header.get_zooms()[:3]


def resample(vol, zooms, centre, order):
    """(X, Y, Z) volume -> (Z, N, N) at PX mm around centre (x, y) in voxel units."""
    out = np.zeros((vol.shape[2], N, N), np.float32)
    ii, jj = np.mgrid[0:N, 0:N].astype(np.float64)
    xs = centre[0] + (ii - (N - 1) / 2) * PX / zooms[0]
    ys = centre[1] + (jj - (N - 1) / 2) * PX / zooms[1]
    for z in range(vol.shape[2]):
        out[z] = map_coordinates(vol[:, :, z], [xs, ys], order=order, mode="constant", cval=0)
    return out


def main():
    iqa = pd.read_csv(os.path.join(SRC, "IQA.csv"))
    q = dict(zip(iqa["Image"], iqa["Label"]))
    centres = {}
    items = []
    for p in range(1, 21):
        for k in range(1, 5):
            for ph in ("ED", "ES"):
                name = f"P{p:03d}-{k}-{ph}"
                base = os.path.join(SRC, "data", f"P{p:03d}-{k}", name)
                lab = base + "-label.nii.gz"
                items.append((p, k, ph, name, base + ".nii.gz", lab if os.path.exists(lab) else None))
                if os.path.exists(lab):
                    l, _ = load(lab)
                    idx = np.argwhere(l > 0)
                    centres.setdefault((p, ph), idx[:, :2].mean(0))
    X, Y, rows = [], [], []
    for p, k, ph, name, ip, lp in items:
        vol, zooms = load(ip)
        c = centres.get((p, ph)) if lp is None else np.argwhere(load(lp)[0] > 0)[:, :2].mean(0)
        if c is None:
            c = centres.get((p, "ED" if ph == "ES" else "ES"))
        lo, hi = np.percentile(vol, 0.5), np.percentile(vol, 99.8)
        vol = np.clip((vol - lo) / (hi - lo + 1e-6), 0, 1)
        im = resample(vol, zooms, c, 1)
        if lp is not None:
            lb = resample(load(lp)[0], zooms, c, 0).round().astype(np.uint8)
        else:
            lb = np.zeros_like(im, np.uint8)
        ref = lb if lp is not None else None
        for z in range(im.shape[0]):
            keep = (lb[z] > 0).sum() >= 20 if lp is not None else (abs(z - im.shape[0] / 2) <= im.shape[0] / 2 - 1)
            if not keep:
                continue
            X.append(ks.normalise(im[z]))
            Y.append(lb[z])
            rows.append(dict(vol=f"P{p:03d}-{k}_{ph.lower()}", subject=f"P{p:03d}", z=z, protocol=PROT[k],
                             has_gt=lp is not None, iqa=int(q.get(name, -1)), dz=float(zooms[2])))
    r = pd.DataFrame(rows)
    np.savez(OUT, X=np.stack(X).astype(np.float16), Y=np.stack(Y).astype(np.uint8),
             vol=r["vol"].values, z=r["z"].values, draw=np.zeros(len(r), int), protocol=r["protocol"].values,
             sev=np.full(len(r), np.nan, np.float32), tp=np.full(len(r), np.nan, np.float32),
             ip=np.full(len(r), np.nan, np.float32), dz=r["dz"].values.astype(np.float32),
             has_gt=r["has_gt"].values, iqa=r["iqa"].values)
    print(OUT, len(r), r.groupby("protocol")["has_gt"].mean().to_dict(), r.groupby("iqa").size().to_dict())


if __name__ == "__main__":
    main()
