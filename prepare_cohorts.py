"""Convert ACDC / M&Ms-2 NIfTI folders into the compact cohort files read by data_prep.py.

Each volume is resampled to 1.5 mm in-plane, cropped to 128 x 128 around the
centroid of the annotated heart, quantised to 8 bits (image) and stored with
its label map.

usage:
  python prepare_cohorts.py acdc  <root with patientXXX/patientXXX_sax_{ed,es}[_gt].nii.gz>  acdc.bin.gz
  python prepare_cohorts.py mnms2 <root with NNN/NNN_sax_{ed,es}[_gt].nii.gz>               mnms2.bin.gz
"""
import glob
import gzip
import json
import os
import re
import struct
import sys
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates

N, PX = 128, 1.5


def resample(vol, zooms, centre, order):
    ii, jj = np.mgrid[0:N, 0:N].astype(np.float64)
    xs = centre[0] + (ii - (N - 1) / 2) * PX / zooms[0]
    ys = centre[1] + (jj - (N - 1) / 2) * PX / zooms[1]
    return np.stack([map_coordinates(vol[:, :, z], [xs, ys], order=order, mode="constant") for z in range(vol.shape[2])])


def main(cohort, root, out):
    pat = re.compile(r"(.+)_sax_(ed|es)\.nii\.gz$")
    chunks, index, pos = [], [], 0
    for f in sorted(glob.glob(os.path.join(root, "**", "*_sax_e[ds].nii.gz"), recursive=True)):
        m = pat.match(os.path.basename(f))
        gt = f.replace(".nii.gz", "_gt.nii.gz")
        if not m or not os.path.exists(gt):
            continue
        im, lb = nib.load(f), nib.load(gt)
        a, l = np.asarray(im.dataobj, np.float32), np.asarray(lb.dataobj)
        a = a[..., 0] if a.ndim == 4 else a
        z = im.header.get_zooms()[:3]
        c = np.argwhere(l > 0)[:, :2].mean(0)
        lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.8)
        I = np.clip(255 * (resample(a, z, c, 1) - lo) / (hi - lo), 0, 255).astype(np.uint8)
        L = resample(l.astype(np.float32), z, c, 0).round().astype(np.uint8)
        tag = "test" if "test" in f.split(os.sep)[-3:-1] else "train"
        vid = f"{cohort}_{m.group(1)}_{m.group(2)}" if cohort == "mnms2" else f"{m.group(1)}_{m.group(2)}"
        index.append(dict(id=vid, tag=tag, nz=int(I.shape[0]), pz=float(z[2]), px=float(z[0]), py=float(z[1]), off=pos))
        chunks += [I.tobytes(), L.tobytes()]
        pos += 2 * I.size
    head = json.dumps(dict(N=N, PX=PX, index=index)).encode()
    with gzip.open(out, "wb") as g:
        g.write(struct.pack("<I", len(head)) + b"\0" * 4 + head)
        for c in chunks:
            g.write(c)
    print(out, len(index), "volumes")


if __name__ == "__main__":
    main(*sys.argv[1:4])
