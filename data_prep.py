"""Read the compact cohort files and build harmonised short-axis stacks.

Each cohort file stores, per volume, a 128 x 128 crop at 1.5 mm in-plane
resolution centred on the heart, the image quantised to 8 bits and the label
map.  Labels are harmonised to 0 background, 1 LV cavity, 2 myocardium, 3 RV.
"""
import gzip
import json
import struct
import numpy as np

LABEL_MAPS = {
    "acdc": {0: 0, 1: 3, 2: 2, 3: 1},      # ACDC: 1 RV, 2 MYO, 3 LV
    "mnms2": {0: 0, 1: 3, 2: 2, 3: 1},     # M&Ms-2 release used here: 1 RV, 2 MYO, 3 LV
    "cmrxmotion": {0: 0, 1: 1, 2: 2, 3: 3},  # CMRxMotion: 1 LV, 2 MYO, 3 RV
}


def read_cohort(path, cohort):
    raw = gzip.open(path, "rb").read()
    hlen = struct.unpack("<I", raw[:4])[0]
    head = json.loads(raw[8:8 + hlen].decode())
    base = 8 + hlen
    N = head["N"]
    lut = np.zeros(256, np.uint8)
    for k, v in LABEL_MAPS[cohort].items():
        lut[k] = v
    vols = []
    for e in head["index"]:
        nz = e["nz"]
        o = base + e["off"]
        img = np.frombuffer(raw, np.uint8, nz * N * N, o).reshape(nz, N, N).astype(np.float32) / 255.0
        lab = np.frombuffer(raw, np.uint8, nz * N * N, o + nz * N * N).reshape(nz, N, N)
        lab = lut[lab]
        vols.append(dict(id=e["id"], tag=e.get("tag", ""), img=img, lab=lab, dz=float(e["pz"]),
                         px=head["PX"], meta=e))
    return vols


def subject_of(vol_id):
    return vol_id.rsplit("_", 1)[0]
