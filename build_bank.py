"""Pre-compute corruption banks so that every model sees identical training and test data.

usage: python build_bank.py <name>
"""
import os
import sys
import json
import numpy as np
from multiprocessing import Pool

import kspace_sim as ks
from breathing import load_recordings, TraceBank, TRAIN_TRACES
from data_prep import read_cohort, subject_of

ROOT = os.environ.get("RESPIRE_DATA", "/home/claude/data")
NK = os.environ.get("RESPIRE_RSP", "/home/claude/data/rsp")
OUT = os.path.join(ROOT, "banks")


def splits_acdc(vols):
    subs = sorted({subject_of(v["id"]) for v in vols if v["tag"] == "train"})
    rng = np.random.default_rng(2026)
    rng.shuffle(subs)
    tr, va, ca = subs[:70], subs[70:80], subs[80:100]
    te = sorted({subject_of(v["id"]) for v in vols if v["tag"] == "test"})
    return dict(train=tr, val=va, cal=ca, test=te)


class HalfBank(TraceBank):
    """Held-out recording split in halves: first half for calibration, second half for testing."""

    def __init__(self, recordings, half):
        r = recordings["heldout"]
        m = len(r) // 2
        self.rec = [r[:m]] if half == 0 else [r[m:]]
        self.names = ["heldout%d" % half]


class SineBank(TraceBank):
    """Ablation: idealised sinusoidal breathing instead of recorded traces."""

    def __init__(self):
        self.rec, self.names = [], ["sine"]

    def window(self, rng, duration_s):
        f = rng.uniform(0.2, 0.3)
        t = np.arange(int(duration_s * 25) + 2) / 25
        return 0.5 * (1 - np.cos(2 * np.pi * f * t + rng.uniform(0, 6.28)))


def _job(args):
    vol, z, seed, protocol, bankkind, opts = args
    rng = np.random.default_rng(seed)
    rec = _job.rec
    tb = {"train": TraceBank(rec, TRAIN_TRACES), "cal": HalfBank(rec, 0), "test": HalfBank(rec, 1),
          "sine": SineBank()}[bankkind]
    img, lab = vol["img"], vol["lab"]
    if opts.get("mode") == "clean":
        c = img[z]
        return ks.normalise(c), lab[z], dict(sev=0.0, tp=0.0, ip=0.0, protocol="clean")
    if opts.get("mode") == "imgwarp":
        c, l2 = ks.image_domain_warp(img[z], lab[z], rng, strength=rng.uniform(0.5, 2.5))
        return ks.normalise(c), l2, dict(sev=0.0, tp=0.0, ip=0.0, protocol="imgwarp")
    if opts.get("mode") == "generic":
        c = ks.generic_kspace_motion(img[z], rng, int(rng.integers(1, 4)))
        return ks.normalise(c), lab[z], dict(sev=0.0, tp=0.0, ip=0.0, protocol="generic")
    c, c0, l, meta = ks.corrupt_slice(img, lab, z, rng, tb, protocol, vol["px"], vol["dz"],
                                      amp_scale=opts.get("amp", 1.0),
                                      through_plane=opts.get("tp", True), multicoil=opts.get("mc", True),
                                      ordering=opts.get("ordering"))
    return ks.normalise(c), l, meta


def _init(rsp_dir):
    _job.rec = load_recordings(rsp_dir)


def build(name, vols, subjects, draws, bankkind, opts, seed0):
    tasks, keys = [], []
    sset = set(subjects)
    k = 0
    for v in vols:
        if subject_of(v["id"]) not in sset:
            continue
        for z in range(v["img"].shape[0]):
            if (v["lab"][z] > 0).sum() < 20:
                continue
            for d in range(draws):
                protocol = ks.PROTOCOLS[d % 4] if opts.get("protocols") is None else opts["protocols"][d % len(opts["protocols"])]
                tasks.append((v, z, seed0 + k, protocol, bankkind, opts))
                keys.append((v["id"], z, d))
                k += 1
    with Pool(int(os.environ.get("NPROC", 2)), initializer=_init, initargs=(NK,)) as p:
        res = p.map(_job, tasks, chunksize=16)
    X = np.stack([r[0] for r in res]).astype(np.float16)
    Y = np.stack([r[1] for r in res]).astype(np.uint8)
    M = [r[2] for r in res]
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, name + ".npz"), X=X, Y=Y,
             sev=np.array([m["sev"] for m in M], np.float32), tp=np.array([m["tp"] for m in M], np.float32),
             ip=np.array([m["ip"] for m in M], np.float32),
             protocol=np.array([m["protocol"] for m in M]),
             vol=np.array([kk[0] for kk in keys]), z=np.array([kk[1] for kk in keys]),
             draw=np.array([kk[2] for kk in keys]),
             dz=np.array([t[0]["dz"] for t in tasks], np.float32))
    print(name, X.shape, flush=True)


SPECS = {
    # name: (cohort, split, draws, bankkind, opts, seed)
    "acdc_train_phys": ("acdc", "train", 8, "train", {}, 1000),
    "acdc_train_clean": ("acdc", "train", 1, "train", {"mode": "clean"}, 2000),
    "acdc_train_imgwarp": ("acdc", "train", 8, "train", {"mode": "imgwarp"}, 3000),
    "acdc_train_generic": ("acdc", "train", 8, "train", {"mode": "generic"}, 4000),
    "acdc_train_notp": ("acdc", "train", 8, "train", {"tp": False}, 5000),
    "acdc_train_1coil": ("acdc", "train", 8, "train", {"mc": False}, 6000),
    "acdc_train_sine": ("acdc", "train", 8, "sine", {}, 7000),
    "acdc_val": ("acdc", "val", 4, "train", {}, 8000),
    "acdc_cal": ("acdc", "cal", 8, "cal", {}, 9000),
    "acdc_test": ("acdc", "test", 4, "test", {}, 10000),
    "acdc_test_clean": ("acdc", "test", 1, "test", {"mode": "clean"}, 11000),
    "acdc_test_stress": ("acdc", "test", 2, "test", {"amp": 1.5, "protocols": ["FB", "IB"]}, 12000),
    "mnms2_test": ("mnms2", "all", 4, "test", {}, 13000),
    "mnms2_cal": ("mnms2", "cal", 4, "cal", {}, 14000),
}


if __name__ == "__main__":
    name = sys.argv[1]
    cohort, split, draws, kind, opts, seed = SPECS[name]
    vols = read_cohort(os.path.join(ROOT, f"{cohort}.bin.gz"), cohort)
    if cohort == "acdc":
        sp = splits_acdc(vols)
        subjects = sp[split]
        json.dump(sp, open(os.path.join(ROOT, "acdc_splits.json"), "w"), indent=1)
    else:
        subs = sorted({subject_of(v["id"]) for v in vols})
        rng = np.random.default_rng(7)
        perm = list(rng.permutation(subs))
        subjects = perm[:40] if split == "cal" else (perm[40:] if split == "all" else perm)
    build(name, vols, subjects, draws, kind, opts, seed)
