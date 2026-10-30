"""Respiratory trace library built from real respiration-belt recordings.

The recordings ship with the NeuroKit2 open data folder
(https://github.com/neuropsychology/NeuroKit/tree/master/data).  Three
recordings are used to drive the simulator during training and calibration;
the fourth (rsp_1000hz.txt) is held out so that test-time corruptions are
driven by a breathing pattern never seen in training.
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly

FS = 25.0  # Hz after resampling


def _clean(sig, fs):
    sig = np.asarray(sig, dtype=np.float64)
    b, a = butter(2, [0.05 / (fs / 2), 1.0 / (fs / 2)], btype="band")
    s = filtfilt(b, a, sig - sig.mean())
    return s


def load_recordings(data_dir):
    """Return dict name -> trace sampled at FS Hz, z-scored (not yet mapped to [0,1])."""
    rec = {}
    for name, fname, col, fs in [
        ("rest5", "bio_resting_5min_100hz.csv", "RSP", 100),
        ("rest8", "bio_resting_8min_100hz.csv", "RSP", 100),
        ("event", "bio_eventrelated_100hz.csv", "RSP", 100),
    ]:
        sig = pd.read_csv(os.path.join(data_dir, fname))[col].values
        s = _clean(sig, fs)
        rec[name] = resample_poly(s, int(FS), fs)
    raw = np.loadtxt(os.path.join(data_dir, "rsp_1000hz.txt"))
    s = _clean(resample_poly(raw, 100, 1000), 100)
    rec["heldout"] = resample_poly(s, int(FS), 100)
    for k in rec:
        r = rec[k]
        rec[k] = (r - np.median(r)) / (np.percentile(r, 95) - np.percentile(r, 5) + 1e-9)
    return rec


TRAIN_TRACES = ("rest5", "rest8", "event")
TEST_TRACES = ("heldout",)


class TraceBank:
    """Draws breathing windows and maps them to a normalised displacement r(t).

    r = 0 corresponds to end-expiration and r = 1 to a typical end-inspiration;
    intensive breathing can exceed 1.
    """

    def __init__(self, recordings, names):
        self.rec = [recordings[n] for n in names]
        self.names = list(names)

    def window(self, rng, duration_s):
        n = int(duration_s * FS) + 2
        i = rng.integers(len(self.rec))
        r = self.rec[i]
        start = rng.integers(0, len(r) - n)
        w = r[start:start + n].copy()
        lo = np.percentile(r, 5)
        w = np.clip((w - lo), 0.0, None)  # 0 = end-expiration level
        return w

    def protocol_trace(self, rng, protocol, n_shots, rr):
        """Respiratory state sampled at each heartbeat (one shot per beat)."""
        dur = n_shots * rr + 1.0
        if protocol == "BH":  # compliant breath-hold: frozen state, slow drift
            level = rng.uniform(0.0, 0.25)
            r = level + np.cumsum(rng.normal(0, 0.004, n_shots))
            return np.clip(r, 0, None)
        if protocol == "IB":  # intensive breathing: deeper and faster
            w = self.window(rng, dur * 1.6)
            t = np.arange(n_shots) * rr * 1.6
            r = np.interp(t, np.arange(len(w)) / FS, w) * rng.uniform(1.5, 2.0)
            return r
        w = self.window(rng, dur)
        t = np.arange(n_shots) * rr
        r = np.interp(t, np.arange(len(w)) / FS, w)
        if protocol == "HBH":  # holds for roughly half the scan, then breathes
            k = int(n_shots * rng.uniform(0.4, 0.6))
            r[:k] = r[k]
            r = r * rng.uniform(0.6, 1.0)
        return r
