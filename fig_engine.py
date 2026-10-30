"""Fig.: corruption engine illustration (grayscale)."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kspace_sim as ks
from breathing import load_recordings
from build_bank import HalfBank
from data_prep import read_cohort

import plotstyle
plotstyle.apply()


class Rec(HalfBank):
    """Wraps the bank so that the drawn per-shot states can be plotted."""

    def protocol_trace(self, rng, protocol, n_shots, rr):
        r = super().protocol_trace(rng, protocol, n_shots, rr)
        self.last = r
        return r


def main(out):
    vols = read_cohort("/home/claude/data/acdc.bin.gz", "acdc")
    v = [x for x in vols if x["id"] == "patient101_ed"][0]
    z = v["img"].shape[0] // 2
    tb = Rec(load_recordings("/home/claude/data/rsp"), 1)
    fig, ax = plt.subplots(2, 5, figsize=(7.16, 3.3), gridspec_kw=dict(height_ratios=[1.0, 0.55], hspace=0.28, wspace=0.08))
    crop = slice(16, 112)
    ax[0, 0].imshow(ks.normalise(v["img"][z])[crop, crop], cmap="gray", vmin=0, vmax=1)
    ax[0, 0].set_title("Source", fontsize=8)
    ax[1, 0].axis("off")
    for j, p in enumerate(ks.PROTOCOLS):
        rng = np.random.default_rng(40 + j)
        c, c0, l, m = ks.corrupt_slice(v["img"], v["lab"], z, rng, tb, p, 1.5, 10.0, ordering="sequential")
        a = ax[0, j + 1]
        a.imshow(ks.normalise(c)[crop, crop], cmap="gray", vmin=0, vmax=1)
        a.set_title(p, fontsize=8)
        a.text(3, 92, f"S = {m['sev']:.1f} mm", color="white", fontsize=7,
               bbox=dict(facecolor="black", edgecolor="none", pad=1.2, alpha=0.75))
        r = tb.last
        segs = ks.segments(128, len(r), "sequential")
        s0 = [i for i, sg in enumerate(segs) if 64 in sg][0]
        b = ax[1, j + 1]
        b.plot(np.arange(len(r)), r, color="black", lw=0.8)
        b.scatter(np.arange(len(r)), r, s=8, color="black", zorder=3)
        b.scatter([s0], [r[s0]], s=55, facecolors="none", edgecolors="black", lw=1.0, zorder=4)
        b.set_ylim(-0.1, max(1.4, r.max() * 1.15))
        b.set_xlabel("Shot index", fontsize=7)
        if j == 0:
            b.set_ylabel("Resp. state $r$", fontsize=7)
        else:
            b.set_yticklabels([])
        b.tick_params(labelsize=6)
    for a in ax[0]:
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    fig.savefig(out, dpi=500, bbox_inches="tight", pad_inches=0.03)


if __name__ == "__main__":
    main(sys.argv[1])
