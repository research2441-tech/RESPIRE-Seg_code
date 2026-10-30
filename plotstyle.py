"""Shared grayscale publication style."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRAYS = ["#000000", "#4d4d4d", "#808080", "#a6a6a6", "#cccccc", "#e6e6e6"]
HATCH = ["", "//", "xx", "..", "\\\\", "++", "--", "oo"]
MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))]


def apply():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0,
        "legend.frameon": False, "savefig.dpi": 500, "axes.spines.top": False, "axes.spines.right": False,
    })
