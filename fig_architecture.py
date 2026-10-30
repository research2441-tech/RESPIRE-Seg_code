"""Fig.: RESPIRE-Seg framework (grayscale vector drawing).

Arrow semantics: solid = forward data flow, dashed = second paired draw and the
reverse-mode gradient of the training loss.
"""
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import plotstyle

plotstyle.apply()
FS = 6.6


def box(ax, x, y, w, h, text, fc="white", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.01", fc=fc, ec="black",
                                lw=0.7, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FS, linespacing=1.2)
    return dict(l=x, r=x + w, b=y, t=y + h, cx=x + w / 2, cy=y + h / 2)


def arrow(ax, p, q, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, ls=ls, color="black",
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def line(ax, xs, ys, ls="-"):
    ax.plot(xs, ys, color="black", lw=0.8, ls=ls, solid_capstyle="butt")


def main(path):
    fig = plt.figure(figsize=(7.16, 4.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    g1, g2 = "#efefef", "#d6d6d6"
    for x, w, t in [(0.010, 0.290, "(a) k-space corruption engine"),
                    (0.330, 0.395, "(b) Motion-state-disentangled network"),
                    (0.745, 0.245, "(c) Severity-conditioned conformal")]:
        ax.add_patch(FancyBboxPatch((x, 0.035), w, 0.885, boxstyle="round,pad=0,rounding_size=0.012",
                                    fc="none", ec="black", lw=0.5, ls=(0, (4, 2))))
        ax.text(x + w / 2, 0.955, t, ha="center", va="center", fontsize=7.4, weight="bold")
    # ---------------- (a)
    a1 = box(ax, 0.025, 0.775, 0.125, 0.095, "Recorded breathing\ntrace $r(t)$", g1)
    a2 = box(ax, 0.165, 0.775, 0.125, 0.095, "SAX stack $V$\nand label $Y$", g1)
    a3 = box(ax, 0.025, 0.605, 0.125, 0.115, "Shot poses $\\mathbf{p}_s$\nrelative to the\n$k_y=0$ shot", "white")
    a4 = box(ax, 0.165, 0.605, 0.125, 0.115, "Through-plane $\\delta z_s$\nand heart-weighted\naffine warp", "white")
    a5 = box(ax, 0.045, 0.445, 0.225, 0.100, "Coil maps $S_c$, segmented\n$k$-space fill $\\Omega_s$, noise", g2)
    a6 = box(ax, 0.045, 0.310, 0.225, 0.075, "RSS reconstruction $\\tilde{x}$", "white")
    a7 = box(ax, 0.025, 0.090, 0.265, 0.160, "Corrupted image $\\tilde{x}$\nexact label $Y$ (no re-annotation)\n"
             "severity $\\mathcal{S}$, $\\delta_{tp}$, $\\delta_{ip}$\npaired draws $\\tilde{x}^{(1)},\\tilde{x}^{(2)}$", g1)
    arrow(ax, (a1["cx"], a1["b"]), (a3["cx"], a3["t"]))
    arrow(ax, (a2["cx"], a2["b"]), (a4["cx"], a4["t"]))
    arrow(ax, (a3["r"], a3["cy"]), (a4["l"], a4["cy"]))
    arrow(ax, (a4["cx"], a4["b"]), (a4["cx"], a5["t"]))
    arrow(ax, (a5["cx"], a5["b"]), (a6["cx"], a6["t"]))
    arrow(ax, (a6["cx"], a6["b"]), (a7["cx"], a7["t"]))
    # ---------------- (b)
    b1 = box(ax, 0.350, 0.800, 0.085, 0.070, "Input $\\tilde{x}^{(1)}$", "white")
    b2 = box(ax, 0.350, 0.675, 0.085, 0.080, "Input $\\tilde{x}^{(2)}$\n(optional)", "white", ls="--")
    b3 = box(ax, 0.465, 0.670, 0.090, 0.210, "Shared\nencoder\n$E_\\theta$", g2)
    b4 = box(ax, 0.590, 0.800, 0.120, 0.075, "Motion head\n$\\mathbf{z}_m\\rightarrow\\hat{\\mathbf{m}}$", g1)
    b5 = box(ax, 0.590, 0.680, 0.120, 0.070, "FiLM $\\boldsymbol{\\gamma}_l,\\boldsymbol{\\beta}_l$", "white")
    b6 = box(ax, 0.455, 0.470, 0.110, 0.115, "Boundary-saliency\ngates $G_l(\\cdot\\,|\\,\\mathbf{z}_m)$", g1)
    b7 = box(ax, 0.600, 0.470, 0.110, 0.115, "Decoder $D_\\theta$", g2)
    b8 = box(ax, 0.455, 0.320, 0.110, 0.075, "Softmax head $\\hat{P}$", "white")
    b9 = box(ax, 0.600, 0.320, 0.110, 0.075, "SDF head $\\hat{\\psi}_k$", "white")
    lo = box(ax, 0.350, 0.090, 0.360, 0.150,
             "$\\mathcal{L}=\\mathcal{L}_{seg}+\\lambda_d\\mathcal{L}_{sdf}+\\lambda_t\\mathcal{L}_{topo}$\n"
             "$+\\,\\lambda_m\\mathcal{L}_{mot}+\\lambda_c\\mathcal{L}_{cons}$", g1)
    # engine -> inputs (gap between panels) and engine -> loss (labels, severity)
    line(ax, [a7["r"], 0.315], [0.205, 0.205]); line(ax, [0.315, 0.315], [0.205, b1["cy"]])
    arrow(ax, (0.315, b1["cy"]), (b1["l"], b1["cy"]))
    arrow(ax, (0.315, b2["cy"]), (b2["l"], b2["cy"]), ls="--")
    arrow(ax, (a7["r"], 0.135), (lo["l"], 0.135))
    ax.text(0.320, 0.148, "$Y,\\mathcal{S}$", fontsize=FS, ha="center")
    # forward pass
    arrow(ax, (b1["r"], b1["cy"]), (b3["l"], b1["cy"]))
    arrow(ax, (b2["r"], b2["cy"]), (b3["l"], b2["cy"]), ls="--")
    arrow(ax, (b3["r"], b4["cy"]), (b4["l"], b4["cy"]))
    arrow(ax, (b4["cx"], b4["b"]), (b5["cx"], b5["t"]))
    arrow(ax, (b5["cx"], b5["b"]), (b5["cx"], b7["t"]))
    arrow(ax, (b6["cx"] - 0.02, b3["b"]), (b6["cx"] - 0.02, b6["t"]))
    ax.text(b6["cx"] - 0.016, 0.625, "skips", fontsize=FS - 0.6, ha="left")
    arrow(ax, (b5["l"], b5["b"] + 0.01), (b6["r"] - 0.012, b6["t"]))
    ax.text(0.578, 0.612, "$\\mathbf{z}_m$", fontsize=FS, ha="left")
    arrow(ax, (b6["r"], b6["cy"]), (b7["l"], b6["cy"]))
    arrow(ax, (b7["cx"], b7["b"]), (b9["cx"], b9["t"]))
    arrow(ax, (b7["l"] + 0.012, b7["b"]), (b8["r"] - 0.012, b8["t"]))
    arrow(ax, (b8["cx"], b8["b"]), (b8["cx"], lo["t"]))
    arrow(ax, (b9["cx"], b9["b"]), (b9["cx"], lo["t"]))
    # reverse-mode gradient
    ax.add_patch(FancyArrowPatch((0.395, lo["t"]), (b3["l"] + 0.012, b3["b"]), arrowstyle="-|>", mutation_scale=7,
                                 lw=0.9, ls=(0, (3, 2)), color="black", connectionstyle="arc3,rad=-0.25",
                                 shrinkA=0, shrinkB=0))
    ax.text(0.372, 0.43, "$\\nabla_\\theta\\mathcal{L}$", fontsize=FS + 0.4, ha="center")
    # ---------------- (c)
    c1 = box(ax, 0.760, 0.780, 0.215, 0.090, "Calibration studies\n(held-out breathing trace)", g1)
    c2 = box(ax, 0.760, 0.640, 0.215, 0.085, "Band loss $\\ell_i(q)$ from $\\hat{\\phi}_k$,\nbinned by $\\hat{\\mathcal{S}}_i$", "white")
    c3 = box(ax, 0.760, 0.500, 0.215, 0.080, "Risk-controlled $\\hat{q}_{k,b}$", g2)
    c4 = box(ax, 0.760, 0.320, 0.215, 0.110, "Inner/outer contours\n$\\hat{\\phi}_k\\leq\\mp\\hat{q}_{k,b}$", "white")
    c5 = box(ax, 0.760, 0.090, 0.215, 0.160, "EDV/ESV bounds $\\rightarrow w_j$\nnormalised LVEF interval\nnon-diagnostic flag", g1)
    for u, v in ((c1, c2), (c2, c3), (c3, c4), (c4, c5)):
        arrow(ax, (u["cx"], u["b"]), (v["cx"], v["t"]))
    arrow(ax, (b9["r"], b9["cy"]), (c4["l"], b9["cy"]))
    fig.savefig(path, dpi=600)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fig_framework.png")
