"""Severity-conditioned (Mondrian) split conformal calibration of contour bands and EF intervals."""
import numpy as np

TGRID = np.arange(-15, 15.01, 0.25)


def conformal_quantile(scores, alpha):
    s = np.sort(np.asarray(scores)[np.isfinite(scores)])
    n = len(s)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else float(s[k - 1])


def bin_edges(pred_cal, n_bins):
    if n_bins <= 1:
        return np.array([])
    return np.quantile(pred_cal, np.linspace(0, 1, n_bins + 1)[1:-1])


def calibrate(scores_cal, pred_cal, alpha, n_bins):
    """Per-bin thresholds; n_bins = 1 reproduces marginal split conformal."""
    edges = bin_edges(pred_cal, n_bins)
    b = np.digitize(pred_cal, edges)
    th = np.array([conformal_quantile(scores_cal[b == j], alpha) for j in range(max(n_bins, 1))])
    return edges, th


def thresholds_for(pred, edges, th):
    return th[np.digitize(pred, edges)]


def area_at(curve, t):
    """Interpolate an area curve (over TGRID) at threshold(s) t; clipped to the grid."""
    t = np.clip(t, TGRID[0], TGRID[-1])
    return np.interp(t, TGRID, curve)


def ef_interval(curves_ed, curves_es, t, dz):
    """Guaranteed LV ejection-fraction interval from inner/outer LV cavity volumes."""
    edv_in = sum(area_at(c, -t) for c in curves_ed) * dz / 1000
    edv_out = sum(area_at(c, t) for c in curves_ed) * dz / 1000
    esv_in = sum(area_at(c, -t) for c in curves_es) * dz / 1000
    esv_out = sum(area_at(c, t) for c in curves_es) * dz / 1000
    lo = 1 - esv_out / max(edv_in, 1e-6)
    hi = 1 - esv_in / max(edv_out, 1e-6)
    return 100 * lo, 100 * hi
