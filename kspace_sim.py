"""Physics-faithful respiratory corruption of segmented cine short-axis slices.

Forward model (per coil c, shot s, phase-encode line k_y in segment Omega_s):
    K_c[k_y, :] = F{ S_c * e^{i phi} * x_s }[k_y, :]
where x_s is the slice content at the cardiac pose of shot s, which combines an
in-plane affine displacement weighted towards the heart and a through-plane
shift resolved by interpolating the short-axis stack.  The image is
reconstructed by root-sum-of-squares.  The label always refers to the pose of
the shot that acquires the k-space centre, so the corrupted image and its
label stay exactly registered.
"""
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter, binary_dilation

PROTOCOLS = ("BH", "HBH", "FB", "IB")


def coil_maps(n, nc, rng):
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    c = (n - 1) / 2
    maps = np.zeros((nc, n, n), np.complex128)
    rad = 0.72 * n
    width = rng.uniform(0.45, 0.6) * n
    off = rng.uniform(0, 2 * np.pi)
    for k in range(nc):
        ang = off + 2 * np.pi * k / nc
        py, px = c + rad * np.sin(ang), c + rad * np.cos(ang)
        mag = np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * width ** 2))
        ph = rng.uniform(-np.pi, np.pi) + 0.6 * ((yy - c) * np.sin(ang) + (xx - c) * np.cos(ang)) / n
        maps[k] = mag * np.exp(1j * ph)
    sos = np.sqrt((np.abs(maps) ** 2).sum(0)) + 1e-8
    return maps / sos


def smooth_phase(n, rng):
    yy, xx = np.mgrid[0:n, 0:n] / n - 0.5
    a = rng.normal(0, 0.8, 5)
    return a[0] * yy + a[1] * xx + a[2] * yy * xx + a[3] * yy ** 2 + a[4] * xx ** 2


def heart_weight(lab2d, sigma_px):
    m = binary_dilation(lab2d > 0, iterations=int(max(2, sigma_px)))
    w = gaussian_filter(m.astype(np.float64), sigma_px)
    return w / (w.max() + 1e-8)


def slice_at(vol, zf):
    """Linear interpolation of the short-axis stack at fractional slice index zf."""
    z0 = int(np.floor(zf))
    t = zf - z0
    z0c = np.clip(z0, 0, vol.shape[0] - 1)
    z1c = np.clip(z0 + 1, 0, vol.shape[0] - 1)
    return (1 - t) * vol[z0c] + t * vol[z1c]


def warp(img, disp_y, disp_x, order=1):
    n0, n1 = img.shape
    yy, xx = np.mgrid[0:n0, 0:n1].astype(np.float64)
    return map_coordinates(img, [yy - disp_y, xx - disp_x], order=order, mode="nearest")


def pose_field(n, pose, wmap, px_mm, bg_frac=0.3):
    """Displacement field (pixels) for an in-plane affine cardiac pose."""
    dy_mm, dx_mm, theta, kappa = pose
    c = (n - 1) / 2
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    y, x = yy - c, xx - c
    ct, st = np.cos(theta), np.sin(theta)
    ay = (kappa * (ct * y - st * x)) - y + dy_mm / px_mm
    ax = (kappa * (st * y + ct * x)) - x + dx_mm / px_mm
    w = wmap + bg_frac * (1 - wmap)
    return ay * w, ax * w


def segments(n_lines, n_shots, ordering):
    ky = np.arange(n_lines)
    if ordering == "interleaved":
        return [ky[s::n_shots] for s in range(n_shots)]
    order = np.argsort(np.abs(ky - n_lines // 2), kind="stable") if ordering == "centric" else ky
    per = int(np.ceil(n_lines / n_shots))
    return [np.sort(order[s * per:(s + 1) * per]) for s in range(n_shots)]


def corrupt_slice(vol, lab, z, rng, trace_bank, protocol, px_mm, dz_mm,
                  nc=8, amp_scale=1.0, through_plane=True, multicoil=True,
                  ordering=None, snr_db=None):
    """Return (corrupted image, clean image, label, meta) for slice z of a stack.

    vol, lab: (Z, N, N) float image and integer label at the cardiac phase.
    """
    n = vol.shape[1]
    n_shots = int(rng.integers(8, 17))
    rr = rng.uniform(0.75, 1.15)
    ordering = ordering or rng.choice(["sequential", "interleaved", "centric"], p=[0.4, 0.3, 0.3])
    r = trace_bank.protocol_trace(rng, protocol, n_shots, rr)
    a_tp = rng.uniform(8.0, 16.0) * amp_scale     # superior-inferior excursion projected through-plane (mm)
    a_ip = rng.uniform(3.0, 8.0) * amp_scale      # in-plane (oblique AP/LR) excursion (mm)
    psi = rng.uniform(0, 2 * np.pi)
    th_max = np.deg2rad(rng.uniform(0.0, 4.0)) * amp_scale
    k_max = rng.uniform(0.0, 0.03) * amp_scale
    segs = segments(n, n_shots, ordering)
    s0 = [i for i, sg in enumerate(segs) if (n // 2) in sg][0]
    rel = r - r[s0]                                  # anchor on the k-space-centre shot
    poses = np.stack([rel * a_ip * np.sin(psi), rel * a_ip * np.cos(psi),
                      rel * th_max, 1.0 + rel * k_max], 1)
    dz = rel * a_tp if through_plane else np.zeros_like(rel)
    wmap = heart_weight(lab[z], sigma_px=6.0 / px_mm)
    maps = coil_maps(n, nc, rng) if multicoil else np.ones((1, n, n), np.complex128)
    phase = np.exp(1j * smooth_phase(n, rng))
    clean = vol[z].astype(np.float64)
    k_clean = np.fft.fftshift(np.fft.fft2(maps * (clean * phase)[None], norm="ortho"), axes=(-2, -1))
    k_corr = np.zeros_like(k_clean)
    for s, sg in enumerate(segs):
        img_s = slice_at(vol, z + dz[s] / dz_mm) if through_plane else clean
        fy, fx = pose_field(n, poses[s], wmap, px_mm)
        img_s = warp(img_s, fy, fx)
        ks = np.fft.fftshift(np.fft.fft2(maps * (img_s * phase)[None], norm="ortho"), axes=(-2, -1))
        k_corr[:, sg, :] = ks[:, sg, :]
    snr_db = snr_db if snr_db is not None else rng.uniform(22, 32)
    sig = np.sqrt(np.mean(np.abs(k_clean) ** 2))
    sd = sig / (10 ** (snr_db / 20)) / np.sqrt(2)
    noise = rng.normal(0, sd, k_clean.shape) + 1j * rng.normal(0, sd, k_clean.shape)
    rss = lambda k: np.sqrt((np.abs(np.fft.ifft2(np.fft.ifftshift(k + noise, axes=(-2, -1)), norm="ortho")) ** 2).sum(0))
    img_c, img_0 = rss(k_corr), rss(k_clean)
    # severity index: k-space-energy-weighted RMS pose deviation from the anchor (mm)
    e_row = (np.abs(k_clean) ** 2).sum((0, 2))
    E = np.array([e_row[sg].sum() for sg in segs])
    radius = 30.0
    dev2 = (poses[:, 0] ** 2 + poses[:, 1] ** 2 + (radius * poses[:, 2]) ** 2
            + (radius * (poses[:, 3] - 1)) ** 2 + dz ** 2)
    sev = float(np.sqrt((E * dev2).sum() / E.sum()))
    tp = float(np.sqrt((E * dz ** 2).sum() / E.sum()))
    ip = float(np.sqrt((E * (poses[:, 0] ** 2 + poses[:, 1] ** 2)).sum() / E.sum()))
    meta = dict(sev=sev, tp=tp, ip=ip, n_shots=n_shots, ordering=str(ordering), protocol=protocol)
    return img_c.astype(np.float32), img_0.astype(np.float32), lab[z].astype(np.uint8), meta


def normalise(img):
    lo, hi = np.percentile(img, 1), np.percentile(img, 99.5)
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1.2).astype(np.float32)


# ---------------------------------------------------------------- baselines
def image_domain_warp(img, lab, rng, strength):
    """Sinusoid + random B-spline-like warp applied to image AND label (image-domain paradigm)."""
    n = img.shape[0]
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    lam = rng.uniform(0.5, 1.5) * n
    ay, ax = rng.uniform(1, 4, 2) * strength
    dy = ay * np.sin(2 * np.pi * xx / lam + rng.uniform(0, 6.3))
    dx = ax * np.sin(2 * np.pi * yy / lam + rng.uniform(0, 6.3))
    for d in (dy, dx):
        coarse = rng.normal(0, 2.0 * strength, (6, 6))
        from scipy.ndimage import zoom
        d += zoom(coarse, n / 6, order=3)[:n, :n]
    return warp(img, dy, dx, 1), warp(lab.astype(np.float64), dy, dx, 0).astype(np.uint8)


def generic_kspace_motion(img, rng, n_events):
    """TorchIO-style rigid in-plane motion: k-space split at random times, single coil, no trace."""
    n = img.shape[0]
    times = np.sort(rng.uniform(0, 1, n_events))
    bounds = np.concatenate([[0], (times * n).astype(int), [n]])
    k = np.zeros((n, n), np.complex128)
    for i in range(len(bounds) - 1):
        if i == 0:
            im = img.astype(np.float64)
        else:
            pose = (rng.uniform(-6, 6), rng.uniform(-6, 6), np.deg2rad(rng.uniform(-4, 4)), 1.0)
            fy, fx = pose_field(n, pose, np.ones((n, n)), 1.5, bg_frac=1.0)
            im = warp(img.astype(np.float64), fy, fx)
        ks = np.fft.fftshift(np.fft.fft2(im, norm="ortho"))
        k[bounds[i]:bounds[i + 1]] = ks[bounds[i]:bounds[i + 1]]
    return np.abs(np.fft.ifft2(np.fft.ifftshift(k), norm="ortho")).astype(np.float32)
