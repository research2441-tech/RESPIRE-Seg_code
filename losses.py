import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

TAU = 10.0  # mm, saturation scale of the signed-distance encoding
T_MIN = 3.0  # mm, minimum myocardial wall thickness used by the topology prior


def structures(lab):
    """Binary masks for LV endocardium, LV epicardium and RV from a harmonised label map."""
    return [lab == 1, (lab == 1) | (lab == 2), lab == 3]


def signed_distance(mask, px):
    if mask.sum() == 0:
        return np.full(mask.shape, 50.0, np.float32)
    inside = distance_transform_edt(mask) * px
    outside = distance_transform_edt(~mask) * px
    return (outside - inside).astype(np.float32)


def sdf_targets(lab_batch, px):
    out = np.zeros((lab_batch.shape[0], 3) + lab_batch.shape[1:], np.float32)
    for b in range(lab_batch.shape[0]):
        for k, m in enumerate(structures(lab_batch[b])):
            out[b, k] = np.tanh(signed_distance(m, px) / TAU)
    return torch.from_numpy(out)


def dice_ce(logits, y, eps=1e-5):
    ce = F.cross_entropy(logits, y)
    p = logits.softmax(1)
    oh = F.one_hot(y, logits.shape[1]).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    dice = 1 - (2 * inter[1:] + eps) / (den[1:] + eps)
    return ce + dice.mean()


def topology_prior(psi):
    """psi: (B,3,H,W) tanh-encoded signed distances (endo, epi, rv); negative inside."""
    endo, epi, rv = psi[:, 0], psi[:, 1], psi[:, 2]
    ring = F.relu(epi - endo).mean()                       # epicardium must enclose endocardium
    band = (endo.abs() < np.tanh(1.5 / TAU)).float()        # pixels on the endocardial contour
    thick = (F.relu(epi + np.tanh(T_MIN / TAU)) * band).sum() / (band.sum() + 1)
    excl = (F.relu(-rv) * F.relu(-epi)).mean()              # RV cavity cannot overlap the LV
    return ring + thick + excl


def paired_consistency(out1, out2):
    p1, p2 = out1["logits"].softmax(1), out2["logits"].softmax(1)
    m = 0.5 * (p1 + p2)
    js = 0.5 * ((p1 * (p1.clamp_min(1e-6) / m.clamp_min(1e-6)).log()).sum(1)
                + (p2 * (p2.clamp_min(1e-6) / m.clamp_min(1e-6)).log()).sum(1)).mean()
    l = js
    if "sdf" in out1:
        l = l + (out1["sdf"] - out2["sdf"]).abs().mean()
    return l
