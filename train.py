"""Train one model on one corruption bank.

python train.py --model respire --bank acdc_train_phys --seed 0 --tag respire_s0
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

import models
from losses import dice_ce, sdf_targets, topology_prior, paired_consistency

DATA = os.environ.get("RESPIRE_DATA", "/home/claude/data")
RUNS = os.environ.get("RESPIRE_RUNS", "/home/claude/runs")
PX = 1.5


CROP = int(os.environ.get("CROP", 96))


def load_bank(name):
    d = np.load(os.path.join(DATA, "banks", name + ".npz"), allow_pickle=True)
    out = {k: d[k] for k in d.files}
    o = (out["X"].shape[-1] - CROP) // 2
    out["X"] = out["X"][..., o:o + CROP, o:o + CROP]
    out["Y"] = out["Y"][..., o:o + CROP, o:o + CROP]
    return out


def sev_bins(sev):
    return np.digitize(sev, [0.5, 2.5])


def augment(x, y, rng, s=None):
    """Same geometric/intensity augmentation for a whole group (keeps pairs aligned)."""
    k = rng.integers(4)
    x, y = np.rot90(x, k, (-2, -1)), np.rot90(y, k, (-2, -1))
    s = None if s is None else np.rot90(s, k, (-2, -1))
    if rng.random() < 0.5:
        x, y = x[..., ::-1], y[..., ::-1]
        s = None if s is None else s[..., ::-1]
    g = rng.uniform(0.7, 1.5)
    x = np.clip(x, 0, None) ** g
    return np.ascontiguousarray(x), np.ascontiguousarray(y), (None if s is None else np.ascontiguousarray(s))


def sdf_cache(name, Y):
    f = os.path.join(DATA, "banks", f"{name}_sdf{CROP}.npy")
    if os.path.exists(f):
        return np.load(f, mmap_mode="r")
    out = np.zeros((len(Y), 3) + Y.shape[1:], np.float16)
    for i in range(0, len(Y), 256):
        out[i:i + 256] = sdf_targets(Y[i:i + 256], PX).numpy().astype(np.float16)
    np.save(f, out)
    return out


@torch.no_grad()
def val_dice(net, val, bs=64):
    net.eval()
    X, Y = val["X"], val["Y"]
    inter = np.zeros(3); den = np.zeros(3)
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs].astype(np.float32))[:, None]
        p = net(x)["logits"].argmax(1).numpy()
        y = Y[i:i + bs]
        for c in (1, 2, 3):
            inter[c - 1] += ((p == c) & (y == c)).sum()
            den[c - 1] += (p == c).sum() + (y == c).sum()
    net.train()
    return float(np.mean(2 * inter / np.maximum(den, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bank", default="acdc_train_phys")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--paired", type=int, default=0)
    ap.add_argument("--lam_sdf", type=float, default=1.0)
    ap.add_argument("--lam_topo", type=float, default=0.5)
    ap.add_argument("--lam_mot", type=float, default=0.2)
    ap.add_argument("--lam_cons", type=float, default=0.5)
    ap.add_argument("--warm", type=float, default=0.0, help="fraction of training over which auxiliary losses ramp up")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    torch.set_num_threads(int(os.environ.get("THREADS", 1)))
    out = os.path.join(RUNS, a.tag)
    os.makedirs(out, exist_ok=True)
    bank, val = load_bank(a.bank), load_bank("acdc_val")
    X, Y = bank["X"], bank["Y"]
    draws = int(bank["draw"].max()) + 1
    n_slices = len(X) // draws
    mot_t = np.log1p(np.stack([bank["sev"], bank["tp"], bank["ip"]], 1)).astype(np.float32)
    sevc = sev_bins(bank["sev"])
    net = models.build(a.model)
    SD = sdf_cache(a.bank, Y) if getattr(net, "use_sdf", False) else None
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    groups = a.bs // 2 if (a.paired and draws > 1) else a.bs
    iters = min(n_slices // (a.bs // 2), int(os.environ.get("MAXIT", 10 ** 9)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * iters, pct_start=0.1)
    best, log = -1, []
    t0 = time.time()
    for ep in range(a.epochs):
        perm = rng.permutation(n_slices)
        tot = 0.0
        for it in range(iters):
            sl = perm[(it * groups) % n_slices: (it * groups) % n_slices + groups]
            if len(sl) < groups:
                sl = rng.choice(n_slices, groups, replace=False)
            if a.paired and draws > 1:
                d = np.stack([rng.choice(draws, 2, replace=False) for _ in sl])
                idx = np.stack([sl * draws + d[:, 0], sl * draws + d[:, 1]], 1)  # (G,2)
            else:
                idx = (sl * draws + rng.integers(0, draws, len(sl)))[:, None]
            xs, ys, ss = [], [], []
            for row in idx:
                xg, yg, sg = augment(X[row].astype(np.float32), Y[row], rng,
                                     None if SD is None else SD[np.sort(row)][np.argsort(np.argsort(row))].astype(np.float32))
                xs.append(xg); ys.append(yg); ss.append(sg)
            xb = torch.from_numpy(np.stack(xs))        # (G,k,H,W)
            yb = np.stack(ys)
            k = xb.shape[1]
            xflat = xb.reshape(-1, 1, *xb.shape[2:])
            yflat = torch.from_numpy(yb.reshape(-1, *yb.shape[2:]).astype(np.int64))
            o = net(xflat)
            step = ep * iters + it
            ramp = min(1.0, step / max(1.0, a.warm * a.epochs * iters)) if a.warm > 0 else 1.0
            loss = dice_ce(o["logits"], yflat)
            if "aux" in o:
                for w, aux in zip((0.5, 0.25), o["aux"]):
                    s = yflat.shape[-1] // aux.shape[-1]
                    loss = loss + w * dice_ce(aux, yflat[:, ::s, ::s])
            if "sdf" in o:
                tgt = torch.from_numpy(np.stack(ss).reshape(-1, 3, *yb.shape[2:]))
                loss = loss + ramp * (a.lam_sdf * F.smooth_l1_loss(o["sdf"], tgt, beta=0.05) + a.lam_topo * topology_prior(o["sdf"]))
            if "mot" in o and getattr(net, "use_motion", False):
                mt = torch.from_numpy(mot_t[idx.reshape(-1)])
                loss = loss + ramp * a.lam_mot * F.smooth_l1_loss(o["mot"], mt)
            if "sev_logits" in o:
                loss = loss + 0.2 * F.cross_entropy(o["sev_logits"], torch.from_numpy(sevc[idx.reshape(-1)]))
            if a.paired and k == 2:
                o1 = {kk: v.reshape(-1, 2, *v.shape[1:])[:, 0] for kk, v in o.items() if kk in ("logits", "sdf")}
                o2 = {kk: v.reshape(-1, 2, *v.shape[1:])[:, 1] for kk, v in o.items() if kk in ("logits", "sdf")}
                loss = loss + ramp * a.lam_cons * paired_consistency(o1, o2)
            opt.zero_grad()
            loss.backward()
            opt.step(); sched.step()
            tot += loss.item()
        vd = val_dice(net, val)
        log.append(dict(epoch=ep, loss=tot / iters, val_dice=vd, t=time.time() - t0))
        print(json.dumps(log[-1]), flush=True)
        if vd > best:
            best = vd
            torch.save(net.state_dict(), os.path.join(out, "best.pt"))
    json.dump(dict(args=vars(a), log=log, best=best, params=sum(p.numel() for p in net.parameters()),
                   train_time=time.time() - t0), open(os.path.join(out, "train.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
