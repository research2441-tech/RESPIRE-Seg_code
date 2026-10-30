"""Segmentation networks: baselines and RESPIRE-Seg.

All networks share the same 4-level encoder width schedule so that parameter
budgets stay comparable; RESPIRE-Seg adds a motion-state head, FiLM conditioning,
boundary-saliency gated skips and a signed-distance head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 4  # background, LV cavity, myocardium, RV cavity


def conv_block(cin, cout, norm="bn", act="relu"):
    Norm = nn.BatchNorm2d if norm == "bn" else (lambda c: nn.InstanceNorm2d(c, affine=True))
    Act = (lambda: nn.ReLU(inplace=True)) if act == "relu" else (lambda: nn.LeakyReLU(0.01, inplace=True))
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), Norm(cout), Act(),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), Norm(cout), Act())


class Encoder(nn.Module):
    def __init__(self, c=16, depth=4, norm="bn", act="relu"):
        super().__init__()
        ch = [c * 2 ** i for i in range(depth + 1)]
        self.blocks = nn.ModuleList([conv_block(1, ch[0], norm, act)] +
                                    [conv_block(ch[i], ch[i + 1], norm, act) for i in range(depth)])
        self.ch = ch

    def forward(self, x):
        feats = []
        for i, b in enumerate(self.blocks):
            x = b(x if i == 0 else F.max_pool2d(x, 2))
            feats.append(x)
        return feats


class UNet(nn.Module):
    """Plain U-Net decoder over the shared encoder."""

    def __init__(self, c=16, depth=4, norm="bn", act="relu", deep_sup=False):
        super().__init__()
        self.enc = Encoder(c, depth, norm, act)
        ch = self.enc.ch
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([conv_block(2 * ch[i], ch[i], norm, act) for i in reversed(range(depth))])
        self.head = nn.Conv2d(ch[0], N_CLASSES, 1)
        self.deep_sup = deep_sup
        if deep_sup:
            self.aux = nn.ModuleList([nn.Conv2d(ch[i], N_CLASSES, 1) for i in (2, 1)])

    def skip(self, s, i):
        return s

    def forward(self, x):
        f = self.enc(x)
        h = f[-1]
        aux = []
        for j, (up, dec) in enumerate(zip(self.up, self.dec)):
            i = len(f) - 2 - j
            h = up(h)
            h = dec(torch.cat([self.skip(f[i], i) if not hasattr(self, "gate") else self.gate(f[i], h, i), h], 1))
            if self.deep_sup and i in (2, 1):
                aux.append(self.aux[(2, 1).index(i)](h))
        out = {"logits": self.head(h)}
        if self.deep_sup and self.training:
            out["aux"] = aux
        return out


class AttentionUNet(UNet):
    """Additive attention gates on skip connections."""

    def __init__(self, c=16, depth=4):
        super().__init__(c, depth)
        ch = self.enc.ch
        self.att = nn.ModuleList([nn.ModuleDict(dict(
            wx=nn.Conv2d(ch[i], ch[i] // 2, 1), wg=nn.Conv2d(ch[i], ch[i] // 2, 1),
            psi=nn.Conv2d(ch[i] // 2, 1, 1))) for i in range(depth)])

    def gate(self, s, g, i):
        a = self.att[i]
        q = torch.sigmoid(a["psi"](F.relu(a["wx"](s) + a["wg"](g))))
        return s * q


class SharpUNet(UNet):
    """Depthwise sharpening kernel applied to encoder skip features (Sharp U-Net)."""

    def __init__(self, c=16, depth=4):
        super().__init__(c, depth)
        k = torch.tensor([[-1., -1., -1.], [-1., 9., -1.], [-1., -1., -1.]])
        self.register_buffer("k", k[None, None])

    def skip(self, s, i):
        w = self.k.expand(s.shape[1], 1, 3, 3)
        return F.conv2d(s, w, padding=1, groups=s.shape[1])


class TransUNetLite(UNet):
    """CNN encoder with a Transformer bottleneck on the 1/8 feature map (hybrid CNN-Transformer)."""

    def __init__(self, c=16, depth=4, layers=4, heads=4):
        super().__init__(c, depth)
        d = self.enc.ch[-1]
        self.pos = nn.Parameter(torch.zeros(1, 64, d))
        el = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(el, layers)

    def forward(self, x):
        f = self.enc(x)
        b, cdim, hh, ww = f[-1].shape
        t = f[-1].flatten(2).transpose(1, 2)
        pos = self.pos if t.shape[1] == self.pos.shape[1] else F.interpolate(
            self.pos.transpose(1, 2), size=t.shape[1]).transpose(1, 2)
        f[-1] = self.tr(t + pos).transpose(1, 2).reshape(b, cdim, hh, ww)
        h = f[-1]
        for j, (up, dec) in enumerate(zip(self.up, self.dec)):
            i = len(f) - 2 - j
            h = dec(torch.cat([f[i], up(h)], 1))
        return {"logits": self.head(h)}


class MultiTaskUNet(UNet):
    """Segmentation plus motion-severity classification from the bottleneck (multi-task competitor)."""

    def __init__(self, c=16, depth=4, n_sev=3):
        super().__init__(c, depth)
        self.cls = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(self.enc.ch[-1], 64),
                                 nn.ReLU(), nn.Linear(64, n_sev))

    def forward(self, x):
        f = self.enc(x)
        cls = self.cls(f[-1])
        h = f[-1]
        for j, (up, dec) in enumerate(zip(self.up, self.dec)):
            i = len(f) - 2 - j
            h = dec(torch.cat([f[i], up(h)], 1))
        return {"logits": self.head(h), "sev_logits": cls}


# --------------------------------------------------------------------- RESPIRE
class SaliencyGate(nn.Module):
    """Boundary-saliency gate: Sobel energy of skip features, channel and spatial
    attention, both modulated by the motion code."""

    def __init__(self, ch, zdim):
        super().__init__()
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
        self.register_buffer("sx", sx[None, None])
        self.register_buffer("sy", sx.t().contiguous()[None, None])
        self.ch_mlp = nn.Sequential(nn.Linear(ch + zdim, max(ch // 4, 8)), nn.ReLU(), nn.Linear(max(ch // 4, 8), ch))
        self.sp = nn.Conv2d(3 + 1, 1, 7, padding=3)
        self.zsp = nn.Linear(zdim, 1)

    def forward(self, s, g, z):
        c = s.shape[1]
        gx = F.conv2d(s, self.sx.expand(c, 1, 3, 3), padding=1, groups=c)
        gy = F.conv2d(s, self.sy.expand(c, 1, 3, 3), padding=1, groups=c)
        edge = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6).mean(1, keepdim=True)
        a_c = torch.sigmoid(self.ch_mlp(torch.cat([s.mean((2, 3)), z], 1)))[:, :, None, None]
        sc = s * a_c
        desc = torch.cat([sc.mean(1, keepdim=True), sc.amax(1, keepdim=True), edge, g.mean(1, keepdim=True)], 1)
        a_s = torch.sigmoid(self.sp(desc) + self.zsp(z)[:, :, None, None])
        return sc * a_s + s  # residual keeps the identity path open


class RespireSeg(nn.Module):
    """Motion-state-disentangled segmentation network.

    flags: use_motion (motion head + FiLM), use_gate (boundary-saliency gates),
           use_sdf (signed-distance head; the topology prior acts on it).
    """

    def __init__(self, c=16, depth=4, zdim=32, use_motion=True, use_gate=True, use_sdf=True):
        super().__init__()
        self.enc = Encoder(c, depth)
        ch = self.enc.ch
        self.use_motion, self.use_gate, self.use_sdf = use_motion, use_gate, use_sdf
        self.zdim = zdim
        self.mot = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch[-1], 64), nn.ReLU(),
                                 nn.Linear(64, zdim))
        self.mot_out = nn.Linear(zdim, 3)  # log1p of [severity, through-plane, in-plane] (mm)
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([conv_block(2 * ch[i], ch[i]) for i in reversed(range(depth))])
        self.film = nn.ModuleList([nn.Linear(zdim, 2 * ch[i]) for i in reversed(range(depth))])
        for f in self.film:
            nn.init.zeros_(f.weight); nn.init.zeros_(f.bias)
        self.gates = nn.ModuleList([SaliencyGate(ch[i], zdim) for i in reversed(range(depth))])
        self.head = nn.Conv2d(ch[0], N_CLASSES, 1)
        self.sdf = nn.Conv2d(ch[0], 3, 1)  # LV endocardium, LV epicardium, RV

    def forward(self, x):
        f = self.enc(x)
        z = self.mot(f[-1])
        if not self.use_motion:
            z = torch.zeros_like(z)
        h = f[-1]
        for j in range(len(self.up)):
            i = len(f) - 2 - j
            hu = self.up[j](h)
            s = self.gates[j](f[i], hu, z) if self.use_gate else f[i]
            h = self.dec[j](torch.cat([s, hu], 1))
            if self.use_motion:
                gb = self.film[j](z)
                gam, bet = gb.chunk(2, 1)
                h = h * (1 + gam[:, :, None, None]) + bet[:, :, None, None]
        out = {"logits": self.head(h), "mot": self.mot_out(z)}
        if self.use_sdf:
            out["sdf"] = self.sdf(h)  # linear output regressing tanh-encoded distances (no saturating activation)
        return out


C = 8  # base width used for every network in the study


def build(name):
    return {
        "unet": lambda: UNet(C),
        "nnunet2d": lambda: UNet(12, 4, norm="in", act="lrelu", deep_sup=True),
        "attunet": lambda: AttentionUNet(C),
        "sharpunet": lambda: SharpUNet(C),
        "transunet": lambda: TransUNetLite(C),
        "multitask": lambda: MultiTaskUNet(C),
        "respire": lambda: RespireSeg(C),
        "respire_noA": lambda: RespireSeg(C, use_motion=False),
        "respire_noG": lambda: RespireSeg(C, use_gate=False),
        "respire_noB": lambda: RespireSeg(C, use_sdf=False),
        "respire_base": lambda: RespireSeg(C, use_motion=False, use_gate=False, use_sdf=False),
        "respire_A": lambda: RespireSeg(C, use_motion=True, use_gate=False, use_sdf=False),
        "respire_B": lambda: RespireSeg(C, use_motion=False, use_gate=False, use_sdf=True),
        "respire_AB": lambda: RespireSeg(C, use_motion=True, use_gate=False, use_sdf=True),
    }[name]()
