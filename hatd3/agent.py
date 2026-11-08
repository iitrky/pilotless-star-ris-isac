"""HA-TD3: hierarchical (zone-structured) attention TD3 with a two-timescale actor.

Action vector a in [-1, 1]^D, laid out as
  [ W (2M(K+L)) | p,q for theta_r (2N) | p,q for theta_t (2N) | beta logits (N) | kappa (1) ]
The last N+1 entries form the slow action a^H.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .action import ActionLayout


def mlp(i, h, o, act_out=None):
    layers = [nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o)]
    if act_out is not None:
        layers.append(act_out)
    return nn.Sequential(*layers)


class ZoneEncoder(nn.Module):
    """Level 1: self-attention restricted to tokens of the same zone, then per-zone attention
    pooling. Level 2: a transformer layer over {z_r, z_t, z_G}."""

    def __init__(self, fv, ft, fg, d, heads, layers, hidden, use_level2=True, extra_glob=0, zone_mask=True):
        super().__init__()
        self.zone_mask = zone_mask
        self.veh = mlp(fv, hidden, d)
        self.tgt = mlp(ft, hidden, d)
        self.glob = mlp(fg + extra_glob, hidden, d)
        self.type_emb = nn.Parameter(torch.zeros(2, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.l1 = nn.TransformerEncoder(layer, layers)
        self.pool_q = nn.Parameter(torch.randn(1, 1, d) / np.sqrt(d))
        self.pool = nn.MultiheadAttention(d, heads, batch_first=True)
        self.use_level2 = use_level2
        if use_level2:
            self.l2 = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        else:
            self.l2_lin = nn.Linear(3 * d, d)
        self.d = d

    def forward(self, veh, tgt, glob, zones, entity_bias=None):
        # veh (B,K,fv) tgt (B,L,ft) glob (B,fg') zones (E,) long tensor, 0 = r, 1 = t
        ev = self.veh(veh) + self.type_emb[0]
        et = self.tgt(tgt) + self.type_emb[1]
        e = torch.cat([ev, et], dim=1)                       # (B,E,d)
        if entity_bias is not None:
            e = e + entity_bias
        B = e.shape[0]
        q = self.pool_q.expand(B, 1, self.d)
        z = []
        if self.zone_mask:
            same = zones[:, None] == zones[None, :]
            u = self.l1(e, mask=~same)                       # True = blocked
            for zid in (0, 1):
                kpm = (zones != zid)[None, :].expand(B, -1)
                zz, _ = self.pool(q, u, u, key_padding_mask=kpm)
                z.append(zz[:, 0])
        else:                                                # flat self-attention (no zone structure)
            u = self.l1(e)
            zz, _ = self.pool(q, u, u)
            z = [zz[:, 0], zz[:, 0]]
        zg = self.glob(glob)
        if self.use_level2:
            tok = self.l2(torch.stack([z[0], z[1], zg], dim=1))
            c = tok.mean(dim=1)
            z_r, z_t = tok[:, 0], tok[:, 1]
        else:
            c = self.l2_lin(torch.cat([z[0], z[1], zg], dim=1))
            z_r, z_t = z
        return u, z_r, z_t, c


class FlatMLPEncoder(nn.Module):
    """MLP-TD3 baseline: all tokens concatenated into one vector (order-dependent)."""

    def __init__(self, fv, ft, fg, d, hidden, K, L, extra_glob=0):
        super().__init__()
        self.net = mlp(K * fv + L * ft + fg + extra_glob, 2 * hidden, d)
        self.d, self.E = d, K + L

    def forward(self, veh, tgt, glob, zones, entity_bias=None):
        x = torch.cat([veh.flatten(1), tgt.flatten(1), glob], dim=1)
        c = self.net(x)
        if entity_bias is not None:
            c = c + entity_bias.mean(1)
        u = c[:, None].expand(-1, self.E, -1)
        return u, c, c, c


def make_encoder(scfg, acfg, fv, ft, fg, extra_glob):
    if acfg.encoder == "mlp":
        return FlatMLPEncoder(fv, ft, fg, acfg.d, acfg.hidden, scfg.K, scfg.L, extra_glob)
    return ZoneEncoder(fv, ft, fg, acfg.d, acfg.heads, acfg.layers_l1, acfg.hidden,
                       acfg.use_level2 and acfg.encoder == "hier", extra_glob,
                       zone_mask=(acfg.encoder == "hier"))


class Actor(nn.Module):
    def __init__(self, scfg, acfg, fv, ft, fg, extra_glob):
        super().__init__()
        self.lay = ActionLayout(scfg)
        d, h = acfg.d, acfg.hidden
        self.enc = make_encoder(scfg, acfg, fv, ft, fg, extra_glob)
        self.flat = acfg.encoder == "mlp"
        self.E = scfg.K + scfg.L
        if self.flat:
            self.beam = mlp(d, h, self.E * 2 * scfg.M, nn.Tanh())
        else:
            self.beam = mlp(2 * d, h, 2 * scfg.M, nn.Tanh())  # shared across entities (equivariant)
        self.ph_r = mlp(2 * d, h, 2 * scfg.N, nn.Tanh())
        self.ph_t = mlp(2 * d, h, 2 * scfg.N, nn.Tanh())
        self.slow = mlp(d, h, scfg.N + 1, nn.Tanh())

    def forward(self, veh, tgt, glob, zones):
        u, z_r, z_t, c = self.enc(veh, tgt, glob, zones)
        E = u.shape[1]
        if self.flat:
            w = self.beam(c)
        else:
            w = self.beam(torch.cat([u, c[:, None].expand(-1, E, -1)], dim=-1))   # (B,E,2M)
        pr = self.ph_r(torch.cat([c, z_r], dim=-1))
        pt = self.ph_t(torch.cat([c, z_t], dim=-1))
        s = self.slow(c)
        return torch.cat([w.flatten(1), pr, pt, s], dim=-1)


class Critic(nn.Module):
    def __init__(self, scfg, acfg, fv, ft, fg, extra_glob):
        super().__init__()
        self.lay = ActionLayout(scfg)
        d, h = acfg.d, acfg.hidden
        self.act_ent = nn.Linear(2 * scfg.M, d)
        self.act_rest = mlp(self.lay.D - self.lay.nW, h, d)
        self.enc = make_encoder(scfg, acfg, fv, ft, fg + d, extra_glob)
        self.q = mlp(d, h, 1)
        self.E = scfg.K + scfg.L

    def forward(self, veh, tgt, glob, zones, a):
        B = a.shape[0]
        w = a[:, self.lay.sl_W].reshape(B, self.E, -1)
        bias = self.act_ent(w)
        rest = self.act_rest(a[:, self.lay.nW:])
        _, _, _, c = self.enc(veh, tgt, torch.cat([glob, rest], dim=-1), zones, entity_bias=bias)
        return self.q(c)


class Replay:
    """Compact buffer: the BS-RIS channel token is constant within an episode, so it is stored
    once per episode and referenced by id."""

    def __init__(self, cap, fv_shape, ft_shape, extra, D):
        self.cap, self.i, self.n = cap, 0, 0
        f16 = np.float16
        self.veh = np.zeros((cap,) + fv_shape, f16)
        self.veh2 = np.zeros((cap,) + fv_shape, f16)
        self.tgt = np.zeros((cap,) + ft_shape, np.float32)
        self.tgt2 = np.zeros((cap,) + ft_shape, np.float32)
        self.ex = np.zeros((cap, extra), np.float32)
        self.ex2 = np.zeros((cap, extra), np.float32)
        self.a = np.zeros((cap, D), np.float32)
        self.r = np.zeros((cap, 1), np.float32)
        self.d = np.zeros((cap, 1), np.float32)
        self.slow_dec = np.zeros((cap, 1), np.float32)
        self.gid = np.zeros(cap, np.int64)
        self.glob = {}

    def add_glob(self, gid, g):
        self.glob[gid] = g

    def add(self, s, a, r, s2, done, gid, slow_dec):
        i = self.i
        self.veh[i], self.tgt[i], self.ex[i] = s
        self.veh2[i], self.tgt2[i], self.ex2[i] = s2
        self.a[i], self.r[i], self.d[i], self.gid[i], self.slow_dec[i] = a, r, done, gid, slow_dec
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)
        if len(self.glob) > 4 * (self.cap // 200 + 1):             # drop unreferenced episodes
            live = set(np.unique(self.gid[: self.n]).tolist())
            self.glob = {k: v for k, v in self.glob.items() if k in live}

    def sample(self, b, dev):
        idx = np.random.randint(0, self.n, b)
        g = np.stack([self.glob[k] for k in self.gid[idx]])
        T = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=dev)
        return (T(self.veh[idx]), T(self.tgt[idx]), T(np.concatenate([g, self.ex[idx]], 1)),
                T(self.a[idx]), T(self.r[idx]),
                T(self.veh2[idx]), T(self.tgt2[idx]), T(np.concatenate([g, self.ex2[idx]], 1)),
                T(self.d[idx]), T(self.slow_dec[idx]))


class HATD3:
    def __init__(self, scfg, acfg, fv, ft, fg, device="cpu"):
        self.s, self.c, self.dev = scfg, acfg, device
        self.lay = ActionLayout(scfg)
        self.extra = self.lay.D + self.lay.D_slow       # previous action + held slow action
        self.actor = Actor(scfg, acfg, fv, ft, fg, self.extra).to(device)
        self.q1 = Critic(scfg, acfg, fv, ft, fg, self.extra).to(device)
        self.q2 = Critic(scfg, acfg, fv, ft, fg, self.extra).to(device)
        self.actor_t, self.q1_t, self.q2_t = (copy.deepcopy(m) for m in (self.actor, self.q1, self.q2))
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=acfg.lr_actor)
        self.opt_c = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=acfg.lr_critic)
        zones = np.concatenate([np.array([0] * scfg.K_r + [1] * scfg.K_t), np.array([0] * scfg.L_r + [1] * scfg.L_t)])
        self.zones = torch.as_tensor(zones, device=device)
        self.it = 0

    @torch.no_grad()
    def act(self, veh, tgt, glob_full, noise=0.0):
        T = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.dev)[None]
        a = self.actor(T(veh), T(tgt), T(glob_full), self.zones)[0].cpu().numpy()
        if noise > 0:
            a = a + noise * np.random.randn(*a.shape)
        return np.clip(a, -1, 1)

    def update(self, buf):
        c = self.c
        self.it += 1
        v, t, g, a, r, v2, t2, g2, d, slow_dec = buf.sample(c.batch, self.dev)
        sl = self.lay.sl_slow
        with torch.no_grad():
            a2 = self.actor_t(v2, t2, g2, self.zones)
            # slow part is held between decision slots: take it from the held-slow slot of s'
            if c.two_timescale:
                held = g2[:, -self.lay.D_slow:]
                a2[:, sl] = held                     # approximation: decision boundary handled by flag
            noise = (torch.randn_like(a2) * c.target_noise).clamp(-c.noise_clip, c.noise_clip)
            a2 = (a2 + noise).clamp(-1, 1)
            y = r + c.gamma * (1 - d) * torch.min(self.q1_t(v2, t2, g2, self.zones, a2),
                                                  self.q2_t(v2, t2, g2, self.zones, a2))
        lc = F.mse_loss(self.q1(v, t, g, self.zones, a), y) + F.mse_loss(self.q2(v, t, g, self.zones, a), y)
        self.opt_c.zero_grad()
        lc.backward()
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 10.0)
        self.opt_c.step()
        out = {"critic_loss": lc.item()}
        if self.it % c.policy_delay == 0:
            ap = self.actor(v, t, g, self.zones)
            if c.two_timescale:
                # slow-head gradient only on decision slots; otherwise use the executed (held) value
                ap = torch.cat([ap[:, : sl.start], slow_dec * ap[:, sl] + (1 - slow_dec) * a[:, sl]], dim=1)
            la = -self.q1(v, t, g, self.zones, ap).mean()
            self.opt_a.zero_grad()
            la.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.opt_a.step()
            for m, mt in ((self.actor, self.actor_t), (self.q1, self.q1_t), (self.q2, self.q2_t)):
                for p, pt in zip(m.parameters(), mt.parameters()):
                    pt.data.mul_(1 - c.tau).add_(c.tau * p.data)
            out["actor_loss"] = la.item()
        return out

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "q1": self.q1.state_dict(), "q2": self.q2.state_dict()}, path)

    def load(self, path):
        sd = torch.load(path, map_location=self.dev)
        self.actor.load_state_dict(sd["actor"])
        self.q1.load_state_dict(sd["q1"])
        self.q2.load_state_dict(sd["q2"])
        self.actor_t = copy.deepcopy(self.actor)
