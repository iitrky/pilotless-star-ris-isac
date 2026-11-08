"""HA-TD3 and belief-state baselines in NumPy (paper Sec. 6, Algorithm 2).

Encoders
  hier : type MLPs -> zone-masked self-attention (level 1) -> per-zone attention pooling ->
         level-2 attention over [zone r, zone t, global] -> context        (HA-TD3)
  flat : one self-attention layer over all entities, mean pooling            (flat-attention TD3)
  mlp  : all tokens flattened into one MLP (fixed entity count)              (MLP-TD3)
Hybrid action
  per vehicle : discrete residual codebook entry j_k (0 = geometric anchor, j >= 1 = image source),
                chosen greedily from the per-vehicle twin Q-values (clipped double Q-learning)
  slow head   : continuous energy-split correction (deterministic actor, TD3 update), every T_h epochs
Critic: same encoder with the action appended to the tokens; Q = sum_k q_k(h_k, c) + q_g(c), where each
q_k is trained on vehicle k's own rate and q_g on the sensing (Lagrangian + shaping) reward.
"""
import numpy as np
from .site_control import N_BEAM
from .npnn import T, Param, Module, Linear, MLP, Adam, concat

NFREQ = (1.5, 3.0, 6.0, 12.0, 24.0, 48.0)                      # position encoding periods [m]


def featurize(env, lam=(0.0, 0.0), beta_held=0.5, phase=0.0):
    """Belief tokens (no channel coefficients) for every tracked entity + a global token."""
    cfg = env.cfg
    ob = env._belief_tokens()
    tok = np.concatenate([ob["veh"], ob["tgt"]])
    x = env.trk.x[:, 0]
    pe = np.concatenate([np.stack([np.sin(2 * np.pi * x / P), np.cos(2 * np.pi * x / P)], 1) for P in NFREQ], 1)
    p = np.concatenate([env.trk.x[:, :2], env.p_ent[:, 2:3]], 1)
    u = env.geo.unit(p - env.geo.p_ris)
    # measured echo attenuation of the last epoch relative to the free-space radar equation (dB),
    # i.e. |alpha_hat_e| / |alpha_e^fs(range)|, with 1 dB measurement error; no channel coefficients
    if not hasattr(env, "feat_rng"):
        env.feat_rng = np.random.default_rng(int(abs(env.p_ent[0, 0]) * 1e6) % (2 ** 32))
    vis = getattr(env, "vis_e", np.ones(len(x)))
    echo_db = 20 * np.log10(vis + 1e-3) + env.feat_rng.normal(0, 1.0, len(x))
    tok = np.concatenate([tok, pe, u[:, [0, 2]], np.sign(env.trk.x[:, 1:2]), (echo_db / 20)[:, None]], 1)
    tok[:, 13] = np.r_[np.full(cfg.K, lam[0]), np.full(cfg.L, lam[1])]          # lambda slot
    glob = np.array([cfg.K_r / cfg.K, cfg.K_t / cfg.K, lam[0], lam[1], beta_held, phase])
    zone = np.concatenate([env.zone_v, env.zone_t])
    return tok.astype(np.float64), glob, zone


class Encoder(Module):
    def __init__(self, kind, F, G, E, K, rng, d=32, act_dim=0, level2=True):
        self.kind, self.K, self.E, self.d, self.level2 = kind, K, E, d, level2
        Fi = F + act_dim
        if kind == "mlp":
            self.net = MLP([E * Fi + G + (1 if act_dim else 0), 128, 128, d], rng)
            return
        self.emb_v = MLP([Fi, 64, d], rng)
        self.emb_t = MLP([Fi, 64, d], rng)
        self.q, self.k, self.v, self.o = (Linear(d, d, rng, bias=False) for _ in range(4))
        self.ff = MLP([d, 64, d], rng, out_scale=0.5)
        self.glob = MLP([G + (1 if act_dim else 0), 32, d], rng)
        if kind == "hier":
            self.pool_q = Param(rng.standard_normal((d, 1)) * 0.1)
            self.q2, self.k2, self.v2, self.o2 = (Linear(d, d, rng, bias=False) for _ in range(4))

    def _attn(self, h, mask, q, k, v, o):
        s = (q(h) @ k(h).T()) * (1.0 / np.sqrt(self.d))
        return h + o(s.softmax(mask) @ v(h))

    def __call__(self, tok, glob, zone, isveh):
        """tok (B,E,Fi) T; glob (B,G') T; zone (B,E) int; isveh (E,) bool. Returns h_veh (B,K,d), c (B,1,d)."""
        B = tok.v.shape[0]
        if self.kind == "mlp":
            x = concat([T(tok.v.reshape(B, -1)) if not tok.req else _flatten(tok), glob], -1)
            c = self.net(x).relu()
            c3 = _expand1(c)
            return None, c3
        m = isveh.astype(float)[None, :, None]
        h = self.emb_v(tok) * T(m) + self.emb_t(tok) * T(1 - m)
        if self.kind == "hier":
            mask = zone[:, :, None] == zone[:, None, :]
        else:
            mask = None
        h = self._attn(h, mask, self.q, self.k, self.v, self.o)
        h = h + self.ff(h)
        g = _expand1(self.glob(glob))
        if self.kind == "flat":
            c = h.mean(1, keepdims=True) + g
        else:
            s = (h @ self.pool_q).T()                                  # (B,1,E)
            zs = []
            for z in (0, 1):
                zm = (zone == z)[:, None, :]
                zs.append(s.softmax(zm) @ h)                           # (B,1,d)
            if self.level2:
                L2 = concat([zs[0], zs[1], g], 1)                      # (B,3,d)
                L2 = self._attn(L2, None, self.q2, self.k2, self.v2, self.o2)
                c = L2.mean(1, keepdims=True)
            else:
                c = (zs[0] + zs[1] + g) * (1.0 / 3)
        return h[:, : self.K, :], c


def _expand1(x):
    """(B,d) -> (B,1,d) with gradient."""
    out = T(x.v[:, None, :], (x,))
    out.back = lambda g: (g[:, 0, :],)
    return out


def _flatten(x):
    B = x.v.shape[0]
    out = T(x.v.reshape(B, -1), (x,))
    out.back = lambda g: (g.reshape(x.v.shape),)
    return out


class Actor(Module):
    """Deterministic actor for the continuous (slow) action."""

    def __init__(self, kind, F, G, E, K, rng, level2=True):
        self.kind, self.K = kind, K
        self.enc = Encoder(kind, F, G, E, K, rng, level2=level2)
        self.slow = MLP([self.enc.d, 64, 1], rng, out_scale=0.1)

    def __call__(self, tok, glob, zone, isveh):
        _, c = self.enc(tok, glob, zone, isveh)
        return self.slow(c[:, 0, :]).tanh()


def _reshape(x, shape):
    out = T(x.v.reshape(shape), (x,))
    out.back = lambda g: (g.reshape(x.v.shape),)
    return out


class Critic(Module):
    """q_k(s, j, b) for every vehicle k and codebook entry j, plus the global value q_g(s, b)."""

    def __init__(self, kind, F, G, E, K, rng, level2=True):
        self.kind, self.K = kind, K
        self.enc = Encoder(kind, F, G + 1, E, K, rng, level2=level2)
        d = self.enc.d
        if kind == "mlp":
            self.head = MLP([d, 128, K * N_BEAM + 1], rng)
        else:
            self.hv = Linear(d, 64, rng); self.hc = Linear(d, 64, rng, bias=False)
            self.qv = Linear(64, N_BEAM, rng)
            self.qg = MLP([d, 64, 1], rng)

    def __call__(self, tok, glob, zone, isveh, b):
        hv, c = self.enc(tok, concat([glob, b], -1), zone, isveh)
        if self.kind == "mlp":
            o = self.head(c[:, 0, :])
            B = o.v.shape[0]
            return _reshape(o[:, : self.K * N_BEAM], (B, self.K, N_BEAM)), o[:, self.K * N_BEAM]
        return self.qv((self.hv(hv) + self.hc(c)).relu()), self.qg(c[:, 0, :])[:, 0]


def _pick(q, j):
    """Select q[b, k, j[b, k]] with gradient."""
    oh = np.eye(q.v.shape[-1])[j]
    return (q * T(oh)).sum(-1)


class TD3:
    def __init__(self, kind, F, G, E, K, seed=0, gamma=0.5, tau=0.01, lr_a=3e-4, lr_c=1e-3, delay=2,
                 tnoise=0.2, tclip=0.5, twin=True, level2=True):
        rng = np.random.default_rng(seed)
        self.K, self.E, self.gamma, self.tau, self.delay = K, E, gamma, tau, delay
        self.tnoise, self.tclip, self.twin = tnoise, tclip, twin
        self.rng = rng
        self.actor = Actor(kind, F, G, E, K, rng, level2)
        self.actor_t = Actor(kind, F, G, E, K, rng, level2); self.actor_t.load(self.actor.state())
        n_c = 2 if twin else 1
        self.critics = [Critic(kind, F, G, E, K, rng, level2) for _ in range(n_c)]
        self.critics_t = [Critic(kind, F, G, E, K, rng, level2) for _ in range(n_c)]
        for c, ct in zip(self.critics, self.critics_t):
            ct.load(c.state())
        self.opt_a = Adam(self.actor.params(), lr_a)
        self.opt_c = Adam(sum([c.params() for c in self.critics], []), lr_c)
        self.isveh = np.r_[np.ones(K, bool), np.zeros(E - K, bool)]
        self.n = 0

    def act(self, tok, glob, zone, eps=0.0):
        b = self.actor(T(tok[None]), T(glob[None]), zone[None], self.isveh)
        qv, _ = self.critics[0](T(tok[None]), T(glob[None]), zone[None], self.isveh, b)
        j = qv.v[0].argmax(-1)
        if eps > 0:
            rnd = self.rng.random(len(j)) < eps
            j[rnd] = self.rng.integers(0, N_BEAM, rnd.sum())
        return j, float(b.v[0, 0])

    def update(self, batch):
        s, g, z, j, b, slow, rv, rg, s2, g2, z2, done = batch
        j = j.astype(int)
        # ---- target: smoothed slow action, double-Q selection of the next codebook entries
        b2 = self.actor_t(T(s2), T(g2), z2, self.isveh)
        nb = np.clip(self.rng.normal(0, self.tnoise, b2.v.shape), -self.tclip, self.tclip) if self.twin else 0
        bb2 = T(np.clip(b2.v + nb, -1, 1))
        j2 = self.critics[0](T(s2), T(g2), z2, self.isveh, bb2)[0].v.argmax(-1)
        qs = [ct(T(s2), T(g2), z2, self.isveh, bb2) for ct in self.critics_t]
        nd = 1 - done
        yv = rv + self.gamma * nd[:, None] * np.min([_pick(q[0], j2).v for q in qs], 0)
        yg = rg + self.gamma * nd * np.min([q[1].v for q in qs], 0)
        loss = None
        for c in self.critics:
            qv, qg = c(T(s), T(g), z, self.isveh, T(b))
            qa = _pick(qv, j)
            l = ((qa - yv) * (qa - yv)).mean() + ((qg - yg) * (qg - yg)).mean()
            loss = l if loss is None else loss + l
        loss.backward(); self.opt_c.step()
        closs = float(loss.v)
        self.n += 1
        if self.n % self.delay == 0:
            ba = self.actor(T(s), T(g), z, self.isveh)
            f = slow[:, None]
            bu = ba * T(f) + T(b * (1 - f))                       # slow head trained only at decision epochs
            qv, qg = self.critics[0](T(s), T(g), z, self.isveh, bu)
            jg = qv.v.argmax(-1)
            ((_pick(qv, jg).sum(1) + qg).mean() * (-1.0)).backward()
            self.opt_a.step(); self.opt_c.zero()
            for net, tgt in [(self.actor, self.actor_t)] + list(zip(self.critics, self.critics_t)):
                for p, pt in zip(net.params(), tgt.params()):
                    pt.v = (1 - self.tau) * pt.v + self.tau * p.v
        return closs


class Replay:
    def __init__(self, cap, rng):
        self.cap, self.rng, self.data, self.i = cap, rng, [], 0

    def add(self, tr):
        if len(self.data) < self.cap:
            self.data.append(tr)
        else:
            self.data[self.i] = tr
        self.i = (self.i + 1) % self.cap

    def sample(self, B):
        idx = self.rng.integers(0, len(self.data), B)
        cols = list(zip(*[self.data[i] for i in idx]))
        return [np.array(c) for c in cols]
