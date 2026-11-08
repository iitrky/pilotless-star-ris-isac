"""Non-learning baselines.

* heuristic_beamformer: regularised ZF for vehicles + matched beams toward each target's cascade.
* random_phase_policy:  random STAR-RIS phases, beta = 0.5, heuristic beamformer.
* pga_benchmark:        penalty-based projected gradient ascent on the per-slot Lagrangian of P1
                        (PyTorch autograd, many restarts). Reaches a stationary point of each
                        snapshot; 'genie' uses true current CSI, 'delayed' uses the agent's CSI.
"""
import numpy as np
from .channels import effective_channels


def heuristic_beamformer(env, th_r, th_t, kappa, g=None, hd=None, G=None):
    cfg = env.cfg
    g = env.g_hat if g is None else g
    hd = env.hd_hat if hd is None else hd
    G = env.G_hat if G is None else G
    theta_users = np.where(env.zone_v[:, None] == 0, th_r[None, :], th_t[None, :])
    H = effective_channels(hd, g, G, theta_users)                       # (K, M)
    reg = cfg.K * env.sigma2 / env.P
    Wc = H.conj().T @ np.linalg.inv(H @ H.conj().T + reg * np.eye(cfg.K))
    phi, el = env.geo.angles_from_ris(env.p_tgt)
    Ws = np.zeros((cfg.M, cfg.L), complex)
    for l in range(cfg.L):
        th = th_r if env.zone_t[l] == 0 else th_t
        a, _ = env.geo.a_ris_angle(env.phi_hat[l], el[l])
        Ws[:, l] = ((th * a) @ G).conj()
    Wc *= np.sqrt(kappa * env.P) / (np.linalg.norm(Wc) + 1e-12)
    Ws *= np.sqrt((1 - kappa) * env.P) / (np.linalg.norm(Ws) + 1e-12)
    return np.concatenate([Wc, Ws], axis=1)


def random_phase_policy(env, rng):
    N = env.cfg.N
    th_r, th_t = rng.uniform(0, 2 * np.pi, N), rng.uniform(0, 2 * np.pi, N)
    beta = np.full(N, 0.5)
    c_r, c_t = env.coefficients(beta, th_r, th_t)
    W = heuristic_beamformer(env, c_r, c_t, 0.7)
    return W, beta, th_r, th_t


def pga_benchmark(env, lam=10.0, iters=400, restarts=4, lr=0.05, genie=True, seed=0):
    """Maximise sum_rate - lam * sum_l relu(CRB_l / Gamma - 1) over (W, beta, theta) for the
    current snapshot. Requires PyTorch."""
    import torch
    cfg = env.cfg
    torch.manual_seed(seed)
    cd = torch.complex128
    G = torch.as_tensor(env.G if genie else env.G_hat, dtype=cd)
    g = torch.as_tensor(env.g if genie else env.g_hat, dtype=cd)
    hd = torch.as_tensor(env.hd if genie else env.hd_hat, dtype=cd)
    zv = torch.as_tensor(env.zone_v)
    phi, el = env.geo.angles_from_ris(env.p_tgt)
    if not genie:
        phi = env.phi_hat
    A, Ad, Bs, Bsd = [], [], [], []
    for l in range(cfg.L):
        a, ad = env.geo.a_ris_angle(phi[l], el[l])
        b, bd = env._sens(phi[l], el[l])
        A.append(torch.as_tensor(a, dtype=cd)); Ad.append(torch.as_tensor(ad, dtype=cd))
        Bs.append(None if b is None else torch.as_tensor(b, dtype=cd))
        Bsd.append(None if bd is None else torch.as_tensor(bd, dtype=cd))
    alpha = torch.as_tensor(env.alpha, dtype=cd)
    s2, T, P, Gam = env.sigma2, cfg.T_snap, env.P, cfg.crb_thresh
    best, best_val = None, -np.inf
    for rs in range(restarts):
        Wr = torch.randn(cfg.M, cfg.K + cfg.L, 2, dtype=torch.float64, requires_grad=True)
        tr = (2 * np.pi * torch.rand(cfg.N, dtype=torch.float64)).requires_grad_()
        tt = (2 * np.pi * torch.rand(cfg.N, dtype=torch.float64)).requires_grad_()
        bl = torch.zeros(cfg.N, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([Wr, tr, tt, bl], lr=lr)
        for it in range(iters):
            W = torch.complex(Wr[..., 0], Wr[..., 1])
            W = W * np.sqrt(P) / torch.linalg.norm(W)
            beta = torch.sigmoid(bl)
            cr = torch.sqrt(beta) * torch.exp(1j * tr)
            ct = torch.sqrt(1 - beta) * torch.exp(1j * tt)
            th_u = torch.where(zv[:, None] == 0, cr[None], ct[None])
            H = hd + (g * th_u) @ G
            Pw = torch.abs(H @ W) ** 2
            sig = torch.diagonal(Pw[:, : cfg.K])
            sinr = sig / (Pw.sum(1) - sig + s2)
            rate = torch.log2(1 + sinr).sum()
            R = W @ W.conj().T
            pen = 0.0
            crbs = []
            for l in range(cfg.L):
                th = cr if env.zone_t[l] == 0 else ct
                v = (th * A[l]) @ G
                vd = (th * Ad[l]) @ G
                if Bs[l] is None:
                    B = torch.outer(v, v); Bd = torch.outer(vd, v) + torch.outer(v, vd)
                else:
                    B = torch.outer(Bs[l], v); Bd = torch.outer(Bsd[l], v) + torch.outer(Bs[l], vd)
                tbb = torch.real(torch.trace(B.conj().T @ B @ R))
                tdd = torch.real(torch.trace(Bd.conj().T @ Bd @ R))
                tdb = torch.trace(Bd.conj().T @ B @ R)
                crb = s2 / (2 * T * torch.abs(alpha[l]) ** 2) * tbb / (tdd * tbb - torch.abs(tdb) ** 2 + 1e-300)
                crbs.append(crb)
                pen = pen + torch.relu(torch.log(crb / Gam))
            obj = rate - lam * pen
            opt.zero_grad()
            (-obj).backward()
            opt.step()
        with torch.no_grad():
            val = obj.item()
            if val > best_val:
                best_val = val
                best = (W.detach().numpy(), beta.detach().numpy(), tr.detach().numpy() % (2 * np.pi),
                        tt.detach().numpy() % (2 * np.pi))
    return best
