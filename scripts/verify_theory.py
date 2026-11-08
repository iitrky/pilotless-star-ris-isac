"""Numerical checks that need only NumPy:
  1. closed-form CRB (crb_angle) == inverse 3x3 Fisher information (finite-difference dB/dphi)
  2. Proposition 2: CRB(beta) * beta^nu is constant (nu = 1 sensors at the surface, nu = 2 BS echo)
  3. Link budget: echo SNR of the two sensing architectures at the default geometry
Writes results/theory_checks.json and results/fig_crb_vs_beta.png
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hatd3.config import SysConfig
from hatd3.env import StarRisIsacEnv
from hatd3.channels import crb_angle, fim_crb_angle, sensor_resp

os.makedirs("results", exist_ok=True)
out = {}
rng = np.random.default_rng(7)

# ---------- 1. closed form vs FIM ----------
errs = []
for trial in range(200):
    mode = "ris_sensors" if trial % 2 == 0 else "bs"
    cfg = SysConfig(sensing_mode=mode)
    env = StarRisIsacEnv(cfg, seed=trial)
    env.reset(eps=0.0)
    M, N = cfg.M, cfg.N
    th = np.sqrt(rng.uniform(0.1, 1, N)) * np.exp(1j * rng.uniform(0, 2 * np.pi, N))
    A = rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))
    R = A @ A.conj().T
    R *= env.P / np.real(np.trace(R))
    phi, el = env.geo.angles_from_ris(env.p_tgt)
    a, ad = env.geo.a_ris_angle(phi[0], el[0])
    b, bd = (None, None) if mode == "bs" else sensor_resp(env.geo, phi[0], el[0])
    c1 = crb_angle(env.G, th, a, ad, R, env.alpha[0], env.sigma2, cfg.T_snap, b, bd)
    afun = lambda p: env.geo.a_ris_angle(p, el[0])
    bfun = None if mode == "bs" else (lambda p: sensor_resp(env.geo, p, el[0]))
    c2 = fim_crb_angle(env.G, th, afun, phi[0], R, env.alpha[0], env.sigma2, cfg.T_snap, bfun, h=1e-7)
    errs.append(abs(c1 - c2) / c2)
out["crb_closed_form_vs_fim"] = {"trials": len(errs), "max_rel_err": float(np.max(errs)),
                                 "median_rel_err": float(np.median(errs))}

# ---------- 2. beta scaling ----------
betas = np.logspace(-2, 0, 25)
curves = {}
for mode, nu in (("ris_sensors", 1), ("bs", 2)):
    cfg = SysConfig(sensing_mode=mode)
    env = StarRisIsacEnv(cfg, seed=11)
    env.reset(eps=0.0)
    N, M, K, L = cfg.N, cfg.M, cfg.K, cfg.L
    ph_r, ph_t = rng.uniform(0, 2 * np.pi, N), rng.uniform(0, 2 * np.pi, N)
    W = rng.standard_normal((M, K + L)) + 1j * rng.standard_normal((M, K + L))
    W *= np.sqrt(env.P) / np.linalg.norm(W)
    crb_t, snr_t = [], []
    for bt in betas:                                  # beta^t = bt, beta^r = 1 - bt
        th_r = np.sqrt(1 - bt) * np.exp(1j * ph_r)
        th_t = np.sqrt(bt) * np.exp(1j * ph_t)
        sinr, _, crb, _ = env.evaluate(W, th_r, th_t)
        crb_t.append(crb[cfg.L_r])                    # transmission-zone target
        snr_t.append(sinr[cfg.K_r:].mean())           # transmission-zone vehicles
    crb_t, snr_t = np.array(crb_t), np.array(snr_t)
    slope = np.polyfit(np.log10(betas), np.log10(crb_t), 1)[0]
    inv = crb_t * betas ** nu
    curves[mode] = (crb_t / crb_t[-1])
    out[f"beta_scaling_{mode}"] = {"nu": nu, "fitted_loglog_slope": float(slope),
                                   "crb_times_beta^nu_rel_spread": float(np.ptp(inv) / np.mean(inv))}

# ---------- 3. link budget ----------
lb = {}
for mode in ("ris_sensors", "bs"):
    cfg = SysConfig(sensing_mode=mode)
    vals = []
    for s in range(50):
        env = StarRisIsacEnv(cfg, seed=100 + s)
        env.reset(eps=0.0)
        M, K, L = cfg.M, cfg.K, cfg.L
        _, _, vh = np.linalg.svd(env.G)
        w = vh[0].conj()
        phi, el = env.geo.angles_from_ris(env.p_tgt)
        a, _ = env.geo.a_ris_angle(phi[0], el[0])
        th = np.exp(-1j * np.angle(a * (env.G @ w)))   # focus surface on target 0
        W = np.zeros((M, K + L), complex)
        W[:, K] = w * np.sqrt(env.P)
        _, _, crb, es = env.evaluate(W, th, th)
        vals.append((10 * np.log10(es[0]), np.degrees(np.sqrt(crb[0]))))
    v = np.array(vals)
    lb[mode] = {"echo_snr_dB_median": float(np.median(v[:, 0])), "sqrt_crb_deg_median": float(np.median(v[:, 1]))}
out["link_budget_focused_beam"] = lb

json.dump(out, open("results/theory_checks.json", "w"), indent=2)
print(json.dumps(out, indent=2))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.loglog(betas, curves["ris_sensors"], "o-", ms=3, label=r"sensors at surface ($\nu$=1)")
    ax.loglog(betas, curves["bs"], "s-", ms=3, label=r"BS echo via surface ($\nu$=2)")
    ax.loglog(betas, betas ** -1.0, "k--", lw=0.8, label=r"$\beta^{-1}$")
    ax.loglog(betas, betas ** -2.0, "k:", lw=0.8, label=r"$\beta^{-2}$")
    ax.set_xlabel(r"transmission-zone energy split $\beta^t$")
    ax.set_ylabel(r"CRB / CRB($\beta^t$=1)")
    ax.grid(True, which="both", lw=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("results/fig_crb_vs_beta.png", dpi=200)
except Exception as e:
    print("plot skipped:", e)
