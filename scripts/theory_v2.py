"""Numerical verification of the 'Sense now, serve later' theory (NumPy only).

Lemma 1   expected cascaded array gain under Gaussian error in the direction cosine u
Theorem 1 pilot-overhead crossover speed
Theorem 2 non-myopic vs myopic sensing allocation (value iteration on the tracking MDP)
Outputs: results/theory_v2.json and figures in results/
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
out = {}
rng = np.random.default_rng(1)


# ---------------------------------------------------------------- Lemma 1
def gbar(var_u, Nx):
    """E|sum_n exp(j*pi*n*du)|^2 / Nx^2 for du ~ N(0, var_u), half-wavelength spacing."""
    m = np.arange(1, Nx)
    var_u = np.atleast_1d(var_u)[:, None]
    return (1.0 / Nx) * (1 + 2 * np.sum((1 - m / Nx) * np.exp(-(np.pi * m) ** 2 * var_u / 2), axis=1))


lemma = {}
for Nx in (8, 16, 32):
    errs = []
    for s in np.logspace(-4, -1, 7):
        du = rng.normal(0, s, 200_000)
        n = np.arange(Nx)
        mc = np.mean(np.abs(np.exp(1j * np.pi * np.outer(du, n)).sum(1)) ** 2) / Nx ** 2
        errs.append(abs(mc - gbar(s ** 2, Nx)[0]) / gbar(s ** 2, Nx)[0])
    lemma[f"Nx={Nx}"] = float(np.max(errs))
out["lemma1_max_rel_err_vs_montecarlo_200k"] = lemma
# small-variance expansion 1 - pi^2 var (Nx^2-1)/12
v = 1e-5
out["lemma1_taylor_check_Nx16"] = {"exact": float(gbar(v, 16)[0]), "taylor": 1 - np.pi ** 2 * v * (16 ** 2 - 1) / 12}

fig, ax = plt.subplots(figsize=(4.2, 3.0))
s = np.logspace(-3.5, -0.5, 200)
for Nx in (8, 16, 32, 64):
    ax.semilogx(s, 10 * np.log10(gbar(s ** 2, Nx)), label=f"$N_x$={Nx}")
ax.set_xlabel(r"std of direction-cosine error $\sigma_u$")
ax.set_ylabel("expected gain loss (dB)")
ax.set_ylim(-15, 0.5); ax.grid(lw=0.3, which="both"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("results/fig_lemma1_gain_loss.png", dpi=200)

# ---------------------------------------------------------------- Theorem 1
c = 299_792_458.0


def pilots_ris(K, N, M):
    """Minimum pilot length for cascaded RIS channel estimation (Wang, Liu, Cui, TWC 2020)."""
    return K + N + max(K - 1, int(np.ceil((K - 1) * N / M)))


def tc(v, fc):
    return 0.423 * c / (v * fc)


fc, scs = 7e9, 120e3
Tsym = (1 / scs) * (1 + 1 / 14)              # normal CP, approx.
cross = {}
for (K, N, M) in [(6, 64, 16), (6, 256, 16), (6, 1024, 64), (2, 64, 16)]:
    tp = pilots_ris(K, N, M)
    v_inf = 0.423 * c / (fc * tp * Tsym)       # speed where pilots fill the whole coherence time
    cross[f"K{K}_N{N}_M{M}"] = {"pilots": tp, "pilot_time_ms": tp * Tsym * 1e3,
                                "v_pilots_fill_Tc_kmh": v_inf * 3.6}
out["theorem1_pilot_limits_7GHz_SCS120k"] = cross


def v_star(ratio, tp):
    """Crossover speed where (1 - tp*Tsym/Tc(v)) * R_csi = R_sense, ratio = R_sense / R_csi."""
    return (1 - ratio) * 0.423 * c / (fc * tp * Tsym)


tp = pilots_ris(6, 64, 16)
out["theorem1_vstar_kmh_examples_N64"] = {f"Rsense/Rcsi={r}": v_star(r, tp) * 3.6 for r in (0.5, 0.7, 0.9)}

fig, ax = plt.subplots(figsize=(4.2, 3.0))
vk = np.linspace(5, 160, 300)
for N in (64, 256):
    tpN = pilots_ris(6, N, 16)
    eff = np.clip(1 - tpN * Tsym / tc(vk / 3.6, fc), 0, None)
    ax.plot(vk, eff, label=f"pilot-based, N={N}")
for r in (0.7, 0.9):
    ax.axhline(r, ls="--", lw=0.8, color="k")
    ax.text(150, r + 0.02, f"sensing-based, $R_s/R_c$={r}", ha="right", fontsize=6.5)
ax.set_xlabel("vehicle speed (km/h)"); ax.set_ylabel("effective rate / CSI rate")
ax.grid(lw=0.3); ax.legend(fontsize=7); ax.set_ylim(0, 1.05)
fig.tight_layout(); fig.savefig("results/fig_thm1_crossover.png", dpi=200)

# ---------------------------------------------------------------- Theorem 2 (tracking MDP)
# State: prior variance P of the direction cosine; action: sensing power fraction rho.
# P_post = (1/P + rho/kappa)^-1 (kappa = CRB at full sensing power), P_next = P_post + q.
# Reward: log2(1 + (1 - rho) * SNR * gbar(P)); constraint P_post <= Gamma (tracking PCRB).
disc = 0.95
Pgrid = np.logspace(-8, 0, 500)
rhos = np.linspace(0, 1, 201)


def solve(Nx, snr_db, kap, q, Gam, iters=500):
    G0 = 10 ** (snr_db / 10)
    Ppost = 1 / (1 / Pgrid[:, None] + rhos[None, :] / kap)
    feas = Ppost <= Gam
    R = np.where(feas, np.log2(1 + (1 - rhos[None, :]) * G0 * gbar(Pgrid, Nx)[:, None]), -np.inf)
    idx = np.clip(np.searchsorted(Pgrid, Ppost + q), 0, len(Pgrid) - 1)
    V = np.zeros(len(Pgrid))
    for _ in range(iters):
        Q = R + disc * V[idx]
        V = Q.max(1)
    pol = Q.argmax(1)
    my = np.array([np.argmax(f) if f.any() else len(rhos) - 1 for f in feas])  # myopic: least sensing that is feasible

    def roll(p):
        i, tot = np.searchsorted(Pgrid, Gam + q), 0.0
        for t in range(300):
            a = p[i]; tot += disc ** t * R[i, a]; i = idx[i, a]
        return tot
    return 100 * (roll(pol) - roll(my)) / roll(my), pol, my


table = {}
kgrid, qgrid = np.logspace(-7, -3, 9), np.logspace(-7, -2, 11)
for Nx in (16, 32, 64):
    for Gam in (1e-3, 1e-2):
        G = np.array([[solve(Nx, 15, k, q, Gam)[0] for q in qgrid] for k in kgrid])
        table[f"Nx={Nx},Gamma={Gam}"] = {"max_pct": float(G.max()), "median_pct": float(np.median(G)),
                                         "min_pct": float(G.min())}
out["theorem2_nonmyopic_gain_over_kappa_q_grid"] = table

g, pol, my = solve(32, 15, 1e-4, 1e-7, 1e-2)
sel = (Pgrid > 1e-7) & (Pgrid < 1e-2)
d = np.diff(rhos[pol][sel])
out["theorem2_example_Nx32"] = {"gain_pct": float(g), "frac_nondecreasing_steps": float(np.mean(d >= -1e-12))}

fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.0))
for Gam, mk in ((1e-3, "o-"), (1e-2, "s-")):
    axs[0].plot([16, 32, 64], [table[f"Nx={n},Gamma={Gam}"]["max_pct"] for n in (16, 32, 64)], mk, label=fr"max, $\Gamma$={Gam:g}")
    axs[0].plot([16, 32, 64], [table[f"Nx={n},Gamma={Gam}"]["median_pct"] for n in (16, 32, 64)], mk, alpha=0.4, label=fr"median, $\Gamma$={Gam:g}")
axs[0].set_xscale("log", base=2); axs[0].set_xlabel(r"surface columns $N_x$"); axs[0].set_ylabel("non-myopic gain over myopic (%)")
axs[0].grid(lw=0.3); axs[0].legend(fontsize=6.5)
axs[1].semilogx(Pgrid[sel], rhos[pol][sel], label="optimal (non-myopic)")
axs[1].semilogx(Pgrid[sel], rhos[my][sel], "--", label="myopic")
axs[1].set_xlabel(r"prior variance $P^-$ of $u$"); axs[1].set_ylabel(r"sensing power fraction $\rho$")
axs[1].grid(lw=0.3, which="both"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig("results/fig_thm2_myopic.png", dpi=200)

json.dump(out, open("results/theory_v2.json", "w"), indent=2)
print(json.dumps(out, indent=2))
