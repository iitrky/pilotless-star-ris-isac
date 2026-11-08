"""Generate every manuscript figure at journal resolution (SN Computer Science artwork rules):
sans-serif lettering (Liberation Sans, metric-compatible with Arial), 600 dpi TIFF (LZW) plus PNG,
no titles inside figures, markers and line styles in addition to colour.

Fig1  system architecture (block diagram)
Fig2  time-division ISAC epoch and processing pipeline (block diagram)
Fig3  HA-TD3 network architecture (block diagram)
Fig4  Lemma 1 expected gain loss
Fig5  Theorem 1 pilot overhead vs speed
Fig6  Proposition 1 CRB vs energy split
Fig7  Theorem 2 abstraction (non-myopic gain, policies)
Fig8  full-system throughput vs speed
Data for Fig4-Fig8 are recomputed here or read from results/*.json produced by the verification scripts.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9, "axes.linewidth": 0.8,
                     "lines.linewidth": 1.4, "lines.markersize": 4.5, "legend.fontsize": 8,
                     "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})
INK, EDGE, ACC, FILL = "#000000", "#333333", "#1f5fa8", "#e8f0fa"


def save(fig, name):
    kw = dict(bbox_inches="tight", pad_inches=0.04)
    fig.savefig(os.path.join(OUT, name + ".tif"), dpi=600, pil_kwargs={"compression": "tiff_lzw"}, **kw)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, **kw)
    plt.close(fig)


def box(ax, x, y, w, h, lines, bold_first=True, accent=False, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                lw=1.6 if accent else 1.0, ec=ACC if accent else EDGE,
                                fc=FILL if accent else "white"))
    n = len(lines)
    for i, t in enumerate(lines):
        yy = y + h / 2 + (n - 1) * 0.11 - i * 0.22
        ax.text(x + w / 2, yy, t, ha="center", va="center", fontsize=fs,
                fontweight="bold" if (i == 0 and bold_first) else "normal", color=INK)


def arrow(ax, pts, label=None, lpos=None, ls="-"):
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=EDGE, lw=1.0, ls=ls)
    a = FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=9, color=EDGE, lw=1.0, ls=ls)
    ax.add_patch(a)
    if label:
        ax.text(*lpos, label, fontsize=7.5, ha="center", va="center", color=INK,
                bbox=dict(fc="white", ec="none", pad=0.5))


# ------------------------------------------------------------------ Fig1 system architecture
def fig1():
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.set_xlim(-0.3, 9.5); ax.set_ylim(0.1, 5.1); ax.axis("off")
    box(ax, -0.25, 2.9, 2.45, 1.1, ["Base station", "M-antenna ISAC", "no pilots sent"], fs=8)
    ax.add_patch(FancyBboxPatch((3.0, 2.2), 2.3, 2.8, boxstyle="round,pad=0.02,rounding_size=0.06", lw=1.0, ec=EDGE, fc="white"))
    ax.text(4.15, 4.75, "STAR-RIS + sensors", ha="center", fontweight="bold", fontsize=8.5)
    ax.plot([3.0, 5.3], [4.5, 4.5], color="#999999", lw=0.6)
    ax.plot([3.0, 5.3], [3.35, 3.35], color="#999999", lw=0.6, ls="--")
    ax.text(4.15, 4.05, "Reflection zone", ha="center", fontsize=8.5); ax.text(4.15, 3.75, r"split $\beta^r$, phase $\theta^r$", ha="center", fontsize=8)
    ax.text(4.15, 2.9, "Transmission zone", ha="center", fontsize=8.5); ax.text(4.15, 2.6, r"split $\beta^t$, phase $\theta^t$", ha="center", fontsize=8)
    box(ax, 6.3, 3.75, 2.4, 0.9, ["Reflection side", "vehicles + road users"])
    box(ax, 6.3, 2.45, 2.4, 0.9, ["Transmission side", "vehicles + road users"])
    box(ax, 6.3, 0.2, 2.4, 1.2, ["Sensors + EKF", "predicted states", "means, covariances"])
    box(ax, 3.0, 0.2, 2.3, 1.2, ["HA-TD3 agent", "belief tokens", "PCRB constraint"], accent=True)
    box(ax, -0.25, 0.2, 2.45, 1.2, ["Control update", r"split $\beta$, anchor or", "image-source beam"], fs=8)
    arrow(ax, [(2.1, 3.45), (3.0, 3.45)], r"$\mathbf{G}$", (2.55, 3.65))
    arrow(ax, [(5.3, 4.2), (6.3, 4.2)], "reflect", (5.8, 4.4))
    arrow(ax, [(5.3, 2.9), (6.3, 2.9)], "transmit", (5.8, 3.1))
    arrow(ax, [(7.5, 2.45), (7.5, 1.4)], "echoes", (7.95, 1.92))
    arrow(ax, [(8.7, 4.2), (9.3, 4.2), (9.3, 0.8), (8.7, 0.8)])
    arrow(ax, [(6.3, 0.8), (5.3, 0.8)], "beliefs", (5.8, 1.0))
    arrow(ax, [(3.0, 0.8), (2.1, 0.8)], r"action $a_n$", (2.55, 1.0))
    arrow(ax, [(0.9, 1.4), (0.9, 2.9)], r"$\mathbf{W}$", (0.65, 2.15))
    arrow(ax, [(1.7, 1.4), (1.7, 1.85), (4.15, 1.85), (4.15, 2.2)], r"$\beta,\theta$", (2.9, 2.03))
    save(fig, "Fig1")


# ------------------------------------------------------------------ Fig2 epoch frame structure
def fig2():
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.0); ax.axis("off")
    x0, y0, h = 0.4, 2.55, 0.7
    segs = [(1.6, "Scan", r"$\rho T$", True)] + [(1.25, f"Data {k}", r"$(1-\rho)T/K$", False) for k in (1, 2)] + \
           [(0.9, "...", "", False)] + [(1.25, "Data K", r"$(1-\rho)T/K$", False)]
    x = x0
    for w, lab, sub, sc in segs:
        ax.add_patch(plt.Rectangle((x, y0), w, h, fc=FILL if sc else "white", ec=EDGE, lw=1.0,
                                   hatch="..." if sc else None))
        bb = dict(fc="white", ec="none", pad=0.6) if sc else None
        ax.text(x + w / 2, y0 + h / 2 + 0.12, lab, ha="center", va="center", fontsize=8.5, fontweight="bold", bbox=bb)
        ax.text(x + w / 2, y0 + h / 2 - 0.18, sub, ha="center", va="center", fontsize=7.5, bbox=bb)
        x += w
    ax.annotate("", xy=(x0, y0 + h + 0.25), xytext=(x, y0 + h + 0.25), arrowprops=dict(arrowstyle="<->", lw=0.9))
    ax.text((x0 + x) / 2, y0 + h + 0.42, r"one epoch $\Delta T$ (T symbols); no pilot symbols", ha="center", fontsize=8)
    ax.text(x + 0.15, y0 + h / 2, "surface steered at all\nentities (scan) or at\none vehicle (data)", fontsize=7.5, va="center")
    steps = ["Echoes of all\nsub-blocks", "EKF update\nposterior, PCRB", "Predict next\nepoch (prior)",
             "HA-TD3 codebook\nentry, split", "Apply in next\nepoch"]
    xs = np.linspace(0.2, 8.15, len(steps))
    for i, (xx, st) in enumerate(zip(xs, steps)):
        box(ax, xx, 0.35, 1.75, 0.9, st.split("\n"), bold_first=False, accent=(i == 3), fs=7.4)
        if i:
            arrow(ax, [(xs[i - 1] + 1.75, 0.8), (xx, 0.8)])
    arrow(ax, [(1.08, y0), (1.08, 1.25)], ls="--")
    save(fig, "Fig2")


# ------------------------------------------------------------------ Fig3 HA-TD3 architecture
def fig3():
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
    box(ax, 0.1, 4.6, 2.0, 1.4, ["Belief tokens", "mean, covariance,", "PCRB, zone, rate,", "echo attenuation"], fs=7.6)
    box(ax, 2.6, 4.6, 1.9, 1.4, ["Type MLPs", "entity embedding", "+ type embedding"], fs=7.6)
    box(ax, 5.0, 5.2, 2.3, 0.85, ["Level 1, zone r", "masked self-attention"], fs=7.6)
    box(ax, 5.0, 4.15, 2.3, 0.85, ["Level 1, zone t", "masked self-attention"], fs=7.6)
    box(ax, 7.8, 4.6, 2.0, 1.2, ["Attention pooling", r"zone summaries $z_r$, $z_t$"], fs=7.6)
    box(ax, 0.1, 2.85, 2.0, 1.1, ["Global token", r"occupancy, $\lambda$,", r"held split $\beta$"], fs=7.6)
    box(ax, 7.8, 2.85, 2.0, 1.1, ["Level 2", "transformer over", r"$\{z_r, z_t, z_G\}$"], fs=7.6, accent=True)
    box(ax, 4.0, 2.05, 2.0, 0.65, ["Context c"], fs=8)
    box(ax, 0.1, 0.15, 2.25, 1.3, ["Slow head (actor)", r"split correction to $K_r/K$", r"every $T_h$ epochs"], fs=7.6)
    box(ax, 2.6, 0.15, 2.25, 1.3, ["Residual codebook", r"per-vehicle $q_k(j)$: anchor", "or image source j"], fs=7.6, accent=True)
    box(ax, 5.1, 0.15, 2.2, 1.3, ["Value heads", r"per-vehicle $q_k$", r"+ global $q_g$"], fs=7.6)
    box(ax, 7.55, 0.15, 2.25, 1.3, ["Twin critics", r"PCRB Lagrangian $\lambda$", "+ PBRS shaping"], fs=7.6)
    arrow(ax, [(2.1, 5.3), (2.6, 5.3)])
    arrow(ax, [(4.5, 5.62), (5.0, 5.62)])
    arrow(ax, [(4.5, 4.95), (4.75, 4.95), (4.75, 4.57), (5.0, 4.57)])
    arrow(ax, [(7.3, 5.62), (7.8, 5.4)]); arrow(ax, [(7.3, 4.57), (7.8, 4.95)])
    arrow(ax, [(8.8, 4.6), (8.8, 3.95)])
    arrow(ax, [(2.1, 3.4), (7.8, 3.4)], r"$z_G$", (4.9, 3.58))
    arrow(ax, [(8.8, 2.85), (8.8, 2.37), (6.0, 2.37)])
    ax.plot([5.0, 5.0], [2.05, 1.75], color=EDGE, lw=1.0)
    ax.plot([1.2, 6.2], [1.75, 1.75], color=EDGE, lw=1.0)
    for xx in (1.2, 3.7, 6.2):
        arrow(ax, [(xx, 1.75), (xx, 1.45)])
    arrow(ax, [(7.3, 0.8), (7.55, 0.8)])
    ax.text(7.42, 1.0, "a", fontsize=7.5, ha="center")
    save(fig, "Fig7")


# ------------------------------------------------------------------ data figures
def gbar(var, Nx):
    m = np.arange(1, Nx)
    var = np.atleast_1d(var)[:, None]
    return (1 / Nx) * (1 + 2 * np.sum((1 - m / Nx) * np.exp(-(np.pi * m) ** 2 * var / 2), 1))


MARK = ["o", "s", "^", "D", "v"]
LS = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]


def fig4():
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    s = np.logspace(-3.5, -0.5, 60)
    for i, Nx in enumerate((8, 16, 32, 64)):
        ax.semilogx(s, 10 * np.log10(gbar(s ** 2, Nx)), ls=LS[i], marker=MARK[i], markevery=8, label=f"$N_x$ = {Nx}")
    ax.set_xlabel(r"Direction-cosine error std $\sigma_u$"); ax.set_ylabel("Expected gain loss (dB)")
    ax.set_ylim(-15, 0.5); ax.grid(lw=0.3, which="both"); ax.legend()
    fig.tight_layout(); save(fig, "Fig3")


def fig5():
    c, fc, Ts = 299_792_458.0, 7e9, (1 / 120e3) * (1 + 1 / 14)
    tp = lambda K, N, M: K + N + max(K - 1, int(np.ceil((K - 1) * N / M)))
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    v = np.linspace(5, 160, 80)
    for i, N in enumerate((64, 256)):
        eff = np.clip(1 - tp(6, N, 16) * Ts * v / 3.6 * fc / (0.423 * c), 0, None)
        ax.plot(v, eff, ls=LS[i], marker=MARK[i], markevery=8, label=f"Pilot-based, N = {N}")
    for j, r in enumerate((0.7, 0.9)):
        ax.axhline(r, color="k", lw=0.8, ls=LS[2 + j], label=fr"Pilot-free, $\eta$ = {r}")
    ax.set_xlabel("Vehicle speed (km/h)"); ax.set_ylabel("Effective rate / CSI rate")
    ax.set_ylim(0, 1.05); ax.grid(lw=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); save(fig, "Fig4")


def fig6():
    from hatd3.config import SysConfig
    from hatd3.env import StarRisIsacEnv
    rng = np.random.default_rng(7)
    betas = np.logspace(-2, 0, 25)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for i, (mode, nu) in enumerate((("ris_sensors", 1), ("bs", 2))):
        cfg = SysConfig(sensing_mode=mode)
        env = StarRisIsacEnv(cfg, seed=11); env.reset(eps=0.0)
        pr, pt = rng.uniform(0, 2 * np.pi, cfg.N), rng.uniform(0, 2 * np.pi, cfg.N)
        W = rng.standard_normal((cfg.M, cfg.K + cfg.L)) + 1j * rng.standard_normal((cfg.M, cfg.K + cfg.L))
        W *= np.sqrt(env.P) / np.linalg.norm(W)
        cr = []
        for bt in betas:
            _, _, crb, _ = env.evaluate(W, np.sqrt(1 - bt) * np.exp(1j * pr), np.sqrt(bt) * np.exp(1j * pt))
            cr.append(crb[cfg.L_r])
        cr = np.array(cr)
        ax.loglog(betas, cr / cr[-1], ls=LS[i], marker=MARK[i], markevery=3,
                  label="Sensors at surface (ν = 1)" if nu == 1 else "BS echo via surface (ν = 2)")
    ax.loglog(betas, betas ** -1.0, "k:", lw=0.8, label=r"$\beta^{-1}$")
    ax.loglog(betas, betas ** -2.0, color="k", ls=(0, (1, 2)), lw=0.8, label=r"$\beta^{-2}$")
    ax.set_xlabel(r"Transmission-zone energy split $\beta^t$"); ax.set_ylabel(r"CRB / CRB($\beta^t$ = 1)")
    ax.grid(lw=0.3, which="both"); ax.legend(fontsize=7)
    fig.tight_layout(); save(fig, "Fig5")


def fig7():
    d = json.load(open(os.path.join(ROOT, "results", "theory_v2.json")))["theorem2_nonmyopic_gain_over_kappa_q_grid"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for i, G in enumerate((0.001, 0.01)):
        axs[0].plot([16, 32, 64], [d[f"Nx={n},Gamma={G}"]["max_pct"] for n in (16, 32, 64)], ls=LS[i], marker=MARK[i], label=fr"max, $\Gamma$ = {G:g}")
        axs[0].plot([16, 32, 64], [d[f"Nx={n},Gamma={G}"]["median_pct"] for n in (16, 32, 64)], ls=LS[i + 2], marker=MARK[i + 2], label=fr"median, $\Gamma$ = {G:g}")
    axs[0].set_xscale("log", base=2); axs[0].set_xticks([16, 32, 64]); axs[0].set_xticklabels(["16", "32", "64"])
    axs[0].set_xlabel(r"Surface columns $N_x$"); axs[0].set_ylabel("Non-myopic gain (%)"); axs[0].grid(lw=0.3); axs[0].legend(fontsize=7)
    # policy shape (recompute, abstraction of Sec. 5.4)
    disc, Pgrid, rhos = 0.95, np.logspace(-8, 0, 500), np.linspace(0, 1, 201)
    G0, kap, q, Gam, Nx = 10 ** 1.5, 1e-4, 1e-7, 1e-2, 32
    Ppost = 1 / (1 / Pgrid[:, None] + rhos[None, :] / kap)
    feas = Ppost <= Gam
    R = np.where(feas, np.log2(1 + (1 - rhos[None, :]) * G0 * gbar(Pgrid, Nx)[:, None]), -np.inf)
    idx = np.clip(np.searchsorted(Pgrid, Ppost + q), 0, len(Pgrid) - 1)
    V = np.zeros(len(Pgrid))
    for _ in range(500):
        Q = R + disc * V[idx]; V = Q.max(1)
    pol = rhos[Q.argmax(1)]
    my = np.array([rhos[np.argmax(f)] if f.any() else 1.0 for f in feas])
    sel = (Pgrid > 1e-7) & (Pgrid < 1e-2)
    axs[1].semilogx(Pgrid[sel], pol[sel], ls="-", label="Optimal (non-myopic)")
    axs[1].semilogx(Pgrid[sel], my[sel], ls="--", label="Myopic")
    axs[1].set_xlabel(r"Prior variance $P^-$ of $u$"); axs[1].set_ylabel(r"Sensing fraction $\rho$")
    axs[1].grid(lw=0.3, which="both"); axs[1].legend(fontsize=7)
    for a, l in zip(axs, "ab"):
        a.text(-0.16, 1.02, f"({l})", transform=a.transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout(); save(fig, "Fig6")


def fig8():
    d = json.load(open(os.path.join(ROOT, "results", "system_eval.json")))["expA"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    v = [30, 60, 90, 120, 150]
    for a, N, l in zip(axs, (64, 256), "ab"):
        a.plot(v, [d[f"N={N},v={x}"]["genie"] for x in v], "k--", marker="^", label="Genie CSI (no pilot cost)")
        a.plot(v, [d[f"N={N},v={x}"]["pilot_free"] for x in v], ls="-", marker="o", label="Pilot-free (beliefs)")
        a.plot(v, [d[f"N={N},v={x}"]["pilot_based"] for x in v], ls="-.", marker="s", label="Pilot-based (overhead counted)")
        a.set_xlabel("Vehicle speed (km/h)"); a.grid(lw=0.3); a.set_ylim(bottom=0)
        a.text(-0.14, 1.02, f"({l}) N = {N}", transform=a.transAxes, fontsize=9, fontweight="bold")
    axs[0].set_ylabel("Throughput (bit/s/Hz)"); axs[0].legend(fontsize=7)
    fig.tight_layout(); save(fig, "Fig8")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        f(); print(f.__name__, "done")
