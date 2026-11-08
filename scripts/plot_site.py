"""Figures 9-11 of the paper from the site experiments (run after site_experiments.py)."""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import scripts.eval_site as ev
from hatd3.site import SITE_A

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8, "axes.linewidth": 0.6,
                     "lines.linewidth": 1.2, "legend.frameon": False})
OUT = "figures"
os.makedirs(OUT, exist_ok=True)
STY = {"hatd3": ("HA-TD3", "C0", "-", "o"), "model": ("Pilot-free model-based", "C1", "--", "s"),
       "genie": ("Genie (perfect CSI)", "k", ":", "^"), "flat": ("Flat-attention TD3", "C2", "-.", "v"),
       "mlp": ("MLP-TD3", "C3", (0, (3, 1, 1, 1)), "d"), "ddpg": ("DDPG", "C4", (0, (5, 2)), "x")}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".tif"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig9():
    log = []
    ev.N_EP = 36
    for p in sorted(glob.glob("runs/hatd3_s[0-9]")):
        ev.eval_agent(p, SITE_A, 6, 1e-3, log=log)
    ev.N_EP = 12
    lanes = [14.0, 10.0, -10.0, -14.0]
    bins = np.arange(-70, 71, 2.0)
    M = np.full((4, len(bins) - 1), np.nan)
    xs, js, ys = [], [], []
    for l in log:
        xs += l["x"]; ys += l["y"]; js += list(np.asarray(l["du"]) > 0)
    xs, ys, js = map(np.asarray, (xs, ys, js))
    for i, ln in enumerate(lanes):
        m = np.abs(ys - ln) < 1
        for b in range(len(bins) - 1):
            mm = m & (xs >= bins[b]) & (xs < bins[b + 1])
            if mm.sum() >= 3:
                M[i, b] = js[mm].mean()
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [2.3, 1]})
    im = ax[0].imshow(M, aspect="auto", cmap="Greys", vmin=0, vmax=1, extent=[bins[0], bins[-1], 3.5, -0.5])
    ax[0].set_yticks(range(4)); ax[0].set_yticklabels(["r, y = 14 m", "r, y = 10 m", "t, y = -10 m", "t, y = -14 m"])
    for s in SITE_A.screens:                                    # obstacle spans on the near side
        row = [0, 1] if s.y > 0 else [2, 3]
        for r in row:
            ax[0].plot([s.x1, s.x2], [r - 0.42, r - 0.42], color="C3" if s.cap_dB > 20 else "C2", lw=2.0)
    ax[0].set_xlabel("Vehicle position x (m)")
    ax[0].set_title("(a)", loc="left", fontsize=8)
    cb = fig.colorbar(im, ax=ax[0], fraction=0.03, pad=0.01); cb.set_label("Share of image-source beams")
    ax[0].plot([], [], color="C3", lw=2, label="truck / shelter"); ax[0].plot([], [], color="C2", lw=2, label="tree canopy")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    kr = np.array([l["K_r"] for l in log]); be = np.array([l["beta"] for l in log])
    K = [2, 3, 4]
    ax[1].bar(np.arange(3) - 0.18, [be[kr == k].mean() for k in K], 0.36, color="C0", label="HA-TD3 (learned)")
    ax[1].bar(np.arange(3) + 0.18, [k / 6 for k in K], 0.36, color="white", edgecolor="k", hatch="///", label=r"Rule $K_r/K$")
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels([f"$K_r$ = {k}" for k in K])
    ax[1].set_ylabel(r"Energy split $\beta^r$"); ax[1].set_ylim(0, 0.8)
    ax[1].set_title("(b)", loc="left", fontsize=8)
    ax[1].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    save(fig, "Fig9")


def fig10(res):
    sw = res["gamma_sweep"]
    G = sorted(float(g) for g in sw)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for key in ("genie", "hatd3", "model"):
        lab, col, ls, mk = STY[key]
        thr = [sw[str(g) if str(g) in sw else repr(g)][key]["throughput"][0] for g in G]
        vio = [sw[str(g) if str(g) in sw else repr(g)][key]["viol"][0] for g in G]
        ax[0].plot(G, thr, color=col, ls=ls, marker=mk, ms=4, label=lab)
        ax[1].plot(G, vio, color=col, ls=ls, marker=mk, ms=4, label=lab)
    ax[0].set_ylabel("Effective throughput (bit/s/Hz)"); ax[1].set_ylabel("PCRB violation rate (%)")
    for a, t in zip(ax, ("(a)", "(b)")):
        a.set_xscale("log"); a.set_xlabel("PCRB threshold Γ"); a.grid(alpha=0.3, lw=0.4)
        a.set_title(t, loc="left", fontsize=8)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "Fig10")


def fig11():
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
    for key in ("hatd3", "flat", "mlp", "ddpg"):
        H = [json.load(open(p + "/train.json"))["hist"] for p in sorted(glob.glob(f"runs/{key}_s[0-9]"))]
        if not H:
            continue
        lab, col, ls, _ = STY[key]
        w = 25
        for i, (field, a) in enumerate((("throughput", ax[0]), ("viol", ax[1]))):
            Y = np.array([[h[field] for h in hh] for hh in H])
            Y = np.array([np.convolve(y, np.ones(w) / w, mode="valid") for y in Y])
            x = np.arange(Y.shape[1]) + w
            a.plot(x, Y.mean(0), color=col, ls=ls, label=lab)
            a.fill_between(x, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0), color=col, alpha=0.15, lw=0)
        if key == "hatd3":
            L = np.array([[h["lam"] for h in hh] for hh in H])
            ax[2].plot(L[:, :, 0].mean(0), color="C0", label="λ vehicles")
            ax[2].plot(L[:, :, 1].mean(0), color="C1", ls="--", label="λ road users")
    ax[0].set_ylabel("Training throughput (bit/s/Hz)"); ax[1].set_ylabel("Violation rate (%)"); ax[2].set_ylabel("Multiplier λ")
    for a, t in zip(ax, ("(a)", "(b)", "(c)")):
        a.set_xlabel("Episode"); a.grid(alpha=0.3, lw=0.4); a.set_title(t, loc="left", fontsize=8)
    ax[0].legend(fontsize=6.5); ax[2].legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, "Fig11")


if __name__ == "__main__":
    res = json.load(open("results/site_results.json"))
    fig9(); fig10(res); fig11()
    print("figures written")
