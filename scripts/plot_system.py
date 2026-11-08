"""Fig. 6 from results/system_eval.json (full-system speed sweep)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.load(open("results/system_eval.json"))["expA"]
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=False)
for ax, N in zip(axs, (64, 256)):
    v = [30, 60, 90, 120, 150]
    g = [d[f"N={N},v={x}"]["genie"] for x in v]
    p = [d[f"N={N},v={x}"]["pilot_based"] for x in v]
    f = [d[f"N={N},v={x}"]["pilot_free"] for x in v]
    ax.plot(v, g, "k--", label="genie CSI (no pilot cost)")
    ax.plot(v, f, "o-", label="pilot-free (beliefs)")
    ax.plot(v, p, "s-", label="pilot-based (overhead counted)")
    ax.set_title(f"N = {N}", fontsize=9); ax.set_xlabel("vehicle speed (km/h)"); ax.grid(lw=0.3)
    ax.set_ylim(bottom=0)
axs[0].set_ylabel("throughput (bit/s/Hz)"); axs[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig("results/fig6_speed_sweep.png", dpi=200)
