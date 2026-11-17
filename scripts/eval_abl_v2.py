"""Ablations re-evaluated on the protocol-v2 test set T_A (60 episodes), paired against HA-TD3 seed 0."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import scripts.eval_v2 as e2
import scripts.train_site as ts

eps = e2.testset("A")
out = {}
ref = e2.run_agent("runs/hatd3_s0", eps)
out["hatd3_s0"] = e2.summ([ref])
for p in sorted(glob.glob("runs/abl_*_s0")):
    rs = e2.run_agent(p, eps)
    out[os.path.basename(p)] = e2.summ([rs]) | {"paired_vs_full": e2.compare(e2.per_ep(rs, "throughput"), e2.per_ep(ref, "throughput"))}
    print(p, out[os.path.basename(p)]["throughput"], flush=True)
# model-based with coupled phases, same episodes
mc = []
for mk in eps:
    env = mk(); env.cfg.coupled_phase = True
    mc.append(e2.summarise([e2.epoch(env, "model", e2.GAM) for _ in range(ts.EPOCHS)], e2.GAM))
out["model_coupled"] = e2.summ([mc])
cp = e2.run_agent("runs/abl_coupled_s0", eps)
out["coupled_vs_model_coupled"] = e2.compare(e2.per_ep(cp, "throughput"), e2.per_ep(mc, "throughput"))
# K = 10 generalisation on 30 episodes of site A
from hatd3.site import SITE_A
eps10 = [lambda i=i: e2.make(950_000 + i, 5, SITE_A, K=10) for i in range(30)]
h10 = [e2.run_agent(p, eps10) for p in sorted(glob.glob("runs/hatd3_s[0-9]"))]
m10 = [e2.run_baseline("model", eps10)]
out["K10"] = {"hatd3": e2.summ(h10), "model": e2.summ(m10),
              "paired": e2.compare(e2.seed_avg(h10, "throughput"), e2.seed_avg(m10, "throughput")),
              "paired_p5": e2.compare(e2.seed_avg(h10, "p5_rate"), e2.seed_avg(m10, "p5_rate"))}
json.dump(out, open("results/eval_abl_v2.json", "w"), indent=1)
print("done")
