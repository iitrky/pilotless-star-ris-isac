#!/usr/bin/env bash
# Reproduces every number, table and figure of the paper (CPU only, about 4 h on 2 cores).
# Usage: bash run_all.sh [parallel jobs, default 2]
set -e
P=${1:-2}
echo "== 1. unit tests";                          python -m pytest -q
echo "== 2. theory (Sec. 5)";                     python scripts/theory_v2.py; python scripts/verify_theory.py; python scripts/anchor_check.py
echo "== 3. full-system simulation (Sec. 9.1-9.2)"; python scripts/system_eval.py
echo "== 4. training: 19 agents + 6 deployment variants"
TRAIN_ONLY=1 bash run_site_training.sh "$P"
{
for s in 0 1 2; do echo "--episodes 400 --site random --seed $s --out runs2/hatd3R_s$s"; done
for s in 0 1 2; do echo "--site B --init runs/hatd3_s$s --episodes 100 --warmup 200 --seed 1$s --out runs2/hatd3FT_s$s"; done
} | xargs -P "$P" -I{} sh -c "python scripts/train_site.py {} > /dev/null"
echo "== 5. first evaluation, Figs. 10-12";      python scripts/site_experiments.py; python scripts/paired_stats.py; python scripts/plot_site.py
echo "== 6. pre-registered evaluation (protocol results/protocol_v2.md), Tables 9-13"
python scripts/eval_v2.py; python scripts/eval_abl_v2.py
echo "== 7. figures 1-9";                         python scripts/make_figures.py
echo "done: results/ and figures/ regenerated"
