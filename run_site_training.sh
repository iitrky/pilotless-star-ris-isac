#!/usr/bin/env bash
# Trains every learned controller of Tables 9-10 on site A (NumPy, CPU). ~11 min per run.
# Usage: bash run_site_training.sh [parallel jobs, default 2]
set -e
P=${1:-2}
EP=400
{
for s in 0 1 2; do
  echo "--encoder hier --seed $s --out runs/hatd3_s$s"
  echo "--encoder flat --seed $s --out runs/flat_s$s"
  echo "--encoder mlp  --seed $s --out runs/mlp_s$s"
  echo "--algo ddpg    --seed $s --out runs/ddpg_s$s"
done
for f in no-level2 no-two-timescale no-anchor no-shaping no-lagrangian no-beta-bound coupled; do
  echo "--$f --seed 0 --out runs/abl_${f//-/}_s0"
done
} | xargs -P "$P" -I{} sh -c "python3 scripts/train_site.py --episodes $EP {} > /dev/null 2>&1 || echo FAILED {}"
[ "$TRAIN_ONLY" = 1 ] && exit 0
python3 scripts/site_experiments.py
python3 scripts/plot_site.py
