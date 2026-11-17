#!/bin/bash
# Hesitation start states from px128_exit2, then PPO from exit2 with them.
cd "$(dirname "$0")/.."
P=data/checkpoints/px128_exit2/ppo_digger_final.pt
for s in 0 1 2 3; do
  .venv/bin/python -m tools.hesitation $P --out-dir data/save/hesitation \
    --states 75 --seed $s > logs/hesitation_$s.log 2>&1 &
done; wait
ls data/save/hesitation | wc -l
.venv/bin/python train_pixel.py --obs-size 128 --arch impala \
  --load-agent $P \
  --total-timesteps 4000000 --num-envs 8 --no-clip-reward --reward-scale 0.02 \
  --episodic-life --value-warmup-updates 5 \
  --bc-traces data/traces/px128_search2_{0,1,2,3}.npz data/traces/px128_search_{0,1}.npz \
  --bc-epochs 0 --bc-anchor-coef 0.3 --bc-anchor-final 0.1 \
  --ent-coef 0.005 --ent-coef-final 0.001 \
  --start-states data/save/levels=0.5,data/save/hesitation=0.2,data/save/bag_kills=0.15,data/save/near_death_rescued=0.15 \
  --start-prob 0.5 --archive-every 50 --archive-prob 0.3 --archive-size 200 \
  --archive-perturb 10 --lr 1e-4 --save-best --run-name px128_hes1
