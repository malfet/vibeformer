#!/bin/zsh
cd /Users/malfet/git/malfet/vibeformer/digger-rl
while pgrep -f "tools.policy_search" >/dev/null; do sleep 30; done
exec .venv/bin/python train_pixel.py --obs-size 128 --arch impala \
  --load-agent data/checkpoints/px128_exit1/ppo_digger_final.pt \
  --total-timesteps 4000000 --num-envs 8 --no-clip-reward --reward-scale 0.02 \
  --episodic-life --value-warmup-updates 5 \
  --bc-traces data/traces/px128_search2_{0,1,2,3}.npz data/traces/px128_search_{0,1}.npz --bc-epochs 0 \
  --bc-anchor-coef 0.5 --bc-anchor-final 0.1 \
  --ent-coef 0.005 --ent-coef-final 0.001 \
  --start-states data/save/levels=0.6,data/save/bag_kills=0.2,data/save/near_death_rescued=0.2 \
  --start-prob 0.5 --archive-every 50 --archive-prob 0.3 --archive-size 200 \
  --archive-perturb 10 --lr 1e-4 --save-best --run-name px128_exit2
