#!/bin/zsh
# impala_coord from scratch: BC on all-level teacher traces -> PPO -> expert iteration.
set -e
cd /Users/malfet/git/malfet/vibeformer/digger-rl
PY=.venv/bin/python
C=data/checkpoints
COMMON=(--obs-size 128 --arch impala_coord --num-envs 8 --no-clip-reward
        --reward-scale 0.02 --episodic-life --start-prob 0.5 --archive-every 50
        --archive-prob 0.3 --archive-size 200 --archive-perturb 10 --lr 1e-4
        --save-best --total-timesteps 4000000)

# 1. BC on the teacher, 300k steps from every level (new teacher: NOOPs while dying)
$PY train_pixel.py --total-timesteps 0 --obs-size 128 --arch impala_coord \
  --aug-shift 4 --bc-traces data/traces/lv_{0..9}.npz \
  --bc-val data/traces/lv_val_{0,1}.npz --bc-epochs 15 --bc-batch-size 256 \
  --run-name px128_coord_bc > logs/px128_coord_bc.log 2>&1
$PY evaluate.py $C/px128_coord_bc/ppo_digger_bc_only.pt --episodes 20 \
  > logs/eval_coord_bc.txt 2>&1

# 2. PPO from the student; existing near-death and bag-kill pools as starts
$PY train_pixel.py $COMMON --load-agent $C/px128_coord_bc/ppo_digger_bc_only.pt \
  --value-warmup-updates 10 \
  --bc-traces data/traces/lv_0.npz data/traces/lv_1.npz --bc-epochs 0 \
  --bc-anchor-coef 0.3 --bc-anchor-final 0.0 \
  --ent-coef 0.01 --ent-coef-final 0.001 \
  --start-states data/save/levels=0.5,data/save/near_death_rescued=0.3,data/save/bag_kills=0.2 \
  --run-name px128_coord_ppo1 > logs/px128_coord_ppo1.log 2>&1
$PY evaluate.py $C/px128_coord_ppo1/ppo_digger_final.pt --episodes 20 \
  > logs/eval_coord_ppo1.txt 2>&1

# 3. Expert iteration: search-corrected play becomes the anchor
for s in 0 1 2 3; do
  $PY -m tools.policy_search $C/px128_coord_ppo1/ppo_digger_final.pt \
    --out data/traces/px128_coord_search_$s.npz --steps 20000 --seed $s \
    > logs/coord_search_$s.log 2>&1 &
done; wait
$PY train_pixel.py $COMMON --load-agent $C/px128_coord_ppo1/ppo_digger_final.pt \
  --value-warmup-updates 5 \
  --bc-traces data/traces/px128_coord_search_{0,1,2,3}.npz --bc-epochs 0 \
  --bc-anchor-coef 0.5 --bc-anchor-final 0.1 \
  --ent-coef 0.005 --ent-coef-final 0.001 \
  --start-states data/save/levels=0.6,data/save/bag_kills=0.2,data/save/near_death_rescued=0.2 \
  --run-name px128_coord_exit1 > logs/px128_coord_exit1.log 2>&1
$PY evaluate.py $C/px128_coord_exit1/ppo_digger_final.pt --episodes 20 \
  > logs/eval_coord_exit1.txt 2>&1
$PY evaluate.py $C/px128_coord_exit1/ppo_digger_final.pt --episodes 10 --argmax \
  > logs/eval_coord_exit1_argmax.txt 2>&1
