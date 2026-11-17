#!/bin/zsh
# coord expert iteration round 2 (mirrors exit2): search traces from coord_exit1.
set -e
cd /Users/malfet/git/malfet/vibeformer/digger-rl
PY=.venv/bin/python
C=data/checkpoints
P=$C/px128_coord_exit1/ppo_digger_final.pt
for s in 0 1 2 3; do
  $PY -m tools.policy_search $P --out data/traces/px128_coord_search2_$s.npz \
    --steps 20000 --seed $((s+10)) > logs/coord_search2_$s.log 2>&1 &
done; wait
$PY train_pixel.py --obs-size 128 --arch impala_coord --load-agent $P \
  --total-timesteps 4000000 --num-envs 8 --no-clip-reward --reward-scale 0.02 \
  --episodic-life --value-warmup-updates 5 \
  --bc-traces data/traces/px128_coord_search2_{0,1,2,3}.npz \
    data/traces/px128_coord_search_{0,1}.npz --bc-epochs 0 \
  --bc-anchor-coef 0.5 --bc-anchor-final 0.1 \
  --ent-coef 0.005 --ent-coef-final 0.001 \
  --start-states data/save/levels=0.6,data/save/bag_kills=0.2,data/save/near_death_rescued=0.2 \
  --start-prob 0.5 --archive-every 50 --archive-prob 0.3 --archive-size 200 \
  --archive-perturb 10 --lr 1e-4 --save-best --run-name px128_coord_exit2 \
  > logs/px128_coord_exit2.log 2>&1
$PY evaluate.py $C/px128_coord_exit2/ppo_digger_final.pt --episodes 20 > logs/eval_coord_exit2.txt 2>&1
$PY evaluate.py $C/px128_coord_exit2/ppo_digger_final.pt --episodes 10 --argmax > logs/eval_coord_exit2_argmax.txt 2>&1
$PY evaluate.py $C/px128_coord_exit2/ppo_digger_final.pt --levels data/save/levels --episodes 2 \
  --label px128_coord_exit2 --force-cpu > logs/evlv_px128_coord_exit2.log 2>&1
