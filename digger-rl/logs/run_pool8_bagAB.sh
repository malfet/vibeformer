#!/bin/zsh
# Ablation of the failed pool8_bag1 round: A keeps the search-trace anchor
# (new speed settings), B drops it but uses the old settings (8x128, fp32).
cd /Users/malfet/git/malfet/vibeformer/digger-rl
PY=.venv/bin/python
C=data/checkpoints
P=$C/px128_pool8_exit2/ppo_digger_final.pt
BASE=(--obs-size 128 --arch impala_pool8 --load-agent $P --total-timesteps 4000000
  --no-clip-reward --reward-scale 0.02 --episodic-life --value-warmup-updates 5
  --ent-coef 0.005 --ent-coef-final 0.001
  --start-states data/save/levels=0.5,data/save/near_death_rescued=0.3,data/save/bag_kills_p8=0.2
  --start-prob 0.5 --archive-every 50 --archive-prob 0.3 --archive-size 200
  --archive-perturb 10 --lr 1e-4 --save-best)
one() {
  local n=$1; shift
  $PY train_pixel.py $BASE "$@" --run-name px128_$n > logs/px128_$n.log 2>&1
  F=$C/px128_$n/ppo_digger_final.pt
  $PY evaluate.py $F --episodes 20 > logs/eval_$n.txt 2>&1
  $PY evaluate.py $F --episodes 10 --argmax > logs/eval_${n}_argmax.txt 2>&1
  $PY evaluate.py $F --levels data/save/levels --episodes 2 --label px128_$n \
    --force-cpu > logs/evlv_px128_$n.log 2>&1
}
one pool8_bagA --num-envs 16 --num-steps 64 --amp \
  --bc-traces data/traces/px128_pool8_search2_{0,1,2,3}.npz --bc-epochs 0 \
  --bc-anchor-coef 0.3 --bc-anchor-final 0.1 &
one pool8_bagB --num-envs 8 --num-steps 128 &
wait
