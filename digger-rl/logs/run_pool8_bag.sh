#!/bin/zsh
# pool8 bag-kill round (mirrors bag1): bag-kill starts from pool8_exit2's
# own play, then PPO with no imitation anchor. New speed settings.
set -e
cd /Users/malfet/git/malfet/vibeformer/digger-rl
PY=.venv/bin/python
C=data/checkpoints
P=$C/px128_pool8_exit2/ppo_digger_final.pt
for s in 0 1 2 3 4 5 6 7 8 9; do
  $PY -m tools.bag_kills $P --out-dir data/save/bag_kills_p8 --kills 20 \
    --max-steps 20000 --seed $((s+20)) > logs/p8_bag_kills_$s.log 2>&1 &
done; wait
ls data/save/bag_kills_p8 | wc -l
$PY train_pixel.py --obs-size 128 --arch impala_pool8 --load-agent $P \
  --total-timesteps 4000000 --num-envs 16 --num-steps 64 --amp \
  --no-clip-reward --reward-scale 0.02 \
  --episodic-life --value-warmup-updates 5 --ent-coef 0.005 --ent-coef-final 0.001 \
  --start-states data/save/levels=0.5,data/save/near_death_rescued=0.3,data/save/bag_kills_p8=0.2 \
  --start-prob 0.5 --archive-every 50 --archive-prob 0.3 --archive-size 200 \
  --archive-perturb 10 --lr 1e-4 --save-best --run-name px128_pool8_bag1 \
  > logs/px128_pool8_bag1.log 2>&1
F=$C/px128_pool8_bag1/ppo_digger_final.pt
$PY evaluate.py $F --episodes 20 > logs/eval_pool8_bag1.txt 2>&1
$PY evaluate.py $F --episodes 10 --argmax > logs/eval_pool8_bag1_argmax.txt 2>&1
$PY evaluate.py $F --levels data/save/levels --episodes 2 --label px128_pool8_bag1 \
  --force-cpu > logs/evlv_px128_pool8_bag1.log 2>&1
