# Digger RL — status & next steps

Reinforcement-learning experiments on the 1983 DOS game **DIGGER** running
inside DOSBox Pure via a libretro pybind11 binding. The goal is to learn
to play it from minimal supervision (score + survival time as reward),
ultimately with a *pixel-only* deployable agent.

## TL;DR — current scoreboard

| Approach | n | Mean (sem) | Max ep | Notes |
|---|---:|---:|---:|---|
| **SmartHeuristic v6 (teacher)** | 10 | **1932** | **4600** | v5 + phantom filter + under-bag transit rule + Dijkstra emerald routing; min ep 1000. Same-session vs v5's 1010; absolute level drifts ±400 between sessions (see caveat) |
| **Symbolic BC clone of v6.4 teacher** | 20 | **1175 ±84** | 2050 | **best ML result so far.** 60k samples, 10 epochs, no PPO / no DAgger; 95.4% teacher action accuracy. Same-session teacher 1362 → 86% of teacher. See "the actual headline" below |
| **Pixel BC + 3 DAGGER iters, v6 teacher, 50-ep eval** | 50 | **1042 ±50** | 1950 | **best ML so far**, pixel-only; same net/data budget as the old 629 recipe — the entire delta is the teacher |
| Pixel PPO on the v6 BC ckpt (best ckpt; sep. critic + value warmup + anchor anneal + save-best) | 50 | 851 ±44 | 1700 | PPO still net-negative on pixels, but −18% now vs −55% in the old recipe |
| Pixel PPO on the v6 BC ckpt (final ckpt) | 50 | 750 ±42 | 1425 | final < best: keep --save-best |
| SmartHeuristic v5 (teacher) | 10 | 1010 | 2250 | fresh 10-ep re-bench; the historical "1475" was a 5-ep number |
| Symbolic PPO + BC pretrain + per-step BC anchor + penalties | 20 | 891 | 1475 | best *symbolic* ML (v5-teacher era) |
| Pixel BC-only (100k labels, 20 epochs, v5 teacher), 50-ep eval | 50 | 629.5 ±32 | 1175 | old pixel ceiling at 92.1% teacher acc |
| Pixel DAGGER (50k warmup + 3×25k student-collected), 50-ep eval | 50 | 625.0 ±34 | 1275 | within noise of pure BC; DAGGER iters did not help |
| Audit's pixel DAGGER (multi-iter, GreedyEmerald teacher) | — | 783 | — | the older audit number; not 50-ep checked |
| Pixel PPO + warmup + live teacher anchor + penalties | 20 | 364 | 750 | PPO **degrades** the BC policy |
| Pixel PPO + live teacher anchor (no warmup) | 20 | 324 | 825 | live anchor only |
| Symbolic PPO + shaping only (vanilla, seed=1) | 20 | 290 | 1050 | local-optimum basin at ~250 |
| Symbolic PPO + shaping only (vanilla, seed=2) | 20 | 235 | 500 | same basin |
| Symbolic Dreamer V3 (500k frames) | 20 | 250 | — | actor entropy collapse |
| Pixel PPO from scratch (exp4_long_500k) | 20 | 75 | — | classic NatureCNN, no teacher |

**Caveat on evaluation noise.** Most numbers above are 10–20 episode
in-trainer ep_end averages, which have ±~100 standard error around the
true mean (we measured this: single-episode std is ~230). The pixel BC
result was reported as "812" from a 10-episode eval; a proper 50-ep
re-eval brought it to **629.5 ±32**. Treat 10-ep numbers as ±100;
prefer the `eval_checkpoint.py` 50-ep numbers for any comparison
worth acting on.

**Caveat on cross-session drift.** Worse: 20-episode teacher means
measured on *identical code* moved from 1634/1932 one day to 1226 the
next (three same-day runs of the new code clustered tightly at
1149–1214 while the old code scored 1226 that same day). Whatever the
mechanism (DOSBox timing/monster behaviour vs. system load), the
between-session spread (~±400) dwarfs the within-session sem (~±150).
**Only same-session A/B comparisons are valid**; never compare a
today-number against a yesterday-number. A chunk of the "teacher v6
doubled v5" delta above is real (same-session ablation grid), but the
absolute teacher level varies by session.

Conclusion: the biggest lever, by far, is **teacher quality** (the
same lesson the snake project learned): upgrading SmartHeuristic v5 →
v6 (1010 → 1932, from an ablation-verified interaction of Dijkstra
routing × the under-bag transit rule) lifted the *unchanged* pixel
BC+DAGGER recipe from 629.5 to 1042 — and made DAGGER iterations
productive for the first time. PPO on top of pixel BC remains
net-negative (851 best vs 1042 BC) even with a separate critic,
value-head warmup and an annealed teacher anchor; the residual pixel
perception errors are not something reward-driven refinement can fix.
See **Lessons from PPO/DAGGER** below.

## Layout

| File | Purpose |
| --- | --- |
| `digger_env.py` | `DiggerEnv` (single env, RGBA frames, score/lives via RAM at 0x282E0/0x259F2) and `DiggerVecEnv` (in-proc for `num_envs=1`, subprocess workers for >1). Now also exposes `save_state` / `load_state` via libretro `retro_serialize`. |
| `train_ppo.py` | Pixel PPO (NatureCNN). Supports BC from `.npz` traces *and* live teacher labeling (`--teacher-policy {smart,greedy,dodge}`), warmup phase (`--warmup-steps`), and BC-only mode (`--total-timesteps 0`). |
| `train_ppo_symbolic.py` | Symbolic-obs PPO with the same recipe: BC traces + per-minibatch anchor + death/time penalty + resume-from-state. |
| `tools/collect_death_states.py` | Harvests emulator save-states from N agent steps *before* each death (ring buffer of `retro_serialize` snapshots, ~1 ms / 678 KB each). Captures several lookbacks per death in one pass since collection is dominated by the ~25 s it takes the driver policy to die. `--probe` measures what fraction of the collected states a policy can escape — the ceiling for any curriculum built on them. |
| `eval_symbolic.py` | Symbolic counterpart to `eval_checkpoint.py` (which only rebuilds the pixel NatureCNN). Runs N full 3-life games and reports **both** game score and per-life length in agent steps — the survival experiments need the second, every earlier scoreboard row quotes the first. `--policy random` / `--policy teacher` give same-session baselines. |
| `train_dagger.py` / `train_dagger_pixel.py` | Earlier DAGGER trainers (multi-iter aggregating teacher rollouts). Superseded by the live-teacher anchor path in `train_ppo.py`. |
| `train_dreamer*.py` | Dreamer V3-ish online and symbolic-obs world models. Plateaued; see lessons. |
| `run_digger.py` | Manual play (matplotlib `--live`) + trace recording. **S / L** keys snapshot and restore emulator state (in-memory or disk via `--save-slot`). `--resume` boots straight into a saved scenario. |
| `tools/game_state.py` | `GameState` dataclass + `extract_state_fast(frame)` — vectorised CV extractor, ~4 ms/frame. Now populates `digger.dir` and `BagPos.moving`. |
| `tools/symbolic_env.py` | `SymbolicDiggerEnv` wraps `DiggerEnv`, emits `(6, 10, 15)` mask tensors. Supports `shaping_coef`, `time_penalty`, and `save_state` / `load_state`. |
| `tools/heuristic_agent.py` | `SmartHeuristic v6.3`: v5 (BFS tunnel-distance safety, LoS FIRE, turn-to-fire) + phantom filter, under-bag transit rule, Dijkstra routing (v6), fall-column hazard model from bags.c ground truth (v6.2), retreat-and-ambush firing on chasers (v6.3, same-session 1399 vs 1242). `DeathBagLogger` (`--death-log`) emits JSONL death forensics + bag timing + level-up events. Ablation knobs: `--no-phantom-filter`, `--underbag`, `--routing`, `--fire-turn-min-dist`, `--safety-cap`, `--predict-monsters`. |
| `tools/gen_symbolic_trace.py` | Generates BC traces from a chosen heuristic into `.npz` for offline BC. (Live teacher path in `train_ppo.py` makes this optional.) |
| `probe_*.py` | One-off diagnostics: MPS-vs-CPU correctness, libretro state save/restore. |
| `interaction-log.txt` | Full chronological log of every prompt; the journey. |

## Lessons from PPO / DAGGER experiments

These are the substantive findings from ~10 distinct training runs. Each
took 30–90 minutes; collectively they map out what *actually* matters.

### 1. Pixel-from-scratch RL is hopeless without a teacher

Vanilla pixel PPO with NatureCNN + entropy bonus + shaping plateaued at
~50–75 score across multiple seeds and budgets (exp1, exp4_long_500k,
exp7_wide). The diagnostic signal is consistent: `pg loss ≈ 0`,
`clip_frac ≈ 0`, top action at 60–76% after collapse. PPO can't
extract a learning signal because (a) the encoder hasn't learned what
a digger / monster / emerald *is* yet, and (b) sparse-emerald rewards
average to noise over a random-action rollout.

### 2. Symbolic obs is a 3× improvement out of the box

Same trainer (`train_ppo_symbolic.py`), same recipe, only the
observation changed: pixel-PPO plateaued at 75, symbolic-PPO at 250.
The encoder no longer has to solve "where is everything"; it can spend
all its capacity on "what to do." But 250 is still a **local-optimum
basin** — both seed=1 and seed=2 land there. The diagnostic is the
same `pg ≈ 0, clip ≈ 0` pattern: policy doesn't visit "successful
exploration of level 2" states, so the gradient never points there.

### 3. Reward shaping alone doesn't escape the basin

Adding `--death-penalty 100` and `--time-penalty 0.01` to symbolic
PPO without any imitation did pull the policy out of NOOP-equivalent
attractors, but the run still landed in the 250–400 range. The basin
is *wide*, not just *deep*; small per-step gradients can't bridge it.

### 4. Bootstrap from a teacher is the single biggest lever

Adding **BC pretrain** (10 epochs of CE on 50k SmartHeuristic-driven
samples) **plus per-PPO-minibatch BC anchor** (0.5 → 0.05 anneal) took
symbolic PPO from 250 → 891 mean, with individual episodes hitting
1475 (teacher mean). BC pretrain hits 95.8% teacher-action accuracy
on the symbolic obs; PPO then refines, occasionally exceeding the
teacher's greedy heuristic.

### 5. PPO updates can *degrade* a BC-pretrained pixel policy

The same recipe on pixels collapsed: pixel PPO+warmup+anchor hit only
364, while a **pure BC pixel run with the same data hit 812**. The
diagnostic is clear: the BC pretrain reaches 88–92% teacher accuracy,
but during PPO updates the value-loss and PG noise pull the actor away
from teacher behavior and the score regresses. Death/time penalty,
which *helped* vanilla PPO escape its basin, *hurt* the BC policy.

  - **Pure-BC pixel** = 812 mean
  - **BC + PPO + anchor + penalty pixel** = 364 mean

Why the difference symbolic vs pixel: the symbolic encoder hits 95.8%
BC accuracy and the residual 4.2% errors are *closer* to teacher's own
greedy mistakes, so PPO refinement on top is net-positive. The pixel
encoder caps at 92% accuracy and the residual 8% are perception
errors compounding — PPO refinement can't fix those, and the noise
hurts.

### 6. Auxiliary symbolic-prediction head doesn't help pixel BC

Audit data: `dagger_pixel_aux` (BCE weight 1.0) got 583, `_aux_w03`
got 408, both worse than vanilla pixel DAGGER at 783. The hypothesis
("teach the encoder to recover the tile grid as an auxiliary task")
was tried twice; both weights hurt. The aux head reaches 99.9%
accuracy but the action-prediction features it competes with degrade.

### 7. Wider net hurts at this dataset scale

`dagger_pixel_w2` (NatureCNN width=2, 6.7M params) scored 483 vs
width=1's 783 on the same data. With ~30k aggregated samples that's
~5 samples per parameter; the FC layer memorizes iteration-0 mistakes
and overfits to states the student won't revisit. Wider would help
*if* we had 10× more samples or proper regularization (dropout, weight
decay, augmentation) — none of which the pixel trainer has.

### 8. The pixel BC ceiling is well below the symbolic one

Updated after a proper 50-episode re-eval. Pixel BC at 100k labels +
20 epochs reaches 92.1% teacher-action accuracy → **mean 629.5 score
(±32 sem)** across 50 episodes. The original 812 figure was a
10-episode artifact at the upper end of the ±100 SEM range.

Pure BC and DAGGER iterations on top (50k warmup + 3×25k
student-collected) land at indistinguishable means (629 vs 625, both
±~33 sem). The 8% per-step error rate from a 92% accurate policy
compounds badly over ~1000-step episodes; the teacher needs to be
~98% imitable for the agent to come close to teacher score.

The "pixel perception is the bottleneck" framing holds. The symbolic
BC ceiling (95.8% acc → mean ~890 score from symbolic + PPO) is real
representation headroom, not a fluke.

### 9. DAGGER iterations don't beat pure BC when the teacher is fallible

Standard DAGGER theory predicts iteration helps under covariate
shift. In our setting, with a *greedy* teacher (SmartHeuristic v5),
the teacher's labels on student-visited states are not reliably
better than its labels on its own trajectories — when the student
ends up in a corner the teacher would have avoided, the teacher's
greedy rule still picks reasonable but not necessarily recovery-
optimal actions. Result: 50k warmup + 3×25k DAGGER iters lands within
noise of plain 100k warmup + 20 epochs. Different mixing strategies
(e.g. disagreement-only sampling) could help; we haven't tried them.

### 10. MPS-vs-CPU is a non-issue

A single-seed PPO run on MPS underperformed CPU by 3×, suggesting a
framework bug. A 3-seed sweep showed MPS *winning* on the same
metric. A 50-trial Categorical-sampling probe found no statistically
significant bias on MPS. Conclusion: MPS's `multinomial` uses an
independent PRNG stream, so device-vs-device seed comparisons are not
meaningful, but within-device training is fine on either. The per-op
forward/backward math is bit-identical between MPS and CPU.

For pixel training (where CE on a NatureCNN dominates wall time), MPS
shaves ~40% off the warmup phase. For symbolic training (where DOSBox
is the bottleneck), MPS is a wash. **No reason to use `--force-cpu`
anymore unless reproducibility against a CPU baseline matters.**

### 11a. Teacher quality is the biggest lever (the snake lesson, confirmed)

SmartHeuristic v6 = v5 + three changes: (a) *phantom filter* — monster
detections on dirt tiles are CV artifacts (nobbins can't be inside
dirt) and are ignored, so no more fake dodges or wasted 50-step FIRE
cooldowns; (b) *under-bag transit rule* — dirt directly below an
intact bag may be dug through horizontally but never entered from
below (UP) and never lingered in (NOOP); (c) *Dijkstra emerald
routing* — moves are scored against a multi-source distance field
(dirt cost 2, bags block, under-bag dirt +6 surcharge) instead of
straight-line Manhattan.

The ablation grid (10 eps each) shows a strong interaction: neither
(b) nor (c) helps alone — each is actively harmful — but together
they double the teacher:

| Config | Mean |
|---|---:|
| phantom + transit + Dijkstra (**v6, shipped default**) | **1932** |
| v5 baseline (fresh re-bench) | 1010 |
| phantom only | 945 |
| phantom + transit + Manhattan | 832 |
| strict under-bag block (emeralds under bags unreachable) | 622 |
| Dijkstra only (routes into under-bag dirt) | 592 |

The router is what *exploits* the transit rule: it plans multi-tile
paths that dig under bags safely and detour around them; Manhattan
can't look far enough ahead to use the rule, and Dijkstra without the
rule plans paths that drop bags on the digger.

Downstream, the same BC recipe (identical net, 100k+3×25k budget,
92% train acc both times) went 629.5 → **1042** — the entire delta is
teacher score at constant imitability.

### 11b. DAGGER works once the teacher is worth imitating

With the v5 teacher, DAGGER iterations were within noise of pure BC
(Lesson 9). With v6, mid-evals climbed 755 → 965 → 1190 across three
iterations, and rollout teacher-agreement rose 59.5% → 64.3%. The
covariate-shift gap was always real; v5's labels on student-visited
states were just not good enough to close it.

### 11c. PPO on pixel BC: still net-negative, but the gap narrowed

PPO from the 1042 BC checkpoint with every anti-degradation lever we
now have — `--separate-critic` (value gradients can't touch the BC
encoder), `--bc-value` MC-return critic warmup + head rescale (no
initial value-loss shockwave), live v6 anchor annealed 0.3 → 0,
`--save-best` — evals at 851 ±44 (best ckpt) / 750 ±42 (final).
Training diagnostics were healthy the whole run (pg ≠ 0, clip
0.05–0.35, entropy 0.5–1.3, tiny value loss, rolling score rising
682 → 832), yet the stochastic-eval score is still ~3σ below BC.
Compare the old recipe's 812 → 364 (−55%): the levers cut the
degradation to −18% but didn't flip the sign. The pixel policy's 8%
per-step perception errors remain something PPO refinement cannot
fix — it can only trade teacher-shaped behaviour away against them.
`--save-best` earned its keep: final < best by 100 points.

### 12. DOSBox save-state is gameplay-valid but not byte-exact

Added libretro `retro_serialize` / `retro_unserialize` to the pybind
binding plus `DiggerEnv.save_state` / `load_state` on top. The visible
game (score, lives, digger position, monsters, dirt) round-trips
faithfully, so we can train from saved scenarios via `--resume-from`.
But two back-to-back restores + the same next action diverge after
~3 frames (1927 pixels differ) — DOSBox's internal timer / audio /
JIT state isn't fully serialized. So Go-Explore-style restart works;
deterministic policy A/B from a saved frame does not.

### 12b. Restore is input-live in gameplay states, dead inside the death sequence

Building the near-death curriculum turned up a sharp qualifier on the
above. Restoring an ordinary mid-play state and then pressing a key
works exactly as it should — controlled against a no-restore baseline
from the same position:

```
no-restore, LEFT (control)   [(1,4) ... (1,3), (1,3), (1,3)]
restore + LEFT               [(1,4) ... (1,3), (1,3), (1,3)]   matches control
restore + RIGHT              [(1,4), (1,5), (1,5) ... (1,6)]   diverges as it should
```

But a state captured a few agent steps before the `lives` counter
decrements is **frozen**: LEFT, RIGHT and NOOP produce identical
trajectories and the life is lost at exactly the same step. Those
states are inside Digger's death sequence — the outcome is already
sealed and input is ignored until the respawn.

The trap is that this is invisible from the `lives` counter. (It is *not* invisible in RAM: the death-stage word at `0x259E4` flips the moment the death starts; see 12c.) `lives` only
decrements when the sequence *finishes*, and the death animation still
renders a digger-coloured sprite, so `GameState.digger.present` stays
true throughout. Trying to measure the animation length by watching for
the sprite to vanish returns a median gap of **0** and tells you
nothing. Only replaying a restored state under different actions
exposes it.

Practical rule: **any curriculum built on save-states must validate
escapability by replaying under at least two different action
sequences.** A scenario set where every policy dies at an identical
fixed step is a set of corpses, not a set of hard problems — and it
looks exactly like a set of hard problems in every metric that doesn't
involve replaying it.

### 12c. ~~Save-states lose the digger's position across processes~~ — retracted: it was a stale framebuffer plus a late death anchor

An earlier version of this section claimed a `retro_serialize` blob
loaded into a freshly booted DOSBox brings the digger back at the spawn
point. **That was wrong.** Reading the game's own RAM settles it:

- **Game RAM round-trips byte-exactly across processes** (0 of 649,120
  bytes differ after `load_state` in a new process), and LEFT/RIGHT from a
  cross-process restore produce exactly the trajectories they produce
  in-process.
- **The framebuffer is not part of the state.** For ~3 emulator frames
  after `unserialize`, `get_frame()` still returns whatever the core last
  drew: the spawn screen in a fresh process (the "(9, 7)" in the old
  table), or the pre-restore screen in-process. `DiggerEnv.load_state`
  now runs 3 frames before returning, so the first observation after a
  restore matches the restored game.
- **The CV extractor sometimes reads a nobbin as the digger** (a frame
  with the digger visibly at (1, 4) parsed as (6, 10), where a nobbin
  was). Checking positions through CV made the confusion worse.
- **What actually sealed every scenario (12b) is the death anchor.**
  The digger's record lives right next to `lives` in RAM (`0x259C8` x,
  `0x259CA` y, `0x259CC` column, `0x259CE` row, `0x259E4` death stage:
  1 = alive). A death *starts* ~100 agent steps (~6 s) before `lives`
  goes down: bag or monster hit, death animation, then a 60-tick
  tombstone countdown, all with input ignored. Anchored on the lives
  decrement, lookbacks of 15/25/40/60 were all inside that window:
  **320 of 320** scenarios in `death_bcclone_far.pkl` load with the
  death stage already off "alive".

`DiggerEnv.step` now returns `digger_rc`, `digger_xy`, `death_stage` and
`dying` in `info`, read straight from RAM. `collect_death_states` anchors
on death onset. Re-collected that way (teacher driving, 12 deaths), the
scenarios respond to input, and escapability rises with lookback as a
real curriculum should:

| lookback before onset | teacher survives 50 steps | random survives |
|---:|---:|---:|
| 3 | 0/12 | 0/12 |
| 5 | 1/12 | 1/12 |
| 10 | 2/12 | 1/12 |
| 20 | **6/12** | 3/12 |

Same-process and cross-process save/restore both work, so a scenario
recorded by hand in `run_digger.py --live` (S with `--save-slot`) is a
valid `--resume-from` start again.

### 12d. Deaths are now detected at onset, not at the lives decrement

`DiggerEnv` reports `death_event` on the frame the RAM death stage
leaves "alive". With `episodic_life` the episode ends there, the death
penalty lands on that step, and `reset()` plays the ~100 input-ignored
steps of the death sequence out before handing back the respawned
digger. A death onset on the last life sets `real_done` at once, since the
game's outcome is already sealed. Both frame-skip helpers OR the one-frame
flag across sub-frames. Without that, 3 deaths in 4 were dropped.
Measured with NOOP: an episodic life went from ~730 to ~350 emulator
frames, and the difference is all dead time.

**Life lengths before this change are inflated.** `eval_symbolic.py`
used to measure from one lives drop to the next, which counts the
previous death sequence. Measured at onset, a random policy's life is
**~96 agent steps** (was 194). The "life length" column in the
survival-reward table below is in the old units: subtract ~100 for
comparison. The ordering is unaffected.

## Lessons ported from ../snake (2026-08-31)

The snake project ran the same recipe on a much cheaper simulator and
got 40× the experiment count out of it. Its README's findings #16-18 are
about *architecture*; here is which of them transfer to Digger and which
do not.

| Snake finding | Transfers? | Why / what it becomes here |
|---|---|---|
| #16 egocentric (head-centred) obs beats allocentric | **yes** | Same failure mode is latent in `SymbolicAgent`: `Flatten -> Linear(64*10*15, 128)` gives every board position private weights, so "monster one tile to my left" has to be relearned at each of the 150 positions. `--egocentric` pastes the 10×15 board into a 19×29 canvas with the digger pinned at the centre, plus an off-board plane. |
| #17 weight-tied conv iterator + head-local readout | **yes** | The 3-conv trunk has a 7-tile receptive field on a 15-wide board — it physically cannot relate the digger to an emerald across the map. `--arch iter` applies one shared residual block `--iter-steps` times (range grows with iterations, not parameters) and reads out a 7×7 window around the centre + a global mean-pool. **232k params vs 4.58M** for the egocentric CNN. |
| #17/#18 canonical (rotate-to-face-up) obs | **no** | Snake is rotation-symmetric; Digger is not. Gravity is real here — bags fall *down*, the under-bag transit rule is a statement about the vertical axis. Rotating the board would destroy the single most safety-relevant structure. Egocentric translation invariance is the part that survives. |
| #18 drop the engineered distance channel, let the net infer routing | **untested here, plausible** | Digger's symbolic obs has no distance channel to drop, but the analogue is `--shaping-coef` (Manhattan-to-emerald reward shaping). If the iterator can route by itself, the shaping term is a crutch at best and a distractor at worst. |
| #9 BC capacity wall — 15k too small, 300k-1.2M memorises, 4.78M generalises | **caution** | Read together with our own Lesson 7 (width-2 NatureCNN scored 483 vs width-1's 783) the resolution is *samples per parameter*, not parameter count: snake's 4.78M win came with 600k samples. Our pixel runs had 30k. Scale the net only alongside the DAgger budget. |
| #11 DAgger overtrains past a saturation point | **yes** | Matches our Lesson 11b, where mid-evals were still climbing at iter 3. The rule is the same: watch rollout agreement, early-stop when it stalls, don't fix the iteration count in advance. |
| #12/#15 `--save-best`, γ=0.997 for long horizons, vectorised envs | **yes** | `--save-best` was already worth 100 points on the pixel PPO run. γ=0.997 matters as soon as the objective is survival (see below). Vectorised symbolic envs are still not wired — one DOSBox process per env makes it the expensive one. |
| #14 eval seeds must be disjoint from collection seeds | **N/A but watch it** | Digger has no seed stream we control; DOSBox timing supplies the variation. Our equivalent hazard is the cross-session drift caveat above. |
| #15 pure reward learns new behaviour *once a navigation core exists* | **the load-bearing one** | Snake's from-scratch PPO never learned to play (ppo04: 1.4), but reward-only *fine-tunes* from a competent base learned genuinely new behaviour (2-apple routing 37 → 53; death rate 82% → 12%). That is exactly the shape of the survival experiment below, and the reason it is run both from scratch and from a BC init. |

## Reward strategies: what is the agent actually being paid for?

Every run before this one was paid in game score (plus emerald-distance
shaping). `SymbolicDiggerEnv` now decomposes the reward so the objective
itself is an experimental variable:

```
reward = score_weight * (game score delta)
       + shaping_coef * (Manhattan distance closed toward an emerald)
       + survival_reward     (per agent step that didn't end a life)
       - time_penalty
       - death_penalty       (on any life loss)
```

`--score-weight 0 --shaping-coef 0 --survival-reward 0.01` is the pure
**"maximise frames survived"** objective: the game's score is invisible
to the agent and the only thing that matters is not dying.

Two mechanics make this work:

- **γ=0.997, not 0.99.** With a constant per-step reward the return *is*
  the (discounted) episode length, so the discount horizon is the
  objective. At γ=0.99 the agent cannot see past ~100 steps — shorter
  than a single teacher life.
- **`--max-episode-steps` + truncation-aware GAE.** A survival-paid
  policy that finds a safe corner would otherwise run one episode for
  the whole job. The cap bounds it, and GAE bootstraps `V(s')` at the
  cut instead of treating it as death — without that the agent is taught
  that surviving to the cap is exactly as bad as dying there.

**The number to beat:** the v6 teacher survives a mean of **280 agent
steps per life** (median 267, max 488, n=60 from
`logs/deathlog_v6_baseline.jsonl`) — and it is *trying* to die, in the
sense that it walks into risk to collect emeralds. A random policy
already survives ~195. Survival-only training that lands between those
has learned nothing; the interesting question is whether it clears the
teacher, and whether it does so by playing well or by hiding.

### Result: survival-only reward is strictly dominated (2026-08-31)

Six 500k-step arms, all launched the same session, evaluated the same
session with `eval_symbolic.py` over 20 full 3-life games each
(stochastic sampling). Life length is agent steps between deaths, n=60
lives per row.

| Policy | Game score | Life length | Notes |
|---|---:|---:|---|
| SmartHeuristic v6.4 (teacher) | **1362 ±133** | **266.6 ±8.8** | same-session reference |
| **Symbolic BC clone, v6.4 teacher** | **1175 ±84** | **259.2 ±8.9** | 95.4% teacher acc, **no PPO at all** |
| `score_ctrl` (old recipe, no BC) | 712 ±21 | 230.6 ±1.1 | see the degenerate-loop note below |
| `surv_ego` (survival only, egocentric) | 312 ±52 | 237.2 ±6.2 | best survival arm |
| `surv_score` (survival + score×0.02) | 282 ±44 | 227.8 ±3.2 | |
| `surv_bcinit` (survival only, **from the BC clone**) | 239 ±65 | 210.8 ±7.0 | started at 1175/259 |
| `surv_iter` (survival only, iterator) | 155 ±28 | 203.3 ±2.9 | collapsed to NOOP @55% |
| `surv_alloc` (survival only, allocentric) | 146 ±29 | 215.8 ±3.8 | never left uniform policy |
| Random policy | 124 ±44 | 194.4 ±4.8 | same-session floor |

Reading the life-length column: **the arms trained to maximise lifetime
are worse at lifetime than the arms that were never trained for it.**
The BC clone, which only ever imitated an emerald-collecting teacher,
survives 259 steps; the best pure-survival arm manages 237, and the
allocentric one 216 against a random floor of 194.

`surv_bcinit` is the decisive run. It *started* from the 1175-score /
259-life BC clone and 500k steps of anchor-free survival PPO took it to
239 / 211 — it made the policy worse at the exact quantity it was being
paid for.

**Why: standing still is the degenerate optimum, and Digger punishes it
only weakly.** A constant `+c` per step is identical across all six
actions, so the only thing that distinguishes them is the truncated tail
at death — a tiny, heavily delayed advantage. Tracking the modal action
across training tells the story:

| Arm | first 20 updates | last 20 updates |
|---|---|---|
| `surv_alloc` | UP @23% | DOWN @24% (still uniform) |
| `surv_ego` | UP @21% | LEFT @27% |
| `surv_iter` | RIGHT @23% | **NOOP @55%** |
| `surv_bcinit` | UP @44% | **NOOP @38%** |
| `score_ctrl` | NOOP @22% | **FIRE @61%** |

The two survival arms that never committed stayed at entropy 1.72-1.74
against a 1.79 maximum — a barely-perturbed uniform policy. The two that
did commit, committed to NOOP. `--max-episode-steps 2000` was never hit
in any of the six runs, so hiding never even survived long enough to be
truncated.

**Snake's finding #15 was mis-ported, and that is the lesson.** Snake's
death-averse fine-tune was `--reward-eat 1 --reward-die -5
--reward-step 0.005` — it *kept the task reward*, and the step bonus was
~8% of the food reward. Snake never ran a zero-task-reward arm. Survival
is a useful *auxiliary* term on top of a task reward; as the sole
objective it selects for inaction.

Two side-findings worth recording:

- **`score_ctrl` is a degenerate loop, not a policy.** Its best-by-score
  checkpoint evals at 712 — respectable next to the README's 250-290 for
  the same recipe — but its life length is 230.6 **±1.1**, median 232,
  max 239. Sixty lives all ending within nine steps of each other is a
  fixed action cycle (FIRE @61%) that dies on a timer, not a policy that
  reacts. In-trainer game score also regressed 175 → 51 over the run;
  `--save-best` is the only reason there is a usable checkpoint at all.
- **The architecture A/B is inconclusive *from these runs*.** Under
  identical survival reward, egocentric CNN (237.2 life) beat
  allocentric (215.8) beat iterator (203.3), which is the predicted
  ordering for the first two — but all three sit in the
  "learned-almost-nothing" band, so the comparison is measuring which
  net degrades most gracefully under a no-signal objective, not which
  encodes Digger better. The architecture question has to be re-asked in
  the BC setting, where there is real signal (in flight).

### The actual headline: symbolic BC on the v6.4 teacher

The control run buried in the table above is the best symbolic result
this project has produced. **A plain BC clone of SmartHeuristic v6.4 —
60k samples, 10 epochs, no PPO, no DAgger — evaluates at 1175 ±84 game
score, 86% of the same-session teacher's 1362.** For comparison the
previous symbolic best was 891 (BC + PPO + anchor, v5-teacher era) and
the best pixel result is 1042.

This confirms Lesson 11a from the other direction: the teacher upgrade
propagates straight through to the student, and on symbolic obs the
imitability is high enough (95.4% action accuracy) that BC alone gets
most of the way. It also re-frames every PPO result here — the bar for
"PPO helped" is now 1175, not 891.

## How to run

### Watch the heuristic play

```bash
python -m tools.heuristic_agent --live --smart
python -m tools.heuristic_agent --episodes 10 --no-episodic-life --smart   # ~1475 mean
```

### BC + DAGGER with the v6 teacher (the best pixel recipe: 1042 ±50)

```bash
python train_ppo.py \
  --total-timesteps 0 \
  --warmup-steps 100000 --warmup-epochs 20 \
  --dagger-iters 3 --dagger-collect-steps 25000 --dagger-epochs 5 \
  --teacher-policy smart --bc-batch-size 256 \
  --episodic-life --run-name pixel_bc_v6teacher

# Then run a tighter eval (the in-trainer 10-ep number is too noisy):
python eval_checkpoint.py \
  data/checkpoints/pixel_bc_v6teacher/ppo_digger_bc_only.pt --episodes 50
```

### PPO on top of the BC checkpoint (evals below BC; see Lesson 11c)

```bash
python train_ppo.py \
  --total-timesteps 600000 \
  --load-agent data/checkpoints/pixel_bc_v6teacher/ppo_digger_bc_only.pt \
  --separate-critic \
  --warmup-steps 25000 --warmup-epochs 5 --bc-value \
  --teacher-policy smart --bc-anchor-coef 0.3 --bc-anchor-final 0.0 \
  --bc-batch-size 256 --episodic-life --save-best \
  --run-name pixel_ppo_v6teacher
```

### Symbolic BC clone of the teacher (best ML result: 1175 ±84)

```bash
python -m tools.gen_symbolic_trace --out data/traces/sym_smart_v64_60k.npz \
    --steps 60000 --teacher smart --frame-stack 4

# --total-timesteps 0 runs BC and skips PPO entirely.
python train_ppo_symbolic.py --total-timesteps 0 \
  --bc-traces data/traces/sym_smart_v64_60k.npz --bc-epochs 10 \
  --run-name ppo_sym_bc_v64

python eval_symbolic.py data/checkpoints/ppo_sym_bc_v64/ppo_sym_final.pt \
    --episodes 20 --label bc_only
# Always collect same-session baselines alongside it -- cross-session
# numbers are not comparable (see the drift caveat above):
python eval_symbolic.py --policy teacher --episodes 20
python eval_symbolic.py --policy random  --episodes 20
```

### Survival-only symbolic PPO (score-blind) — negative result, see above

```bash
python train_ppo_symbolic.py --total-timesteps 500000 \
  --score-weight 0 --shaping-coef 0 --survival-reward 0.01 \
  --gamma 0.997 --max-episode-steps 2000 \
  --save-best --best-metric length --run-name ppo_sym_surv_alloc

# Same objective, egocentric obs (19x29 digger-centred canvas):
#   ... --egocentric --run-name ppo_sym_surv_ego
# Same again on the 232k-param weight-tied iterator:
#   ... --egocentric --arch iter --run-name ppo_sym_surv_iter
```

### Symbolic PPO with BC pretrain + anchor (the best symbolic recipe)

```bash
# 1. Generate offline BC trace from teacher
python -m tools.gen_symbolic_trace --out data/traces/sym_smart_50k.npz \
    --steps 50000 --teacher smart --frame-stack 4

# 2. Train
python train_ppo_symbolic.py \
  --total-timesteps 500000 --num-steps 256 \
  --bc-traces data/traces/sym_smart_50k.npz \
  --bc-epochs 10 --bc-batch-size 256 \
  --bc-anchor-coef 0.5 --bc-anchor-final 0.05 \
  --death-penalty 100 --time-penalty 0.01 \
  --shaping-coef 0.5 --frame-stack 4 \
  --force-cpu --run-name ppo_sym_bc_v1
```

### Capture near-death scenarios and replay/train from them

The capture side harvests emulator save-states from N agent steps
*before* each death; the replay side either probes them or trains from
them. `--lookbacks` takes a list because all of them come out of one
ring buffer — collection is dominated by the ~25 s it takes the driver
policy to die, so extra lookbacks are free.

```bash
# 1. CAPTURE. 80 deaths x 4 lookbacks = 320 scenarios (~217 MB).
#    --policy ckpt collects states the *student* dies in (on-distribution);
#    --policy teacher / random also work.
python -m tools.collect_death_states \
  --out data/scenarios/death_bcclone_far.pkl \
  --deaths 80 --lookbacks 15,25,40,60 \
  --policy ckpt --checkpoint data/checkpoints/ppo_sym_bc_v64/ppo_sym_final.pt

# 2. PROBE -- ALWAYS DO THIS BEFORE TRAINING. Replays each scenario and
#    reports, per lookback, what fraction the policy escapes.
python -m tools.collect_death_states \
  --out data/scenarios/death_bcclone_far.pkl --probe \
  --policy ckpt --checkpoint data/checkpoints/ppo_sym_bc_v64/ppo_sym_final.pt \
  --probe-steps 30 --probe-limit 120

# 3. TRAIN from the scenario set. Each reset() samples one uniformly.
python train_ppo_symbolic.py --total-timesteps 400000 \
  --resume-from data/scenarios/death_bcclone_far.pkl \
  --resume-prob 0.5 \
  --load-agent data/checkpoints/ppo_sym_bc_v64/ppo_sym_final.pt \
  --score-weight 0 --shaping-coef 0 --survival-reward 0.01 \
  --gamma 0.99 --max-episode-steps 40 \
  --save-best --best-metric length --run-name ppo_sym_deathcur
```

**Read the probe output before training on the set.** A survival rate of
0% at *every* lookback with *identical* steps-to-death across policies
means the states are inside the death sequence and no action can change
the outcome (Lesson 12b) — training on them produces a guaranteed null.
Raise `--lookbacks` until the rate goes above zero, and use the shortest
lookback that does.

Two knobs that matter:

- `--resume-prob 0.5` mixes ordinary level-start episodes back in.
  Training purely on near-death states teaches escape at the cost of
  forgetting the rest of the game.
- **Do not add a BC anchor here.** These are precisely the states where
  the driver policy's action was wrong; anchoring to it re-teaches the
  mistake. Use `--load-agent` for the init and leave
  `--bc-anchor-coef` at 0.

### Save / restore game state

```bash
# Play to an interesting scenario, press S, close window
python run_digger.py --live --save-slot scenarios/right_edge.pkl

# Boot straight back into that scenario
python run_digger.py --live --save-slot scenarios/right_edge.pkl --resume

# Train PPO from that scenario
python train_ppo_symbolic.py --resume-from scenarios/right_edge.pkl --force-cpu \
  --total-timesteps 500000 --run-name ppo_sym_right_edge
```

## Open paths

1. ~~DAGGER iterations on the BC-only pixel model~~ — done with the v6
   teacher: 1042 ±50 ("plausibly reach 1000+" confirmed).
2. **Symbolic BC + PPO with the v6 teacher** — the old symbolic recipe
   (891) was built on v5 traces. Symbolic BC reaches 95.8% teacher
   accuracy and PPO refinement is net-*positive* there (Lesson 5); with
   a 1932-mean teacher the symbolic path could plausibly clear 1500.
   Cheapest high-upside experiment left.
3. **Higher-resolution pixel obs** (`--obs-size 168`) — 4× more pixels
   per tile; digger sprite goes from 1–2 px to 4 px. The 92% pixel BC
   accuracy ceiling (unchanged across both teachers) is where the next
   pixel gain lives; more DAGGER rounds won't move it (rollout
   agreement was still only 64% after three).
4. **More / adaptive DAGGER rounds** — mid-evals were still climbing at
   iter 3 (755 → 965 → 1190). Snake's rule applies: keep going while
   rollout agreement rises, early-stop when it stalls.
5. **Curriculum from saved scenarios** — capture "stuck at right edge"
   states via `--live S`, train from those. `--resume-from` already
   wired in `train_ppo_symbolic.py`.

## Known open problems (from the heuristic era)

Both long-standing teacher problems were closed by SmartHeuristic v6
(see Lesson 11a): phantom border monsters are filtered by the
they-can't-be-in-dirt rule (teacher-side; `extract_state_fast` still
emits them, so symbolic obs are unchanged), and the "don't dig under a
bag" rule is wired as the under-bag transit rule. One caveat inherited
by the phantom filter: *hobbins* genuinely dig through dirt, so the
filter also blinds the teacher to a hobbin mid-dirt; level 1 is
nobbin-dominated so this is currently a good trade
(`--no-phantom-filter` to ablate).

## Key constants worth knowing

- Tile grid: 15 cols × 10 rows (`MWIDTH`, `MHEIGHT` in `game_state.py`).
- Score RAM offset: `0x282E0` (int32 LE). Lives: `0x259F2` (uint8).
- Frame dimensions: 640×400 RGBA. Score bar takes top 32 px; play area is `(368, 640)`.
- Palette: exactly 11 RGB triples, no antialiasing. See `PAL_*` constants in `game_state.py`.
- `frame_skip = 4`, `frame_stack = 4`, `obs_size = 84` for pixel; 64 for Dreamer; none for symbolic.
- Action space: `{NOOP, LEFT, RIGHT, UP, DOWN, FIRE}` with FIRE = F1 in the libretro keyboard layout.
- `BASE_OBS_CHANNELS = 6` for symbolic: dirt / emerald / digger / monster / bag / cherry.

## Recent significant commits

- `743ce33` — DAGGER iterations (`--dagger-iters`) for BC-only mode
- `f01ff07` — BC-only mode (`--total-timesteps 0` skips PPO)
- `0d3ee6d` — BC warmup phase before PPO (`--warmup-steps`, `--warmup-epochs`)
- `75f84f5` — live teacher labeling (`--teacher-policy`) for pixel DAGGER
- `9bdcf11` — symbolic PPO: BC pretrain + anchor + reward shaping knobs
- `3c7dd75` — symbolic PPO `--resume-from <pickle>` for scenario training
- `ca15ddf` — `run_digger.py --live` S/L key bindings for state save/restore
- `d0747e1` — `DiggerEnv.save_state` / `load_state` via libretro serialize
- `3e4d51f` — MPS-vs-CPU diagnostic probes (rules out framework bug)
- `0bc186e` — symbolic-obs PPO trainer
- `d5566e1` — SmartHeuristic v5: dirt-aware FIRE LoS + safe turn-to-fire
- `bb41a87` — digger direction + falling-bag detection + turn-to-fire

The `interaction-log.txt` has every prompt in chronological order if
you want to retrace specific decisions.
