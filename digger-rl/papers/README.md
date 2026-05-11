# Papers

Reference papers for the `digger-rl` project: training an RL agent to play
video games (specifically DOS-era titles like Digger) from pixels with a
minimal reward signal (score + survival time).

## Included

### `dreamerv3-2301.04104.pdf`
**Mastering Diverse Domains through World Models** — Hafner, Pasukonis, Ba,
Lillicrap (2023; Nature 2025).
[arXiv:2301.04104](https://arxiv.org/abs/2301.04104) ·
[code](https://github.com/danijar/dreamerv3)

Learns a recurrent latent world model (RSSM) from pixels, then trains an
actor-critic *inside* imagined rollouts of that model. The headline result is
that a single configuration handles 150+ tasks — Atari, DMControl, Crafter,
Minecraft (first method to mine diamonds from scratch, no human data) — with no
per-task tuning. The practical wins for our setting: high sample efficiency
versus model-free methods, robustness to sparse rewards, and a stable training
recipe (symlog rewards, two-hot critic, fixed entropy schedule) that makes
"point it at a new game and let it run" actually work.

### `twister-2503.04416.pdf`
**Learning Transformer-based World Models with Contrastive Predictive Coding**
— (2025). [arXiv:2503.04416](https://arxiv.org/abs/2503.04416)

Replaces the RSSM recurrent backbone with a transformer world model trained
with a contrastive (CPC-style) objective on top of the standard reconstruction
loss. Reports 162% human-normalized mean on Atari 100k without look-ahead
search — current SOTA for that sample-efficient regime. Relevant because (a)
transformer backbones scale better than GRUs if we want a single agent across
many games, and (b) the contrastive auxiliary loss helps with the kind of
sparse-reward, visually-noisy environments DOS games tend to be.

## Suggested additional reading

Listed roughly in the order I'd read them.

### Foundations (read first)
- **Playing Atari with Deep Reinforcement Learning** — Mnih et al., 2013.
  [arXiv:1312.5602](https://arxiv.org/abs/1312.5602). The original DQN. Pixels
  in, score-delta as reward, no hand-engineered features — exactly the regime
  we want.
- **Human-level control through deep RL** — Mnih et al., Nature 2015.
  [link](https://www.nature.com/articles/nature14236). The polished DQN
  (target net, replay, 49 games, same hyperparams).

### Exploration when score is sparse
- **Curiosity-driven Exploration by Self-supervised Prediction** — Pathak et
  al., ICML 2017. [arXiv:1705.05363](https://arxiv.org/abs/1705.05363) ·
  [project](https://pathak22.github.io/noreward-rl/). ICM forward-model
  prediction error as intrinsic reward. Learns Super Mario Level 1 with no
  extrinsic reward at all.
- **Exploration by Random Network Distillation** — Burda et al., 2018.
  [arXiv:1810.12894](https://arxiv.org/abs/1810.12894). Simpler curiosity
  signal than ICM; first to beat human on Montezuma's Revenge.
- **Never Give Up / Agent57** — Badia et al., 2020.
  [arXiv:2003.13350](https://arxiv.org/abs/2003.13350). State-of-the-art on
  the hardest exploration Atari games; combines episodic and lifelong novelty.

### World models lineage (context for what's in this folder)
- **World Models** — Ha & Schmidhuber, 2018.
  [arXiv:1803.10122](https://arxiv.org/abs/1803.10122). The original
  VAE + RNN + controller idea.
- **Dreamer (V1)** — Hafner et al., ICLR 2020.
  [arXiv:1912.01603](https://arxiv.org/abs/1912.01603). Learning behaviors by
  latent imagination, continuous control.
- **DreamerV2** — Hafner et al., ICLR 2021.
  [arXiv:2010.02193](https://arxiv.org/abs/2010.02193). Discrete latents;
  first world-model agent to hit human-level on Atari.
- **IRIS** — Micheli et al., ICLR 2023.
  [arXiv:2209.00588](https://arxiv.org/abs/2209.00588). Transformer world
  model with a discrete autoencoder; useful prior art for TWISTER's design
  choices.

### Alternative SOTA (model-based + search)
- **MuZero** — Schrittwieser et al., Nature 2020.
  [arXiv:1911.08265](https://arxiv.org/abs/1911.08265). Learns a latent model
  and plans with MCTS. Strong on Atari but much heavier than Dreamer.
- **EfficientZero** — Ye et al., NeurIPS 2021.
  [arXiv:2111.00210](https://arxiv.org/abs/2111.00210). MuZero variant
  targeting Atari 100k sample efficiency.

### Practical / infrastructure
- **The Arcade Learning Environment** — Bellemare et al., JAIR 2013.
  [arXiv:1207.4708](https://arxiv.org/abs/1207.4708). The reference for how to
  wrap an emulator as a Gym-style env; we'll need an analogous wrapper for
  DOSBox.
- **Gymnasium docs** — [gymnasium.farama.org](https://gymnasium.farama.org/).
  Modern fork of OpenAI Gym, the API target for our env wrapper.
- **VizDoom** — Kempka et al., 2016.
  [arXiv:1605.02097](https://arxiv.org/abs/1605.02097). Closest prior art for
  instrumenting a non-Atari emulator (ZDoom) for RL — patch points,
  framebuffer access, action injection.
