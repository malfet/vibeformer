# Papers: Learning to solve Sudoku from first principles

A small reading list covering different angles on "teach a neural net to solve Sudoku."

## 1. `recurrent_relational_networks.pdf`

**Recurrent Relational Networks** — Palm, Paquet, Winther (NeurIPS 2018).
[arXiv:1711.08028](https://arxiv.org/abs/1711.08028)

Treats the Sudoku board as a graph (cells = nodes, same row/col/box = edges) and runs
iterative message passing. Each step is effectively a learned constraint-propagation
update; the network discovers the rules from solved/unsolved pairs without being told
them. Solves 96.6% of the hardest published Sudokus. Canonical "neural net learns
Sudoku" baseline.

## 2. `causal_lm_logic_puzzles.pdf`

**Causal Language Modeling Can Elicit Search and Reasoning Capabilities on Logic Puzzles**
— Shah et al. (2024). [arXiv:2409.10502](https://arxiv.org/abs/2409.10502)

A decoder-only transformer trained on Sudoku-solving traces (which cell to fill next,
and the value). The decomposition matters: picking *where* to fill turns out to be
much harder than picking *what* to fill. Reaches ~94% full-puzzle accuracy with beam
search. Closest in spirit to a from-scratch transformer setup like this repo.

## 3. `searchformer.pdf`

**Searchformer: Beyond A\* — Better Planning with Transformers via Search Dynamics
Bootstrapping** — Lehnert et al., Meta (2024).
[arXiv:2402.14083](https://arxiv.org/abs/2402.14083)

Trains a transformer to imitate the *trace* of a symbolic search algorithm (A\*),
then iteratively distills it down to shorter, more efficient traces. The resulting
model plans better than the algorithm it was trained on. Sudoku is a natural target
for the same recipe: imitate a backtracking solver, then compress the trace.

## 4. `satnet.pdf`

**SATNet: Bridging deep learning and logical reasoning using a differentiable
satisfiability solver** — Wang, Donti, Wilder, Kolter (ICML 2019).
[arXiv:1905.12149](https://arxiv.org/abs/1905.12149)

Embeds a differentiable approximate MaxSAT solver as a layer inside a neural network.
Given only (puzzle, solution) pairs — no rules — the model recovers the constraints
of Sudoku from images of the board. The most "first-principles" of the four: the
inductive bias is *logical satisfiability*, not graph structure or token sequences.

## How they relate

| Paper | Inductive bias | Supervision | What it learns |
| --- | --- | --- | --- |
| RRN | Graph + iteration | (puzzle, solution) | Constraint propagation |
| Causal LM | Sequence | Solver traces | Search-like fill order |
| Searchformer | Sequence | A\* traces, then self-distilled | Better-than-A\* planning |
| SATNet | Differentiable SAT | (puzzle, solution) | The rules themselves |

Suggested order: **RRN → Causal LM → Searchformer → SATNet**
(easiest baseline → closest to a transformer setup → search-as-learning → the
intellectually weirdest one).
