# MADDPG PyTorch

This is the standalone PyTorch implementation extracted from the MADDPG
comparison repository. It contains no TensorFlow implementation and does not
vendor MPE, Gym, PettingZoo, or PyTorch.

The exact runtime/test file boundary and the reason for every dependency are
recorded in [`docs/PYTORCH_STANDALONE_PROJECT.md`](docs/PYTORCH_STANDALONE_PROJECT.md).

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Installing the pinned archived MPE commit requires the `git` command and
network access to GitHub.

The default `legacy + official` configuration uses the archived OpenAI MPE
commit pinned in `requirements.txt` and the original Gumbel-Softmax action
semantics. PettingZoo remains available as a separate comparison backend.
Both Gymnasium and Gym are intentionally installed: PettingZoo uses Gymnasium,
while the archived legacy backend imports Gym.

## Training

```bash
python -m experiments.train_torch \
  --env-backend legacy \
  --policy-mode official \
  --scenario simple_spread \
  --tb-log-interval 10000 \
  --checkpoint-eval-episodes 10 \
  --checkpoint-eval-seed 10000 \
  --save-dir ./chkpt
```

`--tb-log-interval` is an aggregation window, not last-value sampling. Every
completed episode and optimizer update contributes to the interval means.
The existing TensorBoard tag set is preserved. Set the interval to `0` to
disable TensorBoard.

`simple_adversary` conditionally reports adversary/nearest-good terminal goal
distances, their distance gap, and good/adversary/tie rates under `task/*`. The
tie distance tolerance is `0.05`; train this scenario with
`--num-adversaries 1`.

Resume a checkpoint produced by this project:

```bash
python -m experiments.train_torch \
  --env-backend legacy \
  --policy-mode official \
  --scenario simple_spread \
  --save-dir ./chkpt \
  --restore
```

Checkpoints are stored under:

```text
<save-dir>/<env-backend>/<policy-mode>/<scenario>/state_steps_<steps>.pt
```

Each save keeps its own step-suffixed file and prints the absolute path. Passing
the scenario directory to `--restore` or `play_torch.py --checkpoint`
automatically selects the greatest numeric step.

Version-2 checkpoints emitted by the immediately preceding PyTorch port remain
loadable. For playback, network width is inferred from weights and other fields
that v2 did not store use its historical defaults. Earlier formats remain
unsupported.

Each checkpoint is reloaded into separate evaluation trainers and evaluated in
a separate environment with deterministic actions. Reports are saved as both
`evaluation_steps_<steps>.json` and the latest `evaluation.json`, and are also
written to TensorBoard under `eval/*`. Set `--checkpoint-eval-episodes 0` to
disable this evaluation.

## Play / standalone evaluation

`play_torch.py` loads all environment and network settings from the checkpoint.
It runs deterministic, headless evaluation by default:

```bash
python -m experiments.play_torch \
  --checkpoint ./chkpt/legacy/official/simple_spread \
  --episodes 20 \
  --report-json ./reports/simple_spread.json
```

Add `--render` to open the GUI. `--fps 10` controls playback speed; `--fps 0`
removes the extra delay. Add `--stochastic` only when sampled policy actions are
desired instead of the default deterministic argmax/mean actions.

## Tests

```bash
python -m unittest discover -s tests -v
```

Long-running multi-seed convergence experiments are not part of the unit test
suite.
