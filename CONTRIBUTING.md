# Contributing to TritonGym

There are three ways to contribute to TritonGym:

1. **Submit a model** to the leaderboard (no code changes required)
2. **Add operators** to the benchmark (broaden coverage)
3. **Improve agents** or evaluation infrastructure

## 1. Submitting a model

See [SUBMISSIONS.md](SUBMISSIONS.md). Two paths:

- **Open-weight HF models**: submit at https://tritongym.picasso-lab.com/submit.html — we run it on our GPU and merge results.
- **Closed-API models / custom setups**: run locally and PR your results into `submissions/<your-model>/`.

## 2. Adding a new operator

Each operator lives in three files:

```
benchmark/data_v2/torch/<op_name>.py     # PyTorch reference (correctness oracle)
benchmark/data_v2/triton/<op_name>.py    # Triton oracle (perf baseline)
benchmark/data_v2/tests/<op_name>.py     # Standalone test harness
```

To add one:

1. Implement `torch_<op_name>(...)` in `torch/<op_name>.py`. This defines correctness — keep it as plain, readable PyTorch.
2. Implement `triton_<op_name>(...)` in `triton/<op_name>.py`. This is the oracle for the Perf@1 ratio. Should be a reasonable hand-written Triton kernel, not necessarily the fastest possible.
3. Add `tests/<op_name>.py` to lock the input shapes you want evaluated.
4. Run `python scripts/sync_torch.py && python scripts/sync_oracles.py` to bake them into `benchmark.json`.
5. Verify: `python -m benchmark.cli --operator <op_name> --agent oneshot --model deepseek-chat ...`

PR title: `[operator] <op_name>`. Description should include what category the op belongs to (Standard / OOD / DSL) and any input-shape rationale.

## 3. Code changes

- Open an issue first for large changes — agents, evaluator, scoring metrics
- Keep PRs focused: one feature / fix per PR
- Don't add abstractions speculatively — three similar lines is better than a premature base class
- For agent changes, include before/after Pass@1 / Perf@1 numbers on a sample of operators

## Setup

```bash
git clone https://github.com/yil384/TritonGym.git
cd TritonGym
pip install -r requirements.txt
```

Requires: Python 3.10+, PyTorch 2.5+, Triton 3.1+, CUDA 12.4+.

A working Triton-capable GPU is needed for evaluation (any single A100 / H100 / consumer 30/40 series).

## First contribution and checks

For the scoped reference-test task, read [the first-contribution guide](docs/first-contribution.md).
It lists the two files to change, the five input cases, the acceptance command,
and the GPU evidence to attach to a PR.

Pull requests run `python -m pytest tests/ci` on a GitHub-hosted CPU runner.
The student starter branch includes additional acceptance tests that initially
fail on the legacy scripts; fixing those failures is the task. CPU checks do
not validate Triton execution on a GPU. Use the separate GPU command in the
guide before reporting GPU validation.

## Code of conduct

Be excellent to each other. Reviews focus on the work, not the person.
