<!-- Pick one: -->
<!-- Title: [submission] <model-name>     — leaderboard submission -->
<!-- Title: [operator] <op-name>          — new operator -->
<!-- Title: [fix] <short-description>     — bug fix -->
<!-- Title: [feat] <short-description>    — new feature -->

## What

<!-- 1-3 sentences on what this PR does -->

## Why

<!-- Motivation / linked issue. -->

## How

<!-- Implementation summary, only the non-obvious parts. -->

## Submission checklist (delete the section that doesn't apply)

### Model submission

- [ ] Logs in `submissions/<model-name>/<agent>/*.json`
- [ ] Model name, parameter count, license described
- [ ] Hardware + Triton/Torch/CUDA versions described
- [ ] Inference setup (vLLM / sglang / API) described
- [ ] OOD numbers look reasonable (no contamination)

### Operator addition

- [ ] `torch/<op_name>.py` with reference implementation
- [ ] `triton/<op_name>.py` with oracle Triton kernel
- [ ] `tests/<op_name>.py` with input shapes
- [ ] `python scripts/sync_torch.py && python scripts/sync_oracles.py` run
- [ ] Verified locally: oracle Triton matches PyTorch within 0.01 abs error

### Code change

- [ ] Tested locally on at least one operator
- [ ] If agent / scoring changed, included before/after Pass@1 / Perf@1

### Reference-test onboarding task (delete for unrelated PRs)

- [ ] `python -m pytest tests/ci -q` passes with the starter branch's complete acceptance tests
- [ ] All five original input cases remain covered; logs show actual shapes
- [ ] Bad values, NaN/Inf, mismatched shapes, and runtime errors produce a nonzero exit
- [ ] The existing `max_abs <= 0.01` rule and reference computations are unchanged
- [ ] Attached `reports/reference-gpu.json` from the reviewed commit on a real GPU, or clearly marked the remaining GPU blocker
- [ ] Included the exact commands and GPU / Torch / Triton / CUDA versions
