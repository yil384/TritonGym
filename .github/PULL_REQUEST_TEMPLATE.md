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
