# Submitting a Model to the TritonGym Leaderboard

There are two ways to get your model onto the leaderboard at
https://tritongym.picasso-lab.com.

## Option A — HuggingFace model (we run it)

If your model has open weights on HuggingFace, just submit the model id at
https://tritongym.picasso-lab.com/submit.html. We'll evaluate it on a free
GPU when one is available — typical wait is hours to days depending on queue
depth and model size.

Constraints:
- Model must be public on HuggingFace.
- We run it via vLLM/sglang in fp16; if your model needs a different runtime
  please use Option B instead.
- We don't accept API keys — closed-API models are Option B only.

## Option B — Pre-computed results (you run it, PR the logs)

This is the only path for closed-API models, fine-tunes you'd rather not
upload, or models that need custom inference setup.

### 1. Run the benchmark on your model

```bash
git clone https://github.com/yil384/TritonGym-public
cd TritonGym-public
pip install -r requirements.txt

# Pick whichever provider matches your model:
export OPENAI_API_KEY=...        # OpenAI
export ANTHROPIC_API_KEY=...     # Claude
export GOOGLE_API_KEY=...        # Gemini
export DEEPSEEK_API_KEY=...      # DeepSeek
export TOGETHER_API_KEY=...      # Together-hosted (Qwen, Llama)

# Run all 4 agents on the full 164-operator benchmark.
# This takes a few hours and consumes meaningful API budget.
bash run_all.sh   # or run benchmark/cli.py directly per agent
```

Logs land in `logs/<your-model-name>/<agent>/<operator>.json`.

### 2. Sanity-check your numbers

```bash
python summarize_results.py --logs logs/<your-model-name>
python web/scripts/aggregate_logs.py --include-logs
```

Confirm Pass@1 / Perf@1 look reasonable (no 100%-pass on Standard, that's
almost certainly answer leakage; expect <60% from frontier models).

### 3. Open a PR

- Move `logs/<your-model-name>` to `submissions/<your-model-name>/`.
- Branch name: `submission/<your-model-name>`.
- PR title: `[submission] <your-model-name>`.
- PR description should include:
  - Model name, parameter count, license
  - Hardware used (GPU model, CUDA, Triton, Torch versions)
  - Inference setup (vLLM / sglang / native HF / API)
  - Anything we should know about the prompt template or sampling params

### 4. Review

We'll do a quick sanity pass:
- Are the OOD numbers in line with the model's general capability? (Wildly
  high OOD pass rate is the canonical sign of training-set contamination.)
- Do the runs reproduce on our hardware for a sample of operators?

If it looks clean, we merge and rerun `web/scripts/aggregate_logs.py
--include-logs`. Your numbers go up with a `[submitted]` tag.

---

## What gets rejected

- Submissions where the OOD split was skipped or the model was clearly
  trained on the operator set.
- Numbers that disagree with our spot-check by more than a few percent.
- Anything that looks like an attempt to game Perf@1 (e.g., kernels that
  don't actually compute the right thing but happen to pass the
  abs-error 0.01 threshold).

We err on the side of accepting borderline cases and adding a note rather
than rejecting outright — but contamination is a hard line.
