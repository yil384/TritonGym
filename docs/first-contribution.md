# First contribution: make two reference tests report failures

## Your goal

Fix two existing test scripts so that their exit status tells us whether the checks passed. A script must fail when the Triton output is wrong, non-finite, or the wrong shape, or when the kernel raises an error.

You will work on exactly these two operators:

- `benchmark/data_v2/tests/token_softmax_llama.py`
- `benchmark/data_v2/tests/attention_forward.py`

The current scripts print an error or a maximum difference, but they can still exit successfully. This makes an automated check report success when a kernel has failed.

You do not need to design a new agent or run an LLM API for this task.

## Start from the starter branch

There are two separate stages:

| Stage | Branch | What it proves |
| --- | --- | --- |
| Mentor's foundation PR | `ci/reference-test-onboarding` | The CI driver and its own tests work. Green checks here do not mean that your task is complete. |
| Your starting point | `onboarding/reference-test-task` | Adds `tests/ci/test_reference_script_acceptance.py`, which checks the two real scripts. The negative cases are expected to fail before your fixes. |

The starter branch is not a separate PR to merge first. Create your work branch from it. Do not start from the foundation branch, because that would leave out the task's acceptance tests.

If you need a fresh clone:

```bash
git clone https://github.com/yil384/TritonGym.git
cd TritonGym
```

Then create your branch:

```bash
git fetch origin
git switch -c fix/reference-test-failures origin/onboarding/reference-test-task
```

You can start work while the foundation PR is under review. Once the foundation PR has merged, open one PR to `main` with your fixes and the acceptance tests supplied by the starter branch. Use your fork if you do not have permission to push to this repository.

## Set up the CPU checks

Run these commands from the repository root on a CPU development machine:

```bash
python3.11 -m venv .venv-ci
source .venv-ci/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ci.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pytest tests/ci -q
```

The CPU acceptance tests replace the kernel calls with controlled outputs. They check whether your scripts catch failures and return the right exit status. They do not run Triton kernels on a GPU.

Use the exact `tests/ci` command above for this task. Running bare `pytest` also
collects the older `benchmark/tests` suite, which is outside this assignment and
includes tests for obsolete interfaces.

Before your changes, the ten negative scenarios should be red. Keep them as ordinary failing tests while you work. Do not add `xfail`, skip them, or weaken the checks to make the suite green.

## Keep these five cases

Keep the existing input shapes and dtypes. The case numbers below start at zero, as in the scripts.

| Script | Case | Actual input shapes | Input dtype |
| --- | --- | --- | --- |
| `token_softmax_llama.py` | 0 | One tensor: `(2, 16)` | `torch.float32` |
| `token_softmax_llama.py` | 1 | One tensor: `(5, 10)` | `torch.float32` |
| `attention_forward.py` | 0 | Q, K, and V: each `(2, 4, 128, 128)` | `torch.float16` |
| `attention_forward.py` | 1 | Q, K, and V: each `(2, 8, 256, 64)` | `torch.float16` |
| `attention_forward.py` | 2 | Q, K, and V: each `(1, 4, 512, 64)` | `torch.float16` |

Use a fixed, documented random seed, such as `42`, before creating the inputs. Report the seed and the actual tensor shapes in the logs. In the token softmax script, replace the hard-coded `batch=2, head=2, max_input_len=8` message: it does not describe these inputs.

## What to change

### 1. Check the outputs before printing success

In each script, after the Torch and Triton calls:

1. Check that their output shapes are exactly equal. Do this before subtraction so broadcasting cannot hide a wrong shape.
2. Check that both outputs contain only finite values. NaN and infinity must cause failure.
3. Compute the maximum absolute difference using the existing reference result.
4. Require `max_abs <= 0.01`.
5. Print success only after all these checks pass.

Keep the existing Torch reference calls and input dtypes. In the attention script, keep the existing conversion of the Torch result to `torch.float16`. Do not change the reference computation, modify the kernel, or increase the error limit to pass a failing case.

Do not turn NaN into zero or ignore non-finite elements when calculating the error.

### 2. Make failures reach the process exit status

The existing `try`/`except` blocks catch errors and then allow the script to finish successfully. Change that behavior.

You may keep running the remaining cases to collect useful information. If you catch an error, record it, print which case failed, and make the final process exit status nonzero. A warning or an `ERROR` line by itself is not enough.

When every case passes, the script must exit with status zero and must have run every listed case: two for token softmax and three for attention.

### 3. Make each result easy to read

For each case, report:

- Operator name and case number.
- Actual input shapes, dtype, device, and random seed.
- PASS or FAIL.
- Maximum absolute difference when it can be computed, or the reason for failure.

For a runtime error, keep enough of the original exception to help someone reproduce it.

### Allowed changes

Change the two test scripts listed above. You may add a small shared helper for their output checks if that avoids repeating code.

Keep the scope small. Do not rewrite the evaluator, change the reference or kernel implementations, add more operators, or change the acceptance tests supplied on the starter branch. If you think an acceptance test is wrong, show the failing example to the mentor before changing it.

## What the acceptance tests require

The suite runs each script with a correct replacement kernel and with five broken replacements.

| Scenario | Expected result for each script |
| --- | --- |
| Correct output | Exit zero and run all of that script's cases. |
| `bad_value`: a value outside the allowed error limit | Exit nonzero. |
| `nan`: NaN in the output | Exit nonzero. |
| `inf`: infinity in the output | Exit nonzero. |
| Wrong output shape that could still broadcast | Exit nonzero. |
| `runtime_error`: the kernel raises an exception | Exit nonzero. |

That gives two positive scenarios and ten negative scenarios across the two operators. A nonzero exit for a broken replacement is the expected behavior; the corresponding pytest test passes when it sees that failure.

Run the full task check after your changes:

```bash
python -m pytest tests/ci -q
```

All tests in this directory should pass. The two scripts must still run all five original cases between them.

## Validate on a real GPU

Use the project's compatible GPU environment: PyTorch 2.5.1, Triton 3.1, and the CUDA 12.4 environment used by the Docker image. Do not install the CPU Torch wheel from the earlier setup into this environment.

From the repository root, run:

```bash
python scripts/check_reference_gpu.py --output reports/reference-gpu.json
```

In the same GPU environment, also run the two edited scripts directly and save
their output and exit status:

```bash
python -m benchmark.data_v2.tests.token_softmax_llama
python -m benchmark.data_v2.tests.attention_forward
```

This command needs a real CUDA GPU. It must exit nonzero if no GPU is available. It separately compares the original reference kernels and records the results; it does not accept a printed `OK` message as proof of correctness.

Keep the JSON report and the command output for your PR. This is a real GPU validation step, not a claim that a GPU runner has been connected to GitHub CI.

If a real kernel does not meet the existing error limit, save the failing shape, dtype, seed, error value, environment, and a minimal reproduction. Share these with the mentor before changing anything else. A failure in an existing kernel is useful evidence; do not hide it by changing the reference computation or relaxing the limit.

If you do not yet have GPU access, finish the CPU work and clearly mark real GPU validation as pending. Green CPU checks alone do not establish that the kernels are numerically correct on a GPU.

## What to include in your PR

- A short explanation of why the scripts previously reported success after failures.
- The two script fixes, plus a small shared helper only if needed.
- The acceptance tests supplied by the starter branch, unchanged.
- The command and result for `python -m pytest tests/ci -q`.
- The real GPU report and environment, or a clear note that this step is still pending.
- Any real numerical mismatch, with a minimal reproduction and the mentor's follow-up plan.

The task is ready for final acceptance when the CPU checks pass, all five original cases run, and real GPU validation has passed or a numerical mismatch has been reviewed with the mentor. A pending GPU run or an unresolved mismatch must remain visible in the PR.

## Suggested timebox

Use up to two weeks for the first contribution. Start by reproducing the failing CPU acceptance tests, then fix the two scripts, and finally validate on a GPU.

Two weeks is a timebox, not a requirement to spend 40 hours. Finish earlier if the checks are complete. If setup or a numerical mismatch blocks progress, report the exact problem early so the mentor can help.
