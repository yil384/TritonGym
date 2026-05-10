# TritonGym

A benchmark framework for evaluating LLM-generated [Triton](https://github.com/triton-lang/triton) GPU kernels. TritonGym measures how well large language models can produce correct and performant GPU code by comparing their outputs against reference PyTorch implementations and oracle Triton kernels.

🏆 **Live leaderboard:** https://tritongym.picasso-lab.com

## Overview

TritonGym provides:

- **164 benchmark operators** across three categories:
  - **Standard** (139 ops): Common GPU kernels including matmul, attention, normalization, activation functions, quantization, and more
  - **Out-of-Distribution** (13 ops): Novel operators not commonly seen in training data
  - **DSL** (12 ops): Operators specified using domain-specific languages (Gluon, TLX)

- **Four code generation agents**:
  - **OneShot**: Single-pass code generation from the operator specification
  - **AlphaEvolve**: Iterative refinement using evaluator feedback (up to 5 attempts)
  - **Geak**: Multi-agent pipeline with generator, compiler, verifier, optimizer, and reflector
  - **Leader**: Diff-based iterative agent that modifies code incrementally

- **Multi-provider LLM support**: OpenAI (GPT-4o, GPT-5), Anthropic (Claude), Google (Gemini), DeepSeek, Meta (Llama via Together AI)

- **Automated evaluation**: Correctness verification (max absolute error ≤ 0.01) and performance benchmarking (latency comparison against oracle Triton implementations)

## Installation

### Docker (Recommended)

```bash
docker build -t tritongym -f Dockerfile .
docker run --name tritongym --gpus all -it -v "$(pwd)":/workspace tritongym
```

### Local

```bash
pip install -r requirements.txt
```

Requires: Python 3.10+, PyTorch 2.5+, Triton 3.1, CUDA 12.4+

## Quick Start

### 1. Set your API key

```bash
# Choose one:
export OPENAI_API_KEY=<your-key>
export ANTHROPIC_API_KEY=<your-key>
export GOOGLE_API_KEY=<your-key>
export DEEPSEEK_API_KEY=<your-key>
export TOGETHER_API_KEY=<your-key>
```

### 2. Run a single operator

```bash
# OneShot with DeepSeek on the "add" operator
python -m benchmark.cli \
    --model deepseek-chat \
    --agent oneshot \
    --operator add \
    --benchmark-json benchmark/data_v2/benchmark.json \
    --json --logs logs

# AlphaEvolve with GPT-4o on the "matmul" operator
python -m benchmark.cli \
    --model gpt-4o \
    --agent alphaevolve \
    --operator matmul \
    --benchmark-json benchmark/data_v2/benchmark.json \
    --json --logs logs
```

### 3. Run the full benchmark

```bash
bash run_all.sh
```

### 4. Summarize results

```bash
python summarize_results.py --logs logs/deepseek-chat
```

Output (Table 1 style):
```
Agent                |     Standard Samples |          OOD Samples |          DSL Samples
                     |    Pass@1     Perf@1 |    Pass@1     Perf@1 |    Pass@1     Perf@1
------------------------------------------------------------------------------------------
alphaevolve          |     48.2%      0.870 |      0.0%      0.000 |     50.0%      0.369
geak                 |     37.4%      0.966 |     23.1%      0.027 |     50.0%      0.410
oneshot              |     33.8%      0.518 |      7.7%      0.000 |     41.7%      0.141
```

## Project Structure

```
TritonGym/
├── benchmark/
│   ├── cli.py                    # Main CLI entry point
│   ├── eval/
│   │   ├── runner.py             # Core evaluator (GenericEvaluator, TaskSpec, RunResult)
│   │   └── tools.py              # Tool functions (evaluator, verifier, compiler)
│   ├── agents/
│   │   ├── base.py               # Abstract CodeGenAgent
│   │   ├── oneshot_agent.py      # Single-pass generation
│   │   ├── alphaevolve_agent.py  # Iterative refinement agent
│   │   ├── geak_agent.py         # Multi-agent pipeline
│   │   ├── leader_agent.py       # Diff-based iterative agent
│   │   └── clients/              # LLM provider clients
│   │       ├── base.py           # Abstract LLM client
│   │       ├── openai_client.py
│   │       ├── anthropic_client.py
│   │       ├── google_client.py
│   │       ├── deepseek_client.py
│   │       └── meta_client.py
│   ├── data_v2/                  # Standard benchmark (139 operators)
│   │   ├── benchmark.json        # Operator specs (torch_operator, triton_operator, input_shapes)
│   │   ├── torch/                # Reference PyTorch implementations
│   │   ├── triton/               # Oracle Triton implementations
│   │   └── tests/                # Standalone test scripts
│   ├── data_ood_v2/              # Out-of-distribution benchmark (13 operators)
│   └── data_dsl_v2/              # DSL benchmark (12 operators)
├── scripts/
│   ├── sync_torch.py             # Sync torch/*.py files into benchmark.json
│   └── sync_oracles.py           # Sync triton/*.py files into benchmark.json
├── run_all.sh                    # Full benchmark runner
├── run_perlmutter.sh             # NERSC Perlmutter HPC runner
├── summarize_results.py          # Results analysis (Table 1 style)
├── Dockerfile
├── requirements.txt
└── README.md
```

## Benchmark Data Format

Each operator in `benchmark.json` contains:

```json
{
  "operator_name": {
    "torch_operator": "import torch\n\ndef torch_operator_name(x):\n    ...",
    "triton_operator": "import triton\n\ndef triton_operator_name(x):\n    ...",
    "input_shapes": [[[128, 128], [128, 128]], [[256, 256], [256, 256]]]
  }
}
```

- **`torch_operator`**: Reference PyTorch implementation (ground truth)
- **`triton_operator`**: Oracle Triton implementation (performance baseline)
- **`input_shapes`**: List of test cases, each containing shapes for input tensors

## Evaluation Metrics

- **Pass@1**: Fraction of operators where the generated kernel produces correct output (max absolute error ≤ 0.01 vs. the PyTorch reference)
- **Perf@1**: Average ratio of oracle Triton latency to generated kernel latency across all operators (0 for failed operators)

## Adding New Operators

1. Create `torch/<op_name>.py` with function `torch_<op_name>(...)`
2. Create `triton/<op_name>.py` with function `triton_<op_name>(...)`
3. Optionally create `tests/<op_name>.py` for standalone testing
4. Run `python scripts/sync_torch.py` and `python scripts/sync_oracles.py` to update `benchmark.json`

## Adding New LLM Providers

1. Create a new client in `benchmark/agents/clients/` inheriting from `BaseLLMClient`
2. Implement the `chat()` method
3. Register the client in the client factory

## HPC Deployment (Perlmutter)

```bash
export DEEPSEEK_API_KEY=<your-key>
shifter --image=docker:yil384/tritongym:latest --volume=$PWD:/workspace /bin/bash
cd /workspace
bash run_perlmutter.sh
```

## Citation

If you use TritonGym in your research, please cite:

```bibtex
@article{tritongym2025,
  title={TritonGym: A Benchmark for Evaluating LLM-Generated GPU Kernels},
  author={...},
  year={2025}
}
```

## Website

A swebench-style leaderboard lives in `web/`. To run it locally:

```bash
pip install -r web/backend/requirements.txt
python3 web/scripts/aggregate_logs.py   # generates web/frontend/public/data/leaderboard.json
PORT=7843 python3 web/backend/main.py   # http://127.0.0.1:7843
```

To deploy it (mirrors the picasso-lab pattern: uvicorn + nginx + certbot):

```bash
cd web
DOMAIN=tritongym.picasso-lab.com bash deploy/deploy.sh
```

See `web/README.md` for details.

## License

MIT — see [LICENSE](LICENSE).
