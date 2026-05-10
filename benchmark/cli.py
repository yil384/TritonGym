from __future__ import annotations
import argparse
import json
import textwrap
import torch
from pathlib import Path
from benchmark.eval.runner import GenericEvaluator, run_single, TaskSpec, EvalConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='claude-sonnet-4-20250514', help='Model name, e.g., gpt-4o, gemini-2.5-pro, claude-sonnet-4-20250514, meta-llama-3-70b-instruct-v2')
    p.add_argument('--agent', default='oneshot', help='Agent type: oneshot, alphaevolve, geak')
    p.add_argument('--operator', required=True, help='Operator name key in benchmark.json')
    p.add_argument('--benchmark-json', default='benchmark/data_v2/benchmark.json', help='Path to JSON spec file')
    p.add_argument('--repeat', type=int, default=50)
    p.add_argument('--warmup', type=int, default=10)
    # No op-specific fallback dims; rely on JSON input_shapes
    p.add_argument('--shape-index', type=int, default=0, help='Index of input_shapes entry in JSON to use')
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--instructions', type=str, default=None)
    p.add_argument('--json', action='store_true')
    p.add_argument('--logs', type=str, default='logs', help='Directory to save logs')
    args = p.parse_args()

    # Load operator spec JSON BEFORE constructing TaskSpec
    spec_path = Path(args.benchmark_json)
    if not spec_path.is_file():
        raise SystemExit(f"Spec file not found: {spec_path}")
    spec = json.loads(spec_path.read_text())
    if args.operator not in spec:
        raise SystemExit(f"Operator '{args.operator}' not found in {spec_path}. Available: {list(spec.keys())}")
    mod_entry = spec[args.operator]

    # Expected callable is now implicit: triton_<operator>

    json_shapes = mod_entry.get('input_shapes')

    def flops_fn(args, kwargs):
        # Generic: unknown FLOPs; return None-equivalent by raising
        raise NotImplementedError

    # Inputs will be materialized in the evaluator from JSON shapes; no loader needed here.

    # Build a reference source that defines the torch function (already compiled); re-exec not needed.
    # We'll reconstruct minimal source for reference: define wrapper calling the loaded function.
    # Simpler: just embed original torch code again for isolation. We re-read from JSON to embed.
    torch_module_src = mod_entry.get('torch_operator') or mod_entry.get('torch') or ''
    # Full file already; ensure stripped
    torch_module_src = textwrap.dedent(torch_module_src).strip('\n') + '\n'

    # Local correctness metrics (max abs/rel error)
    def _correctness(out, ref):
        with torch.no_grad():
            diff = (out - ref).abs()
            denom = ref.abs().clamp_min(1e-8)
            rel = diff / denom
            # Replace non-finite with 0 for stability
            rel = torch.where(torch.isfinite(rel), rel, torch.zeros_like(rel))
            max_abs = torch.nan_to_num(diff, nan=0.0, posinf=float('inf'), neginf=float('inf')).max().item()
            max_rel = rel.max().item()
        return {"max_abs": max_abs, "max_rel": max_rel}

    spec_obj = TaskSpec(
        name=args.operator,
        reference_source=torch_module_src,
        correctness_fn=_correctness,
        input_shapes=json_shapes,
        flops_fn=None,
    )
    evaluator = GenericEvaluator(EvalConfig(warmup=args.warmup, repeat=args.repeat, shape_index=args.shape_index), spec_obj)

    # Prepare inputs once for fair benchmarking and reuse evaluator's materialization
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    bench_args, bench_kwargs = evaluator._materialize_inputs(device)

    def _load_func_from_source(src: str, func_name: str):
        import tempfile as _tempfile
        from importlib import util as _util
        tmpdir = Path(_tempfile.mkdtemp())
        p = tmpdir / 'tmp_impl.py'
        p.write_text(src)
        m_spec = _util.spec_from_file_location('tmp_impl', p)
        module = _util.module_from_spec(m_spec)
        assert m_spec and m_spec.loader
        m_spec.loader.exec_module(module)  # type: ignore
        if not hasattr(module, func_name):
            raise AttributeError(f"symbol {func_name} missing")
        fn = getattr(module, func_name)
        if isinstance(fn, type) or not callable(fn):
            raise TypeError(f"expected a function callable: {func_name}")
        return fn

    # Benchmark torch_operator from JSON
    torch_runtime_ms = None
    torch_error = None
    try:
        torch_fn = _load_func_from_source(torch_module_src, f"torch_{args.operator}")
        _, torch_runtime_ms = evaluator._bench(torch_fn, bench_args, bench_kwargs, args.repeat, args.warmup)
    except Exception as e:
        torch_error = str(e)

    # Benchmark triton_operator from JSON (if any)
    triton_runtime_ms = None
    triton_error = None
    triton_module_src = mod_entry.get('triton_operator')
    if triton_module_src:
        try:
            triton_src = textwrap.dedent(triton_module_src).strip('\n') + '\n'
            triton_fn = _load_func_from_source(triton_src, f"triton_{args.operator}")
            _, triton_runtime_ms = evaluator._bench(triton_fn, bench_args, bench_kwargs, args.repeat, args.warmup)
        except Exception as e:
            triton_error = str(e)

    # Now run code generation benchmark
    if args.agent == 'oneshot':
        from benchmark.agents.oneshot_agent import OneShotAgent
        agent = OneShotAgent(args.model, temperature=args.temperature)
    elif args.agent == 'alphaevolve':
        from benchmark.agents.alphaevolve_agent import AlphaEvolveAgent
        agent = AlphaEvolveAgent(args.model, temperature=args.temperature)
    elif args.agent == 'geak':
        from benchmark.agents.geak_agent import GeakAgent
        agent = GeakAgent(args.model, temperature=args.temperature)
    elif args.agent == 'leader':
        from benchmark.agents.leader_agent import LeaderAgent
        agent = LeaderAgent(args.model, temperature=args.temperature)
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")
    
    result, decoded = run_single(agent, torch_module_src, evaluator, args.instructions)

    print(json.dumps({
            'operator': args.operator,
            'compile_success': result.compile_success,
            'runtime_ms': result.runtime_ms,
            'gflops': result.gflops,
            'metrics': result.metrics,
            'notes': getattr(result, 'notes', ''),
            'model': result.gen.metadata.get('model'),
            'total_input_tokens': result.gen.metadata.get('total_input_tokens', 0),
            'total_output_tokens': result.gen.metadata.get('total_output_tokens', 0),
            'torch_runtime_ms': torch_runtime_ms,
            'torch_error': torch_error,
            'triton_runtime_ms': triton_runtime_ms,
            'triton_error': triton_error,
        }, indent=2))

    # Prepare result data for logging
    result_data = {
        'operator': args.operator,
        'compile_success': result.compile_success,
        'runtime_ms': result.runtime_ms,
        'gflops': result.gflops,
        'metrics': result.metrics,
        'notes': getattr(result, 'notes', ''),
        'model': result.gen.metadata.get('model'),
        'total_input_tokens': result.gen.metadata.get('total_input_tokens', 0),
        'total_output_tokens': result.gen.metadata.get('total_output_tokens', 0),
        'torch_runtime_ms': torch_runtime_ms,
        'torch_error': torch_error,
        'triton_runtime_ms': triton_runtime_ms,
        'triton_error': triton_error,
        'agent': args.agent,
        'shape_index': args.shape_index,
        'metadata': result.gen.metadata if hasattr(result, 'gen') and hasattr(result.gen, 'metadata') else {}
    }

    if args.json:
        # Log directory: logs/model/agent/
        log_dir = Path(args.logs) / args.model / args.agent
        log_dir.mkdir(parents=True, exist_ok=True)

        log_filename = f"{args.operator}.json"
        log_path = log_dir / log_filename

        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        print(f"Log saved to: {log_path}")
    else:
        print(f"Operator: {args.operator}")
        print(f"Model: {result.gen.metadata.get('model')}")
        print(f"Agent: {args.agent}")
        print(f"Compile: {result.compile_success}")
        if result.runtime_ms:
            print(f"Latency: {result.runtime_ms:.3f} ms  GFLOP/s: {result.gflops:.2f}")
        print(f"Metrics: {result.metrics}")
        if torch_runtime_ms is not None:
            print(f"Torch JSON op latency: {torch_runtime_ms:.3f} ms")
        if triton_runtime_ms is not None:
            print(f"Triton JSON op latency: {triton_runtime_ms:.3f} ms")
        if torch_error:
            print(f"Torch JSON op error: {torch_error}")
        if triton_error:
            print(f"Triton JSON op error: {triton_error}")

        # Also save a copy in non-JSON mode
        log_dir = Path(args.logs) / args.model / args.agent
        log_dir.mkdir(parents=True, exist_ok=True)

        log_filename = f"{args.operator}.json"
        log_path = log_dir / log_filename
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        print(f"\nLog saved to: {log_path}")

    # Save generated code
    code_dir = Path(args.logs + "_generated_code") / args.model / args.agent
    code_dir.mkdir(parents=True, exist_ok=True)
    code_filename = f"{args.operator}_code.py"
    code_path = code_dir / code_filename
    with open(code_path, 'w', encoding='utf-8') as f:
        f.write(decoded)

if __name__ == '__main__':
    main()