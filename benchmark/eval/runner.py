from __future__ import annotations
from dataclasses import dataclass
from importlib import util
from pathlib import Path
import tempfile
import torch
import time
from typing import Dict, Any, Optional, Callable, Tuple, List

from benchmark.agents.base import CodeGenAgent, GenerationResult

@dataclass
class RunResult:
    gen: GenerationResult
    compile_success: bool
    runtime_ms: Optional[float]
    gflops: Optional[float]
    metrics: Dict[str, Any]
    notes: str = ""

@dataclass
class TaskSpec:
    """Specification of a benchmark task (operation) independent of model/provider.

    Inputs are derived from `input_shapes` (list of cases), not a provided input_fn.
    reference_fn: optional callable; if absent we compile `reference_source`.
    reference_source: optional Python source defining a reference module/function.
        If it defines a class inheriting nn.Module, the first such class is used.
    correctness_fn: returns dict of metrics comparing (out, ref).
    flops_fn: optional FLOPs computation.
    """
    name: str
    correctness_fn: Callable[[Any, Any], Dict[str, Any]]
    reference_fn: Optional[Callable[..., Any]] = None
    reference_source: Optional[str] = None
    input_shapes: Optional[List[Any]] = None  # list of cases, each case is list of tensor shapes
    flops_fn: Optional[Callable[[Tuple[Any, ...], Dict[str, Any]], float]] = None


@dataclass
class EvalConfig:
    warmup: int = 20
    repeat: int = 100
    shape_index: int = 0


class GenericEvaluator:
    def __init__(self, cfg: EvalConfig, spec: TaskSpec):
        self.cfg = cfg
        self.spec = spec

    def _materialize_inputs(self, device: torch.device) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        # Build input tensors from spec.input_shapes. Supports:
        # - spec.input_shapes = [ [(...shape for input0...), (...shape for input1...), ...],  # one case
        #                         [...],  # another case
        #                       ]
        # - Back-compat: a single case provided directly as list of shapes.
        shapes_raw = self.spec.input_shapes or []
        # Normalize to list of cases
        def to_cases(raw):
            if not raw:
                return []
            # Single tensor shape (one-input op)
            if all(isinstance(x, int) for x in raw):
                return [[tuple(raw)]]
            # Single case with multiple tensors
            if all(isinstance(s, (list, tuple)) and all(isinstance(d, int) for d in s) for s in raw):
                return [[tuple(s) for s in raw]]
            # List of cases
            cases = []
            for case in raw:
                cases.append([tuple(s) for s in case])
            return cases
        cases = to_cases(shapes_raw)
        # Pick case
        idx = max(0, min(self.cfg.shape_index, max(0, len(cases) - 1)))
        chosen = cases[idx] if cases else []
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        torch.manual_seed(42)  # Fixed seed for reproducible evaluation
        args = tuple(torch.randn(*shape, device=device, dtype=dtype) for shape in chosen)
        return args, {}

    def _bench(self, op: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any], repeat: int, warmup: int):
        for _ in range(warmup):
            op(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeat):
                out = op(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / repeat
        else:
            # CPU timing fallback (less precise)
            start_t = time.time()
            for _ in range(repeat):
                out = op(*args, **kwargs)
            ms = (time.time() - start_t) * 1000 / repeat
        return out, ms

    def evaluate_code(self, code: str) -> RunResult:
        # print("code:\n", code)
        spec = self.spec
        cfg = self.cfg
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        args, kwargs = self._materialize_inputs(device)
        # Obtain reference output
        if spec.reference_fn is not None:
            ref = spec.reference_fn(*args, **kwargs)
        elif spec.reference_source is not None:
            # reference_source is now a FULL python file (with imports / optional __main__);
            # execute and discover callable using heuristics similar to loader.
            ref_tmp = Path(tempfile.mkdtemp()) / 'ref_impl.py'
            ref_tmp.write_text(spec.reference_source)
            try:
                r_spec = util.spec_from_file_location('ref_impl', ref_tmp)
                r_mod = util.module_from_spec(r_spec)
                assert r_spec and r_spec.loader
                r_spec.loader.exec_module(r_mod)  # type: ignore
            except Exception as e:
                return RunResult(gen=GenerationResult(raw=code, metadata={}), compile_success=False, runtime_ms=None, gflops=None, metrics={"error": f"reference compile failure: {e}"}, notes="reference failure")
            ref_callable = None
            # Candidate names ordered (generic)
            candidates = [
                f'torch_{spec.name}', f'torch{spec.name}', spec.name,
                'reference', 'forward', 'op'
            ]
            # Pass 1: exact callable functions
            for cand in candidates:
                if hasattr(r_mod, cand) and callable(getattr(r_mod, cand)):
                    ref_callable = getattr(r_mod, cand)
                    break
            # Pass 4: first public callable
            if ref_callable is None:
                for name, val in r_mod.__dict__.items():  # type: ignore
                    if name.startswith('_'): continue
                    if callable(val) and not isinstance(val, type):
                        ref_callable = val
                        break
            if ref_callable is None:
                return RunResult(gen=GenerationResult(raw=code, metadata={}), compile_success=False, runtime_ms=None, gflops=None, metrics={"error": "no reference callable found"}, notes="reference missing")
            try:
                # Call reference directly assuming standard signature (no ctx/self/context handling)
                ref = ref_callable(*args, **kwargs)
            except Exception as e:
                return RunResult(gen=GenerationResult(raw=code, metadata={}), compile_success=False, runtime_ms=None, gflops=None, metrics={"error": f"reference run failure: {e}"}, notes="reference runtime failure")
        else:
            return RunResult(gen=GenerationResult(raw=code, metadata={}), compile_success=False, runtime_ms=None, gflops=None, metrics={"error": "no reference implementation specified"}, notes="spec invalid")
        # materialize module file
        tmpdir = Path(tempfile.mkdtemp())
        code_path = tmpdir / 'gen_impl.py'
        # Assume generated code already contains the correct function with expected name
        code_path.write_text(code)
        try:
            m_spec = util.spec_from_file_location('gen_impl', code_path)
            module = util.module_from_spec(m_spec)
            assert m_spec and m_spec.loader
            m_spec.loader.exec_module(module)  # type: ignore
        except Exception as e:
            return RunResult(gen=GenerationResult(raw=code, metadata={}), compile_success=False, runtime_ms=None, gflops=None, metrics={"error": str(e)}, notes="import failure")
        expected_name = f"triton_{spec.name}"
        if not hasattr(module, expected_name):
            return RunResult(gen=GenerationResult(raw=code, metadata={}), compile_success=False, runtime_ms=None, gflops=None, metrics={"error": f"symbol {expected_name} missing"})
        op = getattr(module, expected_name)
        # Enforce that the generated artifact is a function (not a class/module)
        if isinstance(op, type):
            return RunResult(gen=GenerationResult(raw=code, metadata={}), compile_success=True, runtime_ms=None, gflops=None, metrics={"error": "expected a function, found class"})
        if not callable(op):
            return RunResult(gen=GenerationResult(raw=code, metadata={}), compile_success=True, runtime_ms=None, gflops=None, metrics={"error": "expected a callable function"})
        bench_op = op
        try:
            out, ms = self._bench(bench_op, args, kwargs, cfg.repeat, cfg.warmup)
        except Exception as e:
            return RunResult(gen=GenerationResult(raw=code, metadata={}), compile_success=True, runtime_ms=None, gflops=None, metrics={"error": str(e)}, notes="runtime failure")
        flops = spec.flops_fn(args, kwargs) if spec.flops_fn else None
        gflops = (flops / (ms * 1e-3) / 1e9) if flops is not None else None
        # Support multiple output tensors: normalize to sequences and aggregate metrics.
        def _to_seq(x: Any):
            return list(x) if isinstance(x, (list, tuple)) else [x]
        try:
            out_seq = _to_seq(out)
            ref_seq = _to_seq(ref)
            if len(out_seq) != len(ref_seq):
                return RunResult(
                    gen=GenerationResult(raw=code, metadata={}),
                    compile_success=True,
                    runtime_ms=ms,
                    gflops=gflops,
                    metrics={"error": f"output count mismatch: got {len(out_seq)} vs reference {len(ref_seq)}"},
                    notes="metric aggregation failure",
                )
            # Compute metrics per output using the provided single-output correctness_fn
            per_metrics: List[Dict[str, Any]] = []
            for o_i, r_i in zip(out_seq, ref_seq):
                per_metrics.append(spec.correctness_fn(o_i, r_i))

            # Aggregate numeric metrics across outputs by taking max (sensible for max_abs/max_rel)
            from numbers import Number
            aggregated: Dict[str, Any] = {}
            # Collect all keys
            all_keys = set().union(*[m.keys() for m in per_metrics]) if per_metrics else set()
            for k in all_keys:
                vals = [m[k] for m in per_metrics if k in m]
                if vals and all(isinstance(v, Number) for v in vals):
                    try:
                        aggregated[k] = max(vals)
                    except Exception:
                        # Fallback: keep first value if max fails
                        aggregated[k] = vals[0]
            # Always include detailed breakdown
            aggregated["per_output"] = per_metrics
            aggregated["num_outputs"] = len(per_metrics)
            metrics = aggregated if len(per_metrics) > 1 else per_metrics[0] if per_metrics else {}
        except Exception as e:
            return RunResult(gen=GenerationResult(raw=code, metadata={}), compile_success=True, runtime_ms=ms, gflops=gflops, metrics={"error": f"correctness evaluation failure: {e}"}, notes="correctness failure")
        return RunResult(gen=GenerationResult(raw=code, metadata={}), compile_success=True, runtime_ms=ms, gflops=gflops, metrics=metrics)


def run_single(agent: CodeGenAgent, module_src: str, evaluator: GenericEvaluator | None = None, extra_instructions: str | None = None) -> RunResult:
    gen = agent.generate(
        torch_module_src=module_src,
        evaluator_obj=evaluator,
        instructions=extra_instructions,
    )
    decoded = agent.decode(gen.raw)
    result = evaluator.evaluate_code(decoded)
    result.gen = gen
    return result, decoded
