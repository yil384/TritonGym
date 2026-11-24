"""CPU-only checks of the benchmark evaluator's correctness rule (no GPU, no Triton)."""

import pytest
import torch

from benchmark.eval.correctness import TOLERANCE, compare_outputs, mismatch_reason
from benchmark.eval.runner import EvalConfig, GenericEvaluator, TaskSpec
from benchmark.eval.tools import _make_evaluator, verifier


def accepted(metrics):
    return "mismatch" not in metrics and metrics["max_abs"] <= TOLERANCE


REF = torch.arange(12, dtype=torch.float32).reshape(3, 4)


def test_matching_output_is_accepted():
    assert accepted(compare_outputs(REF.clone(), REF))


def test_difference_at_the_tolerance_is_accepted_and_above_is_rejected():
    ref = torch.zeros(4, dtype=torch.float64)
    assert accepted(compare_outputs(ref + 0.01, ref))
    assert not accepted(compare_outputs(ref + 0.011, ref))


@pytest.mark.parametrize("corrupt", [
    lambda t: torch.full_like(t, float("nan")),
    lambda t: torch.full_like(t, float("inf")),
    lambda t: torch.where(torch.arange(t.numel()).reshape(t.shape) == 5, torch.tensor(float("nan")), t),
], ids=["all_nan", "all_inf", "one_nan"])
def test_non_finite_output_is_rejected(corrupt):
    metrics = compare_outputs(corrupt(REF), REF)
    assert not accepted(metrics)
    assert "NaN or infinity" in metrics["mismatch"]


@pytest.mark.parametrize("reshape", [
    lambda t: t.unsqueeze(0),
    lambda t: t[:1],
    lambda t: t[:, :1],
    lambda t: t.sum(),
], ids=["extra_leading_dim", "first_row", "first_column", "scalar"])
def test_wrong_shape_is_rejected_even_when_it_broadcasts(reshape):
    ref = torch.ones(3, 4)  # every broadcast of a slice of this equals the reference
    out = torch.ones_like(reshape(ref))
    metrics = compare_outputs(out, ref)
    assert not accepted(metrics)
    assert "shape" in metrics["mismatch"]


def test_non_finite_values_shared_with_the_reference_are_accepted():
    ref = torch.tensor([1.0, float("inf"), float("-inf"), float("nan")])
    assert accepted(compare_outputs(ref.clone(), ref))
    assert not accepted(compare_outputs(torch.tensor([1.0, float("-inf"), float("-inf"), float("nan")]), ref))
    assert not accepted(compare_outputs(torch.tensor([1.0, float("inf"), float("-inf"), 0.0]), ref))


def test_integer_and_boolean_outputs_are_compared():
    assert accepted(compare_outputs(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3])))
    assert not accepted(compare_outputs(torch.tensor([1, 2, 4]), torch.tensor([1, 2, 3])))
    assert accepted(compare_outputs(torch.tensor([True, False]), torch.tensor([True, False])))
    assert not accepted(compare_outputs(torch.tensor([True, True]), torch.tensor([True, False])))


def test_non_tensor_output_is_rejected():
    assert "expected a tensor" in compare_outputs([1.0, 2.0], torch.tensor([1.0, 2.0]))["mismatch"]


def test_mismatch_reason_reads_multi_output_metrics():
    assert mismatch_reason({"max_abs": 0.0}) == ""
    assert mismatch_reason({"per_output": [{"max_abs": 0.0}, {"mismatch": "bad"}]}) == "output 1: bad"


IN_PLACE_REFERENCE = """
import torch

def torch_scale(x, y):
    x.mul_(2.0)
    x.add_(y)
    return x
"""


def evaluate(candidate_source, reference_source=IN_PLACE_REFERENCE, name="scale"):
    spec = TaskSpec(name=name, correctness_fn=compare_outputs, reference_source=reference_source,
                    input_shapes=[[[8, 8], [8, 8]]])
    result = GenericEvaluator(EvalConfig(warmup=2, repeat=3), spec).evaluate_code(candidate_source)
    assert "error" not in result.metrics, result.metrics
    return result.metrics


def test_kernel_that_returns_its_input_untouched_is_rejected_for_an_in_place_operator():
    assert not accepted(evaluate("def triton_scale(x, y):\n    return x\n"))


def test_correct_in_place_kernel_is_accepted():
    assert accepted(evaluate("def triton_scale(x, y):\n    x.mul_(2.0)\n    x.add_(y)\n    return x\n"))


def test_correct_out_of_place_kernel_is_accepted():
    assert accepted(evaluate("def triton_scale(x, y):\n    return x * 2.0 + y\n"))


def test_multi_output_wrong_shape_is_rejected():
    reference = "import torch\n\ndef torch_pair(x, y):\n    return x + y, x - y\n"
    good = "def triton_pair(x, y):\n    return x + y, x - y\n"
    bad = "def triton_pair(x, y):\n    return x + y, (x - y).unsqueeze(0)\n"
    assert accepted(evaluate(good, reference, "pair"))
    metrics = evaluate(bad, reference, "pair")
    assert not accepted(metrics)
    assert "shape" in mismatch_reason(metrics)


ADD_STAND_IN = "import torch\n\ndef triton_add(a, b):\n    out = a + b\n    return {}\n"


@pytest.mark.parametrize("expression, expected", [
    ("out", ""),
    ("torch.full_like(out, float('nan'))", "NaN or infinity"),
    ("out.unsqueeze(0)", "shape"),
    ("out + 1", "tolerance exceeded"),
], ids=["correct", "all_nan", "extra_leading_dim", "off_by_1"])
def test_agent_facing_verifier_on_the_add_operator(expression, expected):
    message = verifier(ADD_STAND_IN.format(expression), _make_evaluator("add"))
    assert (message == "") if expected == "" else (expected in message)
