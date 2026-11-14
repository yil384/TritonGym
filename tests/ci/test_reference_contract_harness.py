"""CPU harness smoke checks; this does not certify legacy failure handling."""

import pytest

from tests.support.reference_contract import (
    COMPLETE_PREFIX,
    call_trace,
    describe_result,
    run_reference_script,
)


EXPECTED_INPUTS = {
    "token_softmax_llama": [
        [([2, 16], "torch.float32")],
        [([5, 10], "torch.float32")],
    ],
    "attention_forward": [
        [([2, 4, 128, 128], "torch.float16")] * 3,
        [([2, 8, 256, 64], "torch.float16")] * 3,
        [([1, 4, 512, 64], "torch.float16")] * 3,
    ],
}


@pytest.mark.parametrize("operator", EXPECTED_INPUTS)
def test_positive_control_executes_every_existing_case_on_cpu(operator):
    result = run_reference_script(operator)
    details = describe_result(result)
    assert result.returncode == 0, details
    assert COMPLETE_PREFIX in result.stdout, details
    trace = call_trace(result)
    expected = EXPECTED_INPUTS[operator]
    assert len(trace) == len(expected), details
    for index, (call, inputs) in enumerate(zip(trace, expected), start=1):
        assert call["operator"] == operator, details
        assert call["scenario"] == "correct", details
        assert call["case"] == index, details
        assert [(arg["shape"], arg["dtype"]) for arg in call["inputs"]] == inputs, details
        assert all(arg["device"] == "cpu" for arg in call["inputs"]), details


@pytest.mark.parametrize("operator,scenario", [
    ("missing_operator", "correct"),
    ("token_softmax_llama", "missing_scenario"),
])
def test_invalid_selection_is_rejected(operator, scenario):
    with pytest.raises(ValueError):
        run_reference_script(operator, scenario)


def test_nonpositive_timeout_is_rejected():
    with pytest.raises(ValueError, match="timeout"):
        run_reference_script("token_softmax_llama", timeout=0)
