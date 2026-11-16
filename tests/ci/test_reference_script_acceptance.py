"""Full student acceptance gate: two positive and ten negative CPU controls.

The unmodified legacy scripts on this starter branch are
expected to FAIL all ten negative controls. These are not GPU kernel tests.
"""

import pytest

from tests.support.reference_contract import (
    COMPLETE_PREFIX,
    EXPECTED_CASE_COUNTS,
    SCENARIOS,
    call_trace,
    describe_result,
    run_reference_script,
)


@pytest.mark.parametrize("operator", EXPECTED_CASE_COUNTS)
def test_correct_outputs_complete_all_cases(operator):
    result = run_reference_script(operator)
    details = describe_result(result)
    assert result.returncode == 0, details
    assert COMPLETE_PREFIX in result.stdout, details
    assert len(call_trace(result)) == EXPECTED_CASE_COUNTS[operator], details


@pytest.mark.parametrize("operator", EXPECTED_CASE_COUNTS)
@pytest.mark.parametrize("scenario", [s for s in SCENARIOS if s != "correct"])
def test_bad_outputs_and_runtime_errors_exit_nonzero(operator, scenario):
    result = run_reference_script(operator, scenario)
    details = describe_result(result)
    trace = call_trace(result)
    assert trace, f"The script did not reach the injected kernel.\n{details}"
    assert all(call["operator"] == operator and call["scenario"] == scenario for call in trace), details
    assert result.returncode != 0, details
