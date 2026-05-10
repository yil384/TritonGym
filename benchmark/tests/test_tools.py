import math
import pytest

from benchmark.agents.clients.anthropic_client import AnthropicAgent
from benchmark.eval.tools import compiler, evaluator, verifier, BENCHMARK_JSON


def _ensure_benchmark_json():
	# Rebuild benchmark.json from current sources to include 'add'
	from benchmark.data.save_json import read_ops
	ops = read_ops()
	BENCHMARK_JSON.parent.mkdir(parents=True, exist_ok=True)
	BENCHMARK_JSON.write_text(__import__('json').dumps(ops, indent=2))


class _Gen:
	def __init__(self, raw: str):
		self.raw = raw


def _agent_code(monkeypatch, code_str: str) -> str:
	# Bypass API key check in __init__
	monkeypatch.setattr(AnthropicAgent, "__init__", lambda self, *a, **k: None)
	agent = AnthropicAgent(model="test", temperature=0.0)  # type: ignore[call-arg]
	monkeypatch.setattr(agent, "generate", lambda *a, **k: _Gen(code_str))
	monkeypatch.setattr(agent, "decode", lambda raw: raw)
	gen = agent.generate("")
	return agent.decode(gen.raw)


GOOD_ADD = """
import torch
def triton_add(input, other):
	return torch.add(input, other)
""".strip()

BAD_NAME = """
def not_triton_add(a, b):
	return a + b
""".strip()

BAD_IMPL = """
import torch
def triton_add(input, other):
	return torch.sub(input, other)
""".strip()

SYNTAX_ERR = "def triton_add(:\n    pass\n"


def test_compiler_with_agent_add(monkeypatch):
	_ensure_benchmark_json()
	code = _agent_code(monkeypatch, GOOD_ADD)
	err = compiler(code, op="add")
	assert err == ""


def test_compiler_errors(monkeypatch):
	_ensure_benchmark_json()
	# Missing symbol
	code = _agent_code(monkeypatch, BAD_NAME)
	err = compiler(code, op="add")
	assert "symbol triton_add missing" in err
	# Syntax error
	code2 = _agent_code(monkeypatch, SYNTAX_ERR)
	err2 = compiler(code2, op="add")
	assert err2


def test_evaluator_with_agent_add(monkeypatch):
	_ensure_benchmark_json()
	code = _agent_code(monkeypatch, GOOD_ADD)
	ms = evaluator(code, op="add")
	assert isinstance(ms, float)
	# Could be NaN on environments without proper backend; allow non-negative or NaN
	assert math.isnan(ms) or ms >= 0.0


def test_verifier_true_and_false(monkeypatch):
	_ensure_benchmark_json()
	good = _agent_code(monkeypatch, GOOD_ADD)
	bad = _agent_code(monkeypatch, BAD_IMPL)
	assert verifier(good, op="add") is True
	assert verifier(bad, op="add") is False
	