from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class GenerationResult:
    """Container for the raw string output from a provider plus metadata.

    The provider-specific decoding/parsing into executable code is deferred
    to the agent's `decode` method so we keep the raw artifact intact.
    """
    raw: str
    metadata: Dict[str, str]

class CodeGenAgent(ABC):
    """Abstract agent that generates code (e.g., Triton kernels) from a spec.

    Implementations should encapsulate model/provider specifics.
    """
    @staticmethod
    def _evaluator(src: str, op) -> float:
        from benchmark.eval.tools import evaluator as _evaluator
        return _evaluator(src, op)

    @staticmethod
    def _verifier(src: str, op) -> bool:
        from benchmark.eval.tools import verifier as _verifier
        return _verifier(src, op)

    @staticmethod
    def _compiler(src: str, op) -> str:
        from benchmark.eval.tools import compiler as _compiler
        return _compiler(src, op)

    @abstractmethod
    def generate(self, torch_module_src: str, evaluator_obj = None, instructions: Optional[str] = None, **kwargs) -> GenerationResult:
        """Invoke provider using the provided module source (and optional instructions).

        Implementations internally construct prompts; callers supply only the
        high-level spec rather than a fully formed prompt string.
        """
        raise NotImplementedError
    
    @abstractmethod
    def decode(self, raw: str) -> str:
        """Decode/extract executable Python code from a raw model output string.

        Implementations should remove markdown fences, extraneous commentary,
        and ensure only python code is returned.
        """
        raise NotImplementedError
