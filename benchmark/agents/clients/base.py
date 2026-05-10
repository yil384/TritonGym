from __future__ import annotations
from abc import ABC, abstractmethod

class CodeGenClient(ABC):
    """Abstract model that generates code (e.g., Triton kernels) from a spec."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _orig_init = cls.__init__
        def _wrapped_init(self, *args, **kw):
            _orig_init(self, *args, **kw)
            # Ensure token counters exist after subclass __init__
            if not hasattr(self, 'total_input_tokens'):
                self.total_input_tokens = 0
            if not hasattr(self, 'total_output_tokens'):
                self.total_output_tokens = 0
        cls.__init__ = _wrapped_init

    @abstractmethod
    def create_msg(self, system: str, prompt: str, **kwargs) -> str:
        """
        Create a message to send to the model API.
        Returns the raw string output from the model.
        """
        raise NotImplementedError

    def get_token_usage(self) -> dict:
        """Return cumulative token usage."""
        return {
            "total_input_tokens": getattr(self, 'total_input_tokens', 0),
            "total_output_tokens": getattr(self, 'total_output_tokens', 0),
        }
