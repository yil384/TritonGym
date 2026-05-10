from __future__ import annotations
import os
from time import sleep
import anthropic

from .base import CodeGenClient

class AnthropicClient(CodeGenClient):
    def __init__(self, model: str = "claude-sonnet-4-20250514",
                 api_key: str | None = None,
                 max_tokens: int = 8192,
                 temperature: float = 0.7):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.max_tokens = max_tokens
        self.temperature = temperature

    def create_msg(self, system: str, prompt: str, **kwargs) -> str:
        """Send prompt to Anthropic API and return raw text output."""
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            system=system,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
        if msg.usage:
            self.total_input_tokens += msg.usage.input_tokens or 0
            self.total_output_tokens += msg.usage.output_tokens or 0
        text_parts = [c.text for c in msg.content if getattr(c, "type", None) == "text"]
        raw = "\n".join(text_parts) if text_parts else msg.content[0].text
        sleep(1)
        return raw
