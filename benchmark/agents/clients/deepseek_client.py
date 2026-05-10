from __future__ import annotations
import os
from time import sleep
from openai import OpenAI

from .base import CodeGenClient

class DeepSeekClient(CodeGenClient):
    def __init__(self, model: str = "deepseek-chat",
                 api_key: str | None = None,
                 max_tokens: int = 8192,
                 temperature: float = 0.7):
        """
        model: deepseek-chat, deepseek-coder, or deepseek-reasoner
        """
        self.model = model
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not set")
        # Use OpenAI SDK with base_url pointed to DeepSeek
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com/v1")
        self.max_tokens = max_tokens
        self.temperature = temperature

    def create_msg(self, system: str, prompt: str, **kwargs) -> str:
        msg = self.client.chat.completions.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        if msg.usage:
            self.total_input_tokens += msg.usage.prompt_tokens or 0
            self.total_output_tokens += msg.usage.completion_tokens or 0
        sleep(1)
        return msg.choices[0].message.content
