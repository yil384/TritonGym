from __future__ import annotations
import os
from time import sleep
from together import Together

from .base import CodeGenClient

class MetaClient(CodeGenClient):
    def __init__(self, model: str = "meta-llama/Llama-3-70B-Instruct",
                 api_key: str | None = None,
                 max_tokens: int = 8192,
                 temperature: float = 0.7):
        self.model = model
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY not set")
        self.client = Together(api_key=self.api_key)
        self.max_tokens = max_tokens
        self.temperature = temperature

    def create_msg(self, system: str, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        if response.usage:
            self.total_input_tokens += response.usage.prompt_tokens or 0
            self.total_output_tokens += response.usage.completion_tokens or 0
        sleep(1)
        return response.choices[0].message.content