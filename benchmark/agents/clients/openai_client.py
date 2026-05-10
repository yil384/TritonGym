from __future__ import annotations
import os
from time import sleep
from openai import OpenAI

from .base import CodeGenClient

class OpenAIClient(CodeGenClient):
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None, max_tokens: int = 8192, temperature: float = 0.7):
        self.model = model # gpt-4o, gpt-4.1, gpt-3.5-turbo, gpt 5 etc.
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=self.api_key)
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def create_msg(self, system: str, prompt: str, **kwargs) -> str:
        if self.model == "gpt-5":
            msg = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=kwargs.get("max_tokens", self.max_tokens),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
        else:
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
