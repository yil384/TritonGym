from __future__ import annotations
import os
from time import sleep
import google.generativeai as genai

from .base import CodeGenClient


class GoogleClient(CodeGenClient):
    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        api_key: str | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.7,
    ):
        self.model = model  # gemini-1.5-pro, gemini-1.5-flash, etc.
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model)

        self.max_tokens = max_tokens
        self.temperature = temperature

    def create_msg(self, system: str, prompt: str, **kwargs) -> str:
        # Gemini merges system + user prompt into a single context
        prompt += "Please provide only the code snippet as the answer. Answer as briefly as possible. Please make sure the total_token_count <= 8192."
        full_prompt = f"System: {system}\nUser: {prompt}"
        retry_count = 3
        while retry_count > 0:
            retry_count -= 1

            response = self.client.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
                    temperature=kwargs.get("temperature", self.temperature),
                ),
            )
            sleep(1)
            try:
                print(response)
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    self.total_input_tokens += getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
                    self.total_output_tokens += getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
                return response.text.strip()
            except Exception as e:
                print(f"[WARN] Google Gemini API call failed: {e}, retrying...")
                continue
