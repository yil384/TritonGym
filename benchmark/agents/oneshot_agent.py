from __future__ import annotations

from .base import CodeGenAgent, GenerationResult

class OneShotAgent(CodeGenAgent):
    def __init__(self, model, temperature: float = 0.7):
        if model.startswith('gpt'):
            from benchmark.agents.clients.openai_client import OpenAIClient
            self.client = OpenAIClient(model=model, temperature=temperature)
        elif model.startswith('gemini'):
            from benchmark.agents.clients.google_client import GoogleClient
            self.client = GoogleClient(model=model, temperature=temperature)
        elif model.startswith('claude'):
            from benchmark.agents.clients.anthropic_client import AnthropicClient
            self.client = AnthropicClient(model=model, temperature=temperature)
        elif model.startswith('deepseek'):
            from benchmark.agents.clients.deepseek_client import DeepSeekClient
            self.client = DeepSeekClient(model=model, temperature=temperature)
        elif model.startswith('meta') or model.startswith('Qwen'):
            from benchmark.agents.clients.meta_client import MetaClient
            self.client = MetaClient(model=model, temperature=temperature)
        else:
            raise ValueError(f"Unsupported model: {model}")
    """
    OneShotAgent generates a Triton implementation of a given PyTorch operator in a single attempt, without iterative refinement.
    It is suitable for straightforward operators where a single well-crafted prompt can yield a correct and efficient solution.
    """
    def _build_prompt(self, torch_module_src: str, instructions: str | None = None) -> str:
        base = (
            "You are an expert in writing high-performance GPU kernels using Triton.\n"
            "Your task is to iteratively improve the provided PyTorch operator by generating a Triton implementation.\n"
            "Rules:\n"
            " - Focus on correctness first; then optimize for speed (tiling, blocking, memory hierarchy).\n"
            " - Return ONLY executable Python code (no markdown, no commentary).\n"
            " - Name the function `triton_<operator_name>` where <operator_name> matches the reference `torch_<operator_name>`.\n"
            " - The function signature must match the reference exactly.\n"
            " - If you modify code, use a SEARCH/REPLACE diff style when possible.\n"
            " - If the previous version failed, explicitly fix the cause while keeping other parts intact.\n"
        )
        body = ["# PyTorch function specification:", torch_module_src.strip()]
        if instructions:
            body.append("# Additional instructions:")
            body.append(instructions.strip())
        return base + "\n".join(body) + "\n"
    
    def generate(self, torch_module_src: str, system: str = "You are an expert in writing GPU programs with Triton compiler.",
                 instructions: str | None = None, **kwargs) -> GenerationResult:
        """
        Generic evolutionary generate loop.
        Requires `create_msg(system, prompt, **kwargs)` to be passed by the Agent.
        """

        base_prompt = self._build_prompt(torch_module_src, instructions)
        raw = self.client.create_msg(system, base_prompt, **kwargs)

        return GenerationResult(raw=raw, metadata={"model": self.client.model, **self.client.get_token_usage()})
    
    def decode(self, raw: str) -> str:
        code = raw
        # Find first fenced code block if present
        if "```" in code:
            segments = code.split("```")
            # Heuristic: code likely in first non-empty segment after opening fence
            if len(segments) >= 2:
                # segments[0] is preamble, segments[1] might start with language id
                candidate = segments[1]
                # Drop leading language hint
                first_line, *rest = candidate.splitlines()
                if first_line.strip().startswith("python") or first_line.strip().startswith("py"):
                    code = "\n".join(rest)
                else:
                    code = candidate
        # Remove any residual fences/backticks
        code = code.replace("```", "").strip().strip('`').strip()
        return code