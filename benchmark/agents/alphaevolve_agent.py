from __future__ import annotations
import time
from time import sleep
from collections import deque
import random

from benchmark.eval.runner import GenericEvaluator
from .base import CodeGenAgent, GenerationResult

class AlphaEvolveAgent(CodeGenAgent):
    """
    AlphaEvolve is an evolutionary coding agent that iteratively improves algorithms by directly modifying code and leveraging evaluator feedback. 
    It has demonstrated strong results in optimizing large-scale computational systems (e.g., datacenter scheduling, accelerator design, LLM training) and in discovering novel, 
    provably correct algorithms in math and computer science—showcasing the broad potential of automated code evolution for scientific and engineering breakthroughs.
    """
    def __init__(self, model, temperature: float = 0.7, attempts: int = 5, sample_size: int = 3, max_size: int = 50):
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

        self.prompt_db = PromptDatabase(max_size=max_size)
        self.attempts = attempts
        self.sample_size = sample_size

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
    
    def generate(self, torch_module_src: str, evaluator_obj: GenericEvaluator,
             system: str = "You are an expert in writing GPU programs with Triton compiler.",
             instructions: str | None = None, **kwargs) -> GenerationResult:
        """
        Generic evolutionary generate loop.
        Requires `create_msg(system, prompt, **kwargs)` to be passed by the Agent.
        """

        base_prompt = self._build_prompt(torch_module_src, instructions)
        candidates = []  # Store all attempt results (raw, decoded, eval_result)

        for attempt in range(self.attempts):
            # Sample inspirations from history
            inspirations = self.prompt_db.sample(self.sample_size)
            prompt = self.prompt_db.build(base_prompt, inspirations)

            # Call the Agent-provided interface
            raw = self.client.create_msg(system, prompt, **kwargs)
            decoded = self.decode(raw)

            # Evaluate and verify
            error = self._compiler(decoded, evaluator_obj) or self._verifier(decoded, evaluator_obj)
            if error == "" or error is None:
                ms = self._evaluator(decoded, evaluator_obj)
                eval_result = {"status": "success", "latency_ms": ms}
            else:
                eval_result = {"status": "fail", "error": error}

            # Store in database
            self.prompt_db.add(prompt, decoded, eval_result)

            # Save candidate result
            candidates.append((raw, decoded, eval_result))

            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')} Attempt {attempt+1}]: Eval result: {eval_result}")
            sleep(1)

        # ------- Select best result -------
        # Filter successful candidates
        success_candidates = [c for c in candidates if c[2].get("status") == "success"]

        if success_candidates:
            # Select best by lowest latency_ms
            best_raw, best_decoded, best_eval = min(
                success_candidates, key=lambda c: c[2].get("latency_ms", float("inf"))
            )
        else:
            # No successful candidates; fall back to the last attempt
            best_raw, best_decoded, best_eval = candidates[-1]

        return GenerationResult(raw=best_raw, metadata={"model": self.client.model, "eval": best_eval, **self.client.get_token_usage()})

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

import difflib

class PromptDatabase:
    """store past prompts, generated code diffs, and evaluation results for inspiration"""
    def __init__(self, max_size: int = 50):
        self.db = deque(maxlen=max_size)
        self.last_code = None  # <--- track the last full code

    def add(self, prompt: str, code: str, result: dict):
        if self.last_code is None:
            # First entry, store full code
            entry_code = code
        else:
            # Subsequent entries, store diff from last_code
            diff_lines = difflib.unified_diff(
                self.last_code.splitlines(),
                code.splitlines(),
                fromfile="prev",
                tofile="new",
                lineterm=""
            )
            entry_code = "\n".join(diff_lines) or code  # <--- if no diff, store full code

        self.db.append({"prompt": prompt, "code": entry_code, "result": result})
        self.last_code = code  # <--- update last_code

    def sample(self, k: int = 3):
        if len(self.db) == 0:
            return []
        success = [e for e in self.db if e["result"].get("status") == "success"]
        fail = [e for e in self.db if e["result"].get("status") != "success"]

        success_sorted = sorted(
            success,
            key=lambda e: e["result"].get("latency_ms", float("inf"))
        )
        chosen = success_sorted[:k]

        if len(chosen) < k and fail:
            need = k - len(chosen)
            chosen += random.sample(fail, min(need, len(fail)))

        return chosen
    
    def build(self, base_prompt: str, inspirations: list[dict]) -> str:
        if not inspirations:
            return base_prompt
        
        context_blocks = []
        for i, entry in enumerate(inspirations):
            context = f"\n# Past trial {i+1} (diff or initial code):\n"
            context += f"{entry['code']}\n"
            context += f"# Result: {entry['result']}\n"
            context_blocks.append(context)

        meta_instruction = (
            "\n# Inspirations from past trials:\n"
            + "".join(context_blocks)
            + "\n# Analyze the above past trial carefully. Identify what worked correctly and what caused errors or inefficiencies. "
            + "\nRetain the correct and efficient logic whenever possible, but modify or replace the faulty parts. "
            + "\n# Learn from the past trials: fix correctness errors, preserve working logic, and aim to reduce latency if possible."
            + "\nGenerate a new improved version below:\n"
        )
        return base_prompt + meta_instruction
