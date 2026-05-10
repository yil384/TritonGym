from __future__ import annotations
import time
from time import sleep
from collections import deque
import random
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from benchmark.eval.runner import GenericEvaluator
from .base import CodeGenAgent, GenerationResult

class GeakAgent(CodeGenAgent):
    """
    GEAK (Generating Efficient AI-centric GPU Kernels) is a multi-agent framework for automatic Triton kernel generation.
    It leverages inference-time compute scaling (sequential and parallel) with frontier LLMs, coordinating specialized
    agents (Generator, Evaluator, Reflector, Optimizer) to iteratively improve both correctness and performance of generated code.
    """
    def __init__(self, model: str, temperature: float = 0.7, max_sequential_iters: int = 3, max_parallel_runs: int = 1, max_perf_debug_num: int = 3):
        # Initialize the LLM client (same as AlphaEvolve)
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

        # GEAK specific parameters
        self.max_sequential_iters = max_sequential_iters  # Sequential Scaling: Max reflection/optimization iterations
        self.max_parallel_runs = max_parallel_runs        # Parallel Scaling: Number of independent parallel runs
        self.max_perf_debug_num = max_perf_debug_num      # Max debugging attempts per code snippet to avoid traps
        self.knowledge_base = self._load_triton_knowledge() # Domain knowledge for injection

    def _load_triton_knowledge(self) -> str:
        """Load domain-specific knowledge about writing efficient Triton kernels for AMD GPUs."""
        # This would contain detailed hardware specs and optimization principles
        knowledge = (
            "# AMD GPU (MI250/MI300X) Triton Optimization Knowledge:\n"
            " - Utilize shared memory for frequent data access within a thread block.\n"
            " - Ensure memory coalescing: consecutive threads should access consecutive memory addresses.\n"
            " - Use vectorized loads/stores (tl.load/tl.store) where possible.\n"
            " - Be mindful of bank conflicts in shared memory.\n"
            " - Use tiling to exploit data locality.\n"
            " - Prefer explicit masking (e.g., `mask = idx < size`) for boundary checks.\n"
            " - Kernel signature must include `@triton.jit` decorator and use `tl.constexpr` for compile-time constants.\n"
            " - ... (more specific knowledge) ...\n"
        )
        return knowledge

    def _build_generator_prompt(self, torch_module_src: str, instructions: str | None = None, 
                               feedback: str | None = None, optimization_goal: str | None = None) -> str:
        """Build the prompt for the Generator agent. Incorporates knowledge injection and context."""
        base = (
            "You are an expert in writing high-performance GPU kernels using Triton for AMD GPUs.\n"
            "Your task is to generate a correct and efficient Triton implementation based on the PyTorch operator.\n"
            "Rules:\n"
            " - Focus on correctness first; then optimize for speed (tiling, blocking, memory hierarchy).\n"
            " - Return ONLY executable Python code (no markdown, no commentary).\n"
            " - Name the function `triton_<operator_name>` where <operator_name> matches the reference `torch_<operator_name>`.\n"
            " - The function signature must match the reference exactly.\n"
            " - Use the following hardware-specific knowledge for optimization:\n"
        )
        # Knowledge Injection
        body = [base, self.knowledge_base, "\n# PyTorch function specification:\n", torch_module_src.strip()]
        
        # Additional Instructions & Feedback (Reflector/Optimizer input)
        if instructions:
            body.append("\n# Additional Instructions:\n")
            body.append(instructions.strip())
        if feedback: # From Reflector: error analysis and fix suggestions
            body.append("\n# Feedback from previous run (Analyze and Fix):\n")
            body.append(feedback)
        if optimization_goal: # From Optimizer: performance optimization strategies
            body.append("\n# Performance Optimization Goal:\n")
            body.append(optimization_goal)

        # 1-Shot Prompting (example retrieval could be implemented here)
        # inspirations = self.prompt_db.sample(1) # Get one similar example
        # if inspirations:
        #     body.append("\n# Example Triton Implementation (for reference):\n")
        #     body.append(inspirations[0]['code'])

        body.append("\n# Generate the Triton kernel code below:\n")
        return "".join(body)

    def _evaluator_agent(self, code: str, evaluator_obj: GenericEvaluator) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluator Agent: Cascaded evaluation.
        1. Functionality Test (Compile + Run + Unit Tests)
        2. Performance Test (If functionality passes)
        """
        error = self._compiler(code, evaluator_obj) or self._verifier(code, evaluator_obj)
        if error != "" and error is not None:
            return False, {"status": "fail", "error": error, "stage": "functionality"}
        else:
            try:
                latency_ms = self._evaluator(code, evaluator_obj) # Assuming this returns latency
                return True, {"status": "success", "latency_ms": latency_ms, "stage": "performance"}
            except Exception as e:
                # Handle potential errors during performance evaluation
                return False, {"status": "fail", "error": str(e), "stage": "performance"}

    def _reflector_agent(self, code: str, error_feedback: str, eval_result: Dict[str, Any]) -> str:
        """
        Reflector Agent: Analyzes errors and proposes fixes. Uses LLM for introspective debugging.
        """
        system_msg = "You are a debugging expert. Analyze the Triton kernel code and the error message. Propose concise, specific fixes to resolve the issue. Focus on the root cause."
        prompt = (
            f"# Triton Kernel Code:\n```python\n{code}\n```\n\n"
            f"# Error Feedback (Compilation/Runtime):\n{error_feedback}\n\n"
            f"# Evaluation Result Context:\n{eval_result}\n\n"
            "Analyze the error and suggest the key changes needed to fix the code. Be specific about what to change and why:"
        )
        # For simplicity, we use the same client. Could use a different model/configuration.
        reflection = self.client.create_msg(system_msg, prompt, max_tokens=500)
        return reflection # Returns textual feedback for the Generator

    def _optimizer_agent(self, code: str, latency_ms: float, history: List[Dict[str, Any]]) -> str:
        """
        Optimizer Agent: Proposes performance optimizations based on current code and past results.
        """
        # Prepare history context for the LLM
        history_context = ""
        for i, entry in enumerate(history[-3:]): # Show last 3 attempts for context
            history_context += f"Attempt {i+1}: Latency {entry.get('latency_ms', 'N/A')} ms | Status: {entry.get('status', 'N/A')}\n"
            # Could include code snippets or error summaries if needed

        system_msg = "You are a performance optimization expert for AMD GPU Triton kernels. Suggest specific, actionable optimizations to reduce kernel latency."
        prompt = (
            f"# Current Triton Kernel Code (Latency: {latency_ms} ms):\n```python\n{code}\n```\n\n"
            f"# Recent Generation History:\n{history_context}\n\n"
            "Suggest specific optimizations (e.g., better tiling, memory access patterns, using shared memory, vectorization). Focus on the most promising changes:"
        )
        optimization_advice = self.client.create_msg(system_msg, prompt, max_tokens=400)
        return optimization_advice # Returns textual strategies for the Generator

    def _run_sequential_iteration(self, torch_module_src: str, evaluator_obj: GenericEvaluator, 
                                 initial_instructions: str | None = None) -> GenerationResult:
        """
        Runs one sequential iteration of the GEAK loop (Generate -> Evaluate -> Reflect/Optimize).
        This represents one path of sequential scaling.
        """
        current_code = None
        current_result = None
        feedback = None
        optimization_goal = None
        history = [] # Tracks attempts in this sequential run
        debug_attempts = 0

        for seq_iter in range(self.max_sequential_iters):
            # 1. GENERATE
            prompt = self._build_generator_prompt(torch_module_src, initial_instructions, feedback, optimization_goal)
            raw_code = self.client.create_msg("You are an expert Triton programmer.", prompt)
            current_code = self.decode(raw_code)

            # 2. EVALUATE
            is_correct, eval_result = self._evaluator_agent(current_code, evaluator_obj)
            history.append({"code": current_code, "result": eval_result.copy()}) # Store history for optimizer
            current_result = eval_result

            print(f"[Seq Iter {seq_iter+1}]: Status: {eval_result['status']} (Stage: {eval_result.get('stage', 'N/A')}), Latency: {eval_result.get('latency_ms', 'N/A')} ms, Error: {eval_result.get('error', '')}")

            if is_correct:
                # Success! Now try to optimize if we haven't started already
                if eval_result['stage'] == 'performance' and optimization_goal is None:
                    # First time passing functionality, switch to optimization mode
                    print("Switching to optimization mode.")
                    optimization_goal = self._optimizer_agent(current_code, eval_result['latency_ms'], history)
                    feedback = None # Clear any previous error feedback
                    debug_attempts = 0 # Reset debug counter on success
                elif eval_result['stage'] == 'performance':
                    # We are already in optimization mode, get new advice
                    optimization_goal = self._optimizer_agent(current_code, eval_result['latency_ms'], history)
                    feedback = None
                    debug_attempts = 0
                # Else: Still in functionality stage but passed? (Shouldn't happen normally)
            else:
                # Failure - Use Reflector
                if eval_result['status'] == 'fail':
                    debug_attempts += 1
                    if debug_attempts > self.max_perf_debug_num and optimization_goal:
                        # Debugging Trap: Stuck optimizing failing code. Reset strategy.
                        print("Debugging trap detected. Resetting optimization strategy.")
                        optimization_goal = None
                        feedback = "Previous optimization attempts led to failures. Prioritize correctness again."
                        debug_attempts = 0
                    else:
                        feedback = self._reflector_agent(current_code, eval_result.get('error', 'Unknown error'), eval_result)
                # Reset optimization goal if we fail during optimization phase
                if optimization_goal and not is_correct:
                    # optimization_goal = None // Keep the goal but Reflector will suggest fixes
                    pass

            sleep(1) # Be nice to the API

            # Early termination if we have a good result and aren't optimizing further?
            # if is_correct and eval_result['stage'] == 'performance' and seq_iter > 0:
            #   break

        # Return the best code from the sequential run, or the last attempt
        # A more sophisticated version would track the best performer in history
        best_attempt = max(history, key=lambda x: x['result'].get('latency_ms', float('-inf')) if x['result'].get('status') == 'success' and x['result'].get('stage') == 'performance' else float('-inf'), default=None)
        final_code = best_attempt['code'] if best_attempt else current_code
        final_result = best_attempt['result'] if best_attempt else current_result

        return GenerationResult(
            raw=final_code, # This might not be the *raw* LLM output, but the final code
            metadata={
                "model": self.client.model,
                "sequential_iterations": seq_iter + 1,
                "evaluation_result": final_result,
                "history": history # For debugging and analysis
            }
        )

    def generate(self, torch_module_src: str, evaluator_obj: GenericEvaluator,
                 system: str = "You are an expert in writing GPU programs with Triton compiler.",
                 instructions: str | None = None, **kwargs) -> GenerationResult:
        """
        Main generate method for GEAK. Implements parallel scaling by running multiple
        sequential iterations concurrently and selecting the best result.
        """
        # System message might be used elsewhere, instructions are used in the prompt
        # `system` parameter might override the default in _build_generator_prompt if needed

        results = []
        # Parallel Scaling: Run multiple independent sequential runs
        with ThreadPoolExecutor(max_workers=self.max_parallel_runs) as executor:
            future_to_run = {
                executor.submit(self._run_sequential_iteration, torch_module_src, evaluator_obj, instructions): i 
                for i in range(self.max_parallel_runs)
            }
            for future in as_completed(future_to_run):
                try:
                    result = future.result()
                    results.append(result)
                    run_id = future_to_run[future]
                    print(f"Parallel run {run_id+1} completed. Status: {result.metadata['evaluation_result']['status']}")
                except Exception as exc:
                    print(f'A parallel run generated an exception: {exc}')
                    # Append a failed result or handle accordingly
                    results.append(GenerationResult(
                        raw="",
                        metadata={"model": self.client.model, "error": str(exc), "evaluation_result": {"status": "fail", "error": str(exc)}}
                    ))

        # Select the best result from parallel runs based on latency (if successful) or other criteria
        best_result = None
        successful_results = [r for r in results if r.metadata['evaluation_result'].get('status') == 'success' and r.metadata['evaluation_result'].get('stage') == 'performance']
        if successful_results:
            best_result = min(successful_results, key=lambda x: x.metadata['evaluation_result'].get('latency_ms', float('inf')))
        else:
            # If no successful performance result, maybe choose one that passed functionality?
            functional_results = [r for r in results if r.metadata['evaluation_result'].get('status') == 'success' and r.metadata['evaluation_result'].get('stage') == 'functionality']
            if functional_results:
                best_result = functional_results[0] # Just pick the first one that compiles/verifies
            else:
                best_result = results[0] # Fallback to the first result

        # Add parallel run summary to metadata
        best_result.metadata['parallel_runs'] = self.max_parallel_runs
        best_result.metadata['all_results'] = [r.metadata['evaluation_result'] for r in results] # Summary of all runs
        best_result.metadata.update(self.client.get_token_usage())

        return best_result

    def decode(self, raw: str) -> str:
        # Reuse the decoding logic from AlphaEvolveAgent
        code = raw
        if "```" in code:
            segments = code.split("```")
            if len(segments) >= 2:
                candidate = segments[1]
                first_line, *rest = candidate.splitlines()
                if first_line.strip().startswith("python") or first_line.strip().startswith("py"):
                    code = "\n".join(rest)
                else:
                    code = candidate
        code = code.replace("```", "").strip().strip('`').strip()
        return code
