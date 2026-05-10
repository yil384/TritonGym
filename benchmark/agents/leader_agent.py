from __future__ import annotations
import time
import json
import random
import difflib
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import deque
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from benchmark.eval.runner import GenericEvaluator
from .base import CodeGenAgent, GenerationResult
from ..eval.tools import searcher

class ToolType(Enum):
    LLM = "llm"
    FUNCTION = "function"
    SEARCHER = "searcher"
    VERIFIER = "verifier"
    PROFILER = "profiler"
    DATABASE = "database"
    EVALUATOR = "evaluator"
    ANALYZER = "analyzer"

class TaskState(Enum):
    INIT = "init"
    CODE_GENERATION = "code_generation"
    VERIFICATION = "verification"
    OPTIMIZATION = "optimization"
    ANALYSIS = "analysis"
    SEARCH = "search"
    REFLECTION = "reflection"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Tool:
    name: str
    tool_type: ToolType
    description: str
    function: Callable
    cost: float = 0.0
    success_rate: float = 1.0

@dataclass
class Task:
    goal: str
    state: TaskState
    context: Dict[str, Any]
    dependencies: List[str]
    priority: int = 1
    max_retries: int = 3
    retry_count: int = 0
    created_time: float = None
    
    def __post_init__(self):
        if self.created_time is None:
            self.created_time = time.time()

class LeaderAgent(CodeGenAgent):
    """
    Advanced agent with task planning capabilities that orchestrates multiple tools
    """
    
    def __init__(self, model: str, temperature: float = 0.7, max_budget: float = 40.0, max_iterations: int = 4):
        # Initialize LLM client
        if model.startswith('gpt'):
            from benchmark.agents.clients.openai_client import OpenAIClient
            self.llm_client = OpenAIClient(model=model, temperature=temperature)
        elif model.startswith('gemini'):
            from benchmark.agents.clients.google_client import GoogleClient
            self.llm_client = GoogleClient(model=model, temperature=temperature)
        elif model.startswith('claude'):
            from benchmark.agents.clients.anthropic_client import AnthropicClient
            self.llm_client = AnthropicClient(model=model, temperature=temperature)
        elif model.startswith('deepseek'):
            from benchmark.agents.clients.deepseek_client import DeepSeekClient
            self.llm_client = DeepSeekClient(model=model, temperature=temperature)
        elif model.startswith('meta') or model.startswith('Qwen'):
            from benchmark.agents.clients.meta_client import MetaClient
            self.llm_client = MetaClient(model=model, temperature=temperature)
        else:
            raise ValueError(f"Unsupported model: {model}")

        # Task planning state
        self.current_task: Optional[Task] = None
        self.task_queue: deque[Task] = deque()
        self.completed_tasks: Dict[str, Task] = {}
        self.available_budget: float = max_budget
        self.max_budget: float = max_budget
        self.max_iterations: int = max_iterations
        self.current_iteration: int = 0
        
        # Tool registry
        self.tools: Dict[str, Tool] = {}
        self._register_core_tools()
        
        # Knowledge and history
        self.knowledge_base = KnowledgeBase()
        self.execution_history: List[Dict[str, Any]] = []
        self.generated_codes: List[Dict[str, Any]] = []  # Store all generated codes with results
        
        # Configuration
        self.max_parallel_tasks = 1  # Reduced for sequential workflow
        self.max_reflection_depth = 5

        # Store previous code for diffing
        self.previous_code: Optional[str] = None
        self.code_diffs: List[Dict[str, Any]] = []  # Store code diffs with context

    def _register_core_tools(self):
        """Register core tools available to the task planner"""
        # LLM-based tools
        self.register_tool(Tool(
            name="llm_code_generator",
            tool_type=ToolType.LLM,
            description="Generate Triton kernel code based on specifications and context",
            function=self._llm_generate_code,
            cost=2.0
        ))
        
        self.register_tool(Tool(
            name="llm_analyzer",
            tool_type=ToolType.LLM,
            description="Analyze code, errors, and provide insights and recommendations",
            function=self._llm_analyze,
            cost=1.5
        ))
        
        self.register_tool(Tool(
            name="llm_planner",
            tool_type=ToolType.LLM,
            description="Plan next steps and select appropriate tools based on current state",
            function=self._llm_plan,
            cost=1.0
        ))

        # Functional tools
        self.register_tool(Tool(
            name="code_verifier",
            tool_type=ToolType.FUNCTION,
            description="Verify code correctness through compilation and basic validation",
            function=self._verify_code,
            cost=0.5
        ))
        
        self.register_tool(Tool(
            name="performance_profiler",
            tool_type=ToolType.FUNCTION,
            description="Profile code performance and identify bottlenecks",
            function=self._profile_code,
            cost=1.0
        ))
        
        self.register_tool(Tool(
            name="knowledge_retriever",
            tool_type=ToolType.DATABASE,
            description="Retrieve relevant knowledge and examples from knowledge base",
            function=self._retrieve_knowledge,
            cost=0.3
        ))
        
        self.register_tool(Tool(
            name="execution_evaluator",
            tool_type=ToolType.EVALUATOR,
            description="Comprehensive evaluation of code execution results",
            function=self._evaluate_execution,
            cost=0.7
        ))
        
        self.register_tool(Tool(
            name="web_searcher",
            tool_type=ToolType.SEARCHER,
            description="Search the web for additional information and examples",
            function=self._web_search,
            cost=1.5
        ))

    def register_tool(self, tool: Tool):
        """Register a new tool for the planner to use"""
        self.tools[tool.name] = tool

    def _llm_generate_code(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """LLM tool for code generation"""
        print("\033[94m[DEBUG]\033[0m [llm_generate_code] Generating code with context")

        # Build enhanced prompt with diff information
        prompt = self._build_diff_based_generation_prompt(context)
        print("\033[94m[DEBUG]\033[0m [llm_generate_code] Generated prompt for code generation")
        print(prompt)  # Debug print of the generation prompt
        
        response = self.llm_client.create_msg(
            "You are an expert Triton GPU kernel developer. Generate efficient and correct code based on the provided diff context.",
            prompt,
            max_tokens=8192
        )
        
        code = self.decode(response)
        print("\033[92m[OK]\033[0m [llm_generate_code] Code generated successfully")
        
        # Store the full generated code each time
        timestamp = time.time()
        filename = f"generated_code/aa_leader_{timestamp}.py"
        with open(filename, "w") as f:
            f.write(f"# Prompt context:\n\"\"\"\n{context.get('instructions', '')}\n\"\"\"\n\n")
            f.write(code)
    
        
        return {
            "code": code,
            "raw_response": response,
            "generation_context": context,
            "timestamp": time.time()
        }

    def _build_diff_based_generation_prompt(self, context: Dict[str, Any]) -> str:
        """Build a diff-based generation prompt"""
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
        
        components = [base]
        
        # Add original PyTorch code (key reference)
        if 'torch_code' in context:
            torch_snippet = self._extract_code_snippet(context['torch_code'], 15)
            components.append(f"# PyTorch Reference Code (snippet):\n{torch_snippet}\n\n")
        
        # Add diff information if available
        if 'previous_diff' in context and context['previous_diff']:
            components.append("# Previous Code Changes:\n")
            components.append(context['previous_diff'])
            components.append("\n")
        
        # Add error history if available (condensed display)
        if 'error_history' in context and context['error_history']:
            components.append("# Recent Errors to Fix:\n")
            for i, error in enumerate(context['error_history'][-4:]):  # Show only the 4 most recent errors
                error_summary = error[:300] + "..." if len(error) > 300 else error
                components.append(f"Error {i+1}: {error_summary}\n")
            components.append("\n")
        
        # Add key recommendations if available
        if 'recommendations' in context:
            components.append("# Key Recommendations:\n")
            for rec in context['recommendations'][-2:]:  # Show only the 2 most important recommendations
                rec_summary = rec[:50] + "..." if len(rec) > 50 else rec
                components.append(f"- {rec_summary}\n")
            components.append("\n")
        
        # Add current instructions
        current_instructions = context.get('instructions', 'Improve the Triton implementation.')
        components.append(f"# Current Task:\n{current_instructions}\n\n")
        
        components.append(
            "Rules:\n"
            " - Focus on correctness first, then optimization\n"
            " - Return ONLY executable Python code with Triton imports\n"
            " - Use proper Triton idioms and AMD GPU optimizations\n"
            "Generate the whole and correct Triton kernel code:\n"
        )
        
        return "".join(components)

    def _store_code_history(self, new_code: str, iteration: int, context: Dict[str, Any], test_result: Dict[str, Any]) -> Dict[str, Any]:
        """Store full code history and diff information"""
        # Store full code
        code_info = {
            'iteration': iteration,
            'full_code': new_code,
            'timestamp': time.time(),
            'test_result': test_result,
            'context_snapshot': self._create_lightweight_context(context)
        }
        
         # Compute diff from previous iteration (used in the next iteration's prompt)
        if self.previous_code is not None:
            diff_info = self._compute_meaningful_diff(self.previous_code, new_code)
            code_info['diff_from_previous'] = diff_info
            code_info['diff_length'] = len(diff_info) if diff_info else 0
        else:
            # First iteration, mark as initial version
            code_info['diff_from_previous'] = "# First version\n"
            code_info['diff_length'] = 0
        
        self.generated_codes.append(code_info)
        self.previous_code = new_code
        
        return code_info
    
    def _compute_meaningful_diff(self, old_code: str, new_code: str) -> str:
        """Compute a meaningful diff, filtering out trivial changes"""
        if not old_code or not new_code:
            return "# First version\n"
        
        old_lines = old_code.splitlines()
        new_lines = new_code.splitlines()
        
        # Use unified diff format
        diff = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile='previous.py',
            tofile='current.py',
            lineterm='',
            n=2  # Number of context lines
        ))
        
        if not diff:
            return "# No significant changes\n"
        
        # Filter out changes that are only whitespace or comments
        meaningful_changes = []
        for i, line in enumerate(diff):
            if i < 2:  # Keep file headers
                meaningful_changes.append(line)
                continue
                
            if line.startswith('+') or line.startswith('-'):
                # Check if it is a significant change (not just whitespace/comments)
                content = line[1:].strip()
                if content and not content.startswith('#') and content not in ['', ' ']:
                    meaningful_changes.append(line)
            else:
                meaningful_changes.append(line)
        
        if len(meaningful_changes) <= 3:  # Only file headers remain
            return "# Minor changes (whitespace/comments only)\n"
        
        return '\n'.join(meaningful_changes)

    def _get_previous_diff_for_prompt(self, iteration: int) -> str:
        """Get diff information for the prompt"""
        if iteration == 0 or len(self.generated_codes) < 2:
            return "# First iteration - no previous changes\n"
        
        # Prefer retrieving diff info from generated_codes
        if iteration - 1 < len(self.generated_codes):
            prev_code_info = self.generated_codes[iteration - 1]
            if 'diff_from_previous' in prev_code_info:
                diff_content = prev_code_info['diff_from_previous']
                if diff_content and diff_content.strip() != "# First version":
                    return diff_content
        
        # Fall back to retrieving from code_diffs
        if iteration - 1 < len(self.code_diffs):
            diff_info = self.code_diffs[iteration - 1]
            if diff_info['diff_type'] == 'diff':
                return diff_info['content']
        
        return "# Previous changes not available\n"


    def _llm_analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """LLM tool for analysis and insights"""
        print("\033[94m[DEBUG]\033[0m [llm_analyze] Analyzing context")
        
        analysis_prompt = self._build_analysis_prompt(context)
        print(analysis_prompt)  # Debug print of the analysis prompt
        response = self.llm_client.create_msg(
            "You are a code analysis expert. Provide deep insights and recommendations.",
            analysis_prompt,
            max_tokens=200
        )
        
        recommendations = self._extract_recommendations(response)
        print("\033[92m[OK]\033[0m [llm_analyze] Analysis completed")
        
        return {
            "analysis": response, 
            "recommendations": recommendations,
            "timestamp": time.time()
        }

    def _llm_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """LLM tool for task planning and tool selection"""
        planning_prompt = self._build_planning_prompt(context)
        print(planning_prompt)
        response = self.llm_client.create_msg(
            "You are a strategic planner. Decide the best next steps and tools to use.",
            planning_prompt,
            max_tokens=100
        )
        return self._parse_planning_response(response)

    def _verify_code(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify code correctness - simplified for demo"""
        code = context.get('code', '')
        evaluator = context.get('evaluator')
        
        if not code:
            return {"success": False, "error": "No code provided"}
        
        # Simulate verification process
        try:
            # Check if code looks like valid Python/Triton code
            if "def " not in code and "import " not in code:
                return {"success": False, "error": "Code doesn't contain function definition"}

            error = self._compiler(code, evaluator) or self._verifier(code, evaluator)

            if error == "" or error is None:
                return {"success": True, "stage": "verification"}
            else:
                return {"success": False, "error": error, "stage": "verification"}

            return {"success": True, "stage": "verification"}
        except Exception as e:
            return {"success": False, "error": str(e), "stage": "verification"}

    def _profile_code(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Profile code performance - simplified for demo"""
        code = context.get('code', '')
        
        if not code:
            return {"success": False, "error": "No code provided"}
        
        try:
            latency_ms = self._evaluator(code, context.get('evaluator'))
            
            return {
                "success": True, 
                "latency_ms": latency_ms, 
                "stage": "profiling"
            }
        except Exception as e:
            return {"success": False, "error": str(e), "stage": "profiling"}


    def _store_code_diff(self, new_code: str, iteration: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """Store code diffs instead of full code"""
        if self.previous_code is None:
            # First iteration, store full code
            diff_info = {
                'iteration': iteration,
                'diff_type': 'full',
                'content': new_code,
                'timestamp': time.time(),
                'context_snapshot': self._create_lightweight_context(context)
            }
        else:
            # Compute diff from the previous code
            diff = self._compute_diff(self.previous_code, new_code)
            diff_info = {
                'iteration': iteration,
                'diff_type': 'diff',
                'content': diff,
                'timestamp': time.time(),
                'context_snapshot': self._create_lightweight_context(context)
            }
        
        self.code_diffs.append(diff_info)
        return diff_info

    def _compute_diff(self, old_code: str, new_code: str) -> str:
        """Compute the diff between two code versions"""
        old_lines = old_code.splitlines(keepends=True)
        new_lines = new_code.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            old_lines, 
            new_lines,
            fromfile='previous.py',
            tofile='current.py',
            lineterm=''
        )
        
        return ''.join(diff)

    def _reconstruct_code(self, iteration: int) -> str:
        """Reconstruct the code for a given iteration from the diff history"""
        if iteration < 0 or iteration >= len(self.code_diffs):
            return ""
        
        # Rebuild from the first full code snapshot
        current_code = ""
        for i in range(iteration + 1):
            diff_info = self.code_diffs[i]
            if diff_info['diff_type'] == 'full':
                current_code = diff_info['content']
            else:
                current_code = self._apply_diff(current_code, diff_info['content'])
        
        return current_code

    def _apply_diff(self, base_code: str, diff: str) -> str:
        """Apply a diff to the base code"""
        # Simplified diff application logic
        lines = base_code.splitlines(keepends=True)
        diff_lines = diff.splitlines(keepends=True)
        
        # Parse unified diff format
        current_line = 0
        result_lines = []
        
        i = 0
        while i < len(diff_lines):
            line = diff_lines[i]
            if line.startswith('@@'):
                # Parse line number information
                i += 1
                continue
            elif line.startswith(' '):
                # Unchanged line
                if current_line < len(lines):
                    result_lines.append(lines[current_line])
                current_line += 1
            elif line.startswith('+'):
                # Added line
                result_lines.append(line[1:])
            elif line.startswith('-'):
                # Deleted line
                current_line += 1
                # Do not add to result
            i += 1
        
        # Append remaining lines
        while current_line < len(lines):
            result_lines.append(lines[current_line])
            current_line += 1
        
        return ''.join(result_lines)

    def _get_full_code_history(self) -> List[Dict[str, Any]]:
        """Get the full code history (reconstruct code for all iterations)"""
        full_history = []
        for i in range(len(self.code_diffs)):
            code = self._reconstruct_code(i)
            diff_info = self.code_diffs[i]
            full_history.append({
                'iteration': i,
                'code': code,
                'timestamp': diff_info['timestamp'],
                'context': diff_info['context_snapshot']
            })
        return full_history

    def _get_code_at_iteration(self, iteration: int) -> str:
        """Get the full code at a specific iteration"""
        return self._reconstruct_code(iteration)

    def _create_lightweight_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a lightweight context snapshot"""
        essential_fields = {
            'instructions', 'error_history', 'recommendations', 
            'analysis', 'search_results', 'iteration'
        }
        
        lightweight = {}
        for key in essential_fields:
            if key in context:
                value = context[key]
                if isinstance(value, list) and value:
                    # Keep only the most recent key information
                    lightweight[key] = value[-3:] if len(value) > 3 else value
                elif isinstance(value, str) and len(value) > 150:
                    lightweight[key] = value[:150] + "..."
                else:
                    lightweight[key] = value
        
        return lightweight

    def _retrieve_knowledge(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant knowledge from knowledge base"""
        query = context.get('query', 'Triton kernel optimization')
        examples = self.knowledge_base.retrieve_similar_examples(query, limit=3)
        principles = self.knowledge_base.retrieve_optimization_principles()
        
        return {
            "examples": examples,
            "principles": principles,
            "query": query
        }

    def _evaluate_execution(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive execution evaluation"""
        verification_result = self._verify_code(context)
        profiling_result = self._profile_code(context)
        
        overall_success = verification_result["success"] and profiling_result["success"]
        
        return {
            "verification": verification_result,
            "profiling": profiling_result,
            "overall_success": overall_success,
            "code": context.get('code', '')
        }

    def _web_search(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Web search for additional information - simplified for demo"""
        query = context.get('query', 'Triton AMD GPU optimization')

        result = searcher(query)
        return {
            "query": query,
            "results": result,
            "timestamp": time.time()
        }
        

    def _build_enhanced_generation_prompt(self, context: Dict[str, Any]) -> str:
        """Build enhanced prompt with error history and analysis"""
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
        
        components = [base]
        
        # Add original PyTorch code
        if 'torch_code' in context:
            components.append(f"# PyTorch Reference Code:\n{context['torch_code']}\n\n")
        
        # Add error history if available
        if 'error_history' in context and context['error_history']:
            components.append("# Previous Errors to Avoid:\n")
            for i, error in enumerate(context['error_history'][-3:]):
                components.append(f"Error {i+1}: {error}\n")
            components.append("\n")
        
        # Add analysis insights if available
        if 'analysis' in context:
            components.append(f"# Analysis Insights:\n{context['analysis']}\n\n")
        
        # Add optimization recommendations if available
        if 'recommendations' in context:
            components.append("# Optimization Recommendations:\n")
            for rec in context['recommendations'][-5:]:
                components.append(f"- {rec}\n")
            components.append("\n")
        
        # Add web search results if available
        if 'search_results' in context:
            components.append("# Relevant Information from Search:\n")
            for result in context['search_results'][-3:]:
                components.append(f"- {result}\n")
            components.append("\n")
        
        components.append(
            "Rules:\n"
            " - Focus on correctness first, then optimization\n"
            " - Return ONLY executable Python code with Triton imports\n"
            " - Use proper Triton idioms and AMD GPU optimizations\n"
            "Generate the whole and correct Triton kernel code:\n"
        )
        
        return "".join(components)

    def _build_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for analysis"""
        serializable_context = self._make_context_serializable(context)
        
        return (
            "Analyze the current code generation situation and provide strategic recommendations:\n\n"
            f"Context: {json.dumps(serializable_context, indent=2)}\n\n"
            "Consider:\n"
            "1. What are the main challenges or bottlenecks in the current approach?\n"
            "2. What optimization strategies could be effective?\n"
            "3. What additional information might be needed?\n"
            "4. Any patterns or insights from previous attempts?\n\n"
            "Provide structured analysis and actionable recommendations:"
        )

    def _build_planning_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for task planning"""
        available_tools = "\n".join([f"- {name}: {tool.description}" for name, tool in self.tools.items()])
        serializable_context = self._make_context_serializable(context)
        
        return (
            "You are a strategic planner. Based on the current context, decide the best next steps.\n\n"
            f"Current Iteration: {self.current_iteration}/{self.max_iterations}\n"
            f"Available Budget: {self.available_budget}\n"
            f"Available Tools:\n{available_tools}\n\n"
            "Context:\n"
            f"{json.dumps(serializable_context, indent=2)}\n\n"
            "Decide which tool to use next and what parameters to pass. "
            "Consider cost, success rate, and potential impact.\n\n"
            "Respond in JSON format with: tool_name, parameters, and rationale."
        )

    def _make_context_serializable(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """More aggressive context pruning for serialization"""
        essential_fields = {
            'torch_code', 'instructions', 'error_history', 'recommendations',
            'current_code', 'analysis', 'iteration'
        }
        
        serializable = {}
        for key in essential_fields:
            if key in context:
                value = context[key]
                if key == 'torch_code':
                    # Just store length and line count
                    serializable[key] = f"code_length: {len(value)} chars, lines: {value.count(chr(10)) + 1}"
                elif key == 'error_history':
                    # Just store error statistics, whole history is okay
                    serializable[key] = f"total: {len(value)} errors, recent: {value[-1] if value else 'N/A'}"
                elif isinstance(value, str) and len(value) > 200:
                    # Long text truncation
                    serializable[key] = value[:200] + "..."
                else:
                    serializable[key] = value
        
        return serializable

    def _parse_planning_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM planning response"""
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {
                "tool_name": "llm_analyzer",
                "parameters": {"context": {"error": "Failed to parse planning response"}},
                "rationale": "Fallback due to parsing error"
            }

    def _extract_recommendations(self, analysis: str) -> List[str]:
        """Extract actionable recommendations from analysis text"""
        recommendations = []
        lines = analysis.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['recommend', 'suggest', 'should', 'consider', 'try', 'improve', 'optimize']):
                clean_line = line.strip().replace('- ', '').replace('* ', '')
                if clean_line and len(clean_line) > 10:  # Filter out very short lines
                    recommendations.append(clean_line)
        return recommendations[:5]

    def create_task(self, goal: str, context: Dict[str, Any], priority: int = 1) -> str:
        """Create a new task and add it to the queue"""
        task_id = f"task_{len(self.task_queue)}_{int(time.time())}"
        task = Task(
            goal=goal,
            state=TaskState.INIT,
            context=context,
            dependencies=[],
            priority=priority
        )
        self.task_queue.append(task)
        return task_id

    def execute_workflow(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow: use diffs for prompting, store full code"""
        print("\033[95m[WORKFLOW]\033[0m Starting code generation workflow with diff-based prompting")
        
        context = initial_context.copy()
        error_history = []
        best_result = None
        self.previous_code = None
        self.generated_codes = []  # Reset code history
        self.code_diffs = []  # Reset diff history
        
        for iteration in range(self.max_iterations):
            self.current_iteration = iteration + 1
            print(f"\033[95m[WORKFLOW]\033[0m Iteration {self.current_iteration}/{self.max_iterations}")
            
            if self.available_budget <= 0:
                print("\033[91m[WORKFLOW]\033[0m Budget exhausted")
                break
            
            # Step 1: Analysis (first iteration or after errors)
            if iteration == 0 or error_history:
                print("\033[94m[WORKFLOW]\033[0m Step 1: Code analysis")
                analysis_context = {
                    **context,
                    'error_history': error_history,
                    'iteration': iteration
                }
                
                # Add previous diff information for analysis
                if iteration > 0:
                    analysis_context['previous_diff'] = self._get_previous_diff_for_prompt(iteration)
                
                analysis_result = self._llm_analyze(analysis_context)
                context['analysis'] = analysis_result.get('analysis', '')
                context['recommendations'] = analysis_result.get('recommendations', [])
            
            # Step 2: Web search (selective execution)
            if iteration % 2 == 0 and self.available_budget > 2.0:
                print("\033[94m[WORKFLOW]\033[0m Step 2: Web search for optimization info")
                search_result = self._web_search({
                    'query': f"Triton kernel optimization AMD GPU {context.get('instructions', '')}"
                })
                print("\033[94m[DEBUG]\033[0m [web_search] search results:")
                print(search_result)  # Debug print of search results
                
                print(f"\033[94m[WORKFLOW]\033[0m Search results: {len(search_result.get('results', []))} items found")
                context['search_results'] = search_result.get('results', [])
                self.available_budget -= 1.5
            
            # Step 3: Prepare generation context (including diff information)
            print("\033[94m[WORKFLOW]\033[0m Step 3: Preparing generation context")
            generation_context = {
                **context,
                'error_history': error_history,
                'iteration': iteration
            }
            
            # Add previous diff information for code generation
            generation_context['previous_diff'] = self._get_previous_diff_for_prompt(iteration)
            
            # Step 4: Code generation
            print("\033[94m[WORKFLOW]\033[0m Step 4: Code generation")
            gen_result = self._llm_generate_code(generation_context)
            
            if not gen_result.get('code'):
                print("\033[91m[WORKFLOW]\033[0m Code generation failed")
                continue
                
            current_code = gen_result['code']
            
            # Step 5: Verification and testing
            print("\033[94m[WORKFLOW]\033[0m Step 5: Code verification")
            verification_context = {
                'code': current_code,
                'evaluator': context.get('evaluator')
            }
            test_result = self._evaluate_execution(verification_context)
            
            # Step 6: Store full code history and diff information
            print("\033[94m[WORKFLOW]\033[0m Step 6: Storing code history")
            code_info = self._store_code_history(current_code, iteration, context, test_result)
            diff_info = self._store_code_diff(current_code, iteration, context)
            
            if test_result['overall_success']:
                print("\033[92m[WORKFLOW]\033[0m Code verification PASSED")
                
                # Update best result
                if best_result is None or test_result.get('profiling', {}).get('latency_ms', float('inf')) < best_result.get('profiling', {}).get('latency_ms', float('inf')):
                    best_result = {
                        'success': True,
                        'code': current_code,  # Use full code directly
                        'iteration': iteration,
                        'profiling': test_result.get('profiling', {}),
                        'verification': test_result.get('verification', {})
                    }
                
                # Step 7: Optimization (if budget allows)
                if self.available_budget > 3.0 and iteration < self.max_iterations - 1:
                    print("\033[94m[WORKFLOW]\033[0m Step 7: Optimization analysis")
                    optimization_analysis = self._llm_analyze({
                        **context,
                        'current_performance': test_result.get('profiling', {}),
                        'stage': 'optimization',
                        'previous_diff': self._get_previous_diff_for_prompt(iteration)
                    })
                    context['optimization_analysis'] = optimization_analysis.get('analysis', '')
                    context['optimization_recommendations'] = optimization_analysis.get('recommendations', [])
                    
                    # Prepare instructions for the next iteration
                    context['instructions'] = f"Optimize for better performance. Current latency: {test_result.get('profiling', {}).get('latency_ms', 'unknown')} ms."
                    
                else:
                    print("\033[93m[WORKFLOW]\033[0m Optimization skipped due to budget or iteration limit")
                    break
                    
            else:
                # Code verification failed
                if test_result.get('verification', {}).get('success') is False:
                    print("\033[91m[WORKFLOW]\033[0m Verification failed")
                    error_msg = test_result.get('verification', {}).get('error', 'Unknown error')
                elif test_result.get('profiling', {}).get('success') is False:
                    print("\033[91m[WORKFLOW]\033[0m Profiling failed")
                    error_msg = test_result.get('profiling', {}).get('error', 'Unknown error')

                print(f"\033[91m[WORKFLOW]\033[0m Code verification FAILED: {error_msg}")
                error_history.append(error_msg)
                
                # Update context
                context['previous_errors'] = error_history[-2:]  # Keep only the 2 most recent errors
                context['instructions'] = f"Fix the error: {error_msg}"
            
            # Update budget
            self.available_budget -= 3.0
            
            print(f"\033[95m[WORKFLOW]\033[0m Iteration {self.current_iteration} completed. Remaining budget: {self.available_budget:.1f}")
            print(f"\033[95m[WORKFLOW]\033[0m Code history: {len(self.generated_codes)} iterations stored")
        
        # Return best result
        if best_result:
            print(f"\033[92m[WORKFLOW]\033[0m Workflow completed. Best result from iteration {best_result.get('iteration', 'unknown')}")
            return best_result
        else:
            print("\033[91m[WORKFLOW]\033[0m Workflow completed without successful code generation")
            # Return the code from the last attempt
            last_code = self.generated_codes[-1]['full_code'] if self.generated_codes else ""
            return {
                'success': False,
                'error': 'No successful code generation',
                'code': last_code,
                'error_history': error_history
            }

    def _extract_code_snippet(self, code: str, max_lines: int = 15) -> str:
        """Extract key code snippet"""
        lines = code.split('\n')
        if len(lines) <= max_lines:
            return code
        
        # Extract lines containing keywords
        important_lines = []
        keywords = ['def ', 'class ', '@triton', 'import ', 'from ', 'return ', 'if __name__']
        
        for line in lines:
            if any(keyword in line for keyword in keywords):
                important_lines.append(line)
                if len(important_lines) >= max_lines:
                    break
        
        # If not enough important lines, supplement with the beginning of the file
        if len(important_lines) < max_lines // 2:
            important_lines = lines[:max_lines]
        
        return '\n'.join(important_lines[:max_lines])

    def generate(self, torch_module_src: str, evaluator_obj: GenericEvaluator,
                 system: str = "You are an expert in writing GPU programs with Triton compiler.",
                 instructions: str | None = None, **kwargs) -> GenerationResult:
        """
        Main generation method with complete workflow execution.
        """
        print("\033[95m[GENERATE]\033[0m Starting code generation process")
        
        # Create initial context
        initial_context = {
            "torch_code": torch_module_src,
            "evaluator": evaluator_obj,
            "instructions": instructions or "Your task is to iteratively improve the provided PyTorch operator by generating a Triton implementation.",
            "system_prompt": system,
            "error_history": [],
            "search_results": []
        }
        
        # Execute the complete workflow
        final_result = self.execute_workflow(initial_context)
        
        # Prepare the generation result
        raw_code = final_result.get('code', '')
        # Print the raw code for debugging
        print("\033[94m[DEBUG]\033[0m [GENERATE] Final generated code:")
        print(raw_code)
        if final_result.get('success', False):
            metadata = {
                "model": getattr(self.llm_client, 'model', 'unknown'),
                "final_budget": self.available_budget,
                "iterations_completed": self.current_iteration,
                "total_generations": len(self.generated_codes),
                "best_iteration": final_result.get('iteration', 0),
                "performance": final_result.get('profiling', {}),
                "storage_method": "diff_based",
                "total_diffs_stored": len(self.code_diffs),
                "code_reconstructed_from": f"iteration_{final_result.get('iteration', 0)}"
            }
            
            metadata.update(self.llm_client.get_token_usage())
            return GenerationResult(
                raw=raw_code,
                metadata=metadata
            )
        else:
            # Return a fallback result with error information
            fallback_meta = {
                    "model": getattr(self.llm_client, 'model', 'unknown'),
                    "error": final_result.get('error', 'Unknown error'),
                    "error_history": final_result.get('error_history', []),
                    "iterations_attempted": self.current_iteration,
                    "final_budget": self.available_budget,
                    "storage_method": "diff_based",
                    "total_diffs_stored": len(self.code_diffs)
                }
            fallback_meta.update(self.llm_client.get_token_usage())
            return GenerationResult(
                raw=raw_code,
                metadata=fallback_meta
            )

    def decode(self, raw: str) -> str:
        """Decode LLM response to extract code"""
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

class KnowledgeBase:
    """Knowledge base for storing and retrieving optimization knowledge"""
    def __init__(self):
        self.examples: List[Dict[str, Any]] = []
        self.optimization_principles: List[str] = [
            "Use shared memory for data reused within thread blocks",
            "Ensure memory access coalescing for better bandwidth utilization",
            "Use vectorized loads/stores when possible",
            "Avoid bank conflicts in shared memory",
            "Use tiling to exploit data locality",
            "Minimize thread divergence",
            "Use appropriate block sizes for the target hardware",
            "Optimize memory access patterns for AMD GPU architecture",
            "Use asynchronous memory operations when possible",
            "Profile and optimize the bottleneck operations"
        ]
    
    def add_example(self, code: str, result: Dict[str, Any], metadata: Dict[str, Any] = None):
        """Add a new example to the knowledge base"""
        self.examples.append({
            "code": code,
            "result": result,
            "metadata": metadata or {},
            "timestamp": time.time()
        })
    
    def retrieve_similar_examples(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve examples similar to the query"""
        # Simple time-based retrieval for demo
        return self.examples[-limit:] if self.examples else []
    
    def retrieve_optimization_principles(self) -> List[str]:
        """Retrieve optimization principles"""
        return self.optimization_principles