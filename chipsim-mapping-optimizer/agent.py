"""
CHIPSIM Mapping Optimizer Agent (Merged)
=========================================
Multi-agent LangGraph pipeline with:
  - Pre-simulation validation
  - Runtime error handling with auto-retry
  - Baseline comparison
  - Iterative evaluation with per-metric improvement tracking

Flow:
  START → baseline ──(fail)──→ END
             │
          (success)
             ↓
         analyzer → optimizer → validator ──(invalid)──→ optimizer (retry)
             ↑                      │
             │                   (valid)
             │                      ↓
             │                  injector → runtime_handler
             │                                  │
             │                    ┌──────────────┼──────────────┐
             │                 (error)        (success)     (max retries)
             │                    ↓              ↓              ↓
             │                optimizer      evaluator      evaluator
             │                                  │
             │              ┌───────────────────┼──────────────┐
             │           (improve)          (no improve)   (max iters)
             └──────────────┘                   ↓              ↓
                                             analyzer         END

Usage:
  python agent.py --config config_1 --max-iterations 5
  python agent.py --config config_1
  python agent.py
"""

import os
import re
import ast
import sys
import yaml
import shutil
import argparse
import subprocess
import time
from dotenv import load_dotenv
from typing import TypedDict, Optional, Any
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

# # Prompt for key if missing
# if not os.environ.get("OPENROUTER_API_KEY"):
#     os.environ["OPENROUTER_API_KEY"] = getpass.getpass("OPENROUTER_API_KEY: ")

load_dotenv()

# ---------------------------------------------------------------
# CLI ARGUMENTS
# ---------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="CHIPSIM Mapping Optimizer Agent")
    parser.add_argument("--config", "-c", default="config_1",
                        help="Config name (without .yaml). Default: config_1")
    parser.add_argument("--max-iterations", "-n", type=int, default=3,
                        help="Maximum optimization iterations. Default: 3")
    parser.add_argument("--max-retries", "-r", type=int, default=3,
                        help="Max validation retries per iteration. Default: 3")
    parser.add_argument("--max-runtime-retries", type=int, default=3,
                        help="Max runtime retries per iteration. Default: 3")
    return parser.parse_args()

args = parse_args()

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
CHIPSIM_ROOT = "/home/ECE511-Final-Research-Project/CHIPSIM"
PROJECT_ROOT = "/home/ECE511-Final-Research-Project"
BASELINE_RESULTS_ROOT = f"{PROJECT_ROOT}/performance_results/baseline/heterogenous"
MAPPER_PATH = f"{CHIPSIM_ROOT}/src/mapping/model_mapper.py"
MAPPER_ORIGINAL_PATH = f"{CHIPSIM_ROOT}/src/mapping/model_mapper_original.py"
PARTITIONER_PATH = f"{CHIPSIM_ROOT}/src/mapping/layer_partitioner.py"
CONFIG_NAME = args.config
CONFIG_PATH = f"{CHIPSIM_ROOT}/configs/experiments/{CONFIG_NAME}.yaml"
MAX_ITERATIONS = args.max_iterations
MAX_VALIDATION_RETRIES = args.max_retries
MAX_RUNTIME_RETRIES = args.max_runtime_retries

# Validate config exists
if not os.path.exists(CONFIG_PATH):
    print(f"ERROR: Config file not found: {CONFIG_PATH}")
    available = [f.replace('.yaml', '') for f in os.listdir(f"{CHIPSIM_ROOT}/configs/experiments/")
                 if f.endswith('.yaml')]
    print(f"Available configs: {', '.join(sorted(available))}")
    sys.exit(1)

# Back up the original mapper
if not os.path.exists(MAPPER_ORIGINAL_PATH):
    shutil.copy(MAPPER_PATH, MAPPER_ORIGINAL_PATH)
    print(f"Backed up original mapper to {MAPPER_ORIGINAL_PATH}")
else:
    with open(MAPPER_ORIGINAL_PATH, 'r') as f:
        backup_content = f.read()
    if '        else:\n            try:' not in backup_content:
        print(f"WARNING: {MAPPER_ORIGINAL_PATH} looks corrupted.")
        with open(MAPPER_PATH, 'r') as f:
            current = f.read()
        if '        else:\n            try:' in current:
            shutil.copy(MAPPER_PATH, MAPPER_ORIGINAL_PATH)
            print(f"  Fresh backup created from {MAPPER_PATH}.")
        else:
            print(f"  ERROR: Neither file has the expected pattern. Restore from git.")
            sys.exit(1)


# ---------------------------------------------------------------
# 1. DEFINE STATE
# ---------------------------------------------------------------
class ChipSimState(TypedDict):
    # --- Config context ---
    user_request: str
    config_name: str
    config_path: str

    # --- Analyzer output ---
    config_analysis: str
    mapping_source_code: str

    # --- Optimizer output ---
    mapping_proposal: str
    function_name: str

    # --- Validator output ---
    validation_errors: list
    is_valid: bool
    retry_count: int

    # --- Injector output ---
    injected_file_path: str

    # --- Runtime handler ---
    run_succeeded: bool
    runtime_error: Optional[str]
    last_runtime_error: Optional[str]
    latest_output: Optional[str]
    runtime_retry_count: int
    results_label: str

    # --- Evaluator output ---
    baseline_metrics: dict
    current_best_metrics: dict
    latest_metrics: dict
    improvement_pct: dict
    evaluation_summary: str

    # --- Iteration control ---
    iteration: int
    max_iterations: int
    history: list[dict[str, Any]]
    should_continue: bool


# ---------------------------------------------------------------
# 2. CREATE THE LLM
# ---------------------------------------------------------------

# llm = ChatGroq(
#     model="llama-3.3-70b-versatile",
#     api_key=os.getenv("OPENROUTER_API_KE"),
# )

# llm = ChatOpenAI(
#     model="meta-llama/llama-3.3-70b-instruct",
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.environ["OPENROUTER_API_KEY"]
# )

llm = ChatOpenAI(
    model="deepseek/deepseek-v4-pro",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

#llm = ChatOllama(model="qwen3.6:35b", base_url="http://192.168.50.4:11434", num_ctx=16384)


# ---------------------------------------------------------------
# 3. MAPPER CONTRACT
# ---------------------------------------------------------------
MAPPER_CONTRACT = """
STRICT REQUIREMENTS — your code MUST follow these rules exactly:

1. METHOD SIGNATURE: The method must be named with a leading underscore
   (e.g., def _my_optimized_mapper) and accept exactly these arguments:
   (self, model_layer_info, system, preference,
    current_available_crossbars, layer_mappings=None, shortest_paths=None)

2. RETURN TYPE: Must return a tuple of exactly 4 values:
   (remaining_capacity, action, mapping_failed, failure_reason)
   - remaining_capacity: numpy array (copy of current_available_crossbars, modified)
   - action: list of floats, length = number of chiplets, percentages summing to ~100
             OR None if mapping failed
   - mapping_failed: bool
   - failure_reason: str or None

3. DO NOT MODIFY:
   - The argument list
   - The return type structure
   - The _calculate_layer_requirements method (treat it as a black box you call)
   - Any System or Chiplet class interfaces

4. YOU MAY MODIFY:
   - How the starting chiplet is selected
   - How chiplets are sorted/prioritized for allocation
   - The greedy allocation loop logic
   - Adding look-ahead or global optimization within the method

5. AVAILABLE DATA you can use:
   - model_layer_info['name'], ['crossbars_required'], ['layer_type'],
     ['filter_size'], ['out_channels'], ['in_features'], ['out_features']
   - system.chiplets (list of Chiplet objects)
   - system.chiplet_network (networkx graph)
   - system.is_io_chiplet(chiplet_id) -> bool
   - chiplet.get_capacity_unit_size(), chiplet.get_total_memory()
   - chiplet.type ('IMC' or 'CMOS'), chiplet.crossbar_rows, etc.
   - shortest_paths[src][dst] -> int (hop count)
   - layer_mappings: list of (layer_idx, [(chiplet_id, percentage)]) for prior layers

6. HELPER METHODS you can call via self:
   - self._calculate_layer_requirements(model_layer_info, chiplet) -> int
   - self.system.is_io_chiplet(chiplet_id) -> bool

7. Output format:
   - Output ONLY the complete def _method_name(self, ...) function.
   - No explanation before or after the code.
   - No markdown fences (no ```python).
   - No import statements (numpy as np, math, and networkx as nx are already imported).
"""


# ---------------------------------------------------------------
# 4. NODES
# ---------------------------------------------------------------

# ======================== ANALYZER ========================
def analyze_config(state: ChipSimState) -> dict:
    """Agent 1: Analyze the config, mapper code, and partitioner code."""
    iteration = state.get("iteration", 0)
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration + 1}/{state['max_iterations']}: ANALYZING")
    print(f"{'='*60}")

    with open(MAPPER_ORIGINAL_PATH) as f:
        mapper_code = f.read()
    with open(PARTITIONER_PATH) as f:
        partitioner_code = f.read()

    # Build history context for subsequent iterations
    history_context = ""
    if state.get("history"):
        history_context = (
            "\n\nPREVIOUS OPTIMIZATION ATTEMPTS:\n"
            + "\n".join(
                f"  Iteration {h['iteration']}: {h['outcome']} "
                f"(latency: {h.get('metrics', {}).get('total_latency', 'N/A')}, "
                f"change: {h.get('change_summary', 'N/A')})"
                for h in state["history"]
            )
            + "\n\nBased on these results, try a DIFFERENT optimization strategy."
        )

    messages = [
        SystemMessage(content=(
            "You are an expert in DNN-to-chiplet mapping optimization. "
            "Analyze the mapping algorithm and identify specific weaknesses. "
            "Focus on the _nearest_neighbor_mapper_v3 method in model_mapper.py "
            "— this is the core placement algorithm. "
            "layer_partitioner.py handles splitting layers after placement.\n\n"
            "Structure your analysis as:\n"
            "1. Key weaknesses of the current nearest-neighbor approach\n"
            "2. Specific opportunities for improvement given this config\n"
            "3. What data/signals the mapper currently ignores that it should use"
        )),
        HumanMessage(content=(
            f"CHIPSIM config:\n{state['user_request']}\n\n"
            f"Mapping algorithm (model_mapper.py):\n```python\n{mapper_code}\n```\n\n"
            f"Layer partitioner (layer_partitioner.py):\n```python\n{partitioner_code}\n```\n\n"
            f"Analyze the nearest-neighbor mapper's weaknesses for this config."
            f"{history_context}"
        ))
    ]
    response = llm.invoke(messages)
    print("  Analysis complete.")
    return {
        "config_analysis": response.content,
        "mapping_source_code": mapper_code,
        # Reset per-iteration retry counters
        "validation_errors": [],
        "is_valid": False,
        "retry_count": 0,
        "runtime_retry_count": 0,
        "runtime_error": None,
        "last_runtime_error": None,
    }


# ======================== OPTIMIZER ========================
def propose_mapping(state: ChipSimState) -> dict:
    """Agent 2: Generate a drop-in replacement mapper function.

    Uses error context from either the most recent validation failure or the
    most recent runtime failure. `last_runtime_error` is preserved across
    validation retries so the LLM doesn't forget the original runtime issue
    while it's also fixing validation problems.
    """
    print(f"  Generating mapping proposal (val retry {state.get('retry_count', 0)}, "
          f"rt retry {state.get('runtime_retry_count', 0)})...")

    runtime_error = state.get("runtime_error")
    last_runtime_error = state.get("last_runtime_error")
    validation_errors = state.get("validation_errors") or []

    error_parts = []
    if runtime_error:
        error_parts.append(
            "Your PREVIOUS attempt FAILED AT RUNTIME with this error:\n"
            f"  {runtime_error}\n\n"
            "Fix the logic so it runs without crashing. Pay attention to "
            "attribute access, None checks, and division-by-zero cases."
        )
    elif validation_errors:
        msg = "Your PREVIOUS attempt failed validation with these errors:\n"
        msg += "\n".join(f"  - {e}" for e in validation_errors)
        msg += "\n\nFix ALL of these issues."
        if last_runtime_error:
            msg += (
                "\n\nFor context, the proposal BEFORE that one failed at runtime with:\n"
                f"  {last_runtime_error}\n"
                "Make sure your fix still addresses that runtime issue."
            )
        error_parts.append(msg)

    # Include evaluation feedback from previous iterations
    eval_summary = state.get("evaluation_summary", "")
    if eval_summary:
        error_parts.append(
            f"FEEDBACK FROM PREVIOUS ITERATION'S EVALUATION:\n{eval_summary}\n\n"
            "Use this feedback to improve your approach."
        )

    error_context = "\n\n" + "\n\n".join(error_parts) if error_parts else ""

    messages = [
        SystemMessage(content=(
            "You are a DNN mapping optimizer that generates drop-in replacement "
            "code for CHIPSIM's mapper. You output ONLY Python code — a single "
            "complete method definition. No prose, no markdown.\n\n"
            f"{MAPPER_CONTRACT}"
        )),
        HumanMessage(content=(
            f"Analysis of current mapper weaknesses:\n{state['config_analysis']}\n\n"
            f"Current working code (for reference):\n"
            f"```python\n{state['mapping_source_code']}\n```\n\n"
            f"Original request: {state['user_request']}\n\n"
            "Generate an improved drop-in mapper function following the contract above."
            f"{error_context}"
        ))
    ]
    response = llm.invoke(messages)
    print("  Proposal generated.")

    return {
        "mapping_proposal": response.content,
        "runtime_error": None,
        "last_runtime_error": runtime_error or last_runtime_error,
        "validation_errors": [],
    }


# ======================== VALIDATOR ========================
def validate_proposal(state: ChipSimState) -> dict:
    """Validate that the LLM output meets the mapper contract before injection."""
    print(f"  Validating proposal...")
    raw_code = state["mapping_proposal"]
    errors = []

    # --- Clean up LLM output ---
    code = raw_code.strip()
    code = re.sub(r'^```(?:python)?\s*\n?', '', code)
    code = re.sub(r'\n?```\s*$', '', code)
    lines = code.split('\n')
    def_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('def '):
            def_start = i
            break
    if def_start is not None and def_start > 0:
        code = '\n'.join(lines[def_start:])
    code = code.strip()

    # --- 1. Syntax check ---
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        errors.append(f"Syntax error on line {e.lineno}: {e.msg}")
        return {
            "mapping_proposal": code, "validation_errors": errors,
            "is_valid": False, "function_name": "",
            "retry_count": state.get("retry_count", 0) + 1,
        }

    # --- 2. Find top-level function definition ---
    func_def = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break

    if func_def is None:
        errors.append("No top-level function definition found.")
        return {
            "mapping_proposal": code, "validation_errors": errors,
            "is_valid": False, "function_name": "",
            "retry_count": state.get("retry_count", 0) + 1,
        }

    # --- 3. Name must start with underscore ---
    raw_name = func_def.name
    if not raw_name.startswith("_"):
        errors.append(f"Function name '{raw_name}' must start with underscore.")
    public_name = raw_name.lstrip("_")
    if not public_name:
        errors.append("Function name cannot be just underscores.")

    # --- 4. Argument list must match exactly ---
    expected_args = [
        "self", "model_layer_info", "system", "preference",
        "current_available_crossbars", "layer_mappings", "shortest_paths",
    ]
    actual_args = [a.arg for a in func_def.args.args]
    if actual_args != expected_args:
        errors.append(
            f"Argument list mismatch.\n"
            f"  Expected: {expected_args}\n"
            f"  Got:      {actual_args}"
        )

    # --- 5. Defaults for last two args must be None ---
    defaults = func_def.args.defaults
    if len(defaults) < 2:
        errors.append("layer_mappings and shortest_paths must have default value None.")
    else:
        for i, d in enumerate(defaults[-2:]):
            arg_name = expected_args[-2 + i]
            if not (isinstance(d, ast.Constant) and d.value is None):
                errors.append(f"Default value for '{arg_name}' must be None.")

    # --- 6. At least one return must be a 4-tuple ---
    return_nodes = [n for n in ast.walk(func_def) if isinstance(n, ast.Return)]
    if not return_nodes:
        errors.append("Function has no return statement.")
    else:
        has_valid = any(
            isinstance(rn.value, ast.Tuple) and len(rn.value.elts) == 4
            for rn in return_nodes if rn.value is not None
        )
        if not has_valid:
            errors.append("At least one return must be a 4-tuple: "
                          "(remaining_capacity, action, mapping_failed, failure_reason).")

    # --- 7. No imports ---
    for n in ast.walk(func_def):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            errors.append("Import statements not allowed inside the function.")
            break

    # --- Result ---
    if errors:
        print(f"  Validation FAILED ({len(errors)} errors)")
        for e in errors:
            print(f"    - {e}")
        return {
            "mapping_proposal": code, "validation_errors": errors,
            "is_valid": False, "function_name": "",
            "retry_count": state.get("retry_count", 0) + 1,
        }

    print(f"  Validation PASSED: {raw_name}")
    return {
        "mapping_proposal": code, "validation_errors": [],
        "is_valid": True, "function_name": public_name,
        "retry_count": state.get("retry_count", 0),
    }


# ======================== INJECTOR ========================
def inject_into_mapper(state: ChipSimState) -> dict:
    """Inject the validated function into model_mapper.py with proper wiring."""
    function_code = state["mapping_proposal"]
    public_name = state["function_name"]
    method_name = f"_{public_name}"

    if not public_name:
        raise ValueError("Cannot inject mapper: function_name is empty.")

    with open(MAPPER_ORIGINAL_PATH, 'r') as f:
        modified = f.read()

    # --- 1. Update default mapping_function in __init__ ---
    old_default = re.search(r'mapping_function\s*=\s*["\']([^"\']+)["\']', modified)
    modified = re.sub(
        r'(mapping_function\s*=\s*)["\'][^"\']+["\']',
        f'\\1"{public_name}"', modified, count=1
    )
    new_default = re.search(r'mapping_function\s*=\s*["\']([^"\']+)["\']', modified)
    print(f"  __init__ default: '{old_default.group(1) if old_default else '?'}'"
          f" → '{new_default.group(1) if new_default else '?'}'")

    # --- 2. Add routing in _get_mapper_function ---
    target_elif = f'elif self.mapping_function == "{public_name}":'
    if target_elif not in modified:
        lines = modified.split('\n')
        new_lines = []
        injected = False
        anchor = "return self._nearest_neighbor_mapper_v3"

        for line in lines:
            new_lines.append(line)
            if anchor in line and not injected:
                ret_indent = len(line) - len(line.lstrip())
                elif_indent = max(0, ret_indent - 4)
                new_lines.append(f"{' ' * elif_indent}elif self.mapping_function == \"{public_name}\":")
                new_lines.append(f"{' ' * ret_indent}return self.{method_name}")
                injected = True

        if injected:
            modified = '\n'.join(new_lines)
            print(f"  _get_mapper_function: elif for '{public_name}' injected")
        else:
            print(f"  _get_mapper_function: INJECTION FAILED — anchor not found")
    else:
        print(f"  _get_mapper_function: elif for '{public_name}' already present")

    # --- 3. Indent and append the new method ---
    fn_lines = function_code.split('\n')
    base_indent = 0
    for line in fn_lines:
        if line.strip().startswith('def '):
            base_indent = len(line) - len(line.lstrip())
            break

    indented_lines = []
    for line in fn_lines:
        if not line.strip():
            indented_lines.append('')
        else:
            relative = max(0, (len(line) - len(line.lstrip())) - base_indent)
            indented_lines.append('    ' + ' ' * relative + line.lstrip())

    modified = modified.rstrip() + '\n\n' + '\n'.join(indented_lines) + '\n'

    # --- 4. Write ---
    with open(MAPPER_PATH, 'w') as f:
        f.write(modified)

    print(f"  Injected '{method_name}' into {MAPPER_PATH}")
    return {"injected_file_path": MAPPER_PATH}


# ======================== RUNTIME HANDLER ========================
def runtime_handler(state: ChipSimState) -> dict:
    """Run CHIPSIM. On failure, preserve the error for the optimizer and
    reset the validation retry budget so the next proposal gets a fresh cycle.
    """
    config_name = state["config_name"]
    function_name = state.get("function_name", "unknown")
    iteration = state.get("iteration", 0) + 1
    runtime_retry_count = state.get("runtime_retry_count", 0)
    results_label = f"{config_name}_{function_name}"

    print(f"\n  Running CHIPSIM (iteration {iteration}, runtime attempt {runtime_retry_count + 1})...")

    # Set env var for simplified results directory
    sim_env = os.environ.copy()
    sim_env["CHIPSIM_RESULTS_DIR_NAME"] = results_label

    # Clean previous results for this label
    for subdir in ["raw_results", "formatted_results"]:
        old = os.path.join(CHIPSIM_ROOT, "_results", subdir, results_label)
        if os.path.exists(old):
            shutil.rmtree(old)

    # Clear pycache
    subprocess.run(
        ["find", CHIPSIM_ROOT, "-type", "d", "-name", "__pycache__",
         "-exec", "rm", "-rf", "{}", "+"],
        capture_output=True
    )

    history = list(state.get("history", []))

    def fail(msg: str, stdout: str = None) -> dict:
        history.append({
            "iteration": iteration,
            "runtime_attempt": runtime_retry_count + 1,
            "outcome": "RUNTIME_ERROR",
            "error": msg[:1000],
            "function_name": function_name,
        })
        return {
            "history": history,
            "run_succeeded": False,
            "runtime_error": msg,
            "latest_output": stdout,
            "runtime_retry_count": runtime_retry_count + 1,
            "retry_count": 0,  # fresh validation budget for next proposal
            "results_label": results_label,
        }

    try:
        # result = subprocess.run(
        #     ["python3", "simulate.py", "--mode", "simulate", "--config", config_name],
        #     cwd=CHIPSIM_ROOT,
        #     capture_output=True, text=True, timeout=900,
        #     env=sim_env,
        # )
        stdout_log = os.path.join(CHIPSIM_ROOT, "_results", f"{results_label}_stdout.log")
        stderr_log = os.path.join(CHIPSIM_ROOT, "_results", f"{results_label}_stderr.log")

        with open(stdout_log, 'w') as out, open(stderr_log, 'w') as err:
            result = subprocess.run(
                ["python3", "-u", "simulate.py", "--mode", "simulate", "--config", config_name],
                cwd=CHIPSIM_ROOT,
                stdout=out, stderr=err,
                timeout=900, env=sim_env,
            )



    except subprocess.TimeoutExpired:
        print(f"  TIMED OUT after 900s")
        return fail("Simulation timed out after 900 seconds.")
    except Exception as e:
        print(f"  ERROR: {e}")
        return fail(f"Execution failed unexpectedly:\n{e}")

    # if result.returncode != 0:
    #     error_text = (result.stderr or result.stdout or "")[-3000:]
    #     print(f"  SIMULATION FAILED (exit code {result.returncode})")
    #     return fail(error_text, stdout=result.stdout)
        
    # Read back only what you need after it finishes
    if result.returncode != 0:
        with open(stderr_log, 'r') as f:
            error_text = f.read()[-3000:]
            return fail(error_text, stdout=result.stdout)
    


    # Success
    print(f"  Simulation succeeded")
    history.append({
        "iteration": iteration,
        "runtime_attempt": runtime_retry_count + 1,
        "outcome": "RUNTIME_SUCCESS",
        "function_name": function_name,
    })
    return {
        "history": history,
        "run_succeeded": True,
        "runtime_error": None,
        "latest_output": result.stdout,
        "runtime_retry_count": runtime_retry_count,
        "results_label": results_label,
    }


# ======================== BASELINE ========================
def run_baseline(state: ChipSimState) -> dict:
    """Load pre-computed baseline metrics from performance_results directory.
    
    Reads from:
      <PROJECT_ROOT>/performance_results/baseline/heterogenous/<CONFIG_NAME>/
        formatted_comparison_metrics/Approach_Comparison_Metrics.txt
    """
    config_name = state["config_name"]

    print(f"\n{'='*60}")
    print(f"LOADING BASELINE (pre-computed nearest_neighbor_v3)")
    print(f"{'='*60}")

    # Restore original mapper so subsequent iterations start from clean state
    shutil.copy(MAPPER_ORIGINAL_PATH, MAPPER_PATH)

    baseline_metrics_path = os.path.join(
        BASELINE_RESULTS_ROOT, config_name,
        "formatted_comparison_metrics", "Approach_Comparison_Metrics.txt"
    )

    if not os.path.exists(baseline_metrics_path):
        print(f"  ERROR: Pre-computed baseline not found at:")
        print(f"    {baseline_metrics_path}")
        # List available baselines
        if os.path.exists(BASELINE_RESULTS_ROOT):
            available = [d for d in os.listdir(BASELINE_RESULTS_ROOT)
                         if os.path.isdir(os.path.join(BASELINE_RESULTS_ROOT, d))]
            print(f"  Available baselines: {', '.join(sorted(available)) if available else 'none'}")
        else:
            print(f"  Baseline directory does not exist: {BASELINE_RESULTS_ROOT}")
        return {"run_succeeded": False, "runtime_error": "Baseline results not found", "baseline_metrics": {}}

    print(f"  Reading baseline from: {baseline_metrics_path}")
    baseline_metrics = _parse_metrics_file(baseline_metrics_path)

    if not baseline_metrics:
        print(f"  ERROR: Could not parse any metrics from baseline file")
        return {"run_succeeded": False, "runtime_error": "Failed to parse baseline metrics", "baseline_metrics": {}}

    print(f"  Baseline loaded:")
    print(f"    Total Latency:   {baseline_metrics.get('total_latency', 'N/A')} µs")
    print(f"    Compute Latency: {baseline_metrics.get('compute_latency', 'N/A')} µs")
    print(f"    Comm Latency:    {baseline_metrics.get('comm_latency', 'N/A')} µs")

    return {
        "run_succeeded": True,
        "baseline_metrics": baseline_metrics,
        "current_best_metrics": baseline_metrics,
    }


# ======================== METRICS PARSER ========================
def _parse_metrics_file(filepath: str) -> dict:
    """Parse an Approach_Comparison_Metrics.txt file and return structured metrics.
    
    This is the core parser shared by both baseline loading and iteration evaluation.
    """
    if not os.path.exists(filepath):
        print(f"  WARNING: Metrics file not found: {filepath}")
        return {}

    with open(filepath, 'r') as f:
        content = f.read()

    metrics = {}
    per_model = {}
    current_table = None

    for line in content.split('\n'):
        stripped = line.strip()
        if 'Average Total Latency Comparison' in stripped:
            current_table = 'total_latency'
        elif 'Average Compute Latency Comparison' in stripped:
            current_table = 'compute_latency'
        elif 'Average Communication Latency Comparison' in stripped:
            current_table = 'comm_latency'
        elif 'Percent Difference' in stripped:
            current_table = None

        if current_table and '|' in line and '+' not in line:
            cells = [c.strip() for c in line.split('|') if c.strip()]
            if not cells or 'Model Type' in cells[0]:
                continue
            if len(cells) >= 2:
                model_name = cells[0].strip()
                try:
                    value = float(cells[1].strip())
                except ValueError:
                    continue
                if model_name not in per_model:
                    per_model[model_name] = {}
                per_model[model_name][current_table] = value
                if current_table not in metrics:
                    metrics[current_table] = 0.0
                metrics[current_table] += value

    metrics['per_model'] = per_model
    metrics['num_models'] = len(per_model)
    metrics['metrics_file'] = filepath

    print(f"  Parsed metrics from: {filepath}")
    for model, vals in per_model.items():
        print(f"    {model}: {vals}")
    return metrics


def parse_metrics_from_file(results_label: str) -> dict:
    """Find and parse metrics for a given results label (iteration dump).
    
    Looks in CHIPSIM/_results/formatted_results/<results_label>/.
    """
    metrics_path = os.path.join(
        CHIPSIM_ROOT, "_results", "formatted_results", results_label,
        "formatted_comparison_metrics", "Approach_Comparison_Metrics.txt"
    )

    if not os.path.exists(metrics_path):
        print(f"  WARNING: Metrics file not found: {metrics_path}")
        formatted_root = os.path.join(CHIPSIM_ROOT, "_results", "formatted_results")
        if os.path.exists(formatted_root):
            for candidate in sorted(os.listdir(formatted_root), reverse=True):
                alt = os.path.join(formatted_root, candidate,
                                   "formatted_comparison_metrics", "Approach_Comparison_Metrics.txt")
                if os.path.exists(alt):
                    print(f"  Found metrics at: {alt}")
                    metrics_path = alt
                    break
            else:
                return {}
        else:
            return {}

    result = _parse_metrics_file(metrics_path)
    result['results_label'] = results_label
    return result


# ======================== EVALUATOR ========================
def evaluate_results(state: ChipSimState) -> dict:
    """Evaluate simulation results and decide whether to continue."""
    iteration = state.get("iteration", 0) + 1
    config_name = state["config_name"]
    function_name = state.get("function_name", "unknown")
    results_label = state.get("results_label", f"{config_name}_{function_name}")

    print(f"\n  Evaluating iteration {iteration} results...")

    latest_metrics = parse_metrics_from_file(results_label)
    baseline = state.get("baseline_metrics", {})
    best = state.get("current_best_metrics", {})

    # Calculate improvement for all three metrics
    improvement_pct = {}
    for key in ['total_latency', 'compute_latency', 'comm_latency']:
        b_val = baseline.get(key)
        l_val = latest_metrics.get(key)
        if b_val and l_val and b_val > 0:
            improvement_pct[key] = (b_val - l_val) / b_val * 100
        else:
            improvement_pct[key] = 0.0

    # Primary improvement decision on total_latency
    is_improvement = False
    if latest_metrics.get("total_latency") and best.get("total_latency"):
        is_improvement = latest_metrics["total_latency"] < best["total_latency"]

    print(f"  Improvement over baseline:")
    print(f"    Total:   {improvement_pct['total_latency']:+.2f}%")
    print(f"    Compute: {improvement_pct['compute_latency']:+.2f}%")
    print(f"    Comm:    {improvement_pct['comm_latency']:+.2f}%")
    print(f"  Better than previous best: {is_improvement}")

    # Build evaluation summary for next iteration's optimizer
    eval_summary = (
        f"Iteration {iteration} results:\n"
        f"  Total Latency:   {latest_metrics.get('total_latency', 'N/A')} µs "
        f"({improvement_pct['total_latency']:+.2f}% vs baseline)\n"
        f"  Compute Latency: {latest_metrics.get('compute_latency', 'N/A')} µs "
        f"({improvement_pct['compute_latency']:+.2f}% vs baseline)\n"
        f"  Comm Latency:    {latest_metrics.get('comm_latency', 'N/A')} µs "
        f"({improvement_pct['comm_latency']:+.2f}% vs baseline)\n"
        f"  Function: {function_name}\n"
    )

    if is_improvement:
        best_metric = max(improvement_pct, key=improvement_pct.get)
        worst_metric = min(improvement_pct, key=improvement_pct.get)
        eval_summary += (
            f"\nImproved! Biggest gain: {best_metric} ({improvement_pct[best_metric]:+.2f}%). "
            f"Focus next on improving {worst_metric} ({improvement_pct[worst_metric]:+.2f}%)."
        )
    else:
        worse = [k for k, v in improvement_pct.items() if v < 0]
        eval_summary += (
            f"\nDid NOT improve. Metrics worse: {', '.join(worse) if worse else 'none'}. "
            "Try a fundamentally different strategy."
        )

    history_entry = {
        "iteration": iteration,
        "outcome": "IMPROVED" if is_improvement else "NO_IMPROVEMENT",
        "metrics": latest_metrics,
        "improvement_pct": improvement_pct,
        "function_name": function_name,
        "change_summary": state.get("config_analysis", ""),
    }

    new_best = latest_metrics if is_improvement else best
    if not is_improvement:
        shutil.copy(MAPPER_ORIGINAL_PATH, MAPPER_PATH)
        print(f"  No improvement — reverted mapper.")

    # Early stopping on consecutive failures
    all_history = list(state.get("history", [])) + [history_entry]
    consecutive_failures = 0
    for h in reversed(all_history):
        if h.get("outcome") in ("NO_IMPROVEMENT", "RUNTIME_ERROR"):
            consecutive_failures += 1
        else:
            break

    should_continue = iteration < state["max_iterations"] and consecutive_failures < 3
    if not should_continue:
        reason = "3 consecutive failures" if consecutive_failures >= 3 else f"max iterations ({state['max_iterations']})"
        print(f"  Stopping: {reason}.")

    return {
        "iteration": iteration,
        "history": all_history,
        "latest_metrics": latest_metrics,
        "current_best_metrics": new_best,
        "improvement_pct": improvement_pct,
        "evaluation_summary": eval_summary,
        "should_continue": should_continue,
        # Reset for next iteration
        "validation_errors": [],
        "is_valid": False,
        "retry_count": 0,
        "runtime_retry_count": 0,
        "runtime_error": None,
        "last_runtime_error": None,
    }


# ---------------------------------------------------------------
# 5. ROUTING
# ---------------------------------------------------------------
def route_after_baseline(state: ChipSimState) -> str:
    if not state.get("run_succeeded", False):
        return "abort"
    return "analyze"


def route_after_validation(state: ChipSimState) -> str:
    if state.get("is_valid", False):
        return "inject"
    if state.get("retry_count", 0) >= MAX_VALIDATION_RETRIES:
        return "give_up"
    return "retry"


def route_after_runtime(state: ChipSimState) -> str:
    """Route after runtime handler: success → evaluate, error → retry or evaluate."""
    if state.get("run_succeeded", False):
        return "evaluate"
    if state.get("runtime_retry_count", 0) >= MAX_RUNTIME_RETRIES:
        # Exhausted runtime retries — send to evaluator to record failure and decide
        return "evaluate_failed"
    return "retry"


def route_after_evaluation(state: ChipSimState) -> str:
    if state.get("should_continue", False):
        return "continue"
    return "done"


# ---------------------------------------------------------------
# 6. BUILD THE GRAPH
# ---------------------------------------------------------------
graph_builder = StateGraph(ChipSimState)

# Nodes
graph_builder.add_node("baseline", run_baseline)
graph_builder.add_node("analyzer", analyze_config)
graph_builder.add_node("optimizer", propose_mapping)
graph_builder.add_node("validator", validate_proposal)
graph_builder.add_node("injector", inject_into_mapper)
graph_builder.add_node("runtime_handler", runtime_handler)
graph_builder.add_node("evaluator", evaluate_results)

# Edges
graph_builder.add_edge(START, "baseline")

graph_builder.add_conditional_edges(
    "baseline", route_after_baseline,
    {"analyze": "analyzer", "abort": END}
)

graph_builder.add_edge("analyzer", "optimizer")
graph_builder.add_edge("optimizer", "validator")

graph_builder.add_conditional_edges(
    "validator", route_after_validation,
    {"inject": "injector", "retry": "optimizer", "give_up": END}
)

graph_builder.add_edge("injector", "runtime_handler")

graph_builder.add_conditional_edges(
    "runtime_handler", route_after_runtime,
    {"evaluate": "evaluator", "retry": "optimizer", "evaluate_failed": "evaluator"}
)

graph_builder.add_conditional_edges(
    "evaluator", route_after_evaluation,
    {"continue": "analyzer", "done": END}
)

graph = graph_builder.compile()


# ---------------------------------------------------------------
# 7. RUN
# ---------------------------------------------------------------
if __name__ == "__main__":
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("CHIPSIM Mapping Optimizer Agent")
    print("=" * 60)
    print(f"Config:              {CONFIG_NAME} ({CONFIG_PATH})")
    print(f"Max iterations:      {MAX_ITERATIONS}")
    print(f"Max val retries:     {MAX_VALIDATION_RETRIES}")
    print(f"Max runtime retries: {MAX_RUNTIME_RETRIES}")
    print(f"Mapper:              {MAPPER_PATH}")
    print(f"Backup:              {MAPPER_ORIGINAL_PATH}")
    print(f"Results:             {CHIPSIM_ROOT}/_results/formatted_results/")
    print()

    result = graph.invoke({
        "user_request": (
            f"Analyze this CHIPSIM configuration and suggest mapping improvements:\n\n"
            f"{yaml.dump(config, default_flow_style=False)}\n\n"
            f"The chiplet mapping uses 96 CMOS_Compute chiplets (2.46M weights each) "
            f"and 4 IO chiplets at corners on a 10x10 mesh topology. "
            f"The current mapper uses nearest-neighbor assignment."
        ),
        "config_name": CONFIG_NAME,
        "config_path": CONFIG_PATH,
        "config_analysis": "",
        "mapping_source_code": "",
        "mapping_proposal": "",
        "function_name": "",
        "validation_errors": [],
        "is_valid": False,
        "retry_count": 0,
        "injected_file_path": "",
        "run_succeeded": False,
        "runtime_error": None,
        "last_runtime_error": None,
        "latest_output": None,
        "runtime_retry_count": 0,
        "results_label": "",
        "baseline_metrics": {},
        "current_best_metrics": {},
        "latest_metrics": {},
        "improvement_pct": {},
        "evaluation_summary": "",
        "iteration": 0,
        "max_iterations": MAX_ITERATIONS,
        "history": [],
        "should_continue": True,
    })

    # ---------------------------------------------------------------
    # FINAL REPORT
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(f"Config: {CONFIG_NAME}")
    print(f"Total iterations: {result.get('iteration', 0)}/{MAX_ITERATIONS}")

    history = result.get("history", [])
    # Filter to evaluation entries only for the summary table
    eval_entries = [h for h in history if h.get("outcome") in ("IMPROVED", "NO_IMPROVEMENT")]
    runtime_errors = [h for h in history if h.get("outcome") == "RUNTIME_ERROR"]

    if eval_entries:
        print(f"\nIteration History:")
        print(f"{'Iter':>4}  {'Outcome':<16} {'Function':<25} {'Total Lat':>12} {'Compute Lat':>12} {'Comm Lat':>12}")
        print(f"{'':>4}  {'':16} {'':25} {'Δ Total':>12} {'Δ Compute':>12} {'Δ Comm':>12}")
        print("-" * 95)
        for h in eval_entries:
            m = h.get('metrics', {})
            total = f"{m['total_latency']:.2f}" if isinstance(m.get('total_latency'), (int, float)) else "N/A"
            compute = f"{m['compute_latency']:.2f}" if isinstance(m.get('compute_latency'), (int, float)) else "N/A"
            comm = f"{m['comm_latency']:.2f}" if isinstance(m.get('comm_latency'), (int, float)) else "N/A"
            imp = h.get('improvement_pct', {})
            if isinstance(imp, dict):
                d_t = f"{imp.get('total_latency', 0):+.1f}%"
                d_c = f"{imp.get('compute_latency', 0):+.1f}%"
                d_m = f"{imp.get('comm_latency', 0):+.1f}%"
            else:
                d_t = d_c = d_m = "N/A"
            print(f"{h['iteration']:>4}  {h['outcome']:<16} {h.get('function_name', 'N/A'):<25} {total:>12} {compute:>12} {comm:>12}")
            print(f"{'':>4}  {'':16} {'':25} {d_t:>12} {d_c:>12} {d_m:>12}")

    if runtime_errors:
        print(f"\nRuntime Errors: {len(runtime_errors)} total")
        for h in runtime_errors:
            print(f"  Iter {h.get('iteration', '?')} attempt {h.get('runtime_attempt', '?')}: "
                  f"{h.get('error', 'unknown')[:150]}")

    best = result.get("current_best_metrics", {})
    baseline = result.get("baseline_metrics", {})
    if best or baseline:
        print(f"\n{'Metric':<30} {'Baseline':>14} {'Best':>14} {'Change':>10}")
        print("-" * 72)
        for key, label in [('total_latency', 'Total Latency (µs)'),
                           ('compute_latency', 'Compute Latency (µs)'),
                           ('comm_latency', 'Comm Latency (µs)')]:
            b = baseline.get(key, 'N/A')
            bst = best.get(key, 'N/A')
            b_s = f"{b:.2f}" if isinstance(b, (int, float)) else str(b)
            bst_s = f"{bst:.2f}" if isinstance(bst, (int, float)) else str(bst)
            if isinstance(b, (int, float)) and isinstance(bst, (int, float)) and b > 0:
                ch = f"{(b - bst) / b * 100:+.1f}%"
            else:
                ch = "N/A"
            print(f"{label:<30} {b_s:>14} {bst_s:>14} {ch:>10}")

    # Save artifacts
    output_dir = os.path.dirname(os.path.abspath(__file__))

    code_output = os.path.join(output_dir, "last_generated_mapper.py")
    with open(code_output, 'w') as f:
        f.write(f"# Config: {CONFIG_NAME}\n")
        f.write(f"# Function: {result.get('function_name', 'unknown')}\n")
        f.write(f"# Iteration: {result.get('iteration', 0)}\n")
        imp = result.get('improvement_pct', {})
        if isinstance(imp, dict):
            f.write(f"# Improvement - Total: {imp.get('total_latency', 0):+.1f}%"
                    f"  Compute: {imp.get('compute_latency', 0):+.1f}%"
                    f"  Comm: {imp.get('comm_latency', 0):+.1f}%\n\n")
        else:
            f.write("# Improvement: N/A\n\n")
        f.write(result.get("mapping_proposal", "# No code generated"))
    print(f"\nGenerated code: {code_output}")

    history_output = os.path.join(output_dir, f"optimization_history_{CONFIG_NAME}.yaml")
    with open(history_output, 'w') as f:
        yaml.dump({
            "config": CONFIG_NAME,
            "max_iterations": MAX_ITERATIONS,
            "total_iterations": result.get("iteration", 0),
            "baseline_metrics": {k: v for k, v in baseline.items() if k not in ('per_model',)},
            "best_metrics": {k: v for k, v in best.items() if k not in ('per_model',)},
            "history": history,
        }, f, default_flow_style=False)
    print(f"History: {history_output}")