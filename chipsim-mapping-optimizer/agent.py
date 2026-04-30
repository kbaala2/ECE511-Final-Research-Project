"""
CHIPSIM Mapping Optimizer Agent
================================
Two-agent LangGraph pipeline with pre-simulation validation, automatic
injection into model_mapper.py, and runtime retry on simulator failures.

Flow: START → analyzer → optimizer → validator → (inject or retry)
                                          ↓
                                       injector → runtime_handler → (END or retry → optimizer)
"""

# this code is agent.py

import os
import re
import ast
import yaml
import shutil
import subprocess
from dotenv import load_dotenv
from typing import TypedDict, Optional, Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

# ---------------------------------------------------------------
# CONFIGURATION — update these paths for your environment
# ---------------------------------------------------------------
CHIPSIM_ROOT = "/home/ECE511-Final-Research-Project/CHIPSIM"
MAPPER_PATH = f"{CHIPSIM_ROOT}/src/mapping/model_mapper.py"
MAPPER_ORIGINAL_PATH = f"{CHIPSIM_ROOT}/src/mapping/model_mapper_original.py"
PARTITIONER_PATH = f"{CHIPSIM_ROOT}/src/mapping/layer_partitioner.py"
CONFIG_PATH = f"{CHIPSIM_ROOT}/configs/experiments/config_1.yaml"

MAX_VALIDATION_RETRIES = 3
MAX_RUNTIME_RETRIES = 3

# Back up the original mapper on first run (only once).
# Delete model_mapper_original.py to force a fresh backup.
if not os.path.exists(MAPPER_ORIGINAL_PATH):
    shutil.copy(MAPPER_PATH, MAPPER_ORIGINAL_PATH)
    print(f"Backed up original mapper to {MAPPER_ORIGINAL_PATH}")
else:
    # Verify backup is pristine — must contain the else/try routing pattern
    with open(MAPPER_ORIGINAL_PATH, 'r') as f:
        backup_content = f.read()
    if '        else:\n            try:' not in backup_content:
        print(f"WARNING: {MAPPER_ORIGINAL_PATH} looks corrupted (missing else/try pattern).")
        print(f"  Creating fresh backup from {MAPPER_PATH}...")
        with open(MAPPER_PATH, 'r') as f:
            current = f.read()
        if '        else:\n            try:' in current:
            shutil.copy(MAPPER_PATH, MAPPER_ORIGINAL_PATH)
            print(f"  Fresh backup created.")
        else:
            print(f"  ERROR: Neither file has the expected pattern. Restore model_mapper.py from git.")


# ---------------------------------------------------------------
# 1. DEFINE STATE
# ---------------------------------------------------------------
class ChipSimState(TypedDict):
    # validation pipeline
    user_request: str
    config_analysis: str
    mapping_source_code: str
    mapping_proposal: str          # cleaned LLM output (function code)
    function_name: str
    validation_errors: list
    is_valid: bool
    injected_file_path: str
    retry_count: int               # validation retries within a single proposal cycle

    # runtime pipeline
    target_file_path: str
    run_command: list[str]
    run_cwd: Optional[str]
    run_succeeded: bool
    runtime_error: Optional[str]
    last_runtime_error: Optional[str]   # preserved across validation retries for LLM context
    latest_output: Optional[str]
    runtime_retry_count: int            # total runtime retries across the whole run
    history: list[dict[str, Any]]


# ---------------------------------------------------------------
# 2. CREATE THE LLM
# ---------------------------------------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)


# ---------------------------------------------------------------
# 3. MAPPER CONTRACT — constraints for the optimizer LLM
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
     Returns crossbars needed (IMC) or weight units (CMOS) for a layer on a chiplet.
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
def analyze_config(state: ChipSimState) -> dict:
    """Agent 1: Analyze the config, mapper code, and partitioner code."""
    with open(MAPPER_ORIGINAL_PATH) as f:
        mapper_code = f.read()
    with open(PARTITIONER_PATH) as f:
        partitioner_code = f.read()

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
            "Analyze the nearest-neighbor mapper's weaknesses for this config."
        ))
    ]
    response = llm.invoke(messages)
    return {
        "config_analysis": response.content,
        "mapping_source_code": mapper_code,
    }


def propose_mapping(state: ChipSimState) -> dict:
    """Agent 2: Generate a drop-in replacement mapper function.

    Uses error context from either the most recent validation failure or the
    most recent runtime failure. `last_runtime_error` is preserved across
    validation retries so the LLM doesn't forget the original runtime issue
    while it's also fixing validation problems.
    """
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

    # Promote the active runtime_error to last_runtime_error so it survives
    # across upcoming validation retries, then clear runtime_error so it
    # doesn't get re-applied as the primary error context next round.
    return {
        "mapping_proposal": response.content,
        "runtime_error": None,
        "last_runtime_error": runtime_error or last_runtime_error,
        "validation_errors": [],
    }


def validate_proposal(state: ChipSimState) -> dict:
    """Validate that the LLM output meets the mapper contract before injection."""
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

    # --- 1. Check it parses as valid Python ---
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        errors.append(f"Syntax error on line {e.lineno}: {e.msg}")
        return {
            "mapping_proposal": code,
            "validation_errors": errors,
            "is_valid": False,
            "function_name": "",
            "retry_count": state.get("retry_count", 0) + 1,
        }

    # --- 2. Find the top-level function definition ---
    func_def = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break

    if func_def is None:
        errors.append("No top-level function definition found.")
        return {
            "mapping_proposal": code,
            "validation_errors": errors,
            "is_valid": False,
            "function_name": "",
            "retry_count": state.get("retry_count", 0) + 1,
        }

    # --- 3. Method name must start with underscore ---
    raw_name = func_def.name
    if not raw_name.startswith("_"):
        errors.append(
            f"Function name '{raw_name}' must start with an underscore "
            f"(e.g., '_my_optimized_mapper')."
        )
    public_name = raw_name.lstrip("_")
    if not public_name:
        errors.append("Function name cannot be just underscores.")

    # --- 4. Argument list must match the contract exactly ---
    expected_args = [
        "self",
        "model_layer_info",
        "system",
        "preference",
        "current_available_crossbars",
        "layer_mappings",
        "shortest_paths",
    ]
    actual_args = [a.arg for a in func_def.args.args]
    if actual_args != expected_args:
        errors.append(
            f"Argument list mismatch.\n"
            f"  Expected: {expected_args}\n"
            f"  Got:      {actual_args}"
        )

    # Defaults for layer_mappings and shortest_paths must be None
    defaults = func_def.args.defaults
    if len(defaults) < 2:
        errors.append(
            "layer_mappings and shortest_paths must have default value None."
        )
    else:
        last_two = defaults[-2:]
        for i, d in enumerate(last_two):
            arg_name = expected_args[-2 + i]
            if not (isinstance(d, ast.Constant) and d.value is None):
                errors.append(f"Default value for '{arg_name}' must be None.")

    # --- 5. At least one return must be a 4-tuple ---
    return_nodes = [n for n in ast.walk(func_def) if isinstance(n, ast.Return)]
    if not return_nodes:
        errors.append("Function has no return statement.")
    else:
        has_valid_tuple_return = False
        for rn in return_nodes:
            if rn.value is None:
                continue
            if isinstance(rn.value, ast.Tuple) and len(rn.value.elts) == 4:
                has_valid_tuple_return = True
                break
        if not has_valid_tuple_return:
            errors.append(
                "At least one return statement must return a 4-tuple: "
                "(remaining_capacity, action, mapping_failed, failure_reason)."
            )

    # --- 6. No imports inside the function ---
    for n in ast.walk(func_def):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            errors.append(
                "Import statements are not allowed inside the function. "
                "numpy as np, math, and networkx as nx are already imported."
            )
            break

    # --- 7. Final decision ---
    if errors:
        return {
            "mapping_proposal": code,
            "validation_errors": errors,
            "is_valid": False,
            "function_name": "",
            "retry_count": state.get("retry_count", 0) + 1,
        }

    return {
        "mapping_proposal": code,
        "validation_errors": [],
        "is_valid": True,
        "function_name": public_name,
        "retry_count": state.get("retry_count", 0),
    }


def runtime_handler(state: ChipSimState) -> dict:
    """Run the simulator. On failure, preserve the runtime error for the
    optimizer and reset the validation retry budget so the next proposal
    gets a fresh validation cycle.
    """
    history = state.get("history", [])
    runtime_retry_count = state.get("runtime_retry_count", 0)

    def fail(msg: str, stdout: Optional[str] = None) -> dict:
        history.append({
            "runtime_attempt": runtime_retry_count + 1,
            "outcome": "RUNTIME_ERROR",
            "error": msg[:1000],
        })
        return {
            "history": history,
            "run_succeeded": False,
            "runtime_error": msg,
            "latest_output": stdout,
            "runtime_retry_count": runtime_retry_count + 1,
            "retry_count": 0,                # fresh validation budget for next proposal
        }

    try:
        result = subprocess.run(
            state["run_command"],
            cwd=state.get("run_cwd"),
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return fail("Execution timed out after 20 seconds.")
    except Exception as e:
        return fail(f"Execution failed unexpectedly:\n{e}")

    if result.returncode != 0:
        error_text = (result.stderr or result.stdout or "")[-3000:]
        return fail(error_text, stdout=result.stdout)

    # Success
    history.append({
        "runtime_attempt": runtime_retry_count + 1,
        "outcome": "SUCCESS",
        "output": result.stdout[-3000:],
    })
    return {
        "history": history,
        "run_succeeded": True,
        "runtime_error": None,
        "latest_output": result.stdout,
        "runtime_retry_count": runtime_retry_count,
    }


def inject_into_mapper(state: ChipSimState) -> dict:
    """Inject the validated function into model_mapper.py with proper wiring.
    Always starts from the pristine MAPPER_ORIGINAL_PATH so each retry
    cleanly replaces the previous proposal rather than stacking.
    """
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
        f'\\1"{public_name}"',
        modified,
        count=1
    )
    new_default = re.search(r'mapping_function\s*=\s*["\']([^"\']+)["\']', modified)
    print(f"  __init__ default: '{old_default.group(1) if old_default else '?'}' → '{new_default.group(1) if new_default else '?'}'")

    # --- 2. Add routing in _get_mapper_function ---
    target_elif_statement = f'elif self.mapping_function == "{public_name}":'
    if target_elif_statement not in modified:
        lines = modified.split('\n')
        new_lines = []
        injected = False
        anchor_string = "return self._nearest_neighbor_mapper_v3"

        for line in lines:
            new_lines.append(line)
            if anchor_string in line and not injected:
                return_indent_level = len(line) - len(line.lstrip())
                elif_indent_level = max(0, return_indent_level - 4)
                indent_elif = ' ' * elif_indent_level
                indent_return = ' ' * return_indent_level
                new_lines.append(f'{indent_elif}elif self.mapping_function == "{public_name}":')
                new_lines.append(f'{indent_return}return self.{method_name}')
                injected = True

        if injected:
            modified = '\n'.join(new_lines)
            print(f"  _get_mapper_function: elif for '{public_name}' injected successfully")
        else:
            print(f"  _get_mapper_function: INJECTION FAILED")
            print(f"    Could not find the anchor string: '{anchor_string}'")
    else:
        print(f"  _get_mapper_function: elif for '{public_name}' already present")

    # --- 3. Indent and append the new method to the class ---
    lines = function_code.split('\n')
    base_indent = 0
    for line in lines:
        if line.strip().startswith('def '):
            base_indent = len(line) - len(line.lstrip())
            break

    indented_lines = []
    for line in lines:
        if not line.strip():
            indented_lines.append('')
        else:
            current_indent = len(line) - len(line.lstrip())
            relative_indent = max(0, current_indent - base_indent)
            indented_lines.append('    ' + ' ' * relative_indent + line.lstrip())

    method_block = '\n'.join(indented_lines)
    modified = modified.rstrip() + '\n\n' + method_block + '\n'

    # --- 4. Write the modified file ---
    with open(MAPPER_PATH, 'w') as f:
        f.write(modified)

    print(f"Injected method '{method_name}' into {MAPPER_PATH}")
    print(f"  Default mapping_function set to: '{public_name}'")
    print(f"  Routing added in _get_mapper_function")

    return {"injected_file_path": MAPPER_PATH}


# ---------------------------------------------------------------
# 5. ROUTING
# ---------------------------------------------------------------
def route_after_validation(state: ChipSimState) -> str:
    """Decide whether to inject, retry validation, or give up."""
    if state.get("is_valid", False):
        return "inject"
    if state.get("retry_count", 0) >= MAX_VALIDATION_RETRIES:
        return "give_up"
    return "retry"


def runtime_decision(state: ChipSimState) -> str:
    """Decide whether to end the run, retry the optimizer, or give up."""
    if state.get("run_succeeded", False):
        return "done"
    if state.get("runtime_retry_count", 0) >= MAX_RUNTIME_RETRIES:
        return "done"
    return "retry"


# ---------------------------------------------------------------
# 6. BUILD THE GRAPH
# ---------------------------------------------------------------
graph_builder = StateGraph(ChipSimState)

graph_builder.add_node("analyzer", analyze_config)
graph_builder.add_node("optimizer", propose_mapping)
graph_builder.add_node("validator", validate_proposal)
graph_builder.add_node("injector", inject_into_mapper)
graph_builder.add_node("runtime_handler", runtime_handler)

graph_builder.add_edge(START, "analyzer")
graph_builder.add_edge("analyzer", "optimizer")
graph_builder.add_edge("optimizer", "validator")

graph_builder.add_conditional_edges(
    "validator",
    route_after_validation,
    {
        "inject": "injector",
        "retry": "optimizer",
        "give_up": END,
    },
)

graph_builder.add_edge("injector", "runtime_handler")

graph_builder.add_conditional_edges(
    "runtime_handler",
    runtime_decision,
    {
        "retry": "optimizer",
        "done": END,
    },
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
    print(f"Config: {CONFIG_PATH}")
    print(f"Mapper: {MAPPER_PATH}")
    print(f"Original backup: {MAPPER_ORIGINAL_PATH}")
    print(f"Validation retries per proposal: {MAX_VALIDATION_RETRIES}")
    print(f"Runtime retries total:           {MAX_RUNTIME_RETRIES}")
    print()

    result = graph.invoke({
        "user_request": (
            f"Analyze this CHIPSIM configuration and suggest mapping improvements:\n\n"
            f"{yaml.dump(config, default_flow_style=False)}\n\n"
            f"The chiplet mapping uses 96 CMOS_Compute chiplets (2.46M weights each) "
            f"and 4 IO chiplets at corners on a 10x10 mesh topology. "
            f"The workload is AlexNet (8 layers: 5 conv + 3 FC, 61M total weights, "
            f"needs 25 chiplets). The current mapper uses nearest-neighbor assignment."
        ),
        "config_analysis": "",
        "mapping_source_code": "",
        "mapping_proposal": "",
        "function_name": "",
        "validation_errors": [],
        "is_valid": False,
        "injected_file_path": "",
        "retry_count": 0,

        "target_file_path": MAPPER_PATH,
        "run_command": ["python3", "simulate.py", "--mode", "simulate", "--config", "config_1"],
        "run_cwd": CHIPSIM_ROOT,

        "run_succeeded": False,
        "runtime_error": None,
        "last_runtime_error": None,
        "latest_output": None,
        "runtime_retry_count": 0,
        "history": [],
    })

    # --- Print results ---
    print("\n" + "=" * 60)
    print("CONFIG ANALYSIS:")
    print("=" * 60)
    print(result["config_analysis"])

    print("\n" + "=" * 60)
    print("VALIDATION RESULT:")
    print("=" * 60)
    if result.get("is_valid"):
        print(f"PASSED — function '{result['function_name']}' is valid")
        print(f"Injected into: {result.get('injected_file_path', 'N/A')}")
    else:
        print(f"FAILED after {result.get('retry_count', 0)} validation attempts")
        print("Errors:")
        for e in result.get("validation_errors", []):
            print(f"  - {e}")

    print("\n" + "=" * 60)
    print("RUNTIME RESULT:")
    print("=" * 60)
    if result.get("run_succeeded"):
        print(f"SUCCESS after {result.get('runtime_retry_count', 0)} retries")
    else:
        print(f"FAILED after {result.get('runtime_retry_count', 0)} runtime attempts")
        if result.get("runtime_error"):
            print(f"Last error:\n{result['runtime_error'][:500]}")

    # print("\n" + "=" * 60)
    # print("HISTORY:")
    # print("=" * 60)
    # for entry in result.get("history", []):
    #     outcome = entry.get("outcome", "?")
    #     attempt = entry.get("runtime_attempt", "?")
    #     print(f"  Attempt {attempt}: {outcome}")
    print("\n" + "=" * 60)
    print("HISTORY:")
    print("=" * 60)
    for entry in result.get("history", []):
        outcome = entry.get("outcome", "?")
        attempt = entry.get("runtime_attempt", "?")
        print(f"\n  Attempt {attempt}: {outcome}")
        if "error" in entry:
            print(f"  Error:\n    {entry['error'][:800]}")
        if "output" in entry:
            print(f"  Output (last 500 chars):\n    {entry['output'][-500:]}")

    print("\n" + "=" * 60)
    print("GENERATED CODE:")
    print("=" * 60)
    print(result.get("mapping_proposal", ""))

    # --- Save the generated code separately for reference ---
    output_dir = os.path.dirname(os.path.abspath(__file__))
    code_output = os.path.join(output_dir, "last_generated_mapper.py")
    with open(code_output, 'w') as f:
        f.write(f"# Function name: {result.get('function_name', 'unknown')}\n")
        f.write(f"# Valid: {result.get('is_valid', False)}\n")
        f.write(f"# Validation retries: {result.get('retry_count', 0)}\n")
        f.write(f"# Runtime retries:    {result.get('runtime_retry_count', 0)}\n")
        f.write(f"# Run succeeded:      {result.get('run_succeeded', False)}\n\n")
        f.write(result.get("mapping_proposal", "# No code generated"))
    print(f"\nGenerated code saved to: {code_output}")