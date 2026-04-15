"""
CHIPSIM Mapping Optimizer Agent
================================
Two-agent LangGraph pipeline with pre-simulation validation and
automatic injection into model_mapper.py.

Flow: START → analyzer → optimizer → validate → (inject or retry) → END
"""
import os
import re
import ast
import yaml
import shutil
from dotenv import load_dotenv
from typing import TypedDict, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

# ---------------------------------------------------------------
# CONFIGURATION — update these paths for your environment
# ---------------------------------------------------------------
CHIPSIM_ROOT = "/home/rbaala2/ECE511-Final-Research-Project/CHIPSIM"
MAPPER_PATH = f"{CHIPSIM_ROOT}/src/mapping/model_mapper.py"
MAPPER_ORIGINAL_PATH = f"{CHIPSIM_ROOT}/src/mapping/model_mapper_original.py"
PARTITIONER_PATH = f"{CHIPSIM_ROOT}/src/mapping/layer_partitioner.py"
CONFIG_PATH = f"{CHIPSIM_ROOT}/configs/experiments/config_1.yaml"

# Back up the original mapper on first run (only once)
# Delete model_mapper_original.py to force a fresh backup
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
        # Only overwrite if the current file has the pattern
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
    user_request: str
    config_analysis: str
    mapping_source_code: str
    mapping_proposal: str        # raw LLM output (just the function code)
    function_name: str            # extracted function name
    validation_errors: list
    is_valid: bool
    injected_file_path: str       # path to the modified model_mapper.py
    retry_count: int


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
# 4. DEFINE NODES
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
    """Agent 2: Generate drop-in replacement mapper function."""

    # Include validation errors from previous attempts if any
    error_context = ""
    if state.get("validation_errors"):
        error_context = (
            "\n\nYour PREVIOUS attempt failed validation with these errors:\n"
            + "\n".join(f"  - {e}" for e in state["validation_errors"])
            + "\n\nFix ALL of these issues. Pay careful attention to the method "
            "signature, return type, and naming requirements."
        )

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
    return {"mapping_proposal": response.content}


def validate_proposal(state: ChipSimState) -> dict:
    """Validate that the LLM output meets the mapper contract before injection."""
    raw_code = state["mapping_proposal"]
    errors = []

    # --- Clean up LLM output ---
    code = raw_code.strip()
    # Strip markdown fences
    code = re.sub(r'^```(?:python)?\s*\n?', '', code)
    code = re.sub(r'\n?```\s*$', '', code)
    # Strip any leading prose before the def
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

    # --- 2. Check there's exactly one function definition ---
    func_nodes = [n for n in ast.iter_child_nodes(tree)
                  if isinstance(n, ast.FunctionDef)]
    if len(func_nodes) == 0:
        errors.append("No function definition found. Output must start with 'def _...'")
    elif len(func_nodes) > 1:
        errors.append(f"Found {len(func_nodes)} function definitions, expected exactly 1")

    if not func_nodes:
        return {
            "mapping_proposal": code,
            "validation_errors": errors,
            "is_valid": False,
            "function_name": "",
            "retry_count": state.get("retry_count", 0) + 1,
        }

    func = func_nodes[0]
    func_name = func.name

    # --- 3. Check function name starts with underscore ---
    if not func_name.startswith('_'):
        errors.append(
            f"Function name '{func_name}' must start with underscore "
            f"(e.g., '_{func_name}')"
        )

    # --- 4. Check 'self' is the first argument ---
    args = func.args
    arg_names = [a.arg for a in args.args]
    if not arg_names or arg_names[0] != 'self':
        errors.append(f"First argument must be 'self', got: {arg_names}")

    # --- 5. Check required positional arguments ---
    required_args = [
        'self', 'model_layer_info', 'system', 'preference',
        'current_available_crossbars'
    ]
    for i, req in enumerate(required_args):
        if i >= len(arg_names):
            errors.append(f"Missing required argument '{req}' at position {i}")
        elif arg_names[i] != req:
            errors.append(
                f"Argument at position {i} should be '{req}', got '{arg_names[i]}'"
            )

    # --- 6. Check optional keyword arguments ---
    defaults_names = []
    # keyword-only args (after *)
    for kw in args.kwonlyargs:
        defaults_names.append(kw.arg)
    # or defaults on positional args
    num_defaults = len(args.defaults)
    if num_defaults > 0:
        defaulted_args = arg_names[-num_defaults:]
        defaults_names.extend(defaulted_args)

    expected_optional = {'layer_mappings', 'shortest_paths'}
    for opt in expected_optional:
        if opt not in arg_names and opt not in defaults_names:
            errors.append(f"Missing optional argument '{opt}' (should default to None)")

    # --- 7. Check return statements ---
    returns = [n for n in ast.walk(func) if isinstance(n, ast.Return)]
    if not returns:
        errors.append("No return statement found in function")
    else:
        for ret in returns:
            if ret.value is None:
                errors.append("Found bare 'return' with no value — must return 4-tuple")
            elif isinstance(ret.value, ast.Tuple):
                if len(ret.value.elts) != 4:
                    errors.append(
                        f"Return tuple has {len(ret.value.elts)} elements, "
                        f"expected exactly 4: (remaining_capacity, action, "
                        f"mapping_failed, failure_reason)"
                    )

    # --- 8. Check no import statements ---
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = ""
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = ", ".join(a.name for a in node.names)
            errors.append(f"Contains import statement ({module}) — not allowed")

    # --- 9. Check it references key variables/patterns ---
    code_text = code
    critical_patterns = {
        'remaining_capacity': 'Must track remaining chiplet capacity',
        'action': 'Must build an action list of allocation percentages',
        'mapping_failed': 'Must set mapping_failed flag',
    }
    for pattern, description in critical_patterns.items():
        if pattern not in code_text:
            errors.append(f"Missing '{pattern}' — {description}")

    # --- 10. Check it calls _calculate_layer_requirements ---
    if '_calculate_layer_requirements' not in code_text:
        errors.append(
            "Does not call self._calculate_layer_requirements() — "
            "this is needed to compute chiplet-specific resource requirements"
        )

    # Build result
    # Strip leading underscore for the "public" function name used in wiring
    public_name = func_name.lstrip('_') if func_name.startswith('_') else func_name

    is_valid = len(errors) == 0
    return {
        "mapping_proposal": code,
        "validation_errors": errors,
        "is_valid": is_valid,
        "function_name": public_name,
        "retry_count": state.get("retry_count", 0) + (0 if is_valid else 1),
    }


def inject_into_mapper(state: ChipSimState) -> dict:
    """Inject the validated function into model_mapper.py with proper wiring."""
    function_code = state["mapping_proposal"]
    public_name = state["function_name"]
    method_name = f"_{public_name}"

    # Always start from the original pristine file
    with open(MAPPER_ORIGINAL_PATH, 'r') as f:
        original = f.read()

    modified = original

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
    # Strategy: find _get_mapper_function line by line, locate the "else:" 
    # that belongs to the if/elif chain, and insert our elif before it.
    # This works regardless of what the backup file looks like.
    target_elif_statement = f'elif self.mapping_function == "{public_name}":'
    if target_elif_statement not in modified:
            lines = modified.split('\n')
            new_lines = []
            injected = False
            anchor_string = "return self._nearest_neighbor_mapper_v3"

            for line in lines:
                # Always add the current line first
                new_lines.append(line)
                
                # If we hit our anchor, inject the new routing right after it
                if anchor_string in line and not injected:
                    # Calculate indentation dynamically based on the anchor line
                    return_indent_level = len(line) - len(line.lstrip())
                    elif_indent_level = max(0, return_indent_level - 4)
                    
                    indent_elif = ' ' * elif_indent_level
                    indent_return = ' ' * return_indent_level
                    
                    # Inject the new elif block
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
    # Determine the base indentation from the LLM output's def line
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
            # Strip the LLM's base indentation, add 4-space class method indent
            current_indent = len(line) - len(line.lstrip())
            relative_indent = max(0, current_indent - base_indent)
            indented_lines.append('    ' + ' ' * relative_indent + line.lstrip())

    method_block = '\n'.join(indented_lines)

    # Append after the last method in the class
    modified = modified.rstrip() + '\n\n' + method_block + '\n'

    # --- 4. Write the modified file ---
    with open(MAPPER_PATH, 'w') as f:
        f.write(modified)

    print(f"Injected method '{method_name}' into {MAPPER_PATH}")
    print(f"  Default mapping_function set to: '{public_name}'")
    print(f"  Routing added in _get_mapper_function")

    return {"injected_file_path": MAPPER_PATH}


# ---------------------------------------------------------------
# 5. ROUTING FUNCTIONS
# ---------------------------------------------------------------
MAX_VALIDATION_RETRIES = 3

def route_after_validation(state: ChipSimState) -> str:
    """Decide whether to inject or retry after validation."""
    if state.get("is_valid", False):
        return "inject"
    if state.get("retry_count", 0) >= MAX_VALIDATION_RETRIES:
        return "give_up"
    return "retry"


# ---------------------------------------------------------------
# 6. BUILD THE GRAPH
# ---------------------------------------------------------------
graph_builder = StateGraph(ChipSimState)

# Add nodes
graph_builder.add_node("analyzer", analyze_config)
graph_builder.add_node("optimizer", propose_mapping)
graph_builder.add_node("validator", validate_proposal)
graph_builder.add_node("injector", inject_into_mapper)

# Edges
graph_builder.add_edge(START, "analyzer")
graph_builder.add_edge("analyzer", "optimizer")
graph_builder.add_edge("optimizer", "validator")

# Conditional: validator decides next step
graph_builder.add_conditional_edges(
    "validator",
    route_after_validation,
    {
        "inject": "injector",       # valid → inject into model_mapper.py
        "retry": "optimizer",       # invalid → retry with error feedback
        "give_up": END,             # too many retries → stop
    }
)

graph_builder.add_edge("injector", END)

# Compile
graph = graph_builder.compile()


# ---------------------------------------------------------------
# 7. RUN IT
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
    })

    # --- Print results ---
    print("\n" + "=" * 60)
    print("CONFIG ANALYSIS:")
    print("=" * 60)
    print(result["config_analysis"])

    print("\n" + "=" * 60)
    print("VALIDATION RESULT:")
    print("=" * 60)
    if result["is_valid"]:
        print(f"PASSED — function '{result['function_name']}' is valid")
        print(f"Injected into: {result.get('injected_file_path', 'N/A')}")
    else:
        print(f"FAILED after {result['retry_count']} attempts")
        print("Errors:")
        for e in result.get("validation_errors", []):
            print(f"  - {e}")

    print("\n" + "=" * 60)
    print("GENERATED CODE:")
    print("=" * 60)
    print(result["mapping_proposal"])

    # --- Save the generated code separately for reference ---
    output_dir = os.path.dirname(os.path.abspath(__file__))
    code_output = os.path.join(output_dir, "last_generated_mapper.py")
    with open(code_output, 'w') as f:
        f.write(f"# Function name: {result.get('function_name', 'unknown')}\n")
        f.write(f"# Valid: {result.get('is_valid', False)}\n")
        f.write(f"# Retry count: {result.get('retry_count', 0)}\n\n")
        f.write(result.get("mapping_proposal", "# No code generated"))
    print(f"\nGenerated code saved to: {code_output}")