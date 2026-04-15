# from typing import TypedDict, Optional, Any
# from langgraph.graph import StateGraph, START, END
# from langchain_core.messages import SystemMessage, HumanMessage
# from langchain_ollama import ChatOllama
# import subprocess

# llm = ChatOllama(
#     model="llama3.1:8b",
#     temperature=0,
# )

# class AgenticState(TypedDict, total=False):
#     iteration: int
#     max_iterations: int

#     target_file_path: str
#     run_command: list[str]
#     run_cwd: Optional[str]

#     current_working_code: str
#     proposed_code: str
#     failed_code: Optional[str]

#     run_succeeded: bool
#     runtime_error: Optional[str]
#     latest_output: Optional[str]

#     should_continue: bool
#     history: list[dict[str, Any]]


# def propose_code_change(state: AgenticState) -> dict:
#     error_context = ""

#     if state.get("runtime_error"):
#         error_context = (
#             "\nThe previous candidate failed at runtime.\n\n"
#             f"Failed code:\n```python\n{state.get('failed_code', '')}\n```\n\n"
#             f"Runtime error:\n```text\n{state['runtime_error']}\n```\n\n"
#             "Fix the runtime issue. Return only valid Python code."
#         )

#     ##This is the code we will need to change for CHIPSIM
#     messages = [
#         SystemMessage(content=(
#             "You repair Python code. "
#             "Your goal is to fix runtime errors while preserving intended behavior. "
#             "Return only the full corrected Python file."
#         )),
#         HumanMessage(content=(
#             f"Current working baseline:\n```python\n{state['current_working_code']}\n```\n"
#             f"{error_context}"
#         ))
#     ]

#     response = llm.invoke(messages)
#     return {"proposed_code": response.content}


# def run_candidate(state: AgenticState) -> dict:
#     try:
#         with open(state["target_file_path"], "w", encoding="utf-8") as f:
#             f.write(state["proposed_code"])
#     except Exception as e:
#         return {
#             "run_succeeded": False,
#             "runtime_error": f"Failed to write file:\n{e}",
#             "failed_code": state["proposed_code"],
#             "latest_output": None,
#         }

#     try:
#         result = subprocess.run(
#             state["run_command"],
#             cwd=state.get("run_cwd"),
#             capture_output=True,
#             text=True,
#             timeout=20
#         )
#     except subprocess.TimeoutExpired:
#         return {
#             "run_succeeded": False,
#             "runtime_error": "Execution timed out after 20 seconds.",
#             "failed_code": state["proposed_code"],
#             "latest_output": None,
#         }
#     except Exception as e:
#         return {
#             "run_succeeded": False,
#             "runtime_error": f"Execution failed unexpectedly:\n{e}",
#             "failed_code": state["proposed_code"],
#             "latest_output": None,
#         }

#     if result.returncode != 0:
#         error_text = (result.stderr or result.stdout or "")[-3000:]
#         return {
#             "run_succeeded": False,
#             "runtime_error": error_text,
#             "failed_code": state["proposed_code"],
#             "latest_output": result.stdout,
#         }

#     return {
#         "run_succeeded": True,
#         "runtime_error": None,
#         "failed_code": None,
#         "latest_output": result.stdout,
#     }


# def handle_error(state: AgenticState) -> dict:
#     iteration = state["iteration"] + 1
#     history = state.get("history", [])

#     history.append({
#         "iteration": iteration,
#         "outcome": "RUNTIME_ERROR",
#         "error": (state.get("runtime_error") or "")[:1000],
#     })

#     return {
#         "iteration": iteration,
#         "history": history,
#     }


# def accept_candidate(state: AgenticState) -> dict:
#     iteration = state["iteration"] + 1
#     history = state.get("history", [])

#     history.append({
#         "iteration": iteration,
#         "outcome": "SUCCESS",
#         "output": state.get("latest_output"),
#     })

#     return {
#         "iteration": iteration,
#         "history": history,
#         "current_working_code": state["proposed_code"],
#         "should_continue": False,
#     }


# def route_after_run(state: AgenticState) -> str:
#     if state["run_succeeded"]:
#         return "accept"
#     if state["iteration"] >= state["max_iterations"]:
#         return "done"
#     return "handle_error"


# builder = StateGraph(AgenticState)

# builder.add_node("propose", propose_code_change)
# builder.add_node("run", run_candidate)
# builder.add_node("handle_error", handle_error)
# builder.add_node("accept", accept_candidate)

# builder.add_edge(START, "propose")
# builder.add_edge("propose", "run")

# builder.add_conditional_edges(
#     "run",
#     route_after_run,
#     {
#         "accept": "accept",
#         "handle_error": "handle_error",
#         "done": END,
#     }
# )

# builder.add_edge("handle_error", "propose")
# builder.add_edge("accept", END)

# graph = builder.compile()

# if __name__ == "__main__":
#     target_file = "buggy_script1.py"

#     with open(target_file, "r", encoding="utf-8") as f:
#         original_code = f.read()

#     initial_state = {
#         "iteration": 0,
#         "max_iterations": 3,
#         "target_file_path": target_file,
#         "run_command": ["python3", target_file],
#         "run_cwd": None,
#         "current_working_code": original_code,
#         "history": [],
#     }

#     result = graph.invoke(initial_state)

#     if not result.get("run_succeeded", False):
#         with open(target_file, "w", encoding="utf-8") as f:
#             f.write(original_code)
#         print("Repair failed. Original file restored.")
#     else:
#         print("Repair succeeded.")

#     print("\nFINAL STATE:")
#     print(result)

from typing import TypedDict, Optional, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
import subprocess

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
)

class AgenticState(TypedDict, total=False):
    iteration: int
    max_iterations: int

    target_file_path: str
    run_command: list[str]
    run_cwd: Optional[str]

    current_working_code: str
    proposed_code: str
    failed_code: Optional[str]

    run_succeeded: bool
    runtime_error: Optional[str]
    latest_output: Optional[str]

    should_continue: bool
    history: list[dict[str, Any]]


def propose_code_change(state: AgenticState) -> dict:
    error_context = ""

    if state.get("runtime_error"):
        error_context = (
            "\nThe previous candidate failed at runtime.\n\n"
            f"Failed code:\n```python\n{state.get('failed_code', '')}\n```\n\n"
            f"Runtime error:\n```text\n{state['runtime_error']}\n```\n\n"
            "Fix the runtime issue. Return only valid Python code."
        )

    messages = [
        SystemMessage(content=(
            "Do not replace crashes with new raised exceptions unless absolutely necessary."
            "Prefer making the code robust to the given failing input. "
            "Preserve successful execution of the script. "
            "The repaired file must run successfully on the provided failing example."
            "Return a working implementation, not defensive failure-only validation."
            "Make sure anythinng that is written that is not code is commented out"
        )),
        HumanMessage(content=(
            f"Current working baseline:\n```python\n{state['current_working_code']}\n```\n"
            f"{error_context}"
        ))
    ]

    response = llm.invoke(messages)
    return {"proposed_code": response.content}


def run_candidate_and_update(state: AgenticState) -> dict:
    history = state.get("history", [])
    iteration = state["iteration"] + 1

    try:
        with open(state["target_file_path"], "w", encoding="utf-8") as f:
            f.write(state["proposed_code"])
    except Exception as e:
        history.append({
            "iteration": iteration,
            "outcome": "RUNTIME_ERROR",
            "error": f"Failed to write file:\n{e}"[:1000],
        })
        return {
            "iteration": iteration,
            "history": history,
            "run_succeeded": False,
            "runtime_error": f"Failed to write file:\n{e}",
            "failed_code": state["proposed_code"],
            "latest_output": None,
        }

    try:
        result = subprocess.run(
            state["run_command"],
            cwd=state.get("run_cwd"),
            capture_output=True,
            text=True,
            timeout=20
        )
    except subprocess.TimeoutExpired:
        history.append({
            "iteration": iteration,
            "outcome": "RUNTIME_ERROR",
            "error": "Execution timed out after 20 seconds."[:1000],
        })
        return {
            "iteration": iteration,
            "history": history,
            "run_succeeded": False,
            "runtime_error": "Execution timed out after 20 seconds.",
            "failed_code": state["proposed_code"],
            "latest_output": None,
        }
    except Exception as e:
        msg = f"Execution failed unexpectedly:\n{e}"
        history.append({
            "iteration": iteration,
            "outcome": "RUNTIME_ERROR",
            "error": msg[:1000],
        })
        return {
            "iteration": iteration,
            "history": history,
            "run_succeeded": False,
            "runtime_error": msg,
            "failed_code": state["proposed_code"],
            "latest_output": None,
        }

    if result.returncode != 0:
        error_text = (result.stderr or result.stdout or "")[-3000:]
        history.append({
            "iteration": iteration,
            "outcome": "RUNTIME_ERROR",
            "error": error_text[:1000],
        })
        return {
            "iteration": iteration,
            "history": history,
            "run_succeeded": False,
            "runtime_error": error_text,
            "failed_code": state["proposed_code"],
            "latest_output": result.stdout,
        }

    history.append({
        "iteration": iteration,
        "outcome": "SUCCESS",
        "output": result.stdout,
    })

    return {
        "iteration": iteration,
        "history": history,
        "run_succeeded": True,
        "runtime_error": None,
        "failed_code": None,
        "latest_output": result.stdout,
        "current_working_code": state["proposed_code"],
        "should_continue": False,
    }


def route_after_run(state: AgenticState) -> str:
    if state["run_succeeded"]:
        return "done"
    if state["iteration"] >= state["max_iterations"]:
        return "done"
    return "retry"


builder = StateGraph(AgenticState)

builder.add_node("propose", propose_code_change)
builder.add_node("run", run_candidate_and_update)

builder.add_edge(START, "propose")
builder.add_edge("propose", "run")

builder.add_conditional_edges(
    "run",
    route_after_run,
    {
        "retry": "propose",
        "done": END,
    }
)

graph = builder.compile()

if __name__ == "__main__":
    target_file = "buggy_script1.py"

    with open(target_file, "r", encoding="utf-8") as f:
        original_code = f.read()

    initial_state = {
        "iteration": 0,
        "max_iterations": 3,
        "target_file_path": target_file,
        "run_command": ["python3", target_file],
        "run_cwd": None,
        "current_working_code": original_code,
        "history": [],
    }

    result = graph.invoke(initial_state)

    if not result.get("run_succeeded", False):
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(original_code)
        print("Repair failed. Original file restored.")
    else:
        print("Repair succeeded.")

    print("\nFINAL STATE:")
    print(result)