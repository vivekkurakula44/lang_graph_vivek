import os
import sys
import io
import traceback
from typing import TypedDict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

import google.generativeai as genai

# ==========================================
# 1. LLM INITIALIZATION
# ==========================================
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("Error: GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=api_key)
print("API Key configured successfully.")

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key,
)
llm = llm_flash


# ==========================================
# 2. STATE DEFINITION
# ==========================================
class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]


# ==========================================
# 3. TOOLS
# ==========================================
@tool
def run_python_code(code: str) -> str:
    """Execute python code and return the standard output or error trace."""
    if not isinstance(code, str):
        code = str(code)
    clean_code = code.replace("```python", "").replace("```", "").strip()

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        local_scope = {}
        exec(clean_code, {}, local_scope)
        result = new_stdout.getvalue()
    except Exception:
        result = f"Execution Error:\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout

    return result.strip() if result.strip() else "Success (no terminal output)"


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate specific test scenarios for a given coding task."""
    prompt = (
        f"You are a Senior QA Engineer. Generate 3 to 5 highly specific test scenarios "
        f"for the following coding task: '{task_description}'.\n"
        f"Include standard cases and edge cases. Return them as a numbered list."
    )
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


# ==========================================
# 4. GRAPH NODES
# (task_input_node and manager_decision_node no longer use input();
#  the task is now passed in via the API request, and the graph
#  runs straight through to the report — no interactive loop.)
# ==========================================
def real_time_developer(state: CrewState):
    print("\n[Developer] Writing dynamic code using LLM...")
    task = state["messages"][-1].content
    dev_prompt = (
        f"Write a clean Python script to solve this: {task}. "
        f"Only return the code, no explanation or markdown formatting."
    )

    response = llm_flash.invoke(dev_prompt)
    content = response.content
    if isinstance(content, list):
        code_str = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
    else:
        code_str = str(content)

    print(code_str)
    return {"code": code_str}


def real_time_tester(state: CrewState):
    print("\n[Tester] Generating dynamic tests and executing code...")
    task = state["messages"][-1].content

    test_cases = generate_test_cases.invoke(task)
    content = test_cases
    if isinstance(content, list):
        cases_str = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
    else:
        cases_str = str(content)

    execution_result = run_python_code.invoke({"code": state["code"]})

    report = (
        f"### EXECUTION OUTPUT:\n{execution_result}\n\n"
        f"### TEST SCENARIOS EVALUATED:\n{cases_str}"
    )
    return {"report": report}


# ==========================================
# 5. GRAPH CONSTRUCTION
# (Simplified: task -> developer -> tester -> END.
#  The manager/archiver "store or retry" loop depended on input(),
#  so it's removed here. Each API call runs exactly one task through
#  once and returns the report. If you want the store/retry behavior
#  back, it should become separate API endpoints instead of input().)
# ==========================================
rt_workflow = StateGraph(CrewState)

rt_workflow.add_node("developer", real_time_developer)
rt_workflow.add_node("tester", real_time_tester)

rt_workflow.add_edge(START, "developer")
rt_workflow.add_edge("developer", "tester")
rt_workflow.add_edge("tester", END)

rt_app = rt_workflow.compile()
print("Pipeline compiled and ready.")


# ==========================================
# 6. FASTAPI APP
# ==========================================
app = FastAPI(title="Agentic Coding Crew API")


class TaskRequest(BaseModel):
    task: str


class TaskResponse(BaseModel):
    code: str
    report: str


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Agentic coding crew is running."}


@app.post("/run-task", response_model=TaskResponse)
def run_task(request: TaskRequest):
    initial_state = {
        "messages": [HumanMessage(content=request.task)],
        "next_step": None,
        "code": None,
        "report": None,
    }
    result = rt_app.invoke(initial_state, config={"recursion_limit": 50})
    return TaskResponse(code=result.get("code", ""), report=result.get("report", ""))
