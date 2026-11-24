"""Minimal coding agent loop — one tool: bash."""
import json, operator, os, sys, subprocess
from typing import Annotated, Literal, TypedDict

import requests
from langgraph.graph import END, START, StateGraph


LLM_BASE_URL = os.environ["LLM_BASE_URL"]
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ["LLM_MODEL"]
LLM_HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {LLM_API_KEY}"}
MAX_TURNS = 1000

SYSTEM_PROMPT = """\
You are a coding agent. Your job is to help the user with programming tasks.

You have access to ONE tool: `bash` — which executes shell commands and returns stdout/stderr.

Workflow:
1. Plan what needs to be done.
2. Use `bash` to read files, run commands, write code, etc.
3. After gathering enough information or completing the task, give your final answer in natural language.
4. To finish, reply with a regular message (no tool call).

Be concise. Explain what you're doing before each command."""

LLM_TOOLS = [
    {"type": "function",
     "function": {"name": "bash",
                  "description": "Execute a shell command and return the output.",
                  "parameters": {"type": "object",
                                 "properties": {
                                     "command": {"type": "string", "description": "The bash command to execute."}
                                 },
                                 "required": ["command"]}
                 }
            }]


def run_bash(command: str) -> str:
    try:
        command_result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        out = command_result.stdout + (f"\nSTDERR:\n{command_result.stderr}" if command_result.stderr else "")
        return f"Exit code: {command_result.returncode}\n{out}"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 120s"


def call_tool(name: str, arguments: dict) -> str:
    func = {"bash": run_bash}.get(name)
    if not func:
        return f"Error: unknown tool '{name}'"
    try:
        return func(**arguments)
    except Exception as e:
        return f"Error calling {name}: {e}"


def call_llm(messages):
    payload = {"model": LLM_MODEL, "messages": messages, "tools": LLM_TOOLS, "tool_choice": "auto",
               "temperature": 0.1, "max_tokens": 4096}
    llm_http_response = requests.post(f"{LLM_BASE_URL}/chat/completions", json=payload, headers=LLM_HEADERS)
    llm_http_response.raise_for_status()
    msg = llm_http_response.json()["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    tool_calls = msg.get("tool_calls") or []
    return content, tool_calls


class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    turn: int
    content: str
    tool_calls: list[dict]


def llm_node(state: AgentState) -> dict:
    turn = state["turn"] + 1
    print(f"\n{'='*60}\n🔄 Turn {turn}\n{'='*60}")
    content, tool_calls = call_llm(state["messages"])
    if content:
        print(f"\n🤖 {content}")
    return {"turn": turn, "content": content, "tool_calls": tool_calls}


def tools_node(state: AgentState) -> dict:
    content = state["content"]
    prefix = "\n" if content else ""
    new_messages = []
    for tool_call in state["tool_calls"]:
        function = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])
        tool_call_id = tool_call["id"]
        print(f"{prefix}🔧 Tool: {function}({json.dumps(arguments, ensure_ascii=False)})")
        result = call_tool(function, arguments)
        print(f"   → {result[:500]}{'...' if len(result)>500 else ''}")
        new_messages.append({"role": "assistant", "content": content or None, "tool_calls": [tool_call]})
        new_messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result})
    return {"messages": new_messages}


def finish_node(state: AgentState) -> dict:
    print("(no text output)" if not state["content"] else "")
    print("✅ Agent finished")
    return {}


def max_turns_node(state: AgentState) -> dict:
    print(f"\n⚠️  Max turns ({MAX_TURNS}) reached. Stopping.")
    return {}


def route_after_llm(state: AgentState) -> Literal["tools", "finish"]:
    return "tools" if state["tool_calls"] else "finish"


def route_after_tools(state: AgentState) -> Literal["llm", "max_turns"]:
    return "max_turns" if state["turn"] >= MAX_TURNS else "llm"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tools_node)
    graph.add_node("finish", finish_node)
    graph.add_node("max_turns", max_turns_node)
    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", route_after_llm)
    graph.add_conditional_edges("tools", route_after_tools)
    graph.add_edge("finish", END)
    graph.add_edge("max_turns", END)
    return graph.compile()


def agent_loop(user_message: str) -> None:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}]
    build_graph().invoke(
        {"messages": messages, "turn": 0, "content": "", "tool_calls": []},
        {"recursion_limit": 2 * MAX_TURNS + 2},
    )


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    if not prompt.strip():
        print("No task provided. Exiting.")
        sys.exit(1)
    agent_loop(prompt)
