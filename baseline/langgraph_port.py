"""Minimal coding agent loop — one tool: bash."""
import json, operator, os, sys, subprocess
from typing import Annotated, Literal, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


LLM_BASE_URL = os.environ["LLM_BASE_URL"]
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ["LLM_MODEL"]
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


llm = ChatOpenAI(model=LLM_MODEL, base_url=LLM_BASE_URL, api_key=LLM_API_KEY,
                 max_tokens=4096, max_retries=0).bind_tools(LLM_TOOLS, tool_choice="auto")


def to_openai_tool_call(tool_call_id: str, name: str, arguments: str) -> dict:
    return {"id": tool_call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def to_usage(usage_metadata: dict) -> dict:
    return {"input": usage_metadata["input_tokens"], "output": usage_metadata["output_tokens"],
            "cached": usage_metadata.get("input_token_details", {}).get("cache_read", 0)}


def call_llm(messages):
    msg = llm.invoke(messages)
    content = msg.text.strip()
    tool_calls = [to_openai_tool_call(tc["id"], tc["name"], json.dumps(tc["args"])) for tc in msg.tool_calls]
    tool_calls += [to_openai_tool_call(tc["id"], tc["name"], tc["args"]) for tc in msg.invalid_tool_calls]
    return content, tool_calls, to_usage(msg.usage_metadata)


class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    turn: int
    content: str
    tool_calls: list[dict]
    usage: Annotated[list[dict], operator.add]
    stop_reason: str


def llm_node(state: AgentState) -> dict:
    turn = state["turn"] + 1
    print(f"\n{'='*60}\n🔄 Turn {turn}\n{'='*60}")
    content, tool_calls, usage = call_llm(state["messages"])
    if content:
        print(f"\n🤖 {content}")
    return {"turn": turn, "content": content, "tool_calls": tool_calls, "usage": [usage]}


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
    return {"stop_reason": "finished"}


def max_turns_node(state: AgentState) -> dict:
    print(f"\n⚠️  Max turns ({MAX_TURNS}) reached. Stopping.")
    return {"stop_reason": "max_turns"}


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


def summarize_usage(state: AgentState) -> dict:
    usage = state["usage"]
    return {"stop_reason": state["stop_reason"], "turns": state["turn"],
            "input_tokens": sum(u["input"] for u in usage), "output_tokens": sum(u["output"] for u in usage),
            "cached_tokens": sum(u["cached"] for u in usage), "peak_context": max(u["input"] for u in usage)}


def agent_loop(user_message: str) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}]
    state = build_graph().invoke(
        {"messages": messages, "turn": 0, "content": "", "tool_calls": [], "usage": [], "stop_reason": ""},
        {"recursion_limit": 2 * MAX_TURNS + 2},
    )
    metrics = summarize_usage(state)
    print(f"\n📊 {json.dumps(metrics)}")
    return metrics


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    if not prompt.strip():
        print("No task provided. Exiting.")
        sys.exit(1)
    agent_loop(prompt)
