"""Coding agent harness with dynamic tool and skill loading (modes: eager | lazy | native)."""
import json, operator, os, sys, subprocess
from typing import Annotated, Literal, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from harness.prompt import TOOL_MODES, initial_tools, system_prompt
from harness.registry import META_HANDLERS, TOOLS, run_catalog_tool, spec_tool


LLM_BASE_URL = os.environ["LLM_BASE_URL"]
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ["LLM_MODEL"]
MAX_TURNS = 1000

def run_bash(command: str) -> str:
    try:
        command_result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        out = command_result.stdout + (f"\nSTDERR:\n{command_result.stderr}" if command_result.stderr else "")
        return f"Exit code: {command_result.returncode}\n{out}"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 120s"


def call_tool(name: str, arguments: dict, available: set[str]) -> str:
    func = run_bash if name == "bash" else META_HANDLERS.get(name) or catalog_handler(name)
    if not func or name not in available:
        return f"Error: unknown tool '{name}'"
    try:
        return func(**arguments)
    except Exception as e:
        return f"Error calling {name}: {e}"


def catalog_handler(name: str):
    if name not in TOOLS.specs:
        return None
    return lambda **arguments: run_catalog_tool(name, arguments)


llm = ChatOpenAI(model=LLM_MODEL, base_url=LLM_BASE_URL, api_key=LLM_API_KEY,
                 max_tokens=4096, max_retries=0)


def to_openai_tool_call(tool_call_id: str, name: str, arguments: str) -> dict:
    return {"id": tool_call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def to_usage(usage_metadata: dict) -> dict:
    return {"input": usage_metadata["input_tokens"], "output": usage_metadata["output_tokens"],
            "cached": usage_metadata.get("input_token_details", {}).get("cache_read", 0)}


def call_llm(messages, tools):
    msg = llm.bind_tools(tools, tool_choice="auto").invoke(messages)
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
    tool_mode: str
    tools: list[dict]


def llm_node(state: AgentState) -> dict:
    turn = state["turn"] + 1
    print(f"\n{'='*60}\n🔄 Turn {turn}\n{'='*60}")
    content, tool_calls, usage = call_llm(state["messages"], state["tools"])
    if content:
        print(f"\n🤖 {content}")
    return {"turn": turn, "content": content, "tool_calls": tool_calls, "usage": [usage]}


def tools_node(state: AgentState) -> dict:
    content = state["content"]
    prefix = "\n" if content else ""
    new_messages = []
    tools = state["tools"]
    for tool_call in state["tool_calls"]:
        function = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])
        tool_call_id = tool_call["id"]
        print(f"{prefix}🔧 Tool: {function}({json.dumps(arguments, ensure_ascii=False)})")
        result = call_tool(function, arguments, {tool["function"]["name"] for tool in tools})
        if state["tool_mode"] == "native" and function == "describe":
            tools = bind_described_tool(tools, arguments.get("name"))
        print(f"   → {result[:500]}{'...' if len(result)>500 else ''}")
        new_messages.append({"role": "assistant", "content": content or None, "tool_calls": [tool_call]})
        new_messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result})
    return {"messages": new_messages, "tools": tools}


def bind_described_tool(tools: list[dict], name: str | None) -> list[dict]:
    if name not in TOOLS.specs or any(tool["function"]["name"] == name for tool in tools):
        return tools
    return [*tools, spec_tool(TOOLS.specs[name])]


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


def agent_loop(user_message: str, tool_mode: str = "lazy") -> dict:
    assert tool_mode in TOOL_MODES, tool_mode
    messages = [{"role": "system", "content": system_prompt(tool_mode)}, {"role": "user", "content": user_message}]
    state = build_graph().invoke(
        {"messages": messages, "turn": 0, "content": "", "tool_calls": [], "usage": [], "stop_reason": "",
         "tool_mode": tool_mode, "tools": initial_tools(tool_mode)},
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
    agent_loop(prompt, os.environ.get("TOOL_MODE", "lazy"))
