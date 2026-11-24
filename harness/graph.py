import json, time
from dataclasses import dataclass, field
from typing import Literal, TypedDict

import openai
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from harness.config import HarnessConfig
from harness.context import Candidate, ContextManager, build_prompt
from harness.protocol import protocol_errors
from harness.executor import ToolExecutor, failure
from harness.llm import ContextOverflow, LLMClient
from harness.memory import MEMORY_TOOL_SCHEMAS, read_artifact, search_history
from harness.metrics import MetricsCollector
from harness.output import OutputProcessor, fingerprint
from harness.prompt import initial_tools, system_prompt
from harness.registry import SKILLS, TOOLS, spec_tool
from harness.store import ArtifactStore, EventStore, new_run_dir
from harness.tokens import count_messages

DISCOVERY_TOOLS = {"discover", "describe", "load_skill", "read_skill_resource"}
RECOVERY_TOOLS = {"read_artifact", "search_history"}


class AgentState(TypedDict):
    task: str
    active_messages: list[dict]
    summary: dict
    observations: dict
    active_skills: dict
    active_tools: list[dict]
    completed_actions: list[dict]
    pending_actions: list[dict]
    context_epoch: int
    status: str
    turn: int
    request: list[dict]
    request_tokens: int
    calibration: float
    overflow_attempts: int
    response: dict
    executor: dict


@dataclass
class Run:
    config: HarnessConfig
    events: EventStore
    artifacts: ArtifactStore
    metrics: MetricsCollector = field(default_factory=MetricsCollector)
    llm: LLMClient | None = None
    started: float = field(default_factory=time.monotonic)


def mode_name(config: HarnessConfig) -> str:
    if not config.lazy_tools:
        return "eager"
    return "native" if config.native_binding else "lazy"


def parsed_arguments(raw: str) -> dict:
    try:
        arguments = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return arguments if isinstance(arguments, dict) else {}


def build_request(state: AgentState, config: HarnessConfig) -> list[dict]:
    return build_prompt(system_prompt(mode_name(config)), state["task"], state["summary"], state["active_messages"])


def to_candidate(state: AgentState) -> Candidate:
    return Candidate(state["active_messages"], state["observations"], state["summary"], state["active_skills"],
                     state["context_epoch"], {})


def from_candidate(candidate: Candidate) -> dict:
    return {"active_messages": candidate.messages, "observations": candidate.observations, "summary": candidate.summary,
            "active_skills": candidate.active_skills, "context_epoch": candidate.epoch}


def budget_status(state: AgentState, run: Run) -> str | None:
    counters, config = run.metrics.counters, run.config
    if state["turn"] >= config.max_turns:
        return "max_turns"
    if counters["tool_calls"] >= config.max_tool_calls:
        return "max_tool_calls"
    if counters["input_tokens"] + counters["output_tokens"] >= config.max_total_tokens:
        return "max_tokens"
    if time.monotonic() - run.started >= config.max_seconds:
        return "max_seconds"
    return None


def mark_stale(observations: dict, turn: int, environment_changed: bool) -> dict:
    for observation in observations.values():
        if observation["stale"] or observation["created_at_turn"] == turn:
            continue
        path = observation["resource_path"]
        if path and observation["kind"] == "read" and fingerprint(path) != observation["resource_version"]:
            observation["stale"] = True
        if observation["kind"] == "test" and environment_changed:
            observation["stale"] = True
    return observations


def build_graph(run: Run):
    processor = OutputProcessor(run.config, run.artifacts)

    def context_window_manager(state: AgentState) -> ContextManager:
        summarize = lambda request: run.llm.complete(request, [], purpose="summarizer")[0]  # noqa: E731
        return ContextManager(run.config, run.artifacts, system_prompt(mode_name(run.config)), state["task"],
                              state["active_tools"], summarize, state["completed_actions"])

    def compact(state: AgentState, force: bool) -> tuple[dict, dict]:
        candidate, report = context_window_manager(state).manage(to_candidate(state), state["calibration"], force)
        if report["steps"] or report["error"] not in (None, "insufficient_reclaim"):
            run.events.append("compaction", turn=state["turn"], force=force, **report)
        if report.get("summarizer_error"):
            run.metrics.count("summarizer_failures")
        if candidate is None:
            return {}, report
        run.metrics.count("compactions")
        for strategy, saved in candidate.reclaimed.items():
            run.metrics.count(f"reclaimed_{strategy}", saved)
        print(f"🗜️  Compaction epoch {candidate.epoch}: {report['before']} -> {report['after']} tokens "
              f"({', '.join(step['strategy'] for step in report['steps'])})")
        return from_candidate(candidate), report

    def context_manager(state: AgentState) -> dict:
        update, report = compact(state, force=False)
        if report["error"] == "context_budget_exceeded":
            return {"status": "context_budget_exceeded"}
        return update

    def route_after_context(state: AgentState) -> Literal["prompt_builder", "__end__"]:
        return "prompt_builder" if state["status"] == "running" else END

    def emergency(state: AgentState) -> dict:
        attempts = state["overflow_attempts"] + 1
        run.metrics.count("emergency_compactions")
        if attempts > run.config.emergency_attempts:
            return {"status": "context_budget_exceeded", "overflow_attempts": attempts}
        update, report = compact(state, force=True)
        if not update:
            return {"status": "context_budget_exceeded", "overflow_attempts": attempts}
        return {**update, "status": "running", "overflow_attempts": attempts}

    def prompt_builder(state: AgentState) -> dict:
        request = build_request(state, run.config)
        return {"request": request, "request_tokens": count_messages(request, state["active_tools"])}

    def llm(state: AgentState) -> dict:
        turn = state["turn"] + 1
        try:
            content, tool_calls, usage = run.llm.complete(state["request"], state["active_tools"])
        except ContextOverflow as error:
            run.events.append("context_overflow", turn=turn, error=str(error)[:500])
            return {"status": "context_overflow"}
        except (openai.AuthenticationError, openai.PermissionDeniedError) as error:
            run.events.append("fatal_llm_error", turn=turn, error=str(error)[:500])
            return {"turn": turn, "status": "llm_unauthorized"}
        except openai.APIError as error:
            run.events.append("fatal_llm_error", turn=turn, error=str(error)[:500])
            return {"turn": turn, "status": "llm_unavailable"}
        run.events.append("assistant", turn=turn, content=content, tool_calls=tool_calls, usage=usage)
        calibration = state["calibration"]
        if state["request_tokens"] and usage["input"]:
            calibration = min(max(usage["input"] / state["request_tokens"], 0.8), 1.5)
        print(f"\n{'=' * 60}\n🔄 Turn {turn}\n{'=' * 60}" + (f"\n\n🤖 {content}" if content else ""))
        return {"turn": turn, "response": {"content": content, "tool_calls": tool_calls}, "calibration": calibration,
                "overflow_attempts": 0}

    def route_after_llm(state: AgentState) -> Literal["tool_router", "final_verification", "emergency", "__end__"]:
        if state["status"] == "context_overflow":
            return "emergency"
        if state["status"] != "running":
            return END
        return "tool_router" if state["response"]["tool_calls"] else "final_verification"

    def tool_router(state: AgentState) -> dict:
        pending = [{"id": call["id"], "name": call["function"]["name"], "arguments": call["function"]["arguments"],
                    "parsed_arguments": parsed_arguments(call["function"]["arguments"])}
                   for call in state["response"]["tool_calls"]]
        return {"pending_actions": pending}

    def tool_executor(state: AgentState) -> dict:
        memory_tools = {"read_artifact": (MEMORY_TOOL_SCHEMAS["read_artifact"],
                                          lambda **a: read_artifact(run.artifacts, **a)),
                        "search_history": (MEMORY_TOOL_SCHEMAS["search_history"],
                                           lambda **a: search_history(run.events, run.artifacts, **a))}
        executor = ToolExecutor(available={tool["function"]["name"] for tool in state["active_tools"]},
                                loop_limit=run.config.loop_limit, extra_tools=memory_tools, **state["executor"])
        results = []
        for action in state["pending_actions"]:
            if run.metrics.counters["tool_calls"] >= run.config.max_tool_calls:
                outcome = failure("BUDGET_EXHAUSTED", "The tool-call budget of this run is exhausted; this call was "
                                  "not executed.", model_can_repair=False)
                results.append({**action, **vars(outcome)})
                run.metrics.count("tool_calls_refused")
                continue
            outcome = executor.execute(action["name"], action["arguments"])
            run.metrics.count("tool_calls")
            run.metrics.count(f"tool_code_{outcome.code}")
            if action["name"] in DISCOVERY_TOOLS:
                run.metrics.count("discovery_calls")
            if action["name"] in RECOVERY_TOOLS:
                run.metrics.count("artifact_retrieval_calls")
            print(f"🔧 Tool: {action['name']}({action['arguments'][:300]})\n   → {outcome.content[:500]}"
                  f"{'...' if len(outcome.content) > 500 else ''}")
            results.append({**action, **vars(outcome)})
        executed_state = {"environment_version": executor.environment_version, "failures": executor.failures}
        return {"pending_actions": results, "executor": executed_state}

    def output_processor(state: AgentState) -> dict:
        available = run.config.input_budget - state["request_tokens"]
        processed, observations = processor.process(state["pending_actions"], state["observations"], state["turn"], available)
        for action in processed:
            observations[action["id"]]["environment_version"] = state["executor"]["environment_version"]
            saved = observations[action["id"]]["original_tokens"] - observations[action["id"]]["token_count"]
            if saved > 0:
                run.metrics.count("output_tokens_saved", saved)
        return {"pending_actions": processed, "observations": mark_stale(observations, state["turn"],
                                                                          any(a["changed_environment"] for a in processed))}

    def state_update(state: AgentState) -> dict:
        response, actions = state["response"], state["pending_actions"]
        messages = [*state["active_messages"],
                    {"role": "assistant", "content": response["content"] or None, "tool_calls": response["tool_calls"]}]
        tools, skills, completed = list(state["active_tools"]), dict(state["active_skills"]), list(state["completed_actions"])
        for action in actions:
            messages.append({"role": "tool", "tool_call_id": action["id"], "content": action["visible"]})
            run.events.append("tool_result", turn=state["turn"], tool_call_id=action["id"], name=action["name"],
                              arguments=action["arguments"], code=action["code"], content=action["content"],
                              artifact_id=state["observations"].get(action["id"], {}).get("artifact_id"))
            completed.append({"turn": state["turn"], "tool": action["name"], "code": action["code"]})
            arguments = parsed_arguments(action["arguments"])
            if action["name"] == "load_skill" and action["ok"] and arguments.get("name") in SKILLS.skills:
                skill = SKILLS.skills[arguments["name"]]
                skills[skill.name] = {"skill_id": skill.name, "version": skill.version, "loaded_at_turn": state["turn"],
                                      "visible_in_context": True, "message_tool_call_id": action["id"]}
            if run.config.native_binding and action["name"] == "describe" and action["ok"]:
                name = arguments.get("name")
                if name in TOOLS.specs and all(tool["function"]["name"] != name for tool in tools):
                    tools.append(spec_tool(TOOLS.specs[name]))
        update = {"active_messages": messages, "active_tools": tools, "active_skills": skills,
                  "completed_actions": completed, "pending_actions": []}
        if (status := budget_status(state, run)) is not None:
            run.events.append("budget_exhausted", turn=state["turn"], status=status)
            update["status"] = status
        return update

    def route_after_emergency(state: AgentState) -> Literal["prompt_builder", "__end__"]:
        return "prompt_builder" if state["status"] == "running" else END

    def route_after_update(state: AgentState) -> Literal["context_manager", "__end__"]:
        return "context_manager" if state["status"] == "running" else END

    def final_verification(state: AgentState) -> dict:
        messages = [*state["active_messages"], {"role": "assistant", "content": state["response"]["content"]}]
        errors = protocol_errors(messages)
        run.events.append("final", turn=state["turn"], protocol_errors=errors)
        print("✅ Agent finished" if not errors else f"❌ Protocol errors: {errors}")
        return {"active_messages": messages, "status": "finished" if not errors else "protocol_error"}

    graph = StateGraph(AgentState)
    for name, node in [("context_manager", context_manager), ("prompt_builder", prompt_builder), ("llm", llm),
                       ("tool_router", tool_router), ("tool_executor", tool_executor),
                       ("output_processor", output_processor), ("state_update", state_update),
                       ("final_verification", final_verification), ("emergency", emergency)]:
        graph.add_node(name, node)
    graph.add_edge(START, "context_manager")
    graph.add_conditional_edges("context_manager", route_after_context)
    graph.add_conditional_edges("emergency", route_after_emergency)
    graph.add_edge("prompt_builder", "llm")
    graph.add_conditional_edges("llm", route_after_llm)
    graph.add_edge("tool_router", "tool_executor")
    graph.add_edge("tool_executor", "output_processor")
    graph.add_edge("output_processor", "state_update")
    graph.add_conditional_edges("state_update", route_after_update)
    graph.add_edge("final_verification", END)
    return graph.compile(checkpointer=InMemorySaver())


def initial_state(task: str, config: HarnessConfig) -> AgentState:
    return AgentState(task=task, active_messages=[], summary={}, observations={}, active_skills={},
                      active_tools=initial_tools(mode_name(config)), completed_actions=[], pending_actions=[],
                      context_epoch=0, status="running", turn=0, request=[], request_tokens=0, calibration=1.0,
                      overflow_attempts=0, response={},
                      executor={"environment_version": 0, "failures": {}})


def run_agent(task: str, config: HarnessConfig, llm: LLMClient | None = None) -> dict:
    run_dir = new_run_dir()
    run = Run(config=config, events=EventStore(run_dir), artifacts=ArtifactStore(run_dir))
    run.llm = llm or LLMClient(run.metrics, config.llm_retries, config.llm_deadline_seconds)
    run.llm.metrics = run.metrics
    run.events.append("task", task=task)
    state = build_graph(run).invoke(initial_state(task, config),
                                    {"recursion_limit": 8 * config.max_turns + 20, "configurable": {"thread_id": "run"}})
    metrics = {"stop_reason": state["status"], "turns": state["turn"], "context_epochs": state["context_epoch"],
               **run.metrics.snapshot(),
               "events_path": str(run.events.path)}
    print(f"\n📊 {json.dumps(metrics)}")
    return metrics
