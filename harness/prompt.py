from harness.config import HarnessConfig
from harness.memory import MEMORY_TOOL_SCHEMAS
from harness.registry import BASH_TOOL, META_TOOL_SCHEMAS, SKILLS, TOOLS, load_skill, spec_tool

HEAD = "You are a coding agent. Your job is to help the user with programming tasks.\n\n"
WORKFLOW = """\
Workflow:
1. Plan what needs to be done.
2. Use the tools to read files, run commands, write code, etc.
3. After gathering enough information or completing the task, give your final answer in natural language.
4. To finish, reply with a regular message (no tool call).

Be concise. Explain what you're doing before each command."""
CATALOG_HINT = (f"Besides `bash`, a catalog of {len(TOOLS.specs)} tools (files, search, tests, SQLite, JSON, CSV, git, "
                "and external services) is available but not loaded.")
LAZY_USAGE = "Find tools with discover(query), read their parameters with describe(name), call them with invoke(name, arguments)."
NATIVE_USAGE = "Find tools with discover(query); describe(name) makes that tool directly callable from your next turn."
MEMORY_USAGE = ("Long or old tool outputs may be shortened to a reference with an artifact id: read_artifact(id) shows the "
                "original; search_history(query) finds earlier outputs of this session.")


HISTORY_USAGE = "search_history(query) finds earlier outputs of this session."


def skills_metadata() -> str:
    return "Skills (load_skill(name) before working on a matching task):\n" + "\n".join(
        f"- {name}: {skill.description} Use when: {skill.when_to_use}" for name, skill in SKILLS.skills.items())


def full_skills() -> str:
    return "Skills:\n\n" + "\n\n".join(load_skill(name) for name in SKILLS.skills)


def system_prompt(config: HarnessConfig) -> str:
    if not config.lazy_tools:
        tools = "Tools: `bash` plus the catalog tools described in their schemas."
    else:
        tools = f"{CATALOG_HINT} {NATIVE_USAGE if config.native_binding else LAZY_USAGE}"
    memory = MEMORY_USAGE if config.artifact_offloading else HISTORY_USAGE
    skills = skills_metadata() if config.lazy_skills else full_skills()
    return f"{HEAD}{tools} {memory}\n\n{skills}\n\n{WORKFLOW}"


def initial_tools(config: HarnessConfig) -> list[dict]:
    names = []
    if config.lazy_tools:
        names += ["discover", "describe"] + ([] if config.native_binding else ["invoke"])
    if config.lazy_skills:
        names += ["load_skill", "read_skill_resource"]
    memory = (["read_artifact"] if config.artifact_offloading else []) + ["search_history"]
    catalog = [] if config.lazy_tools else list(map(spec_tool, TOOLS.specs.values()))
    return [BASH_TOOL, *(META_TOOL_SCHEMAS[name] for name in names), *(MEMORY_TOOL_SCHEMAS[n] for n in memory), *catalog]
