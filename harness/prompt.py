from harness.memory import MEMORY_TOOL_SCHEMAS
from harness.registry import BASH_TOOL, META_TOOL_SCHEMAS, SKILLS, TOOLS, load_skill, spec_tool

TOOL_MODES = ("eager", "lazy", "native")

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


def skills_metadata() -> str:
    return "Skills (load_skill(name) before working on a matching task):\n" + "\n".join(
        f"- {name}: {skill.description} Use when: {skill.when_to_use}" for name, skill in SKILLS.skills.items())


def system_prompt(mode: str) -> str:
    if mode == "eager":
        skills = "\n\n".join(load_skill(name) for name in SKILLS.skills)
        return (f"{HEAD}Tools: `bash` plus the catalog tools described in their schemas. {MEMORY_USAGE}\n\n"
                f"Skills:\n\n{skills}\n\n{WORKFLOW}")
    usage = LAZY_USAGE if mode == "lazy" else NATIVE_USAGE
    return f"{HEAD}{CATALOG_HINT} {usage} {MEMORY_USAGE}\n\n{skills_metadata()}\n\n{WORKFLOW}"


def initial_tools(mode: str) -> list[dict]:
    if mode == "eager":
        return [BASH_TOOL, *MEMORY_TOOL_SCHEMAS.values(), *map(spec_tool, TOOLS.specs.values())]
    names = ["discover", "describe", "invoke", "load_skill", "read_skill_resource"]
    if mode == "native":
        names.remove("invoke")
    return [BASH_TOOL, *(META_TOOL_SCHEMAS[name] for name in names), *MEMORY_TOOL_SCHEMAS.values()]
