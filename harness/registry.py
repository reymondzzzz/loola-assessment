import json, math, re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from harness.catalog import CATALOG, ToolSpec, schema, string

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, documents: dict[str, str]):
        self.documents = {key: tokenize(text) for key, text in documents.items()}
        self.frequency = Counter(token for tokens in self.documents.values() for token in set(tokens))
        self.average_length = sum(map(len, self.documents.values())) / max(len(self.documents), 1)

    def search(self, query: str, top_k: int) -> list[str]:
        query_tokens = set(tokenize(query))
        total = len(self.documents)
        scored = []
        for key, tokens in self.documents.items():
            counts = Counter(tokens)
            score = 0.0
            for token in query_tokens & counts.keys():
                idf = math.log(1 + (total - self.frequency[token] + 0.5) / (self.frequency[token] + 0.5))
                score += idf * counts[token] * 2.2 / (counts[token] + 1.2 * (0.25 + 0.75 * len(tokens) / self.average_length))
            if score > 0:
                scored.append((score, key))
        return [key for _, key in sorted(scored, reverse=True)[:top_k]]


def openai_tool(name: str, description: str, parameters: dict) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


def spec_tool(spec: ToolSpec) -> dict:
    return openai_tool(spec.name, spec.description, spec.parameters)


def validation_error(arguments: dict, parameters: dict) -> str | None:
    properties = parameters["properties"]
    missing = [key for key in parameters["required"] if key not in arguments]
    unknown = [key for key in arguments if key not in properties]
    wrong_type = [key for key, value in arguments.items()
                  if key in properties and "type" in properties[key]
                  and (not isinstance(value, JSON_TYPES[properties[key]["type"]])
                       or (isinstance(value, bool) and properties[key]["type"] in ("integer", "number")))]
    problems = [f"missing required {missing}" if missing else "", f"unknown {unknown}" if unknown else "",
                f"wrong type for {wrong_type}" if wrong_type else ""]
    return "; ".join(filter(None, problems)) or None


class ToolRegistry:
    def __init__(self, specs: dict[str, ToolSpec]):
        self.specs = specs
        self.index = BM25({name: f"{name} {spec.summary} {spec.description} {' '.join(spec.keywords)}"
                           for name, spec in specs.items()})

    def discover(self, query: str, top_k: int = 5) -> list[str]:
        return self.index.search(query, top_k)

    def describe(self, name: str) -> dict:
        return spec_tool(self.specs[name])["function"]


@dataclass(frozen=True)
class Skill:
    name: str
    version: str
    description: str
    when_to_use: str
    directory: Path


def read_skill(directory: Path) -> Skill:
    front_matter = (directory / "SKILL.md").read_text().split("---")[1]
    meta = dict(line.split(": ", 1) for line in front_matter.strip().splitlines())
    return Skill(meta["name"], meta["version"], meta["description"], meta["when_to_use"], directory)


class SkillRegistry:
    def __init__(self, root: Path):
        self.skills = {path.name: read_skill(path) for path in sorted(root.iterdir()) if (path / "SKILL.md").exists()}
        self.index = BM25({name: f"{name} {skill.description} {skill.when_to_use}" for name, skill in self.skills.items()})

    def discover(self, query: str, top_k: int = 3) -> list[str]:
        return self.index.search(query, top_k)

    def references(self, name: str) -> list[str]:
        directory = self.skills[name].directory
        return sorted(str(path.relative_to(directory)) for path in directory.rglob("*")
                      if path.is_file() and path.name != "SKILL.md")

    def instructions(self, name: str) -> str:
        return (self.skills[name].directory / "SKILL.md").read_text()

    def resource(self, name: str, path: str) -> str | None:
        directory = self.skills[name].directory.resolve()
        target = (directory / path).resolve()
        return target.read_text() if target.is_relative_to(directory) and target.is_file() else None


TOOLS = ToolRegistry(CATALOG)
SKILLS = SkillRegistry(SKILLS_DIR)

BASH_TOOL = openai_tool("bash", "Execute a shell command and return the output.",
                        schema({"command": string("The bash command to execute.")}, ["command"]))

META_TOOL_SCHEMAS = {
    "discover": openai_tool(
        "discover", "Search the tool or skill catalog by keywords. Returns a few matching names with one-line summaries.",
        schema({"query": string("What you want to do, in a few words."),
                "kind": {"type": "string", "enum": ["tool", "skill"], "description": "Catalog to search. Default 'tool'."}},
               ["query"])),
    "describe": openai_tool("describe", "Get the full description and JSON parameters of a catalog tool.",
                            schema({"name": string("Tool name returned by discover.")}, ["name"])),
    "invoke": openai_tool("invoke", "Call a catalog tool by name with arguments matching its parameters.",
                          schema({"name": string("Tool name."),
                                  "arguments": {"type": "object", "description": "Arguments object for the tool."}},
                                 ["name", "arguments"])),
    "load_skill": openai_tool("load_skill", "Load a skill's instructions (SKILL.md) by name.",
                              schema({"name": string("Skill name.")}, ["name"])),
    "read_skill_resource": openai_tool("read_skill_resource", "Read a reference file of a skill.",
                                       schema({"name": string("Skill name."),
                                               "path": string("Reference path listed by load_skill.")}, ["name", "path"])),
}


def discover(query: str, kind: str = "tool") -> str:
    if kind == "skill":
        hits = SKILLS.discover(query)
        lines = [f"{name} — {SKILLS.skills[name].description}" for name in hits]
        follow_up = "Call load_skill(name) to read one."
    else:
        hits = TOOLS.discover(query)
        lines = [f"{name} — {TOOLS.specs[name].summary}" for name in hits]
        follow_up = "Call describe(name) for parameters, then invoke(name, arguments)."
    return "\n".join(lines + [follow_up]) if hits else f"No matching {kind}s. Try other keywords or bash."


def describe(name: str) -> str:
    if name not in TOOLS.specs:
        return f"Error: unknown tool '{name}'. Use discover(query) to find tools."
    return json.dumps(TOOLS.describe(name), indent=2)


def run_catalog_tool(name: str, arguments: dict) -> str:
    spec = TOOLS.specs.get(name)
    if spec is None:
        return f"Error: unknown tool '{name}'. Use discover(query) to find tools."
    if (problem := validation_error(arguments, spec.parameters)) is not None:
        return f"Error: invalid arguments for {name}: {problem}. Schema: {json.dumps(spec.parameters)}"
    if spec.handler is None:
        return f"Error: {name} is not available in this environment. Use another tool or bash."
    return spec.handler(**arguments)


def invoke(name: str, arguments: dict | None = None) -> str:
    return run_catalog_tool(name, arguments or {})


def load_skill(name: str) -> str:
    if name not in SKILLS.skills:
        return f"Error: unknown skill '{name}'. Use discover(query, kind='skill')."
    return SKILLS.instructions(name) + f"\nReferences: {', '.join(SKILLS.references(name)) or 'none'}"


def read_skill_resource(name: str, path: str) -> str:
    content = SKILLS.resource(name, path) if name in SKILLS.skills else None
    if content is None:
        return f"Error: no reference '{path}' in skill '{name}'. load_skill(name) lists the references."
    return content


META_HANDLERS = {"discover": discover, "describe": describe, "invoke": invoke,
                 "load_skill": load_skill, "read_skill_resource": read_skill_resource}
