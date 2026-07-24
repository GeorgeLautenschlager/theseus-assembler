"""agent-assembler — assemble a Theseus agent from a Markdown manifest.

Reads a manifest (see `theseus/agent_manifests/alty_mcgee.md`) and stands up a running
agent equivalent to the hand-wired reference `AltyMcGee`: an `OODACore` with model
providers, tools, an `AgenticMemory`, and observers — all named by the manifest.

The manifest is intentionally human-readable Markdown. Sections:

    # <Agent Name>
    ## Core Cognition
    Core: OODA
    ContextAssembler: MonoMemory
    ### Memory
    | Type | Role | Inference |
    | ---  | ---  | --- |
    | AgenticMemory | episodic | primary: Ollama(gemma4:e4b) <br> embeddings: Ollama(nomic-embed-text) |
    ## Interfaces
    Tools: list, read, edit, write, terminal_chat
    Observers: terminal_chat
    ## Inference
    primary: Ollama(gemma4:e4b)
    backups: LmStudio("gemma-4-e4b-it-qat-nvfp4")
    ## Background
    <constitution text>
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from theseus.agentic_memory import AgenticMemory
from theseus.chat_observer import TerminalChatObserver
from theseus.memory_store import MemoryStore
from theseus.model_providers.claude_provider import ClaudeProvider
from theseus.model_providers.lm_studio_provider import LmStudioProvider
from theseus.model_providers.model_provider import ModelProvider
from theseus.model_providers.ollama_provider import OllamaProvider
from theseus.ooda_core import OODACore
from theseus.stimulus_log import StimulusLog
from theseus.tools.registry import all_tools
from theseus.tools.terminal_chat import TerminalChat

# --- vocabulary maps ----------------------------------------------------------

_PROVIDERS: dict[str, type[ModelProvider]] = {
    "Ollama": OllamaProvider,
    "LmStudio": LmStudioProvider,
    "Claude": ClaudeProvider,
}
_CORES = {"OODA": OODACore}
_MEMORIES = {"AgenticMemory": AgenticMemory}
# The manifest's ContextAssembler value; OODACore builds MonoMemory itself, so this is
# validated, not constructed.
_CONTEXT_ASSEMBLERS = {"MonoMemory"}
# Friendly manifest tool names that differ from the registry key.
_TOOL_ALIASES = {"list": "ls"}


# --- parsed manifest ----------------------------------------------------------

@dataclass
class Manifest:
    name: str
    core: str
    context_assembler: str
    memory_type: str
    memory_role: str
    memory_inference: dict[str, str]      # {"primary": "Ollama(...)", "embeddings": "Ollama(...)"}
    tools: list[str]
    observers: list[str]
    inference_primary: str
    inference_backups: list[str]
    background: str = ""


def _provider(spec: str) -> ModelProvider:
    """`Ollama(gemma4:e4b)` / `LmStudio("gemma-4-…")` -> a ModelProvider instance."""
    match = re.fullmatch(r'\s*(\w+)\(\s*"?(.+?)"?\s*\)\s*', spec)
    if not match:
        raise ValueError(f"Cannot parse inference spec: {spec!r} (expected Name(model))")
    name, model = match.group(1), match.group(2)
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown provider {name!r} in {spec!r}; known: {sorted(_PROVIDERS)}")
    return _PROVIDERS[name](model=model)


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_manifest(text: str) -> Manifest:
    """Parse the Markdown manifest into a `Manifest`. Section-aware so that `primary:`
    inside the Memory table and inside `## Inference` don't collide."""
    title: str | None = None
    section = ""                       # current "## ..." (lowercased)
    kv: dict[str, dict[str, str]] = {}  # section -> {key: value}
    memory_row: list[str] | None = None
    background: list[str] = []

    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("# ") and title is None:
            title = line[2:].strip()
            continue
        header = re.match(r"^(#{2,})\s+(.*)$", line)
        if header:
            section = header.group(2).strip().lower()
            continue
        if section == "background":
            background.append(raw)
            continue
        if line.lstrip().startswith("|"):   # a Markdown table row
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            # skip header (`Type`) and separator (`---`) rows; keep the data row
            if cells and cells[0] not in ("Type", "") and not cells[0].startswith("-"):
                memory_row = cells
            continue
        m = re.match(r"^\s*([A-Za-z][\w ]*?)\s*:\s*(.+?)\s*$", line)
        if m:
            kv.setdefault(section, {})[m.group(1).strip().lower()] = m.group(2).strip()

    if title is None:
        raise ValueError("Manifest has no `# <name>` title")

    core_cog = kv.get("core cognition", {})
    interfaces = kv.get("interfaces", {})
    inference = kv.get("inference", {})

    if memory_row is None or len(memory_row) < 3:
        raise ValueError("Manifest `### Memory` table is missing its data row")
    memory_inference = {}
    for part in memory_row[2].split("<br>"):
        if ":" in part:
            k, v = part.split(":", 1)
            memory_inference[k.strip().lower()] = v.strip()

    return Manifest(
        name=title,
        core=_require(core_cog, "core", "## Core Cognition"),
        context_assembler=_require(core_cog, "contextassembler", "## Core Cognition"),
        memory_type=memory_row[0],
        memory_role=memory_row[1],
        memory_inference=memory_inference,
        tools=_split_list(_require(interfaces, "tools", "## Interfaces")),
        observers=_split_list(_require(interfaces, "observers", "## Interfaces")),
        inference_primary=_require(inference, "primary", "## Inference"),
        inference_backups=_split_list(inference.get("backups", "")),
        background="\n".join(background).strip(),
    )


def _require(section_kv: dict[str, str], key: str, where: str) -> str:
    if key not in section_kv:
        raise ValueError(f"Manifest is missing `{key}` under {where}")
    return section_kv[key]


def build_tools(names: list[str]) -> tuple[dict, TerminalChat | None]:
    """Map manifest tool names to tool instances. Returns (tools_by_name, terminal_chat)."""
    registry = all_tools()
    tools: dict = {}
    terminal_chat: TerminalChat | None = None
    for raw in names:
        key = _TOOL_ALIASES.get(raw, raw)
        if key == "terminal_chat":
            terminal_chat = TerminalChat()
            tools[terminal_chat.name] = terminal_chat
        elif key in registry:
            tools[key] = registry[key]
        else:
            raise ValueError(f"Unknown tool {raw!r}; known: {sorted(registry) + ['terminal_chat']}")
    return tools, terminal_chat


class AssembledAgent:
    """A running agent built from a manifest — same shape as the reference `AltyMcGee`
    (`.core`, `.stimulus_log`, `.terminal_chat`, `.memory`, `.run()`) so it is a drop-in
    for the reference agent in tests and scripts."""

    def __init__(
        self,
        manifest: Manifest,
        *,
        stimulus_log: StimulusLog | None = None,
        memory_store: MemoryStore | None = None,
    ):
        if manifest.core not in _CORES:
            raise ValueError(f"Unknown core {manifest.core!r}; known: {sorted(_CORES)}")
        if manifest.context_assembler not in _CONTEXT_ASSEMBLERS:
            raise ValueError(
                f"Unknown context assembler {manifest.context_assembler!r}; "
                f"known: {sorted(_CONTEXT_ASSEMBLERS)}"
            )
        if manifest.memory_type not in _MEMORIES:
            raise ValueError(f"Unknown memory {manifest.memory_type!r}; known: {sorted(_MEMORIES)}")

        # `is not None`, not `or`: an empty MemoryStore is falsy (it defines __len__).
        if stimulus_log is None:
            stimulus_log = StimulusLog(f"{_slug(manifest.name)}_stimulus_log.jsonl")
        if memory_store is None:
            memory_store = MemoryStore(f"{_slug(manifest.name)}_a_mem.jsonl")
        self.stimulus_log = stimulus_log

        self.memory = AgenticMemory(
            model_providers=[_provider(manifest.memory_inference["primary"])],
            embedding_providers=[_provider(manifest.memory_inference["embeddings"])],
            store=memory_store,
            stimulus_log=self.stimulus_log,
        )

        tools, self.terminal_chat = build_tools(manifest.tools)

        providers = [_provider(manifest.inference_primary)]
        providers += [_provider(b) for b in manifest.inference_backups]

        self.core = _CORES[manifest.core](
            constitution=manifest.background,
            model_providers=providers,
            tools=tools,
            stimulus_log=self.stimulus_log,
            memory=self.memory,
            name=manifest.name,
        )

        self.observers = self._build_observers(manifest.observers)

    def _build_observers(self, names: list[str]) -> list:
        observers = []
        for name in names:
            if name == "terminal_chat":
                observers.append(
                    TerminalChatObserver(
                        stimulus_log=self.stimulus_log,
                        orient_chat_message_callback=self.core.orient,
                    )
                )
            else:
                raise ValueError(f"Unknown observer {name!r}; known: ['terminal_chat']")
        return observers

    def run(self) -> None:
        """Run the agent's observe loop (blocks, like AltyMcGee.run())."""
        if not self.observers:
            raise RuntimeError("No observers to run")
        while True:
            for observer in self.observers:
                observer.observe_chat_message()


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "agent"


def assemble(
    manifest_path: str | Path,
    *,
    stimulus_log: StimulusLog | None = None,
    memory_store: MemoryStore | None = None,
) -> AssembledAgent:
    """Read and parse a manifest file, then build the agent."""
    text = Path(manifest_path).read_text(encoding="utf-8")
    return AssembledAgent(
        parse_manifest(text), stimulus_log=stimulus_log, memory_store=memory_store
    )


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: agent-assembler <manifest.md>", file=sys.stderr)
        raise SystemExit(2)
    assemble(sys.argv[1]).run()


if __name__ == "__main__":
    main()
