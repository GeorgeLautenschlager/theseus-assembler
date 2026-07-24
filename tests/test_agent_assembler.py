from __future__ import annotations

import textwrap

import pytest

from agent_assembler import (
    AssembledAgent,
    Manifest,
    _provider,
    assemble,
    build_tools,
    parse_manifest,
)
from theseus.model_providers.lm_studio_provider import LmStudioProvider
from theseus.model_providers.ollama_provider import OllamaProvider
from theseus.ooda_core import OODACore
from theseus.tools.terminal_chat import TerminalChat

# A self-contained manifest fixture in the canonical format (mirrors
# theseus/agent_manifests/alty_mcgee.md, but with the Background filled).
MANIFEST = textwrap.dedent(
    """\
    # Alty McGee

    ## Core Cognition

    Core: OODA

    ContextAssembler: MonoMemory

    ### Memory
    | Type | Role | Inference |
    | --- | --- | --- |
    | AgenticMemory | episodic | primary: Ollama(gemma4:e4b) <br> embeddings: Ollama(nomic-embed-text) |

    ## Interfaces
    Tools: list, read, edit, write, terminal_chat
    Observers: terminal_chat

    ## Inference
    primary: Ollama(gemma4:e4b)
    backups: LmStudio("gemma-4-e4b-it-qat-nvfp4")

    ## Background
    You are the crash-test dummy of Theseus Agents.
    """
)


class TestProviderSpec:
    def test_ollama_unquoted(self):
        p = _provider("Ollama(gemma4:e4b)")
        assert isinstance(p, OllamaProvider)
        assert p.model == "gemma4:e4b"

    def test_lmstudio_quoted(self):
        p = _provider('LmStudio("gemma-4-e4b-it-qat-nvfp4")')
        assert isinstance(p, LmStudioProvider)
        assert p.model == "gemma-4-e4b-it-qat-nvfp4"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            _provider("Groq(llama)")


class TestParseManifest:
    def setup_method(self):
        self.m = parse_manifest(MANIFEST)

    def test_name_and_core(self):
        assert self.m.name == "Alty McGee"
        assert self.m.core == "OODA"
        assert self.m.context_assembler == "MonoMemory"

    def test_memory(self):
        assert self.m.memory_type == "AgenticMemory"
        assert self.m.memory_role == "episodic"
        assert self.m.memory_inference == {
            "primary": "Ollama(gemma4:e4b)",
            "embeddings": "Ollama(nomic-embed-text)",
        }

    def test_interfaces(self):
        assert self.m.tools == ["list", "read", "edit", "write", "terminal_chat"]
        assert self.m.observers == ["terminal_chat"]

    def test_inference(self):
        assert self.m.inference_primary == "Ollama(gemma4:e4b)"
        assert self.m.inference_backups == ['LmStudio("gemma-4-e4b-it-qat-nvfp4")']

    def test_background(self):
        assert "crash-test dummy" in self.m.background

    def test_missing_section_raises(self):
        with pytest.raises(ValueError, match="title"):
            parse_manifest("## Core Cognition\nCore: OODA")


class TestBuildTools:
    def test_maps_names_and_aliases_list_to_ls(self):
        tools, terminal_chat = build_tools(["list", "read", "terminal_chat"])
        assert set(tools) == {"ls", "read", "terminal_chat"}
        assert isinstance(terminal_chat, TerminalChat)
        assert tools["terminal_chat"] is terminal_chat

    def test_no_terminal_chat_returns_none(self):
        tools, terminal_chat = build_tools(["read", "write"])
        assert terminal_chat is None
        assert set(tools) == {"read", "write"}

    def test_unknown_tool_raises(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            build_tools(["telepathy"])


class TestAssemble:
    def _agent(self, tmp_path) -> AssembledAgent:
        from theseus.memory_store import MemoryStore
        from theseus.stimulus_log import StimulusLog

        return AssembledAgent(
            parse_manifest(MANIFEST),
            stimulus_log=StimulusLog(path=tmp_path / "log.jsonl"),
            memory_store=MemoryStore(tmp_path / "a_mem.jsonl"),
        )

    def test_builds_ooda_core_named_from_manifest(self, tmp_path):
        agent = self._agent(tmp_path)
        assert isinstance(agent.core, OODACore)
        assert agent.core.name == "Alty McGee"
        assert agent.core.constitution.strip().startswith("You are the crash-test dummy")

    def test_core_provider_order_primary_then_backups(self, tmp_path):
        agent = self._agent(tmp_path)
        order = [type(p).__name__ for p in agent.core.model_providers]
        assert order == ["OllamaProvider", "LmStudioProvider"]

    def test_tools_and_terminal_chat_wired(self, tmp_path):
        agent = self._agent(tmp_path)
        assert set(agent.core.tools) == {"ls", "read", "edit", "write", "terminal_chat"}
        assert isinstance(agent.terminal_chat, TerminalChat)

    def test_memory_and_observer_wired(self, tmp_path):
        agent = self._agent(tmp_path)
        assert agent.core.memory is agent.memory
        assert type(agent.memory.model_providers[0]).__name__ == "OllamaProvider"
        assert type(agent.memory.embedding_providers[0]).__name__ == "OllamaProvider"
        assert len(agent.observers) == 1
        # The observer drives the same core.
        assert agent.observers[0].orient_chat_message_callback == agent.core.orient

    def test_isolated_store_not_default(self, tmp_path):
        agent = self._agent(tmp_path)
        assert str(agent.memory.store.path).startswith(str(tmp_path))

    def test_assemble_from_file(self, tmp_path):
        from theseus.memory_store import MemoryStore
        from theseus.stimulus_log import StimulusLog

        path = tmp_path / "alty.md"
        path.write_text(MANIFEST, encoding="utf-8")
        agent = assemble(
            path,
            stimulus_log=StimulusLog(path=tmp_path / "log.jsonl"),
            memory_store=MemoryStore(tmp_path / "a_mem.jsonl"),
        )
        assert isinstance(agent.core, OODACore)
