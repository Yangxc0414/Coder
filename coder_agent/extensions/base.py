"""Base classes for extensions (MCP tools, Skills, Subagents)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ExtensionResult:
    """Result from an extension tool call.

    Mirrors ToolResult's interface (success / to_message_content) so
    extension tools can flow through the ReAct loop unchanged.
    """
    output: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None

    def to_message_content(self) -> str:
        if self.error:
            return f"(error) {self.error}"
        return self.output or "(no output)"


class ExtensionTool(ABC):
    """Base class for all extension tools (MCP, Skills, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]: ...

    @abstractmethod
    def execute(self, args: dict[str, Any]) -> ExtensionResult: ...

    def to_tool_def(self) -> dict:
        """Convert to OpenAI function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Skill(ExtensionTool):
    """A skill is a reusable prompt + tool combination.

    Skills are like mini-prompts that the model can invoke.
    They can restrict which tools are available and provide
    domain-specific behavior.
    """

    def __init__(
        self,
        name: str,
        description: str,
        command: str,
        when_to_use: str | None = None,
        allowed_tools: tuple[str, ...] = (),
        context: str = "inline",
    ) -> None:
        self._name = name
        self._description = description
        self._command = command
        self._when_to_use = when_to_use
        self._allowed_tools = allowed_tools
        self._context = context

    @property
    def name(self) -> str:
        return f"skill_{self._name}"

    @property
    def description(self) -> str:
        desc = self._description
        if self._when_to_use:
            desc += f"\n\nUse this when: {self._when_to_use}"
        return desc

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Target file, function, or pattern to apply the skill to",
                },
                "context": {
                    "type": "string",
                    "description": "Additional context or instructions",
                },
            },
            "required": ["target"],
        }

    def execute(self, args: dict[str, Any]) -> ExtensionResult:
        """Execute the skill by returning its command for the model to follow."""
        target = args.get("target", "")
        context = args.get("context", "")
        command = self._command.format(target=target, context=context)
        return ExtensionResult(
            output=f"[Skill: {self._name}]\n{command}",
            metadata={"skill_name": self._name, "target": target},
        )

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        return self._allowed_tools

    @property
    def context_mode(self) -> str:
        return self._context


class McpTool(ExtensionTool):
    """A tool discovered from an MCP (Model Context Protocol) server."""

    def __init__(
        self,
        server_name: str,
        name: str,
        description: str,
        parameters: dict[str, Any],
        executor: Callable[[dict[str, Any]], dict[str, str]] | None = None,
    ) -> None:
        self._server_name = server_name
        self._name = name
        self._description = description
        self._parameters = parameters
        self.executor = executor

    @property
    def name(self) -> str:
        return f"mcp_{self._server_name}_{self._name}"

    @property
    def description(self) -> str:
        return f"[MCP: {self._server_name}] {self._description}"

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    def execute(self, args: dict[str, Any]) -> ExtensionResult:
        if self.executor:
            result = self.executor(args)
            return ExtensionResult(
                output=result.get("output", ""),
                error=result.get("error"),
            )
        return ExtensionResult(output=f"Executed {self._name} with args: {args}")


class SubagentDefinition:
    """Definition of a subagent type."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        when_to_use: str,
        tools: tuple[str, ...] = ("*",),
        disallowed_tools: tuple[str, ...] = (),
        max_steps: int | None = None,
        model: str | None = None,
        read_only: bool = False,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.when_to_use = when_to_use
        self.tools = tools
        self.disallowed_tools = disallowed_tools
        self.max_steps = max_steps or 20
        self.model = model
        self.read_only = read_only


@dataclass
class SubagentRequest:
    """Request to spawn a subagent."""
    prompt: str
    subagent_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubagentResult:
    """Result from a subagent execution."""
    subagent_type: str
    final_output: str
    is_error: bool = False
    steps_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class SubagentRunner:
    """Runs subagent requests and returns results."""

    def __init__(self, parent_agent) -> None:
        self._parent = parent_agent
        self._definitions: dict[str, SubagentDefinition] = {}
        self._results: dict[str, SubagentResult] = {}

    def register(self, definition: SubagentDefinition) -> None:
        """Register a subagent type."""
        self._definitions[definition.name] = definition

    def run(self, request: SubagentRequest) -> SubagentResult:
        """Run a subagent request synchronously."""
        defn = self._definitions.get(request.subagent_type)
        if not defn:
            return SubagentResult(
                subagent_type=request.subagent_type,
                final_output=f"Unknown subagent type: {request.subagent_type}",
                is_error=True,
            )

        from coder_agent.agent import Agent
        from coder_agent.llm.client import LLMClient
        from coder_agent.tools.registry import ToolRegistry

        registry = ToolRegistry()
        all_tools = self._parent.registry.list_names()
        if defn.tools != ("*",):
            all_tools = [t for t in all_tools if t in defn.tools]
        if defn.disallowed_tools:
            all_tools = [t for t in all_tools if t not in defn.disallowed_tools]
        # Anti-recursion invariant (structural, from my-pi-agent tasks.py):
        # children can NEVER re-delegate or touch memory — enforced by
        # filtering, not by prompt-level trust.
        for forbidden in ("task", "memory"):
            if forbidden in all_tools:
                all_tools.remove(forbidden)

        for tool_name in all_tools:
            try:
                tool = self._parent.registry.get(tool_name)
                registry.register(tool)
            except KeyError:
                pass

        llm = LLMClient(
            model=defn.model or self._parent.llm.model,
            base_url=getattr(self._parent.llm, "base_url", None),
        )

        # Per-instance max_steps — no global monkey-patching
        subagent = Agent(
            llm_client=llm,
            registry=registry,
            workspace=self._parent.workspace,
            mode=self._parent.mode,
            max_steps=defn.max_steps,
        )
        result = subagent.run(request.prompt)
        return SubagentResult(
            subagent_type=request.subagent_type,
            final_output=result,
            steps_used=subagent.state.step,
            metadata={"workspace": str(self._parent.workspace)},
        )
