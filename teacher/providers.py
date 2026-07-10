"""Pluggable teacher-model clients for trajectory generation.

SparkDistill Phase 1 supports two teachers only:
- Anthropic Claude Fable 5 (`claude-fable-5`)
- OpenAI GPT 5.6 (`gpt-5.6`)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

ANTHROPIC_TEACHER_MODEL = "claude-fable-5"
OPENAI_TEACHER_MODEL = "gpt-5.6"
_ALLOWED_OPENAI_MODELS = frozenset({OPENAI_TEACHER_MODEL, "gpt-5.6-sol"})
_SUPPORTED_PROVIDERS = frozenset({"anthropic", "openai"})


@dataclass(frozen=True)
class Trajectory:
    """A single prompt/response pair captured from a teacher model.

    `reasoning` is the teacher's captured chain-of-thought/thinking trace, kept
    separate from the final `response` — SparkDistill trains students to reproduce
    reasoning, not just answers, so the raw trajectory must preserve that distinction
    even before any training-format decision is made (see `teacher/format.py`).
    `reasoning` is `None` when a teacher provides no capturable trace (e.g. GPT 5.6
    over chat-completions, which may not expose reasoning tokens).
    """

    prompt: str
    response: str
    provider: str
    model: str
    system: str | None = None
    reasoning: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "provider": self.provider,
            "model": self.model,
            "system": self.system,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
        }


class TeacherClient(Protocol):
    """Anything that can turn a prompt into a captured `Trajectory`."""

    name: str
    model: str

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        thinking_budget: int | None = None,
    ) -> Trajectory: ...


class AnthropicTeacher:
    """Teacher backed by the Anthropic API (Claude Fable 5 only)."""

    name = "anthropic"

    def __init__(self, model: str = ANTHROPIC_TEACHER_MODEL, api_key: str | None = None) -> None:
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        thinking_budget: int | None = None,
    ) -> Trajectory:
        kwargs: dict[str, Any] = {}
        if system is not None:
            kwargs["system"] = system
        if thinking_budget is not None:
            # Anthropic requires max_tokens > thinking.budget_tokens, and extended
            # thinking is incompatible with a fixed sampling temperature.
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
            max_tokens = max(max_tokens, thinking_budget + 1024)
            temperature = 1.0
        message = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        reasoning = "".join(block.thinking for block in message.content if block.type == "thinking") or None
        response = "".join(block.text for block in message.content if block.type == "text")
        return Trajectory(
            prompt=prompt,
            response=response,
            provider=self.name,
            model=self.model,
            system=system,
            reasoning=reasoning,
            metadata={"stop_reason": message.stop_reason, "usage": message.usage.model_dump()},
        )


class OpenAICompatibleTeacher:
    """Teacher backed by the OpenAI API (GPT 5.6 only)."""

    name = "openai"

    def __init__(self, model: str = OPENAI_TEACHER_MODEL, api_key: str | None = None) -> None:
        import openai

        self.model = model
        self._client = openai.OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        thinking_budget: int | None = None,  # not used for GPT 5.6 chat completions
    ) -> Trajectory:
        messages: list[ChatCompletionMessageParam] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        completion = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        choice = completion.choices[0]
        reasoning = getattr(choice.message, "reasoning_content", None)
        return Trajectory(
            prompt=prompt,
            response=choice.message.content or "",
            provider=self.name,
            model=self.model,
            system=system,
            reasoning=reasoning,
            metadata={
                "finish_reason": choice.finish_reason,
                "usage": completion.usage.model_dump() if completion.usage else {},
            },
        )


def get_teacher(provider: str, model: str | None = None) -> TeacherClient:
    """Construct a configured teacher client by provider name.

    Each provider is pinned to a single model (Fable 5 or GPT 5.6). Reads credentials
    from the environment (see `.env.example`).
    """
    if provider not in _SUPPORTED_PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}, expected one of {sorted(_SUPPORTED_PROVIDERS)}")

    if provider == "anthropic":
        if model is not None and model != ANTHROPIC_TEACHER_MODEL:
            raise ValueError(
                f"anthropic teacher is fixed to {ANTHROPIC_TEACHER_MODEL!r}; got {model!r}"
            )
        return AnthropicTeacher(model=ANTHROPIC_TEACHER_MODEL)

    if model is not None and model not in _ALLOWED_OPENAI_MODELS:
        raise ValueError(
            f"openai teacher is fixed to {OPENAI_TEACHER_MODEL!r} "
            f"(alias {sorted(_ALLOWED_OPENAI_MODELS)}); got {model!r}"
        )
    return OpenAICompatibleTeacher(model=OPENAI_TEACHER_MODEL)
