"""RAGAS judge wired to our own `LLMProvider`.

RAGAS defaults to OpenAI for both its judge model and its embeddings, and the
`[evals]` extra pulls `openai` in transitively whether we want it or not. That
default is the thing to avoid: evaluation traffic would leave the machine, to a
provider this project does not otherwise use, carrying the retrieved documents
with it.

`ProviderJudge` implements RAGAS' `InstructorBaseRagasLLM` interface directly on
top of `LLMProvider`, so the judge is whatever `LLM_PROVIDER` selects — the same
Claude or Ollama the pipeline uses (optionally a different model, via
`--judge-model`). Nothing here constructs an OpenAI client.

The interface RAGAS needs is `agenerate(prompt, response_model) -> response_model`,
i.e. structured output. Instructor gets that from function calling; we get it by
asking for JSON against the model's own JSON Schema and validating the reply,
retrying on malformed output. Small local models do produce malformed JSON, so
the retry is not decoration — `JudgeError` after the last attempt is what keeps
a bad judge reply from silently scoring as zero.

**The retry doubles the token budget each time.** A reasoning model spends part
of `num_predict` on `message.thinking` (the Phase 3 finding), and judge prompts
are long: faithfulness asks a model to restate every claim in an answer. When
the budget runs out inside the thinking block the reply comes back *empty*, and
retrying with the same budget just produces another empty reply. Doubling turns
that into a scored row.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, TypeVar

from evals.compat import patch_langchain_community

patch_langchain_community()

from ragas.llms.base import InstructorBaseRagasLLM  # noqa: E402

from app.core.llm_provider import LLMProvider  # noqa: E402

logger = logging.getLogger("evals.judge")

T = TypeVar("T")

JUDGE_SYSTEM = """You are a strict evaluation judge. You reply with JSON only.

Rules:
- Output exactly one JSON value matching the requested schema.
- No prose, no markdown fences, no explanation outside the JSON.
- Judge only what the input says. Do not use outside knowledge."""

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class JudgeError(RuntimeError):
    """The judge model never produced output matching the requested schema."""


def extract_json(text: str) -> Any:
    """Pull the first JSON value out of a model reply.

    Handles the three things models actually do: clean JSON, JSON in a fenced
    block, and JSON with a sentence in front of it.
    """
    if not text:
        raise JudgeError("empty judge reply")

    candidates: list[str] = [text.strip()]
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise JudgeError(f"no JSON found in judge reply: {text[:200]!r}")


class ProviderJudge(InstructorBaseRagasLLM):
    """RAGAS judge backed by `LLMProvider`. No OpenAI, no instructor client."""

    def __init__(
        self, provider: LLMProvider, *, max_attempts: int = 3, max_tokens: int = 4096
    ) -> None:
        self._provider = provider
        self._max_attempts = max_attempts
        self._max_tokens = max_tokens
        self.calls = 0
        self.retries = 0
        self.empty_replies = 0

    @property
    def model(self) -> str:
        return getattr(self._provider, "model", "")

    def _prompt(self, prompt: str, response_model: type) -> str:
        schema = json.dumps(response_model.model_json_schema(), indent=None)
        return f"{prompt}\n\nReply with JSON matching this schema:\n{schema}\n\nJSON:"

    def budget(self, attempt: int) -> int:
        """Token budget for an attempt: doubled each retry."""
        return self._max_tokens * 2 ** (attempt - 1)

    async def agenerate(self, prompt: str, response_model: type[T]) -> T:
        last: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            self.calls += 1
            if attempt > 1:
                self.retries += 1
            result = await self._provider.chat(
                [{"role": "user", "content": self._prompt(prompt, response_model)}],
                system=JUDGE_SYSTEM,
                max_tokens=self.budget(attempt),
            )
            if not (result.text or "").strip():
                self.empty_replies += 1
                last = JudgeError(f"empty reply at a budget of {self.budget(attempt)} tokens")
                logger.debug("judge attempt %d/%d: empty reply", attempt, self._max_attempts)
                continue
            try:
                return response_model.model_validate(extract_json(result.text))
            except Exception as exc:  # malformed JSON, or JSON of the wrong shape
                last = exc
                logger.debug("judge attempt %d/%d failed: %s", attempt, self._max_attempts, exc)
        raise JudgeError(
            f"judge produced no valid {response_model.__name__} in "
            f"{self._max_attempts} attempts: {last}"
        )

    def generate(self, prompt: str, response_model: type[T]) -> T:
        """Sync entry point. RAGAS' async path is what we actually use."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.agenerate(prompt, response_model))
        raise RuntimeError("ProviderJudge.generate() called from a running loop; use agenerate()")
