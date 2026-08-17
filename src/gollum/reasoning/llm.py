"""Thin, uniform clients for the LLM APIs the reasoning agent can call.

Each client exposes a single ``complete(prompt) -> (text, reasoning)`` method,
where ``reasoning`` is the model's native thinking/chain-of-thought trace (not
a requested summary) — Claude extended thinking, Gemini thought parts, or the
``reasoning_content`` field returned by DeepSeek-style / OpenAI-compatible
reasoning-serving endpoints. Empty string if the provider/model exposes none.
Instantiated via ``gollum.utils.config.instantiate_class`` from a
class_path/init_args config, the same pattern used for acquisition functions
and surrogate models (see gollum.bo.optimizer).
"""
import os
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Common interface for all LLM providers used by the reasoning agent."""

    @abstractmethod
    def complete(self, prompt: str) -> tuple:
        """Send `prompt` to the LLM; return (text, reasoning)."""
        raise NotImplementedError


class AnthropicLLM(LLMClient):
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        max_tokens: int = 4096,
        thinking_budget_tokens: int = 1024,
        api_key: str = None,
    ):
        import anthropic

        self.model = model
        self.max_tokens = max_tokens
        self.thinking_budget_tokens = thinking_budget_tokens
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, prompt: str) -> tuple:
        # Temperature is fixed at 1 (the API default) when thinking is enabled.
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "enabled", "budget_tokens": self.thinking_budget_tokens},
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        reasoning = "\n".join(b.thinking for b in response.content if b.type == "thinking")
        return text, reasoning


class GeminiLLM(LLMClient):
    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        temperature: float = 0.0,
        api_key: str = None,
    ):
        from google import genai

        self.model = model
        self.temperature = temperature
        self._client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))

    def complete(self, prompt: str) -> tuple:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                thinking_config=types.ThinkingConfig(include_thoughts=True),
            ),
        )
        text, reasoning = [], []
        for part in response.candidates[0].content.parts:
            (reasoning if getattr(part, "thought", False) else text).append(part.text)
        return "".join(text), "\n".join(reasoning)


class OpenAILLM(LLMClient):
    def __init__(
        self,
        model: str = "gpt-5",
        temperature: float = 0.0,
        api_key: str = None,
    ):
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def complete(self, prompt: str) -> tuple:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        message = response.choices[0].message
        return message.content, getattr(message, "reasoning_content", "") or ""
