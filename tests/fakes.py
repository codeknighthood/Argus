"""Test doubles: FakeLLM (replaces LLMClient) and FakeOllama (replaces the ollama.Client inside LLMClient)."""
import json

from .fake_rules import fake_reply


class FakeLLM:
    """Same `structured()` interface as LLMClient, answering from the keyword rules."""

    model = "fake-model"

    def __init__(self, fail_when=None):
        self.calls: list[str] = []
        self.fail_when = fail_when      # optional callable(prompt) -> bool, to simulate a failing call

    def check_ready(self) -> None:
        pass

    def structured(self, messages, schema):
        prompt = messages[-1]["content"]
        self.calls.append(prompt)
        if self.fail_when and self.fail_when(prompt):
            raise RuntimeError("simulated LLM failure")
        keys = set(schema.model_json_schema()["properties"])
        return schema.model_validate(fake_reply(prompt, keys))


class FakeOllama:
    """Scriptable stand-in for ollama.Client. `replies` are the raw texts returned by successive chat() calls."""

    def __init__(self, replies=None, models=("qwen2.5:7b",), list_error=None, chat_error=None):
        self.replies = list(replies or [])
        self.models = list(models)
        self.list_error = list_error
        self.chat_error = chat_error
        self.chat_calls: list[dict] = []

    def list(self):
        if self.list_error:
            raise self.list_error
        return {"models": [{"model": m} for m in self.models]}

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        if self.chat_error:
            raise self.chat_error
        reply = self.replies.pop(0) if self.replies else json.dumps({"stance": "Favor", "explanation": "x"})
        return {"message": {"content": reply}}
