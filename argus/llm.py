"""Local LLM client (Ollama) with schema-constrained output and an on-disk cache.

Why schema-constrained: we hand Ollama the pydantic JSON schema, so the model can
only produce valid labels (Favor / Against / Neutral). We still validate every
reply with pydantic, as a second safety net.

Why a cache: local inference is free but slow. Every validated reply is saved to
disk keyed by (model, temperature, schema, prompt), so re-running an experiment
only recomputes what changed, and results stay reproducible.
"""
import hashlib
import json
import time
from pathlib import Path
from typing import Type, TypeVar

import ollama
import yaml
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_client(model: str | None = None, temperature: float | None = None,
                cache_salt: str | None = None) -> "LLMClient":
    """Client from config.yaml, optionally overriding the model (to compare models), the temperature and a
    cache salt. A different salt gives separate cache entries, so repeated runs at a non-zero temperature each
    ask the model again instead of replaying the first answer (used by the self-consistency check)."""
    cfg = load_config()
    if model:
        cfg["model"] = model
    if temperature is not None:
        cfg["temperature"] = temperature
    if cache_salt:
        cfg["cache_salt"] = cache_salt
    return LLMClient(cfg)


def ensure_ready(llm: "LLMClient") -> "LLMClient":
    """For scripts: stop with a plain message (no traceback) if Ollama is down or the model is missing."""
    try:
        llm.check_ready()
    except RuntimeError as e:
        raise SystemExit(str(e))
    return llm


def make_ready_client(model: str | None = None) -> "LLMClient":
    return ensure_ready(make_client(model))


class LLMClient:
    def __init__(self, config: dict | None = None):
        self.cfg = config or load_config()
        self.model = self.cfg["model"]
        self.temperature = self.cfg.get("temperature", 0.0)
        self.num_ctx = self.cfg.get("num_ctx", 8192)
        self.max_retries = self.cfg.get("max_retries", 3)
        self.cache_salt = self.cfg.get("cache_salt") or ""
        self.client = ollama.Client(host=self.cfg.get("host", "http://localhost:11434"))

        self.cache_dir = ROOT / self.cfg.get("cache_dir", ".cache")
        self.cache_dir.mkdir(exist_ok=True)

        # Running totals, so a script can report how much real model work a run needed.
        self.stats = {"model_calls": 0, "cache_hits": 0}

    # ---- readiness check -----------------------------------------------
    def check_ready(self) -> None:
        """Fail early with a clear message if Ollama is down or the model is missing."""
        try:
            resp = self.client.list()
        except Exception as e:
            raise RuntimeError(
                "Cannot reach Ollama. Install it from https://ollama.com and make sure "
                f"it is running (the llama icon in your system tray). Details: {e}"
            )
        names = []
        for m in resp["models"]:
            names.append(m.get("model") or m.get("name"))
        if not any(n == self.model or n == f"{self.model}:latest" for n in names):
            raise RuntimeError(
                f"Model '{self.model}' is not downloaded. Run:  ollama pull {self.model}"
            )

    # ---- cache helpers -------------------------------------------------
    def _cache_path(self, messages: list[dict], schema: Type[BaseModel]) -> Path:
        key = [self.model, self.temperature, schema.__name__, messages]
        if self.cache_salt:                      # only when set, so caches from earlier phases stay valid
            key.append(self.cache_salt)
        blob = json.dumps(key, sort_keys=True)
        return self.cache_dir / (hashlib.sha256(blob.encode()).hexdigest() + ".json")

    # ---- core call -----------------------------------------------------
    def _raw_chat(self, messages: list[dict], schema: Type[BaseModel]) -> str:
        """One real model call; returns the raw JSON text. No caching here."""
        self.stats["model_calls"] += 1
        resp = self.client.chat(
            model=self.model,
            messages=messages,
            format=schema.model_json_schema(),
            options={"temperature": self.temperature, "num_ctx": self.num_ctx},
        )
        return resp["message"]["content"] or ""

    def structured(self, messages: list[dict], schema: Type[T]) -> T:
        """Call the model and return a validated pydantic object.

        Retries on errors, and on invalid output by telling the model what was
        wrong. Only VALID replies are cached, always under the original prompt.
        """
        path = self._cache_path(messages, schema)
        if path.exists():
            try:
                cached = schema.model_validate_json(
                    json.loads(path.read_text(encoding="utf-8"))["reply"]
                )
                self.stats["cache_hits"] += 1
                return cached
            except (ValidationError, KeyError, json.JSONDecodeError):
                path.unlink(missing_ok=True)  # stale or corrupt entry: re-ask the model

        last_error = None
        convo = list(messages)
        for attempt in range(1, self.max_retries + 1):
            try:
                reply = self._raw_chat(convo, schema)
                result = schema.model_validate_json(reply)
                path.write_text(json.dumps({"reply": reply}), encoding="utf-8")
                return result
            except ValidationError as e:
                last_error = e
                convo = list(messages) + [
                    {"role": "user", "content": (
                        "Your previous reply was not valid. Reply with ONLY a JSON "
                        f"object matching the required fields. Problem: {e.errors()[:2]}"
                    )}
                ]
            except Exception as e:  # server hiccup, model still loading, etc.
                last_error = e
                time.sleep(2 * attempt)
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_error}")
