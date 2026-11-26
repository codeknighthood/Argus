"""Phase 7: check that this computer is ready to run ARGUS, and say exactly how to fix what is not.

Each check returns {"name", "status", "detail", "fix"} where status is "ok", "warn" (works, but something is
missing that you may want) or "fail" (ARGUS cannot run until this is fixed).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable, Optional

MIN_PYTHON = (3, 10)
REQUIRED_PACKAGES = [("ollama", "ollama"), ("pydantic", "pydantic"), ("yaml", "pyyaml"), ("pysbd", "pysbd")]


def _result(name: str, status: str, detail: str, fix: str = "") -> dict:
    return {"name": name, "status": status, "detail": detail, "fix": fix}


def check_python(version=None) -> dict:
    v = tuple(version or sys.version_info[:3])
    text = ".".join(str(p) for p in v[:3])
    if v[:2] < MIN_PYTHON:
        return _result("Python", "fail", f"version {text} is too old (need {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer)",
                       "Install Python 3.11 from https://www.python.org/downloads/ and tick 'Add python.exe to PATH'.")
    return _result("Python", "ok", f"version {text}")


def check_packages(importer: Callable[[str], object] = importlib.import_module) -> list[dict]:
    out = []
    for module, pip_name in REQUIRED_PACKAGES:
        try:
            importer(module)
            out.append(_result(f"package {pip_name}", "ok", "installed"))
        except ImportError:
            out.append(_result(f"package {pip_name}", "fail", "not installed",
                               "pip install -r requirements.txt   (with the virtual environment active, see INSTALL.md step 2)"))
    try:
        importer("streamlit")
        out.append(_result("package streamlit", "ok", "installed (needed only for the demo)"))
    except ImportError:
        out.append(_result("package streamlit", "warn", "not installed (needed only for the demo)",
                           "pip install -r requirements-demo.txt"))
    return out


def check_config(root: Path) -> tuple[dict, Optional[dict]]:
    path = Path(root) / "config.yaml"
    try:
        import yaml
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict) or "model" not in cfg:
            raise ValueError("no 'model:' line")
    except Exception as e:
        return _result("config.yaml", "fail", f"cannot read {path}: {e}", "Restore config.yaml from the zip."), None
    return _result("config.yaml", "ok", f"model {cfg['model']}, host {cfg.get('host', 'http://localhost:11434')}"), cfg


def check_ollama(cfg: dict, client_factory: Optional[Callable[[dict], object]] = None) -> dict:
    """Is the Ollama server running and is the configured model downloaded?"""
    try:
        if client_factory is None:
            from .llm import LLMClient
            client_factory = LLMClient
        client = client_factory(cfg)
        client.check_ready()
    except RuntimeError as e:
        message = str(e)
        if "not downloaded" in message:
            return _result("Ollama model", "fail", message, f"ollama pull {cfg['model']}")
        return _result("Ollama", "fail", "cannot reach the Ollama server",
                       "Install Ollama from https://ollama.com, start it (llama icon in the system tray), then run this again.")
    except Exception as e:
        return _result("Ollama", "fail", f"unexpected problem: {e}", "Make sure Ollama is installed and running.")
    return _result("Ollama", "ok", f"running, model {cfg['model']} is downloaded"), client  # type: ignore[return-value]


def check_live_call(client) -> dict:
    """One real, tiny model call, to prove the model can answer in the format ARGUS needs."""
    try:
        from .stance import predict_stance
        result = predict_stance(client, "zero_shot", "the minimum wage",
                                "Raising it would help families who work full time and still cannot pay rent.")
        return _result("Live model call", "ok", f"answered '{result.stance.value}' in the required format")
    except Exception as e:
        return _result("Live model call", "fail", f"the model did not return a valid answer: {str(e)[:200]}",
                       "Try again; if it keeps failing, restart Ollama or run:  ollama run <model>  once to load it.")


def check_data(root: Path) -> list[dict]:
    root = Path(root)
    raw = [(root / "data" / "raw" / "vast" / n) for n in ("vast_dev.csv", "vast_test.csv")]
    processed = [(root / "data" / "processed" / n) for n in ("dev.jsonl", "test.jsonl")]
    out = []
    if all(p.exists() for p in raw):
        out.append(_result("VAST data downloaded", "ok", "vast_dev.csv and vast_test.csv are in data/raw/vast"))
    else:
        out.append(_result("VAST data downloaded", "warn", "not downloaded yet", "python scripts/download_data.py"))
    if all(p.exists() for p in processed):
        out.append(_result("Data prepared", "ok", "data/processed/dev.jsonl and test.jsonl exist"))
    else:
        out.append(_result("Data prepared", "warn", "not prepared yet", "python scripts/prepare_data.py"))
    return out


def run_checks(root: Path, live: bool = False, client_factory: Optional[Callable[[dict], object]] = None) -> list[dict]:
    """All checks in the order a person should fix them. Stops asking the model if the basics are broken."""
    results = [check_python()]
    results += check_packages()
    config_result, cfg = check_config(root)
    results.append(config_result)
    basics_ok = all(r["status"] != "fail" for r in results)
    if cfg is not None and basics_ok:
        outcome = check_ollama(cfg, client_factory)
        if isinstance(outcome, tuple):
            results.append(outcome[0])
            if live:
                results.append(check_live_call(outcome[1]))
        else:
            results.append(outcome)
    results += check_data(root)
    return results


def format_results(results: list[dict]) -> str:
    tag = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]"}
    lines = []
    for r in results:
        lines.append(f"{tag[r['status']]} {r['name']}: {r['detail']}")
        if r["fix"] and r["status"] != "ok":
            lines.append(f"       fix: {r['fix']}")
    failed = sum(r["status"] == "fail" for r in results)
    warned = sum(r["status"] == "warn" for r in results)
    lines.append("")
    if failed:
        lines.append(f"{failed} problem(s) must be fixed before ARGUS can run. Fix the first [FAIL] and run this again.")
    elif warned:
        lines.append("ARGUS can run. The [WARN] items are the next steps in INSTALL.md.")
    else:
        lines.append("Everything is ready.")
    return "\n".join(lines)
