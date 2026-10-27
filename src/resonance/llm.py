"""Minimal OpenAI-compatible chat client for Meta Model API and Baseten (no SDK needed)."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

PROVIDERS = {
    # Muse Spark on Meta Model API (the inspector, once credits are active)
    "meta": {"base": "https://api.meta.ai/v1", "model": "muse-spark-1.3", "key": "MODEL_API_KEY"},
    # Muse Spark Open on Baseten (hackathon key works only with this exact model id)
    "baseten": {"base": "https://inference.baseten.co/v1", "model": "meta-models/muse-spark-open-1026",
                "key": "BASETEN_API_KEY"},
}


def load_env(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line.strip())
        if m and m.group(2) and m.group(1) not in os.environ:
            os.environ[m.group(1)] = m.group(2).strip()


def chat(messages: list[dict], tools: list[dict] | None = None, provider: str = "baseten",
         model: str | None = None, max_tokens: int = 4000, retries: int = 3, timeout: int = 180) -> dict:
    """Return {"message": ..., "usage": ..., "model": ...}; raises on persistent failure."""
    load_env()
    cfg = PROVIDERS[provider]
    key = os.environ.get(cfg["key"], "")
    if not key:
        raise RuntimeError(f"{cfg['key']} is not set (see .env.example)")
    body = {"model": model or cfg["model"], "messages": messages, "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
    data = json.dumps(body).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(cfg["base"] + "/chat/completions", data=data,
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=timeout))
            return {"message": r["choices"][0]["message"], "finish": r["choices"][0].get("finish_reason"),
                    "usage": r.get("usage", {}), "model": r.get("model", body["model"])}
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:300]!r}"
            if e.code not in (408, 429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError) as e:
            last = str(e)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{provider} chat failed: {last}")
