from __future__ import annotations
import itertools
import os
import random
import threading
import time
from openai import OpenAI

class ModelPool:
    def __init__(self):
        urls = [x.strip() for x in os.environ["QWEN_CHAT_BASE_URL"].split(",") if x.strip()]
        key = os.environ.get("QWEN_CHAT_API_KEY", "EMPTY")
        self.model = os.environ.get("QWEN_CHAT_MODEL", "qwen3.6-35b-a3b")
        self.temperature = float(os.environ.get("QWEN_CHAT_TEMPERATURE", "0"))
        self.timeout = float(os.environ.get("QWEN_CHAT_TIMEOUT", "600"))
        self.retry_rounds = max(1, int(os.environ.get("QWEN_CHAT_RETRY_ROUNDS", "2")))
        self.retry_backoff = max(0.0, float(os.environ.get("QWEN_CHAT_RETRY_BACKOFF", "2")))
        # Disable the SDK's hidden retry layer: retries and endpoint failover are
        # handled explicitly below so logs show exactly what happened.
        self.clients = [
            OpenAI(base_url=url, api_key=key, timeout=self.timeout, max_retries=0)
            for url in urls
        ]
        self.urls = urls
        self._counter = itertools.count()
        self._lock = threading.Lock()

    def complete(self, messages, *, tools=None, max_tokens=8192):
        with self._lock:
            start = next(self._counter)
        errors = []
        for round_idx in range(self.retry_rounds):
            for offset in range(len(self.clients)):
                index = (start + round_idx + offset) % len(self.clients)
                client = self.clients[index]
                try:
                    kwargs = dict(model=self.model, messages=messages, temperature=self.temperature,
                                  max_completion_tokens=max_tokens)
                    if tools:
                        kwargs.update(tools=tools, tool_choice="auto")
                    return client.chat.completions.create(**kwargs).choices[0].message
                except Exception as exc:
                    detail = (
                        f"round={round_idx + 1}/{self.retry_rounds} "
                        f"endpoint={self.urls[index]} {type(exc).__name__}: {exc}"
                    )
                    errors.append(detail)
                    print(f"[model-retry] {detail}", flush=True)
            if round_idx + 1 < self.retry_rounds:
                delay = self.retry_backoff * (2 ** round_idx) + random.random()
                print(f"[model-retry] all endpoints failed; sleeping {delay:.1f}s", flush=True)
                time.sleep(delay)
        raise RuntimeError(
            "all model endpoints failed after "
            f"{len(errors)} explicit attempt(s): " + " | ".join(errors)
        )
