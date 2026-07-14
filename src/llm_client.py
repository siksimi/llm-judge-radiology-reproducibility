"""
llm_client.py — provider-agnostic LLM call wrapper.

Features:
  - Multi-provider adapters: anthropic / openai / google / openai_compatible (self-hosted Qwen, etc.)
  - Models without temperature support (judge.temperature_capable is False) receive no temperature/top_p
  - Exponential-backoff retries (non-retryable errors are not retried)
  - Disk caching (cache_tag keeps repeated runs independent -> preserves variance)
  - Raw logging of every call (runs/calls.log.jsonl)
  - MOCK mode (YH2_MOCK=1): returns fake JSON without API keys
Parsing/validation happens in response_parser.py (raw output preserved).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import random
import datetime as dt
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# permanent (non-retryable) error keywords -> stop immediately
_NON_RETRYABLE = ("authentication", "invalid_request", "deprecated", "not_found",
                  "permission", "ModuleNotFoundError", "BadRequestError", "NotFoundError",
                  "insufficient_quota", "credit balance", "billing", "exceeded your current quota")


@dataclass
class JudgeSpec:
    id: str
    family: str
    provider: str
    model: str
    type: str = "closed"
    base_url: Optional[str] = None
    temperature_capable: Optional[bool] = None
    min_interval_s: float = 0.0
    daily_cap: int = 0


@dataclass
class CallResult:
    judge_id: str
    raw_text: str
    model_version: str
    access_date: str
    params: dict
    cache_hit: bool
    latency_s: float
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


def _cache_key(judge_id, system, prompt, params, cache_tag=""):
    blob = json.dumps({"judge": judge_id, "system": system, "prompt": prompt,
                       "params": params, "tag": cache_tag},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _send_temp(judge) -> bool:
    """If temperature_capable is False, do not send it; if True/None, send it (None = attempt, then decide)."""
    return judge.temperature_capable is not False


class LLMClient:
    def __init__(self, cache_dir="runs/cache", log_path="runs/calls.log.jsonl",
                 max_retries=5, backoff_base_s=2.0, request_timeout_s=120.0, use_cache=True,
                 min_interval_s=0.0):
        self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_path); self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.request_timeout_s = request_timeout_s
        self.use_cache = use_cache
        self.min_interval_s = min_interval_s
        self._last_call = {}
        self.mock = os.environ.get("YH2_MOCK") == "1"

    def cached_exists(self, judge, system, prompt, temperature=0.0, top_p=1.0,
                      max_tokens=1024, cache_tag=""):
        """Check whether a call is already cached (= no re-call needed). Used for daily_cap logic."""
        if not self.use_cache:
            return False
        params = {"max_tokens": max_tokens}
        if _send_temp(judge):
            params["temperature"] = temperature; params["top_p"] = top_p
        key = _cache_key(judge.id, system, prompt, params, cache_tag)
        return (self.cache_dir / f"{key}.json").exists()

    def complete(self, judge, system, prompt, temperature=0.0, top_p=1.0,
                 max_tokens=1024, cache_tag=""):
        # actually-transmitted parameters (unsupported models omit temperature/top_p -> logged as such)
        params = {"max_tokens": max_tokens}
        if _send_temp(judge):
            params["temperature"] = temperature
            params["top_p"] = top_p
        key = _cache_key(judge.id, system, prompt, params, cache_tag)
        cache_file = self.cache_dir / f"{key}.json"

        if self.use_cache and not self.mock and cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                cached["cache_hit"] = True
                res = CallResult(**cached)
                self._log(res, key); return res
            except Exception:
                # corrupt/incomplete cache file -> ignore and re-call (so the run does not die)
                try: cache_file.unlink()
                except Exception: pass

        interval = judge.min_interval_s or self.min_interval_s
        if interval and not self.mock:
            wait = interval - (time.time() - self._last_call.get(judge.id, 0.0))
            if wait > 0: time.sleep(wait)
            self._last_call[judge.id] = time.time()
        t0 = time.time()
        raw, version, err = self._dispatch_with_retry(judge, system, prompt, params, cache_tag)
        latency = round(time.time() - t0, 3)
        res = CallResult(judge_id=judge.id, raw_text=raw, model_version=version,
                         access_date=dt.date.today().isoformat(), params=params,
                         cache_hit=False, latency_s=latency, error=err)
        if err is None and self.use_cache and not self.mock:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(res.as_dict(), ensure_ascii=False, indent=2),
                                      encoding="utf-8")
            except Exception:
                pass  # cache-write failure is non-fatal (re-called on the next run)
        self._log(res, key); return res

    def _dispatch_with_retry(self, judge, system, prompt, params, cache_tag=""):
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw, version = self._dispatch(judge, system, prompt, params, cache_tag)
                return raw, version, None
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                if any(tok in last_err for tok in _NON_RETRYABLE) or attempt == self.max_retries:
                    break
                # exponential backoff + jitter (avoid retry stampede on 529/429 overload)
                base = min(self.backoff_base_s * (2 ** (attempt - 1)), 60)
                time.sleep(base + random.uniform(0, base * 0.5))
        return "", "unknown", last_err

    def _dispatch(self, judge, system, prompt, params, cache_tag=""):
        if self.mock:
            return self._mock_response(judge, prompt, params, cache_tag), f"{judge.model}@MOCK"
        p = judge.provider
        if p == "anthropic":
            return self._call_anthropic(judge, system, prompt, params)
        if p == "openai":
            return self._call_openai(judge, system, prompt, params)
        if p == "google":
            return self._call_google(judge, system, prompt, params)
        if p == "openai_compatible":
            return self._call_openai(judge, system, prompt, params, compatible=True)
        raise ValueError(f"Unknown provider: {p}")

    def _call_anthropic(self, judge, system, prompt, params):
        import anthropic
        client = anthropic.Anthropic(timeout=self.request_timeout_s)
        kw = dict(model=judge.model, system=system, max_tokens=params["max_tokens"],
                  messages=[{"role": "user", "content": prompt}])
        if "temperature" in params:
            kw["temperature"] = params["temperature"]; kw["top_p"] = params["top_p"]
        msg = client.messages.create(**kw)
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return text, getattr(msg, "model", judge.model)

    def _call_openai(self, judge, system, prompt, params, compatible=False):
        from openai import OpenAI
        kwargs = {}
        if compatible and judge.base_url:
            kwargs["base_url"] = judge.base_url
            # self-host (e.g. vLLM) is kept separate from OPENAI_API_KEY: dummy "EMPTY" if QWEN_API_KEY unset
            kwargs["api_key"] = os.environ.get("QWEN_API_KEY", "EMPTY")
        kwargs["timeout"] = self.request_timeout_s
        client = OpenAI(**kwargs)
        kw = dict(model=judge.model,
                  messages=[{"role": "system", "content": system},
                            {"role": "user", "content": prompt}])
        # real OpenAI (GPT-5 family) uses max_completion_tokens; self-hosted compatible servers use max_tokens
        if compatible:
            kw["max_tokens"] = params["max_tokens"]
            # self-hosted Qwen 3.5: disable thinking -> direct JSON (reproducibility/speed/parse stability); recorded per MI-CLEAR-LLM
            kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        else:
            kw["max_completion_tokens"] = params["max_tokens"]
        if "temperature" in params:
            kw["temperature"] = params["temperature"]; kw["top_p"] = params["top_p"]
        resp = client.chat.completions.create(**kw)
        return resp.choices[0].message.content or "", getattr(resp, "model", judge.model)

    def _call_google(self, judge, system, prompt, params):
        # use the new google-genai SDK; minimize thinking + force JSON output to
        # avoid thinking tokens eating the output budget / code-fence issues.
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"),
                              http_options=types.HttpOptions(timeout=int(self.request_timeout_s*1000)))
        cfg = dict(system_instruction=system,
                   max_output_tokens=params["max_tokens"],
                   response_mime_type="application/json")
        if "temperature" in params:
            cfg["temperature"] = params["temperature"]; cfg["top_p"] = params["top_p"]
        # thinking_level is Gemini 3.x only -> not sent to 2.5 etc. (avoids 400)
        if "gemini-3" in judge.model:
            try:
                cfg["thinking_config"] = types.ThinkingConfig(thinking_level="low")
            except Exception:
                pass
        resp = client.models.generate_content(
            model=judge.model, contents=prompt,
            config=types.GenerateContentConfig(**cfg))
        return (resp.text or ""), judge.model

    @staticmethod
    def _mock_response(judge, prompt, params, cache_tag=""):
        import random
        temp = params.get("temperature", 0.0)
        seed = hashlib.sha256(f"{judge.id}|{prompt}|{temp}|{cache_tag}".encode()).hexdigest()
        rng = random.Random(seed)
        base = {k: rng.randint(2, 5) for k in
                ["finding_completeness", "diagnostic_accuracy",
                 "clarity_structure", "clinical_appropriateness"]}
        if temp > 0 and rng.random() < temp:
            k = rng.choice(list(base)); base[k] = max(1, min(5, base[k] + rng.choice([-1, 1])))
        base["overall_global_quality"] = max(1, min(5, round(sum(base.values()) / 4)))
        return json.dumps(base)

    def _log(self, res, prompt_key):
        rec = res.as_dict()
        rec["prompt_key"] = prompt_key
        rec["ts"] = dt.datetime.now(dt.timezone.utc).isoformat()
        rec["raw_len"] = len(rec.pop("raw_text", ""))
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


_PROVIDER_BY_FAMILY = {
    "claude": ("anthropic", None),
    "gpt": ("openai", None),
    "gemini": ("google", None),
    "qwen": ("openai_compatible", "http://localhost:8000/v1"),
}


def judges_from_config(judges_cfg):
    specs = []
    for j in judges_cfg:
        provider, base_url = _PROVIDER_BY_FAMILY[j["family"]]
        # self-host endpoint override: QWEN_BASE_URL env or the config judge's endpoint.
        # (e.g. the Vast.ai vLLM template uses port 18000; the AMD serve script uses 8000.)
        if provider == "openai_compatible":
            base_url = j.get("endpoint") or os.environ.get("QWEN_BASE_URL", base_url)
        specs.append(JudgeSpec(id=j["id"], family=j["family"], provider=provider,
                               model=j.get("model", j["id"]), type=j.get("type", "closed"),
                               base_url=base_url, temperature_capable=j.get("temperature_capable"),
                               min_interval_s=j.get("min_interval_s", 0.0),
                               daily_cap=j.get("daily_cap", 0)))
    return specs


if __name__ == "__main__":
    os.environ.setdefault("YH2_MOCK", "1")
    c = LLMClient(cache_dir="runs/cache_demo", log_path="runs/demo.log.jsonl", use_cache=False)
    j = JudgeSpec(id="qwen-3.5", family="qwen", provider="openai_compatible", model="qwen-3.5",
                  base_url="http://localhost:8000/v1")
    for run in range(3):
        print(run, c.complete(j, "sys", "REPORT x", cache_tag=f"C1|v1|T0.0|run{run}|R1").raw_text)
