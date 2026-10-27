"""Cached, rate-limited Gemini client. The only file that imports google-genai.

Every response is cached on disk under a hash of model name + prompt, every call is logged to
outputs/llm_log.jsonl, and the number of new (uncached) calls per process is capped by
`llm.max_calls`. Without GEMINI_API_KEY, `available()` is False and callers fall back to rules.

Run `python -m inclination.llm models` to list the models the key can use.
"""
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

from pydantic import BaseModel, ValidationError

from inclination.config import load_config, path

RETRY_CODES = {429, 500, 503}


class LLMStopped(RuntimeError):
    """A job cannot continue: the call budget is used up or the API keeps failing.
    Everything answered so far is cached, so re-running the job resumes where it stopped."""


class BudgetExceeded(LLMStopped):
    """Raised when a job would make more new calls than `llm.max_calls` allows."""


def available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


class GeminiClient:
    def __init__(self, cfg: dict | None = None, max_calls: int | None = None):
        from google import genai

        self.cfg = (cfg or load_config())["llm"]
        self.max_calls = self.cfg["max_calls"] if max_calls is None else max_calls
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.cache_dir = path("llm_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = path("llm_log")
        self.new_calls = 0
        self.last_call = 0.0
        self.usage = {}  # model -> {"calls", "cache_hits", "input_tokens", "output_tokens"}

    # ---- bookkeeping ----------------------------------------------------------------------

    def _cache_file(self, model: str, prompt: str):
        key = hashlib.sha256(f"{model}\n{prompt}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _record(self, job: str, model: str, input_tokens: int, output_tokens: int, hit: bool) -> None:
        totals = self.usage.setdefault(
            model, {"calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0}
        )
        totals["cache_hits" if hit else "calls"] += 1
        if not hit:
            totals["input_tokens"] += input_tokens
            totals["output_tokens"] += output_tokens
        entry = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "job": job,
            "model": model, "input_tokens": input_tokens, "output_tokens": output_tokens,
            "cache": "hit" if hit else "miss",
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _wait_for_rate_limit(self) -> None:
        pause = 60.0 / self.cfg["rpm"] - (time.time() - self.last_call)
        if pause > 0:
            time.sleep(pause)
        self.last_call = time.time()

    # ---- calls ----------------------------------------------------------------------------

    def is_cached(self, model: str, prompt: str) -> bool:
        return self._cache_file(model, prompt).exists()

    def generate(self, prompt: str, model: str, schema: type[BaseModel], job: str):
        """Returns (validated `schema` instance or None, raw response text).

        None means the model's answer did not fit the schema; it is cached like any other
        answer so the call is not repeated.
        """
        from google.genai import errors, types

        cache_file = self._cache_file(model, prompt)
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            self._record(job, model, cached["input_tokens"], cached["output_tokens"], hit=True)
            return _parse(cached["text"], schema), cached["text"]

        if self.new_calls >= self.max_calls:
            raise BudgetExceeded(f"llm.max_calls = {self.max_calls} reached; nothing more was sent")
        generation = types.GenerateContentConfig(
            temperature=self.cfg["temperature"], response_mime_type="application/json",
            response_schema=schema, **self._thinking(types),
        )
        for attempt in range(self.cfg["max_retries"] + 1):
            self._wait_for_rate_limit()
            try:
                response = self.client.models.generate_content(
                    model=model, contents=prompt, config=generation
                )
                break
            except errors.APIError as error:
                if error.code not in RETRY_CODES or attempt == self.cfg["max_retries"]:
                    raise LLMStopped(f"Gemini call failed ({error.code}): {str(error)[:300]}") from error
                delay = self.cfg["backoff_base_s"] * 2 ** attempt
                print(f"    Gemini returned {error.code}; retrying in {delay:.0f}s", flush=True)
                time.sleep(delay)
        self.new_calls += 1
        usage = response.usage_metadata
        input_tokens = int(usage.prompt_token_count or 0)
        # Thinking tokens are billed as output.
        output_tokens = int(usage.candidates_token_count or 0) + int(usage.thoughts_token_count or 0)
        text = response.text or ""
        cache_file.write_text(json.dumps({
            "model": model, "text": text, "input_tokens": input_tokens, "output_tokens": output_tokens,
        }), encoding="utf-8")
        self._record(job, model, input_tokens, output_tokens, hit=False)
        return _parse(text, schema), text

    def _thinking(self, types) -> dict:
        """Optional thinking setting from config (`llm.thinking_level`); omitted when null.
        Thinking tokens are billed as output, so this is the main cost control."""
        level = self.cfg.get("thinking_level")
        if level is None:
            return {}
        return {"thinking_config": types.ThinkingConfig(thinking_level=level)}

    def list_models(self) -> list[str]:
        return sorted(m.name.removeprefix("models/") for m in self.client.models.list())

    def summary(self) -> str:
        """Totals for this process, with an estimated cost from config `llm.prices_per_million`."""
        prices = self.cfg.get("prices_per_million", {})
        lines, total_cost = [], 0.0
        for model, t in self.usage.items():
            price = prices.get(model)
            cost = None
            if price:
                cost = (t["input_tokens"] * price["input"] + t["output_tokens"] * price["output"]) / 1e6
                total_cost += cost
            lines.append(
                f"  {model}: {t['calls']} new calls, {t['cache_hits']} cache hits, "
                f"{t['input_tokens']:,} input + {t['output_tokens']:,} output tokens"
                + (f", about ${cost:.4f}" if cost is not None else ", price not configured")
            )
        return "LLM usage:\n" + "\n".join(lines) + f"\n  estimated total: ${total_cost:.4f} (paid-tier list prices)"


def _parse(text: str, schema: type[BaseModel]):
    try:
        return schema.model_validate_json(text)
    except ValidationError:
        return None


if __name__ == "__main__":
    if not available():
        sys.exit("GEMINI_API_KEY is not set (put it in .env).")
    if sys.argv[1:] == ["models"]:
        cfg = load_config()["llm"]
        names = GeminiClient().list_models()
        print("\n".join(names))
        for role in ("model", "tagging_model"):
            print(f"configured llm.{role} = {cfg[role]}: {'FOUND' if cfg[role] in names else 'NOT FOUND'}")
