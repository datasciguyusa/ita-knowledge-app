"""All calls to Claude go through here (model: Claude Haiku 5.5 by default)."""
import base64
import json
import re

import anthropic

from usage import BudgetExceeded

DEFAULT_MODEL = "claude-haiku-5-5"
TRANSCRIBE_MAX_TOKENS = 16_000


def _parse_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(t[start : end + 1])


def friendly_error(exc: Exception) -> str:
    """Plain-English message for common API failures."""
    if isinstance(exc, BudgetExceeded):
        return str(exc)
    if isinstance(exc, anthropic.AuthenticationError):
        return ("Claude rejected the API key (401 Unauthorized). Create a key under "
                "console.anthropic.com → API keys (it should start with sk-ant-api) and put it in "
                ".env as ANTHROPIC_API_KEY=…, then restart the app.")
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "The API key doesn't have permission for this model or workspace."
    if isinstance(exc, anthropic.NotFoundError):
        return "Model not found — check MODEL in app.py."
    if isinstance(exc, anthropic.RateLimitError):
        return "Claude rate limit or spend limit reached — wait a minute or raise the limit in the console."
    if "anthropic-workspace-id" in str(exc):
        return ("This API key isn't tied to a workspace. Either create a key inside a workspace at "
                "console.anthropic.com → API keys, or add ANTHROPIC_WORKSPACE_ID=<workspace ID> to .env.")
    if isinstance(exc, anthropic.BadRequestError) and "credit" in str(exc).lower():
        return "Your Anthropic account is out of credit — add credit at console.anthropic.com → Billing."
    return f"{type(exc).__name__}: {exc}"


class LLM:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, budget=None, workspace_id: str = ""):
        # User-level keys (sk-ant-usr…) must say which workspace to bill; workspace keys don't need it.
        headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=3, default_headers=headers)
        self.model = model
        self.budget = budget            # usage.Budget: checks the monthly limit and records spend

    def _check(self):
        if self.budget is not None:
            self.budget.check()

    def _record(self, kind: str, usage):
        if self.budget is not None and usage is not None:
            self.budget.record(kind, usage)

    def _create(self, system: str, messages: list, max_tokens: int, kind: str):
        # Note: anthropic SDK 2.x has no `temperature` parameter, so none is sent.
        self._check()
        resp = self.client.messages.create(model=self.model, max_tokens=max_tokens,
                                           system=system, messages=messages)
        self._record(kind, getattr(resp, "usage", None))
        return resp

    @staticmethod
    def _text(resp) -> str:
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    def ask_text(self, system: str, prompt: str, max_tokens: int = 1500, kind: str = "explain") -> str:
        resp = self._create(system, [{"role": "user", "content": prompt}], max_tokens, kind)
        return self._text(resp).strip()

    def ask_json(self, system: str, prompt: str, max_tokens: int = 16_000, kind: str = "json") -> dict:
        prompt += "\n\nRespond with the JSON object only — no other text, no code fences."
        resp = self._create(system, [{"role": "user", "content": prompt}], max_tokens, kind)
        return _parse_json(self._text(resp))

    def transcribe(self, pdf_bytes: bytes) -> str:
        from exam_bank import TRANSCRIBE_PROMPT
        content = [
            {"type": "document",
             "source": {"type": "base64", "media_type": "application/pdf",
                        "data": base64.standard_b64encode(pdf_bytes).decode()}},
            {"type": "text", "text": TRANSCRIBE_PROMPT},
        ]
        resp = self._create("You transcribe documents faithfully.",
                            [{"role": "user", "content": content}], TRANSCRIBE_MAX_TOKENS, "transcribe")
        return self._text(resp)

    def stream_chat(self, system: str, history: list[dict], user_msg: str, max_tokens: int = 2000):
        """Yield text pieces. history = [{"role": "user"|"assistant", "content": str}, ...]"""
        msgs = [m for m in history if m.get("content")]
        while msgs and msgs[0]["role"] != "user":       # Claude needs to start with a user turn
            msgs.pop(0)
        merged = []
        for m in msgs:                                   # and roles must alternate
            if merged and merged[-1]["role"] == m["role"]:
                merged[-1]["content"] += "\n\n" + m["content"]
            else:
                merged.append({"role": m["role"], "content": m["content"]})
        if merged and merged[-1]["role"] == "user":
            merged[-1]["content"] += "\n\n" + user_msg
        else:
            merged.append({"role": "user", "content": user_msg})
        self._check()
        with self.client.messages.stream(model=self.model, max_tokens=max_tokens,
                                         system=system, messages=merged) as stream:
            for text in stream.text_stream:
                yield text
            self._record("chat", getattr(stream.get_final_message(), "usage", None))
