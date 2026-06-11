"""Token counter callback for LangChain LLM calls. Tracks tokens per pipeline run."""
from __future__ import annotations
import logging

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

# Groq Llama 3 70B pricing (public, per 1M tokens)
INPUT_COST_PER_M = 0.59
OUTPUT_COST_PER_M = 0.79
DAILY_FREE_TIER_LIMIT = 100_000


class TokenCounter(BaseCallbackHandler):
    """Accumulates token counts from all LLM calls during a graph run."""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def on_llm_end(self, response, **kwargs):
        # Primary path: llm_output["token_usage"] (standard LangChain/Groq)
        u = None
        llm_out = getattr(response, "llm_output", None) or {}
        if "token_usage" in llm_out:
            u = llm_out["token_usage"]

        # Fallback path: generation_info["usage"] (some ChatGroq versions)
        if u is None:
            try:
                gen_info = response.generations[0][0].generation_info or {}
                u = gen_info.get("usage") or gen_info.get("token_usage")
            except (IndexError, AttributeError):
                pass

        if u:
            self.prompt_tokens += u.get("prompt_tokens", 0)
            self.completion_tokens += u.get("completion_tokens", 0)
            self.total_tokens += u.get("total_tokens", 0)

    @property
    def estimated_cost_usd(self) -> float:
        return round(
            self.prompt_tokens * INPUT_COST_PER_M / 1_000_000
            + self.completion_tokens * OUTPUT_COST_PER_M / 1_000_000,
            6,
        )

    @property
    def daily_limit_pct(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return round(self.total_tokens / DAILY_FREE_TIER_LIMIT * 100, 1)

    def summary(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "daily_limit_pct": self.daily_limit_pct,
        }
