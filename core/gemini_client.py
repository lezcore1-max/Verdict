"""
core/gemini_client.py — Thin wrapper around the Gemini API.

JSON extraction pipeline (in order):
  1. json.loads on raw text
  2. Regex extract between first { and last }
  3. Single retry with correction prompt
  4. Return None on complete failure
"""
import json
import re
import time
import logging
from typing import Optional

import google.generativeai as genai

from core.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)


class GeminiClient:
    """
    Single-responsibility Gemini caller.  One instance per agent call.
    All JSON extraction + retry logic lives here.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        system_prompt: str,
        api_key: Optional[str] = None,
    ) -> None:
        key = api_key or GEMINI_API_KEY
        if not key:
            raise ValueError("GEMINI_API_KEY is not set.")
        genai.configure(api_key=key)
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = system_prompt
        self._model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def call(self, user_prompt: str) -> Optional[dict]:
        """
        Call the model and return a parsed dict.
        Returns None if JSON extraction fails after all retries.
        """
        raw = self._call_with_backoff(user_prompt)
        if raw is None:
            return None

        parsed = self._extract_json(raw)
        if parsed is not None:
            return parsed

        # Single correction retry
        correction = (
            "Your previous response could not be parsed as JSON.\n"
            "Here is what you returned:\n"
            f"{raw}\n\n"
            "Please respond ONLY with a valid JSON object. "
            "No markdown fences, no explanation, no preamble."
        )
        raw2 = self._call_with_backoff(correction)
        if raw2 is None:
            return None

        return self._extract_json(raw2)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _call_with_backoff(self, prompt: str) -> Optional[str]:
        """Call the Gemini API with exponential backoff on quota errors."""
        delays = [2, 4, 8]
        for attempt, delay in enumerate(delays + [None]):
            try:
                # 4.1s strict delay to guarantee < 15 RPM
                time.sleep(4.1)
                t0 = time.perf_counter()
                response = self._model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature,
                    ),
                )
                latency = (time.perf_counter() - t0) * 1000
                text = response.text.strip()
                logger.debug(
                    "Gemini %s responded in %.0f ms (%d chars)",
                    self.model_name, latency, len(text)
                )
                return text
            except Exception as exc:
                exc_name = type(exc).__name__
                if attempt < len(delays):
                    logger.warning(
                        "Gemini call failed (%s): %s — retrying in %ds",
                        exc_name, exc, delays[attempt]
                    )
                    time.sleep(delays[attempt])
                else:
                    logger.error("Gemini call failed after all retries: %s", exc)
                    return None
        return None

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """
        Try to parse text as JSON using three strategies:
          1. Direct json.loads
          2. Regex: extract from first { to last }
          3. Regex: extract from first [ to last ] (for array roots)
        """
        # Strategy 1: direct
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: extract object
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Strategy 3: extract array (then wrap)
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                arr = json.loads(match.group())
                return {"items": arr}
            except json.JSONDecodeError:
                pass

        return None
