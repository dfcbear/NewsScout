"""newsscout.llm.json_repair
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Robust JSON extraction and syntax repair for LLM responses.
Handles markdown fences, conversational wrappers, and trailing commas.
"""
from __future__ import annotations

import json
import re
from typing import Any


def clean_and_parse_json(text: str) -> dict[str, Any]:
    """Robustly extracts, repairs, and parses JSON from LLM output.

    Handles:
    - Outer markdown code fences (```json ... ```)
    - Surrounding conversational commentary (preamble and sign-off)
    - Single and multi-level trailing commas before } or ]
    - Preservation of inner backticks inside string values

    Args:
        text: Raw text string emitted by an LLM.

    Returns:
        Parsed Python dictionary.

    Raises:
        ValueError: If JSON is completely unrecoverable.
    """
    if not text or not text.strip():
        raise ValueError("Failed to parse model output as JSON: Empty model output.")

    candidate = text.strip()

    # Step 1: Strip outermost markdown code fences if present
    first_fence = candidate.find("```")
    last_fence = candidate.rfind("```")
    if first_fence != -1 and last_fence != -1 and last_fence > first_fence:
        inner = candidate[first_fence + 3 : last_fence]
        if inner.lstrip().lower().startswith("json"):
            inner = re.sub(r"^json\s*", "", inner.lstrip(), flags=re.IGNORECASE)
        candidate = inner.strip()

    # Step 2: Extract outermost JSON object if wrapped in conversational text
    start_idx = candidate.find("{")
    end_idx = candidate.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate = candidate[start_idx : end_idx + 1]

    # Step 3: Fast-path clean parse attempt
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Step 4: Iterative syntax repair: trailing commas before } or ]
    repaired = candidate
    for _ in range(3):
        cleaned = re.sub(r",\s*([\}\]])", r"\1", repaired)
        if cleaned == repaired:
            break
        repaired = cleaned

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Step 5: Advanced repair for unescaped newlines in raw multi-line string fields
    try:
        repaired_newlines = re.sub(
            r'("(?:[^"\\]|\\.)*")',
            lambda m: m.group(0).replace("\n", "\\n").replace("\r", "\\r"),
            repaired,
            flags=re.DOTALL,
        )
        return json.loads(repaired_newlines)
    except json.JSONDecodeError:
        pass

    # Step 6: Unrecoverable syntax error: raise descriptive ValueError
    raise ValueError(f"Failed to parse model output as JSON: Unrecoverable syntax\nRaw text:\n{text[:300]}")
