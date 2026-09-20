"""newsscout.filtering
~~~~~~~~~~~~~~~~~~~~
Two-Stage Filtering Pipeline for NewsScout:
- Stage 1: Deterministic Local Heuristics & Anti-Hype Blacklists
- Stage 2: Deep LLM Evaluation & 1-Minute Decision Card Generation
"""

from __future__ import annotations

from newsscout.filtering.card_gen import (
    CardValidationError,
    DecisionCardGenerator,
    DecisionCardValidator,
    ValidationResult,
)
from newsscout.filtering.prompts import (
    STAGE2_JSON_SCHEMA,
    SYSTEM_INSTRUCTION,
    build_evaluation_user_prompt,
    format_few_shot_calibration,
)
from newsscout.filtering.stage1 import (
    BLACKLIST_PATTERNS,
    BLACKLISTED_DOMAINS,
    MAX_INACTIVITY_DAYS,
    NON_COMMERCIAL_LICENSES,
    OSI_APPROVED_LICENSES,
    RUNNABLE_FILENAMES,
    Stage1Filter,
)
from newsscout.filtering.stage2 import (
    DecisionCardPayload,
    Stage2EvaluationResponse,
    Stage2Evaluator,
    clean_and_parse_json,
)

__all__ = [
    "Stage1Filter",
    "Stage2Evaluator",
    "Stage2EvaluationResponse",
    "DecisionCardPayload",
    "clean_and_parse_json",
    "SYSTEM_INSTRUCTION",
    "STAGE2_JSON_SCHEMA",
    "build_evaluation_user_prompt",
    "format_few_shot_calibration",
    "MAX_INACTIVITY_DAYS",
    "RUNNABLE_FILENAMES",
    "OSI_APPROVED_LICENSES",
    "NON_COMMERCIAL_LICENSES",
    "BLACKLIST_PATTERNS",
    "BLACKLISTED_DOMAINS",
    "DecisionCardGenerator",
    "DecisionCardValidator",
    "CardValidationError",
    "ValidationResult",
]
