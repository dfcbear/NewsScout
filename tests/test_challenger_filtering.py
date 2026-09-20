"""tests/test_challenger_filtering.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Empirical Challenger Test Suite for NewsScout Filtering Pipeline:
1. Boundary score validation (out of bounds, non-numeric, negative, SQLite CHECKs)
2. Adversarial LLM JSON outputs (fences, nested backticks, trailing commas, missing/extra fields)
3. License boundary checking (non-commercial, BSL, SSPL, ambiguous, multi-license, dict metadata)
4. Anti-hype blacklist evasion (obfuscation, spacing, landing page builders, drop-shipping)
5. Concurrent Stage 1 & Stage 2 evaluations writing to SQLite WAL under high throughput
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import sqlite3
from typing import Any, Optional

import httpx
from pydantic import ValidationError
import pytest

from newsscout.config import Settings
from newsscout.filtering.prompts import STAGE2_JSON_SCHEMA
from newsscout.filtering.stage1 import (
    BLACKLIST_PATTERNS,
    BLACKLISTED_DOMAINS,
    NON_COMMERCIAL_LICENSES,
    OSI_APPROVED_LICENSES,
    Stage1Filter,
)
from newsscout.filtering.stage2 import (
    DecisionCardPayload,
    Stage2EvaluationResponse,
    Stage2Evaluator,
    clean_and_parse_json,
)
from newsscout.storage.db import Database
from newsscout.storage.models import (
    Breakthrough,
    DecisionCard,
    DecisionCardData,
    RawItem,
    Stage1Evaluation,
    Stage2Category,
)


# ============================================================================
# 1. Boundary Score Validation Tests
# ============================================================================

class TestChallengerBoundaryScoreValidation:
    """Stress-tests score validation at Pydantic schema and SQLite DB levels."""

    @pytest.mark.parametrize("invalid_score", [
        0.99, 0.0, -0.01, -1.0, -10.0, -999.0, 10.01, 10.1, 15.0, 100.0
    ])
    def test_breakthrough_score_out_of_bounds_raises_validation_error(self, invalid_score: float) -> None:
        """Scores strictly outside [1.0, 10.0] must be rejected by Pydantic."""
        with pytest.raises(ValidationError) as exc_info:
            Stage2EvaluationResponse(
                breakthrough_score=invalid_score,
                roi_score=7.5,
                category=Stage2Category.CORE,
                relevance_justification="Boundary test",
                decision_card=DecisionCardPayload(
                    tldr="TLDR sentence.",
                    use_case="Use case description.",
                    comparison="Comparison benchmark.",
                    quickstart="docker run tool",
                    hardware_requirements="24GB VRAM",
                    license="Apache-2.0",
                ),
            )
        assert "breakthrough_score" in str(exc_info.value)

    @pytest.mark.parametrize("invalid_score", [
        0.99, 0.0, -0.1, -5.0, 10.01, 11.0, 50.0
    ])
    def test_roi_score_out_of_bounds_raises_validation_error(self, invalid_score: float) -> None:
        """ROI scores strictly outside [1.0, 10.0] must be rejected by Pydantic."""
        with pytest.raises(ValidationError) as exc_info:
            Stage2EvaluationResponse(
                breakthrough_score=8.0,
                roi_score=invalid_score,
                category=Stage2Category.CORE,
                relevance_justification="Boundary test",
                decision_card=DecisionCardPayload(
                    tldr="TLDR sentence.",
                    use_case="Use case description.",
                    comparison="Comparison benchmark.",
                    quickstart="docker run tool",
                    hardware_requirements="24GB VRAM",
                    license="Apache-2.0",
                ),
            )
        assert "roi_score" in str(exc_info.value)

    @pytest.mark.parametrize("non_numeric", [
        "nine", "8.5/10", "10 out of 10", "high", None, float("nan"), float("inf"), float("-inf")
    ])
    def test_non_numeric_and_special_float_scores_rejected(self, non_numeric: Any) -> None:
        """Non-numeric, NaN, and Infinite float values must fail validation."""
        with pytest.raises(ValidationError):
            Stage2EvaluationResponse(
                breakthrough_score=non_numeric,
                roi_score=8.0,
                category=Stage2Category.CORE,
                relevance_justification="Testing non-numeric inputs",
                decision_card=DecisionCardPayload(
                    tldr="TLDR sentence.",
                    use_case="Use case description.",
                    comparison="Comparison benchmark.",
                    quickstart="docker run tool",
                    hardware_requirements="24GB VRAM",
                    license="Apache-2.0",
                ),
            )

    def test_exact_boundaries_accepted(self) -> None:
        """Exact boundaries (1.0 and 10.0) must be accepted."""
        res_min = Stage2EvaluationResponse(
            breakthrough_score=1.0,
            roi_score=1.0,
            category=Stage2Category.DISCARD,
            relevance_justification="Minimum boundary score",
            decision_card=DecisionCardPayload(
                tldr="TLDR.",
                use_case="None.",
                comparison="None.",
                quickstart="None.",
                hardware_requirements="None.",
                license="None.",
            ),
        )
        assert res_min.breakthrough_score == 1.0
        assert res_min.roi_score == 1.0

        res_max = Stage2EvaluationResponse(
            breakthrough_score=10.0,
            roi_score=10.0,
            category=Stage2Category.CORE,
            relevance_justification="Maximum boundary score",
            decision_card=DecisionCardPayload(
                tldr="TLDR.",
                use_case="Use case.",
                comparison="Comparison.",
                quickstart="docker run tool",
                hardware_requirements="24GB VRAM",
                license="MIT",
            ),
        )
        assert res_max.breakthrough_score == 10.0
        assert res_max.roi_score == 10.0

    @pytest.mark.asyncio
    async def test_sqlite_check_constraints_enforce_score_bounds(self, migrated_db: Database) -> None:
        """Direct insertion into SQLite breakthroughs table must fail if score outside [1.0, 10.0]."""
        await migrated_db.execute(
            """INSERT INTO raw_items (id, source, source_id, title, url, raw_content, metadata_json, ingested_at)
            VALUES (9001, 'github_releases', 'tool:9001', 'Test Tool', 'https://github.com/tool/9001', 'Content', '{}', '2026-09-17T12:00:00Z');"""
        )

        base_sql = """
        INSERT INTO breakthroughs (
            raw_item_id, title, repo_url, breakthrough_score, roi_score, category,
            tldr, use_case, comparison, quickstart, hardware_requirements, license,
            card_markdown, evaluation_raw_json, evaluated_at
        ) VALUES (
            9001, 'Test Tool', 'https://github.com/tool/9001', :bt_score, :roi_score, 'core',
            'tldr', 'use_case', 'comparison', 'quickstart', 'hw', 'MIT',
            'markdown', '{}', '2026-09-17T12:00:00Z'
        );
        """

        # 1. Negative score -> CHECK failure
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
            await migrated_db.execute(base_sql, {"bt_score": -1.0, "roi_score": 7.0})

        # 2. Score > 10.0 -> CHECK failure
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
            await migrated_db.execute(base_sql, {"bt_score": 10.5, "roi_score": 7.0})

        # 3. ROI score < 1.0 -> CHECK failure
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
            await migrated_db.execute(base_sql, {"bt_score": 8.0, "roi_score": 0.5})

        # 4. Valid boundaries 1.0 and 10.0 -> Success
        row_id = await migrated_db.execute(base_sql, {"bt_score": 1.0, "roi_score": 10.0})
        assert row_id > 0


# ============================================================================
# 2. Adversarial LLM JSON Output Tests
# ============================================================================

class TestChallengerAdversarialLLMOutputs:
    """Stress-tests clean_and_parse_json and Stage2Evaluator with malformed LLM outputs."""

    def test_markdown_fence_variants(self) -> None:
        """Tests varying code fences: ```json, ```JSON, and raw ``` fences."""
        payload = {"breakthrough_score": 8.5, "roi_score": 8.0, "category": "core"}

        raw_fenced = f"```json\n{json.dumps(payload)}\n```"
        assert clean_and_parse_json(raw_fenced)["breakthrough_score"] == 8.5

        raw_fenced_upper = f"```JSON\n{json.dumps(payload)}\n```"
        assert clean_and_parse_json(raw_fenced_upper)["breakthrough_score"] == 8.5

        raw_no_lang = f"```\n{json.dumps(payload)}\n```"
        assert clean_and_parse_json(raw_no_lang)["breakthrough_score"] == 8.5

    def test_trailing_commas_variations(self) -> None:
        """Tests single and nested trailing commas before } and ]."""
        raw_json = """
        {
            "breakthrough_score": 8.5,
            "roi_score": 8.0,
            "category": "core",
            "tags": ["agent", "rag", ],
            "decision_card": {
                "tldr": "Some sentence.",
                "use_case": "Some use case.",
            },
        }
        """
        parsed = clean_and_parse_json(raw_json)
        assert parsed["breakthrough_score"] == 8.5
        assert parsed["tags"] == ["agent", "rag"]
        assert parsed["decision_card"]["tldr"] == "Some sentence."

    def test_conversational_chatter_wrapping_json(self) -> None:
        """Models often output conversational greeting and sign-off around JSON."""
        raw_text = """
        Hello! I have thoroughly evaluated the candidate repository according to your criteria.
        Here is the evaluation result:

        ```json
        {
            "breakthrough_score": 9.2,
            "roi_score": 8.8,
            "category": "core",
            "relevance_justification": "Direct fit.",
            "decision_card": {
                "tldr": "Tldr.",
                "use_case": "Use case.",
                "comparison": "Comp.",
                "quickstart": "docker run ...",
                "hardware_requirements": "24GB",
                "license": "MIT"
            }
        }
        ```

        Let me know if you would like any follow-up adjustments!
        """
        parsed = clean_and_parse_json(raw_text)
        assert parsed["breakthrough_score"] == 9.2
        assert parsed["category"] == "core"

    def test_nested_markdown_backticks_inside_json_string_vulnerability(self) -> None:
        """Empirically prove that nested backticks inside string fields break clean_and_parse_json.
        LLMs frequently put ```bash or `code` inside 'quickstart' or 'comparison'.
        """
        raw_text = (
            "```json\n"
            "{\n"
            '  "breakthrough_score": 9.0,\n'
            '  "roi_score": 8.5,\n'
            '  "category": "core",\n'
            '  "relevance_justification": "Valid item",\n'
            '  "decision_card": {\n'
            '    "tldr": "TLDR sentence.",\n'
            '    "use_case": "Workflow application.",\n'
            '    "comparison": "Compared with baseline.",\n'
            '    "quickstart": "```bash\\ndocker run -p 8080:8080 vllm:latest\\n```",\n'
            '    "hardware_requirements": "24GB VRAM",\n'
            '    "license": "Apache-2.0"\n'
            "  }\n"
            "}\n"
            "```"
        )
        # Remediated: Outer fence extraction correctly preserves inner code fences inside string fields
        parsed = clean_and_parse_json(raw_text)
        assert parsed["breakthrough_score"] == 9.0
        assert parsed["category"] == "core"
        assert "docker run" in parsed["decision_card"]["quickstart"]

    def test_missing_required_decision_card_field_raises_validation_error(self) -> None:
        """Missing any required card field ('tldr', 'use_case', 'comparison', etc.) must fail."""
        incomplete_data = {
            "breakthrough_score": 8.0,
            "roi_score": 7.5,
            "category": "core",
            "relevance_justification": "Justification",
            "decision_card": {
                # "tldr" is missing!
                "use_case": "Use case",
                "comparison": "Comparison",
                "quickstart": "docker run ...",
                "hardware_requirements": "Hardware",
                "license": "MIT",
            },
        }
        with pytest.raises(ValidationError) as exc:
            Stage2EvaluationResponse.model_validate(incomplete_data)
        assert "tldr" in str(exc.value)

    def test_missing_category_or_invalid_category_raises(self) -> None:
        """Top-level category missing or not in ('core', 'serendipity', 'discard') must fail."""
        invalid_cat_data = {
            "breakthrough_score": 8.0,
            "roi_score": 7.5,
            "category": "hallucinated_category",
            "relevance_justification": "Justification",
            "decision_card": {
                "tldr": "Tldr",
                "use_case": "Use case",
                "comparison": "Comparison",
                "quickstart": "docker run ...",
                "hardware_requirements": "Hardware",
                "license": "MIT",
            },
        }
        with pytest.raises(ValidationError) as exc:
            Stage2EvaluationResponse.model_validate(invalid_cat_data)
        assert "category" in str(exc.value)

    def test_hallucinated_extra_fields_behavior(self) -> None:
        """Verify behavior when LLM adds unexpected extra fields."""
        data_with_extras = {
            "breakthrough_score": 8.0,
            "roi_score": 7.5,
            "category": "core",
            "relevance_justification": "Justification",
            "hallucinated_confidence": 0.99,
            "unsolicited_tags": ["ai", "super_fast"],
            "decision_card": {
                "tldr": "Tldr",
                "use_case": "Use case",
                "comparison": "Comparison",
                "quickstart": "docker run ...",
                "hardware_requirements": "Hardware",
                "license": "MIT",
                "hallucinated_rating": "5_stars",
            },
        }
        response = Stage2EvaluationResponse.model_validate(data_with_extras)
        assert response.breakthrough_score == 8.0
        dumped = response.model_dump()
        assert "hallucinated_confidence" not in dumped
        assert "unsolicited_tags" not in dumped
        assert "hallucinated_rating" not in dumped["decision_card"]

    def test_truncated_json_raises_value_error(self) -> None:
        """Truncated output from token exhaustion must raise ValueError."""
        truncated = '{"breakthrough_score": 8.5, "roi_score": 7.0, "category": "core", "decision_card": {"tldr": "Something'
        with pytest.raises(ValueError, match="Failed to parse model output as JSON"):
            clean_and_parse_json(truncated)

    def test_double_encoded_json_decision_card_fails(self) -> None:
        """LLMs occasionally string-encode nested objects ('{\"tldr\": ...}'). Pydantic must reject."""
        double_encoded = {
            "breakthrough_score": 8.5,
            "roi_score": 8.0,
            "category": "core",
            "relevance_justification": "Good tool",
            "decision_card": '{"tldr": "string instead of dict"}',
        }
        with pytest.raises(ValidationError) as exc:
            Stage2EvaluationResponse.model_validate(double_encoded)
        assert "decision_card" in str(exc.value)


# ============================================================================
# 3. License Boundary Checking Tests
# ============================================================================

class TestChallengerLicenseBoundaryChecking:
    """Stress-tests Stage 1 license detection across non-commercial, BSL, SSPL, and multi-licenses."""

    @pytest.mark.parametrize("license_name", [
        "CC-BY-NC-4.0", "cc-by-nc-4.0", "CC-BY-NC-3.0", "CC-BY-NC-SA-4.0",
        "CC-BY-NC-ND-4.0", "BSL 1.1", "bsl-1.1", "SSPL-1.0", "sspl-1.0",
        "Commons-Clause", "Proprietary", "all-rights-reserved",
    ])
    def test_known_non_commercial_licenses_rejected(self, license_name: str) -> None:
        """Standard non-commercial and source-available licenses must be rejected."""
        filter_engine = Stage1Filter()
        raw_item = RawItem(
            source="github_releases",
            source_id="test/noncomm",
            title="Non-Commercial Model",
            url="https://github.com/test/noncomm",
            metadata={"root_files": ["Dockerfile"], "license": license_name},
        )
        res = filter_engine.evaluate(raw_item)
        assert res.passed is False
        assert "Non-commercial or proprietary license" in (res.drop_reason or "")

    def test_written_out_business_source_license_evasion(self) -> None:
        """Empirically test whether 'Business Source License 1.1' written out evades Stage 1 filter.
        The NON_COMMERCIAL_LICENSES set has 'bsl-1.1', but 'business-source-license-1.1' is not in it.
        """
        filter_engine = Stage1Filter(strict_licenses=False)
        raw_item = RawItem(
            source="github_releases",
            source_id="test/bsl_written",
            title="BSL Database Tool",
            url="https://github.com/test/bsl_written",
            metadata={"root_files": ["Dockerfile"], "license": "Business Source License 1.1"},
        )
        res = filter_engine.evaluate(raw_item)
        assert res.detected_license == "Business Source License 1.1"
        assert res.passed is False
        assert "Non-commercial or proprietary license" in (res.drop_reason or "")

    def test_sspl_without_version_evasion(self) -> None:
        """Empirically test 'SSPL' or 'Server Side Public License' without '-1.0'."""
        filter_engine = Stage1Filter(strict_licenses=False)
        raw_item = RawItem(
            source="github_releases",
            source_id="test/sspl_unversioned",
            title="SSPL Database",
            url="https://github.com/test/sspl_unversioned",
            metadata={"root_files": ["Dockerfile"], "license": "SSPL"},
        )
        res = filter_engine.evaluate(raw_item)
        assert res.passed is False
        assert "Non-commercial or proprietary license" in (res.drop_reason or "")

    def test_creative_commons_noncommercial_spelled_out_evasion(self) -> None:
        """Empirically test 'Creative Commons Attribution-NonCommercial 4.0' without hyphen in NonCommercial."""
        filter_engine = Stage1Filter(strict_licenses=False)
        raw_item = RawItem(
            source="github_releases",
            source_id="test/cc_noncomm",
            title="CC Model",
            url="https://github.com/test/cc_noncomm",
            metadata={"root_files": ["Dockerfile"], "license": "Creative Commons Attribution-NonCommercial 4.0"},
        )
        res = filter_engine.evaluate(raw_item)
        assert res.passed is False
        assert "Non-commercial or proprietary license" in (res.drop_reason or "")

    def test_multi_licensed_permissive_or_permissive(self) -> None:
        """Dual-licensed repos like 'MIT OR Apache-2.0' should not be rejected."""
        filter_engine = Stage1Filter(strict_licenses=False)
        raw_item = RawItem(
            source="github_releases",
            source_id="test/dual_open",
            title="Dual Licensed Tool",
            url="https://github.com/test/dual_open",
            metadata={"root_files": ["Dockerfile"], "license": "MIT OR Apache-2.0"},
        )
        res = filter_engine.evaluate(raw_item)
        assert res.passed is True
        assert res.detected_license == "MIT OR Apache-2.0"

    def test_github_api_license_dict_metadata_defect(self) -> None:
        """GitHub API returns `license` as a dict: `{'key': 'mit', 'name': 'MIT License', 'spdx_id': 'MIT'}`.
        Empirically verify that when only `license` dict is provided (no top-level spdx_id),
        Stage1Filter successfully extracts the license string and instantiates Stage1Evaluation cleanly.
        """
        filter_engine = Stage1Filter(strict_licenses=True)
        # Case A: spdx_id provided directly in metadata
        raw_item_a = RawItem(
            source="github_releases",
            source_id="test/dict_license_a",
            title="Tool A",
            url="https://github.com/test/dict_license_a",
            metadata={"root_files": ["Dockerfile"], "spdx_id": "MIT", "license": {"name": "MIT"}},
        )
        res_a = filter_engine.evaluate(raw_item_a)
        assert res_a.passed is True
        assert res_a.detected_license == "MIT"

        # Case B: Only `license` dict is provided without top-level spdx_id
        raw_item_b = RawItem(
            source="github_releases",
            source_id="test/dict_license_b",
            title="Tool B",
            url="https://github.com/test/dict_license_b",
            metadata={"root_files": ["Dockerfile"], "license": {"key": "mit", "spdx_id": "MIT"}},
        )
        # Remediated: check_license extracts spdx_id string from dict, so evaluation succeeds cleanly
        res_b = filter_engine.evaluate(raw_item_b)
        assert res_b.passed is True
        assert res_b.detected_license == "MIT"

    def test_strict_licenses_drops_unapproved_or_missing(self) -> None:
        """When strict_licenses=True, missing or unknown licenses must be dropped."""
        strict_filter = Stage1Filter(strict_licenses=True)

        # Missing license
        raw_no_lic = RawItem(
            source="github_releases",
            source_id="test/no_lic",
            title="No Lic",
            url="https://github.com/test/no_lic",
            metadata={"root_files": ["Dockerfile"]},
        )
        assert strict_filter.evaluate(raw_no_lic).passed is False

        # Unknown license
        raw_unk_lic = RawItem(
            source="github_releases",
            source_id="test/unk_lic",
            title="Unk Lic",
            url="https://github.com/test/unk_lic",
            metadata={"root_files": ["Dockerfile"], "license": "Custom-Foo-License"},
        )
        assert strict_filter.evaluate(raw_unk_lic).passed is False


# ============================================================================
# 4. Anti-Hype Blacklist Evasion Tests
# ============================================================================

class TestChallengerAntiHypeBlacklistEvasion:
    """Stress-tests Stage 1 blacklist filters against obfuscation, spacing, and subtle marketing."""

    def test_direct_saas_wrapper_rejected(self) -> None:
        """Standard 'ChatGPT wrapper' or 'thin UI for Claude' is rejected."""
        filter_engine = Stage1Filter()
        raw_item = RawItem(
            source="github_releases",
            source_id="test/wrap1",
            title="Thin UI for OpenAI API",
            url="https://github.com/test/wrap1",
            metadata={"root_files": ["Dockerfile"], "license": "MIT"},
        )
        res = filter_engine.evaluate(raw_item)
        assert res.passed is False
        assert "saas_wrapper" in (res.drop_reason or "")

    def test_obfuscated_saas_boilerplate_evasion(self) -> None:
        """Empirically test whether 'AI SaaS Boilerplate' or 'S-A-A-S Starter Kit' evades Stage 1.
        The saas_wrapper patterns only match '(chatgpt|openai|claude) wrapper', 'wrapper app', etc.
        They do NOT match 'SaaS' alone or 'S-A-A-S'!
        """
        filter_engine = Stage1Filter()
        raw_item = RawItem(
            source="github_releases",
            source_id="test/saas_boiler",
            title="NextGen AI SaaS Boilerplate Starter Kit",
            url="https://github.com/test/saas_boiler",
            raw_content="Build profitable AI SaaS apps in minutes with this starter template.",
            metadata={"root_files": ["Dockerfile"], "license": "MIT"},
        )
        res = filter_engine.evaluate(raw_item)
        assert res.passed is False
        assert "saas_wrapper" in (res.drop_reason or "")

    def test_no_code_with_space_evasion(self) -> None:
        """Empirically test whether 'no code website builder' with a space evades Stage 1.
        Regex is: r'\b(?:no[\s-]code|nocode|low-code)\s+(?:website|landing\s+page|web\s+app)\s+builder\b'
        """
        filter_engine = Stage1Filter()
        raw_item = RawItem(
            source="github_releases",
            source_id="test/nocode_space",
            title="AI Powered No Code Website Builder",
            url="https://github.com/test/nocode_space",
            raw_content="Visual drag and drop builder for everyone.",
            metadata={"root_files": ["Dockerfile"], "license": "MIT"},
        )
        res = filter_engine.evaluate(raw_item)
        assert res.passed is False
        assert "no_code_builder" in (res.drop_reason or "")

    def test_dropshipping_ai_evasion(self) -> None:
        """Empirically test whether 'drop-shipping ai' or 'dropshipping automation' evades Stage 1."""
        filter_engine = Stage1Filter()
        raw_item = RawItem(
            source="github_releases",
            source_id="test/dropship",
            title="Autonomous Drop-Shipping AI Store Automation",
            url="https://github.com/test/dropship",
            raw_content="Automate your dropshipping store products and fulfillment.",
            metadata={"root_files": ["Dockerfile"], "license": "MIT"},
        )
        res = filter_engine.evaluate(raw_item)
        assert res.passed is False
        assert "no_code_builder" in (res.drop_reason or "")

    def test_subtle_ai_landing_page_generator_evasion(self) -> None:
        """Empirically test whether 'AI Landing Page Generator' evades Stage 1."""
        filter_engine = Stage1Filter()
        raw_item = RawItem(
            source="github_releases",
            source_id="test/lp_gen",
            title="AI Landing Page Generator with Tailwind",
            url="https://github.com/test/lp_gen",
            raw_content="Generate landing pages automatically from a prompt.",
            metadata={"root_files": ["Dockerfile"], "license": "MIT"},
        )
        res = filter_engine.evaluate(raw_item)
        assert res.passed is False
        assert "no_code_builder" in (res.drop_reason or "")

    def test_blacklisted_domain_and_subdomain_enforcement(self) -> None:
        """Blacklisted domains and their subdomains must be rejected."""
        filter_engine = Stage1Filter()

        # Exact domain
        raw_item_exact = RawItem(
            source="github_releases",
            source_id="test/domain1",
            title="Crypto Tool",
            url="https://pump.fun/coin/123",
            metadata={"root_files": ["Dockerfile"], "license": "MIT"},
        )
        assert filter_engine.evaluate(raw_item_exact).passed is False

        # Subdomain
        raw_item_sub = RawItem(
            source="github_releases",
            source_id="test/domain2",
            title="Storefront",
            url="https://cool-ai-store.myshopify.com/products/template",
            metadata={"root_files": ["Dockerfile"], "license": "MIT"},
        )
        assert filter_engine.evaluate(raw_item_sub).passed is False

    def test_false_positive_prevention_cryptography(self) -> None:
        """Legitimate low-level cryptography and microcontrollers must not be false positives."""
        filter_engine = Stage1Filter()
        raw_item = RawItem(
            source="github_releases",
            source_id="test/crypto_lib",
            title="Fast Elliptic Curve Cryptography Library",
            url="https://github.com/test/crypto_lib",
            raw_content="Zero-dependency constant-time C++ cryptographic primitives.",
            metadata={"root_files": ["CMakeLists.txt"], "license": "Apache-2.0"},
        )
        res = filter_engine.evaluate(raw_item)
        assert res.passed is True
        assert res.has_runnable_code is True

    def test_non_string_root_files_metadata_defect(self) -> None:
        """Empirically test whether non-string elements in metadata['root_files'] (e.g. None or int)
        are handled gracefully without AttributeError: 'NoneType' object has no attribute 'lower'.
        """
        filter_engine = Stage1Filter()
        raw_item = RawItem(
            source="github_releases",
            source_id="test/none_root_files",
            title="Corrupted Metadata Repo",
            url="https://github.com/test/none_root_files",
            metadata={"root_files": [None, "Dockerfile"], "license": "MIT"},
        )
        # Remediated: root_files list comprehension filters non-string/None elements
        res = filter_engine.evaluate(raw_item)
        assert res.passed is True
        assert res.has_docker is True


# ============================================================================
# 5. Concurrent Stage 1 & Stage 2 High-Throughput WAL Concurrency Tests
# ============================================================================

class TestChallengerHighThroughputWALConcurrency:
    """Stress-tests SQLite WAL under high-throughput concurrent Stage 1 and Stage 2 workloads."""

    @pytest.mark.asyncio
    async def test_concurrent_stage1_and_stage2_evaluations_under_load(
        self, migrated_db: Database, test_settings: Settings
    ) -> None:
        """Spawns 50 concurrent tasks performing interleaved Stage 1 and Stage 2 operations:
        - Inserting raw_items
        - Saving Stage 1 evaluations
        - Evaluating and persisting Stage 2 breakthroughs
        - Performing concurrent reads
        Verifies zero SQLite lock contention (OperationalError) and 100% data integrity.
        """
        num_items = 40
        filter_engine = Stage1Filter(db=migrated_db)
        evaluator = Stage2Evaluator(db=migrated_db, settings=test_settings)

        # 1. Concurrently insert 40 raw items
        async def _insert_raw(i: int) -> int:
            is_core = (i % 2 == 0)
            raw = RawItem(
                source="github_releases",
                source_id=f"org/high-concurrency-repo-{i}:v1",
                title=f"vLLM Concurrency Runner {i}" if is_core else f"Robot Sensor Node {i}",
                url=f"https://github.com/org/high-concurrency-repo-{i}",
                raw_content="vLLM NVFP4 CUDA kernels" if is_core else "ROS2 robotics control loop",
                metadata={"root_files": ["Dockerfile", "pyproject.toml"], "license": "MIT"},
            )
            return await migrated_db.execute(
                """INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at)
                VALUES (:source, :source_id, :title, :url, :raw_content, :metadata_json, :ingested_at);""",
                raw.to_db_params(),
            )

        insert_tasks = [_insert_raw(i) for i in range(1, num_items + 1)]
        raw_item_ids = await asyncio.gather(*insert_tasks)
        assert len(raw_item_ids) == num_items

        # Fetch all raw items from DB
        raw_rows = await migrated_db.fetch_all("SELECT * FROM raw_items ORDER BY id ASC;")
        assert len(raw_rows) == num_items
        raw_items = [RawItem.from_row(r) for r in raw_rows]

        # 2. Concurrently execute Stage 1 evaluations, Stage 2 evaluations, and concurrent reader queries
        async def _run_stage1_worker(item: RawItem) -> Stage1Evaluation:
            eval_res = filter_engine.evaluate(item)
            eval_res.raw_item_id = item.id or 0
            await filter_engine.save_evaluation(eval_res)
            return eval_res

        async def _run_stage2_worker(item: RawItem, s1: Stage1Evaluation) -> Optional[Breakthrough]:
            await asyncio.sleep(0.01)
            return await evaluator.evaluate_and_persist(item, s1)

        async def _run_reader_worker(interval: float, iterations: int = 15) -> int:
            total_read = 0
            for _ in range(iterations):
                await asyncio.sleep(interval)
                rows = await migrated_db.fetch_all(
                    "SELECT COUNT(*) as cnt FROM stage1_evaluations UNION ALL SELECT COUNT(*) FROM breakthroughs;"
                )
                total_read += sum(r[0] for r in rows)
            return total_read

        # Start readers in background
        reader_task1 = asyncio.create_task(_run_reader_worker(0.015, 20))
        reader_task2 = asyncio.create_task(_run_reader_worker(0.025, 15))

        # Run Stage 1 evaluations concurrently
        s1_tasks = [_run_stage1_worker(item) for item in raw_items]
        s1_results = await asyncio.gather(*s1_tasks)
        assert len(s1_results) == num_items

        # Run Stage 2 evaluations concurrently
        s2_tasks = [_run_stage2_worker(raw_items[i], s1_results[i]) for i in range(num_items)]
        s2_results = await asyncio.gather(*s2_tasks)

        # Wait for readers to finish
        reads1 = await reader_task1
        reads2 = await reader_task2
        assert reads1 > 0
        assert reads2 > 0

        # 3. Verify Database Integrity and Row Counts
        s1_count = await migrated_db.fetch_val("SELECT COUNT(*) FROM stage1_evaluations;")
        assert s1_count == num_items

        bt_count = await migrated_db.fetch_val("SELECT COUNT(*) FROM breakthroughs;")
        assert bt_count == num_items

        # 4. Idempotency test under concurrency: Re-running evaluate_and_persist on all 40 concurrently
        re_s2_tasks = [_run_stage2_worker(raw_items[i], s1_results[i]) for i in range(num_items)]
        re_s2_results = await asyncio.gather(*re_s2_tasks)
        assert len(re_s2_results) == num_items

        bt_count_after = await migrated_db.fetch_val("SELECT COUNT(*) FROM breakthroughs;")
        assert bt_count_after == num_items, "Concurrent re-evaluation must maintain exact count (UPSERT idempotency)"
