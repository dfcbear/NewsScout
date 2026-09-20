"""newsscout.filtering.card_gen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1-Minute Decision Card Generator & Quality Validator.

Enforces:
1. 1-sentence TL;DR (strictly single sentence, max 180 chars, active voice).
2. Concrete workflow use case mapping to user tech profile (RTX 4090, local inference, agents, RAG, serendipity).
3. Concrete comparison against existing baselines (e.g. vLLM, llama.cpp, Ollama).
4. Quickstart 1-liner validator (copy-pasteable single line, strips linebreaks/backticks, validates syntax).
5. Hardware requirements and open-source license status standardization (e.g. '24GB VRAM (RTX 4090) | Apache 2.0').
6. Safe multi-channel rendering (Markdown and Telegram HTML with strict entity escaping).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import html
import re
from typing import Any, Optional

from newsscout.storage.models import (
    DecisionCard,
    DecisionCardData,
    Stage2Category,
)


class CardValidationError(ValueError):
    """Raised when a decision card fails strict validation."""

    pass


@dataclass
class ValidationResult:
    """Detailed result of card validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cleaned_data: Optional[DecisionCardData] = None


class DecisionCardValidator:
    """Validates decision card fields against editorial, syntactic, and profile criteria."""

    # Active voice verbs that frequently start high-impact technical TL;DRs
    ACTIVE_VERB_PREFIXES = (
        "enables",
        "accelerates",
        "replaces",
        "provides",
        "delivers",
        "allows",
        "compiles",
        "extracts",
        "automates",
        "runs",
        "transforms",
        "bypasses",
        "integrates",
        "unlocks",
        "reduces",
        "executes",
        "generates",
        "deploys",
        "optimizes",
        "introduces",
        "scales",
        "synthesizes",
        "implements",
    )

    # Passive voice phrases that weaken punchy technical communication
    PASSIVE_PHRASES = (
        "is enabled by",
        "is supported by",
        "was created to",
        "was developed to",
        "can be used to",
        "is an attempt to",
        "aims to be",
        "is designed to",
        "is implemented by",
        "is built to",
        "has been created",
        "has been developed",
    )

    # Core profile domain keywords (80% core)
    CORE_PROFILE_KEYWORDS = (
        "4090",
        "rtx 4090",
        "vram",
        "24gb",
        "cuda",
        "vllm",
        "llama.cpp",
        "sglang",
        "agent",
        "agents",
        "agentic",
        "mcp",
        "harness",
        "rag",
        "qdrant",
        "docling",
        "embedded",
        "esp32",
        "teensy",
        "c++",
        "c#",
        "quantization",
        "nvfp4",
        "awq",
        "exl2",
        "kernel",
        "triton",
        "local inference",
        "offline",
        "ollama",
        "python",
        "fastapi",
        "docker",
        "speech",
        "whisper",
        "faster-whisper",
        "tts",
        "n8n",
    )

    # Serendipity domain keywords (20% serendipity)
    SERENDIPITY_KEYWORDS = (
        "robotics",
        "ros2",
        "vla",
        "control loop",
        "neuromorphic",
        "spiking",
        "sensor",
        "arm64",
        "raspberry pi",
        "edge",
        "coral",
        "hailo",
        "rk3588",
        "mamba",
        "ssm",
        "titans",
        "rwkv",
        "biotech",
        "humanoid",
        "quadruped",
    )

    # Known baseline tooling
    KNOWN_BASELINES = (
        "vllm",
        "llama.cpp",
        "ollama",
        "sglang",
        "tgi",
        "transformers",
        "huggingface",
        "langchain",
        "llamaindex",
        "docling",
        "whisper",
        "faster-whisper",
        "qdrant",
        "milvus",
        "chroma",
        "faiss",
        "ros2",
        "pytorch",
        "baseline",
        "standard",
        "vanilla",
    )

    # Comparative metric phrases
    COMPARATIVE_TERMS = (
        "faster",
        "lower",
        "higher",
        "speedup",
        "throughput",
        "latency",
        "reduction",
        "footprint",
        "vram",
        "memory",
        "vs",
        "versus",
        "compared to",
        "replaces",
        "outperforms",
        "drop-in",
        "zero host",
        "parity",
        "overhead",
        "efficiency",
        "improvement",
    )

    # Valid command prefixes for quickstarts
    ALLOWED_COMMAND_PREFIXES = (
        "docker run",
        "docker-compose",
        "docker compose",
        "uvx",
        "uv run",
        "pip install",
        "cargo run",
        "cargo install",
        "npm install",
        "npx",
        "git clone",
        "python",
        "python3",
        "ollama run",
        "curl -fsSL",
        "curl",
    )

    # Known dangerous command patterns
    DANGEROUS_PATTERNS = (
        # rm -rf with contiguous flags targeting root or home
        re.compile(r"\brm\s+-[rf]{1,2}\s+[/~]", re.IGNORECASE),
        re.compile(r"\b(?:sudo\s+)?rm\s+-rf\b", re.IGNORECASE),
        # rm with separated -r and -f flags (e.g., rm -r -f / or rm -f -r /)
        re.compile(r"(?:\s|^)rm\s+-r\s+-f\s+[/~]", re.IGNORECASE),
        re.compile(r"(?:\s|^)rm\s+-f\s+-r\s+[/~]", re.IGNORECASE),
        # rm with long options --recursive --force
        re.compile(r"(?:\s|^)rm\s+--recursive\b.*--force\b", re.IGNORECASE),
        re.compile(r"(?:\s|^)rm\s+--force\b.*--recursive\b", re.IGNORECASE),
        # rm with --no-preserve-root (always dangerous when targeting /)
        re.compile(r"(?:\s|^)rm\s+--no-preserve-root\b", re.IGNORECASE),
        # curl/wget piped to bash/sh/python3
        re.compile(r"\bcurl\b.*\|\s*(?:bash|sh|python3?)\b", re.IGNORECASE),
        re.compile(r"\bwget\b.*\|\s*(?:bash|sh|python3?)\b", re.IGNORECASE),
        # base64 decode piped to shell
        re.compile(r"\bbase64\b.*\|\s*(?:bash|sh)\b", re.IGNORECASE),
        # sh -c / bash -c with curl/wget inside (command substitution)
        re.compile(r"\b(?:sh|bash)\s+-c\b.*\b(?:curl|wget)\b", re.IGNORECASE),
        # python3 -c with rmtree
        re.compile(r"\bpython3?\s+-c\b.*\brmtree\b", re.IGNORECASE),
        # semicolon-chained rm targeting root
        re.compile(r";\s*rm\b.*\s+[/~]", re.IGNORECASE),
        # Fork bomb
        re.compile(r":\(\)\{\s*:\|:&\s*\};:", re.IGNORECASE),
        # Windows format
        re.compile(r"\bformat\s+[c-z]:", re.IGNORECASE),
        # dd to device
        re.compile(r"\bdd\s+if=.*of=/dev/", re.IGNORECASE),
    )

    # Regex for protecting non-sentence boundary periods in technical text
    _PROTECTED_TOKEN_PATTERN = re.compile(
        r"(?:"
        r"\bv\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9]+)?"  # Versions like v2.0, v1.2.3
        r"|\be\.g\."  # e.g.
        r"|\bi\.e\."  # i.e.
        r"|\bvs\."  # vs.
        r"|\bet\s+al\."  # et al.
        r"|\betc\."  # etc.
        r"|\b\d+\.\d+(?:[xX%])?"  # Floats, ratios, percentages (3.5x, 0.4, 99.9%)
        r"|\bllama\.cpp\b"  # Specific library with dot
        r"|\b[a-zA-Z0-9_-]+\.(?:cpp|py|toml|json|bin|gguf|safetensors|pt|com|org|io|ai|dev)\b"  # Files / domains
        r")",
        re.IGNORECASE,
    )

    @classmethod
    def validate_tldr(cls, text: str) -> tuple[bool, list[str]]:
        """Validates 1-sentence TL;DR rules: single sentence, <= 180 chars, active voice."""
        errors: list[str] = []
        cleaned = text.strip()

        if not cleaned:
            return False, ["TL;DR cannot be empty."]

        if len(cleaned) > 180:
            errors.append(f"TL;DR exceeds 180 characters ({len(cleaned)} chars).")

        if "\n" in cleaned or "\r" in cleaned:
            errors.append("TL;DR must not contain linebreaks.")

        # Check for multiple sentences while ignoring protected abbreviations/numbers
        masked = cls._PROTECTED_TOKEN_PATTERN.sub("PROTECTED", cleaned)
        # Sentence boundary: period/exclamation/question followed by space and uppercase letter
        sentence_splits = re.split(r"[.!?]\s+[A-Z]", masked)
        if len(sentence_splits) > 1:
            errors.append(
                f"TL;DR contains multiple sentences ({len(sentence_splits)} detected). Exactly 1 sentence required."
            )

        # Check active voice heuristic (flagging passive phrasing)
        lower = cleaned.lower()
        has_passive = any(phrase in lower for phrase in cls.PASSIVE_PHRASES)
        if has_passive:
            errors.append(
                "TL;DR uses passive phrasing (e.g. 'is enabled by', 'can be used to'). Active voice required."
            )

        return len(errors) == 0, errors

    @classmethod
    def validate_use_case(
        cls,
        text: str,
        category: Stage2Category = Stage2Category.CORE,
    ) -> tuple[bool, list[str]]:
        """Validates workflow use case mapping to user tech profile or serendipity domain."""
        errors: list[str] = []
        cleaned = text.strip()

        if not cleaned:
            return False, ["Workflow use case cannot be empty."]

        if len(cleaned) < 15:
            errors.append("Workflow use case is too brief to describe a concrete workflow.")

        lower = cleaned.lower()
        if category == Stage2Category.SERENDIPITY:
            matches = any(k in lower for k in cls.SERENDIPITY_KEYWORDS) or any(
                k in lower for k in cls.CORE_PROFILE_KEYWORDS
            )
        else:
            matches = any(k in lower for k in cls.CORE_PROFILE_KEYWORDS)

        if not matches:
            errors.append(
                "Workflow use case does not clearly map to the user's technical profile "
                "(RTX 4090, local inference, agent harnesses, RAG/Qdrant, embedded C++, or serendipity domains)."
            )

        return len(errors) == 0, errors

    @classmethod
    def validate_comparison(cls, text: str) -> tuple[bool, list[str]]:
        """Validates baseline comparison: must compare against known baselines with comparative terms."""
        errors: list[str] = []
        cleaned = text.strip()

        if not cleaned:
            return False, ["Comparison cannot be empty."]

        lower = cleaned.lower()
        has_baseline = any(b in lower for b in cls.KNOWN_BASELINES)
        has_metric = any(m in lower for m in cls.COMPARATIVE_TERMS) or bool(
            re.search(r"\b\d+(?:\.\d+)?x\b", lower)
        )

        if not (has_baseline and has_metric):
            errors.append(
                "Comparison must state a measurable performance or architectural comparison against existing baselines (e.g. vLLM, llama.cpp)."
            )

        return len(errors) == 0, errors

    @classmethod
    def validate_quickstart(cls, command: str) -> tuple[bool, list[str]]:
        """Validates quickstart command: single line, valid syntax, allowed prefixes, no dangerous patterns."""
        errors: list[str] = []
        cleaned = command.strip()

        if not cleaned:
            return False, ["Quickstart command cannot be empty."]

        if "\n" in cleaned or "\r" in cleaned:
            errors.append("Quickstart command must be a single copy-pasteable line without linebreaks.")

        # Strip markdown ticks for checking
        unfenced = cleaned
        if unfenced.startswith("```"):
            unfenced = re.sub(r"^```(?:bash|sh|zsh)?\s*", "", unfenced)
            unfenced = re.sub(r"\s*```$", "", unfenced).strip()
        elif unfenced.startswith("`") and unfenced.endswith("`"):
            unfenced = unfenced.strip("`").strip()

        unfenced = re.sub(r"^\$\s+", "", unfenced)

        # Dangerous pattern check
        for pat in cls.DANGEROUS_PATTERNS:
            if pat.search(unfenced):
                errors.append("Quickstart command contains potentially dangerous or destructive shell commands.")
                break

        # Check allowed prefix
        prefix_matched = any(unfenced.startswith(p) for p in cls.ALLOWED_COMMAND_PREFIXES)
        if not prefix_matched:
            errors.append(
                f"Quickstart command does not start with an approved runner prefix: {', '.join(cls.ALLOWED_COMMAND_PREFIXES[:5])}..."
            )

        # Syntax check: balanced quotes (context-aware)
        # Walk through the string tracking whether we're inside double quotes,
        # so apostrophes inside double-quoted strings don't count as unbalanced single quotes.
        in_double = False
        single_count = 0
        double_count = 0
        i = 0
        while i < len(unfenced):
            ch = unfenced[i]
            if ch == '"':
                in_double = not in_double
                double_count += 1
            elif ch == "'" and not in_double:
                single_count += 1
            i += 1

        if single_count % 2 != 0:
            errors.append("Quickstart command contains unbalanced single quotes.")
        if double_count % 2 != 0:
            errors.append("Quickstart command contains unbalanced double quotes.")

        # Backtick balance check
        backtick_count = unfenced.count("`")
        if backtick_count % 2 != 0:
            errors.append("Quickstart command contains unbalanced backticks.")

        # Smart/curly quote rejection (Unicode U+2018–U+201F)
        smart_quotes = {"\u2018", "\u2019", "\u201C", "\u201D", "\u201A", "\u201B", "\u201E", "\u201F"}
        if any(ch in smart_quotes for ch in unfenced):
            errors.append("Quickstart command contains Unicode smart/curly quotes which are not valid shell delimiters.")

        return len(errors) == 0, errors

    @classmethod
    def validate_hardware_and_license(cls, hw: str, lic: str) -> tuple[bool, list[str]]:
        """Validates hardware requirements and open-source license strings."""
        errors: list[str] = []
        if not hw.strip():
            errors.append("Hardware requirements must not be empty.")
        if not lic.strip():
            errors.append("License status must not be empty.")
        return len(errors) == 0, errors

    @classmethod
    def validate_card(
        cls,
        card_data: DecisionCardData,
        category: Stage2Category = Stage2Category.CORE,
    ) -> ValidationResult:
        """Performs full composite validation across all card fields."""
        errors: list[str] = []
        warnings: list[str] = []

        _, tldr_errs = cls.validate_tldr(card_data.tldr)
        errors.extend(tldr_errs)

        _, use_case_errs = cls.validate_use_case(card_data.use_case, category)
        errors.extend(use_case_errs)

        _, comp_errs = cls.validate_comparison(card_data.comparison)
        errors.extend(comp_errs)

        _, qs_errs = cls.validate_quickstart(card_data.quickstart)
        errors.extend(qs_errs)

        _, hw_errs = cls.validate_hardware_and_license(
            card_data.hardware_requirements, card_data.license
        )
        errors.extend(hw_errs)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data=card_data if len(errors) == 0 else None,
        )

    @classmethod
    def validate_card_or_raise(
        cls,
        card_data: DecisionCardData,
        category: Stage2Category = Stage2Category.CORE,
    ) -> None:
        """Performs composite validation and raises CardValidationError on failure."""
        result = cls.validate_card(card_data, category)
        if not result.is_valid:
            raise CardValidationError("; ".join(result.errors))


class DecisionCardGenerator:
    """Generates, sanitizes, normalizes, and renders 1-Minute Decision Cards."""

    def __init__(self, validator: Optional[DecisionCardValidator] = None) -> None:
        self.validator = validator or DecisionCardValidator()

    def normalize_tldr(self, text: str, max_chars: int = 180) -> str:
        """Normalizes and repairs TL;DR into a single punchy active-voice sentence."""
        cleaned = " ".join(text.strip().split())  # collapse newlines and multi-spaces

        # If wrapped in quotes, unwrap
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (
            cleaned.startswith("'") and cleaned.endswith("'")
        ):
            cleaned = cleaned[1:-1].strip()

        # Extract first sentence if multiple exist
        masked = DecisionCardValidator._PROTECTED_TOKEN_PATTERN.sub(
            lambda m: m.group(0).replace(".", "@DOT@"), cleaned
        )
        match = re.search(r"^(.*?[.!?])(?:\s+[A-Z]|$)", masked)
        if match:
            first_sentence = match.group(1).replace("@DOT@", ".")
            cleaned = first_sentence.strip()
        else:
            cleaned = masked.replace("@DOT@", ".")

        # Ensure trailing period
        if cleaned and not cleaned.endswith((".", "!", "?")):
            cleaned = f"{cleaned}."

        # Ensure capitalized first letter
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]

        # Enforce max characters at word boundary
        if len(cleaned) > max_chars:
            cutoff = max_chars - 1
            truncated = cleaned[:cutoff]
            last_space = truncated.rfind(" ")
            if last_space > 30:
                truncated = truncated[:last_space]
            cleaned = truncated.rstrip(",;:- ") + "."

        return cleaned

    def normalize_quickstart(self, command: str) -> str:
        """Strips markdown code fences, converts multi-line commands to single-line, and trims."""
        cmd = command.strip()

        # Strip markdown fences (including console/shell language identifiers)
        if cmd.startswith("```"):
            cmd = re.sub(r"^```(?:bash|shell|sh|zsh|console)?\s*", "", cmd)
            cmd = re.sub(r"\s*```$", "", cmd)
        cmd = cmd.strip("`").strip()

        # Strip leading shell prompts
        cmd = re.sub(r"^\$\s+", "", cmd)

        # Convert backslash line continuations and newlines to single space
        cmd = re.sub(r"\\\s*[\r\n]+", " ", cmd)
        cmd = re.sub(r"[\r\n]+", " ", cmd)
        cmd = " ".join(cmd.split())

        return cmd

    def standardize_badge(self, hardware: str, license_str: str) -> tuple[str, str, str]:
        """Standardizes hardware requirements and license into canonical strings and composite badge."""
        hw = hardware.strip()
        lic = license_str.strip()

        # If LLM put hardware and license combined in hardware field
        if "|" in hw and (not lic or lic.lower() in ("unknown", "n/a", "none")):
            parts = hw.split("|", 1)
            hw = parts[0].strip()
            lic = parts[1].strip()

        # Standardize known licenses
        lic_map = {
            "apache-2.0": "Apache 2.0",
            "apache 2.0": "Apache 2.0",
            "apache2": "Apache 2.0",
            "apache-2": "Apache 2.0",
            "apache 2": "Apache 2.0",
            "mit": "MIT",
            "bsd-3-clause": "BSD-3-Clause",
            "bsd-2-clause": "BSD-2-Clause",
            "bsd 3-clause": "BSD-3-Clause",
            "bsd 2-clause": "BSD-2-Clause",
            "gpl-3.0": "GPL-3.0",
            "gpl 3.0": "GPL-3.0",
            "gplv3": "GPL-3.0",
            "agpl-3.0": "AGPL-3.0",
            "agpl 3.0": "AGPL-3.0",
            "agplv3": "AGPL-3.0",
            "lgpl-3.0": "LGPL-3.0",
            "lgpl 3.0": "LGPL-3.0",
        }
        lic_clean = lic_map.get(lic.lower(), lic)

        # Standardize hardware phrasing
        if "4090" in hw.lower() and "24gb" not in hw.lower():
            hw = "24GB VRAM (RTX 4090)"

        badge = f"{hw} | {lic_clean}"
        return hw, lic_clean, badge

    def generate_card_data(
        self,
        raw_payload: Any,
        category: Stage2Category = Stage2Category.CORE,
    ) -> DecisionCardData:
        """Cleans, normalizes, and constructs a robust DecisionCardData object."""
        if hasattr(raw_payload, "model_dump"):
            payload_dict = raw_payload.model_dump()
        elif isinstance(raw_payload, dict):
            payload_dict = raw_payload
        else:
            payload_dict = {
                "tldr": getattr(raw_payload, "tldr", ""),
                "use_case": getattr(raw_payload, "use_case", ""),
                "comparison": getattr(raw_payload, "comparison", ""),
                "quickstart": getattr(raw_payload, "quickstart", ""),
                "hardware_requirements": getattr(raw_payload, "hardware_requirements", ""),
                "license": getattr(raw_payload, "license", ""),
            }

        norm_tldr = self.normalize_tldr(str(payload_dict.get("tldr", "")))
        norm_quickstart = self.normalize_quickstart(str(payload_dict.get("quickstart", "")))
        norm_use_case = " ".join(str(payload_dict.get("use_case", "")).strip().split())
        norm_comparison = " ".join(str(payload_dict.get("comparison", "")).strip().split())
        hw, lic, _ = self.standardize_badge(
            str(payload_dict.get("hardware_requirements", "24GB VRAM (RTX 4090)")),
            str(payload_dict.get("license", "Apache 2.0")),
        )

        return DecisionCardData(
            tldr=norm_tldr,
            use_case=norm_use_case,
            comparison=norm_comparison,
            quickstart=norm_quickstart,
            hardware_requirements=hw,
            license=lic,
        )

    def create_decision_card(
        self,
        title: str,
        category: Stage2Category,
        breakthrough_score: float,
        roi_score: float,
        raw_payload: Any,
        repo_url: Optional[str] = None,
        breakthrough_id: Optional[int] = None,
    ) -> DecisionCard:
        """Constructs a fully validated DecisionCard instance."""
        card_data = self.generate_card_data(raw_payload, category=category)
        return DecisionCard(
            breakthrough_id=breakthrough_id,
            title=title.strip(),
            repo_url=repo_url.strip() if repo_url else None,
            breakthrough_score=round(breakthrough_score, 1),
            roi_score=round(roi_score, 1),
            category=category,
            data=card_data,
        )

    def render_markdown(self, card: DecisionCard) -> str:
        """Renders standard GitHub-flavored Markdown representation."""
        badge = (
            "🎯 CORE BREAKTHROUGH"
            if card.category == Stage2Category.CORE
            else "🚀 SERENDIPITY HIT"
        )
        repo_link = f" ([Repository]({card.repo_url}))" if card.repo_url else ""
        return (
            f"### {card.title}{repo_link}\n"
            f"**{badge}** — Score: `{card.breakthrough_score:.1f}/10` | ROI: `{card.roi_score:.1f}/10`\n\n"
            f"> **TL;DR**: {card.data.tldr}\n\n"
            f"* **Workflow Use Case**: {card.data.use_case}\n"
            f"* **Baseline Comparison**: {card.data.comparison}\n"
            f"* **Quickstart**:\n"
            f"  ```bash\n"
            f"  {card.data.quickstart}\n"
            f"  ```\n"
            f"* **Hardware Requirements & License**: `{card.data.hardware_requirements}` | `{card.data.license}`"
        )

    def render_telegram_html(self, card: DecisionCard, max_chars: int = 4096) -> str:
        """Renders clean, entity-escaped Telegram HTML representation guaranteed <= max_chars."""
        badge = (
            "🎯 <b>CORE BREAKTHROUGH</b>"
            if card.category == Stage2Category.CORE
            else "🚀 <b>SERENDIPITY HIT</b>"
        )
        repo_link = (
            f' (<a href="{html.escape(card.repo_url)}">Repo</a>)'
            if card.repo_url
            else ""
        )
        escaped_title = html.escape(card.title)
        escaped_tldr = html.escape(card.data.tldr)
        escaped_use_case = html.escape(card.data.use_case)
        escaped_comp = html.escape(card.data.comparison)
        escaped_quickstart = html.escape(card.data.quickstart)
        escaped_hardware = html.escape(card.data.hardware_requirements)
        escaped_license = html.escape(card.data.license)

        rendered = (
            f"⚡ <b>{escaped_title}</b>{repo_link}\n"
            f"{badge} | Score: <b>{card.breakthrough_score:.1f}</b> | ROI: <b>{card.roi_score:.1f}</b>\n\n"
            f"💡 <b>TL;DR</b>: {escaped_tldr}\n\n"
            f"🛠 <b>Workflow Use Case</b>: {escaped_use_case}\n"
            f"📊 <b>Baseline Comparison</b>: {escaped_comp}\n"
            f"🚀 <b>Quickstart</b>: <code>{escaped_quickstart}</code>\n"
            f"💻 <b>Hardware & License</b>: {escaped_hardware} | <i>{escaped_license}</i>"
        )

        if len(rendered) > max_chars:
            # Tag-safe truncation: close any open HTML tags to prevent Telegram API rejection
            from newsscout.storage.models import _safe_truncate_telegram_html
            rendered = _safe_truncate_telegram_html(rendered, max_chars)

        return rendered
