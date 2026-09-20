"""newsscout.filtering.stage1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Stage 1 Deterministic Heuristic Filter running locally on Raspberry Pi 5.
Evaluates ingested raw items against:
- Anti-hype / SaaS wrapper / SEO / Web3 blacklist
- Credibility & repository staleness (>180 days) / artificial star spikes
- OSI-approved permissive/copyleft licenses vs non-commercial (CC-BY-NC)
- Runnable code & Dockerfile / build artifact presence
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import re
from typing import Any, Optional, Sequence
from urllib.parse import urlparse

import httpx

from newsscout.storage.db import Database
from newsscout.storage.models import RawItem, Stage1Evaluation

logger = logging.getLogger(__name__)


# ============================================================================
# Heuristic Configuration Constants
# ============================================================================

MAX_INACTIVITY_DAYS: int = 180  # 6 months staleness threshold

RUNNABLE_FILENAMES: set[str] = {
    # Docker / Containers
    "dockerfile", "containerfile", "docker-compose.yml", "docker-compose.yaml",
    "compose.yml", "compose.yaml",
    # Python
    "pyproject.toml", "requirements.txt", "setup.py", "setup.cfg",
    "pipfile", "environment.yml", "conda.yaml",
    # Rust
    "cargo.toml",
    # C / C++
    "cmakelists.txt", "makefile", "meson.build", "build.bazel", "configure.ac",
}

DOCKER_FILENAMES: set[str] = {
    "dockerfile", "containerfile", "docker-compose.yml", "docker-compose.yaml",
    "compose.yml", "compose.yaml",
}

BINARY_ASSET_EXTENSIONS: set[str] = {
    ".whl", ".deb", ".rpm", ".appimage", ".pkg", ".msi", ".exe", ".apk",
    ".tar.gz", ".tar.xz", ".tgz", ".zip",
}

OSI_APPROVED_LICENSES: set[str] = {
    # Permissive
    "mit", "mit-0", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "bsd-4-clause",
    "isc", "0bsd", "unlicense", "cc0-1.0", "wtfpl", "zlib",
    # Copyleft
    "gpl-2.0", "gpl-2.0-only", "gpl-2.0-or-later",
    "gpl-3.0", "gpl-3.0-only", "gpl-3.0-or-later",
    "agpl-3.0", "agpl-3.0-only", "agpl-3.0-or-later",
    "lgpl-2.1", "lgpl-3.0", "mpl-2.0", "epl-2.0",
}

NON_COMMERCIAL_LICENSES: set[str] = {
    "cc-by-nc-4.0", "cc-by-nc-3.0", "cc-by-nc-2.5", "cc-by-nc-2.0", "cc-by-nc-1.0",
    "cc-by-nc-sa-4.0", "cc-by-nc-sa-3.0", "cc-by-nc-sa",
    "cc-by-nc-nd-4.0", "cc-by-nc-nd-3.0", "cc-by-nc-nd",
    "creative-commons-attribution-noncommercial",
    "creative-commons-attribution-noncommercial-4.0",
    "bsl-1.1", "business source license 1.1", "business-source-license-1.1",
    "sspl", "sspl-1.0", "commons-clause", "proprietary", "all-rights-reserved",
    "nc-1.0", "non-commercial",
}

BLACKLIST_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "saas_wrapper": [
        re.compile(r"\b(?:chatgpt|openai|claude)\s+wrapper\b", re.IGNORECASE),
        re.compile(r"\bwrapper\s+(?:app|service|tool|for\s+chatgpt|for\s+claude)\b", re.IGNORECASE),
        re.compile(r"\bthin\s+(?:ui|client)\s+for\s+(?:openai|anthropic)\b", re.IGNORECASE),
        re.compile(r"\bchrome\s+extension\s+for\s+chatgpt\b", re.IGNORECASE),
        re.compile(r"\b(?:ai\s+)?saas\s+boilerplate\b", re.IGNORECASE),
        re.compile(r"\bmicro-?saas\b", re.IGNORECASE),
        re.compile(r"\b(?:s-a-a-s|saas)\s+starter\s+kit\b", re.IGNORECASE),
    ],
    "no_code_builder": [
        re.compile(r"\b(?:no[\s-]code|nocode|low-code)\s+(?:website|landing\s+page|web\s+app)\s+builder\b", re.IGNORECASE),
        re.compile(r"\bno[\s-]code\b", re.IGNORECASE),
        re.compile(r"\bai\s+landing\s+page\b", re.IGNORECASE),
        re.compile(r"\bdrop-?shipping\b", re.IGNORECASE),
        re.compile(r"\bbuild\s+(?:websites|landing\s+pages)\s+in\s+(?:seconds|minutes)\b", re.IGNORECASE),
        re.compile(r"\b(?:shopify|wordpress)\s+(?:app|plugin|theme)\b", re.IGNORECASE),
        re.compile(r"\b(?:wix|squarespace|framer|webflow)\s+(?:alternative|clone|template)\b", re.IGNORECASE),
        re.compile(r"\bdrag\s+and\s+drop\s+website\b", re.IGNORECASE),
    ],
    "seo_spam": [
        re.compile(r"\bseo\s+(?:article|content|blog|post)?\s*generator\b", re.IGNORECASE),
        re.compile(r"\b(?:rank\s+#?1\s+on\s+google|parasite\s+seo|backlink\s+generator)\b", re.IGNORECASE),
        re.compile(r"\bautomated\s+blogging\s+for\s+affiliates\b", re.IGNORECASE),
        re.compile(r"\bauto-?dm\s+(?:bot|tool|automation)\b", re.IGNORECASE),
    ],
    "crypto_web3": [
        re.compile(r"\btokenomics\b", re.IGNORECASE),
        re.compile(r"\bmemecoin\b", re.IGNORECASE),
        re.compile(r"\b(?:airdrop|token\s+launch|presale)\b", re.IGNORECASE),
        re.compile(r"\b(?:crypto\s+token|web3\s+ai|solana\s+bot|pump\.fun)\b", re.IGNORECASE),
        re.compile(r"\b(?:nft\s+marketplace|dao\s+governance\s+token)\b", re.IGNORECASE),
        re.compile(r"\bdecentralized\s+compute\s+token\b", re.IGNORECASE),
    ],
}

BLACKLISTED_DOMAINS: set[str] = {
    "pump.fun", "dexscreener.com", "coinmarketcap.com", "myshopify.com",
    "gumroad.com", "buymeacoffee.com", "linktr.ee",
}

# Regex for detecting command snippets in release notes / README markdown
DOCKER_CMD_REGEX = re.compile(r"(?i)\b(?:docker\s+run|docker\s+compose|docker\s+build|containerfile)\b")
RUN_CMD_REGEX = re.compile(r"(?i)\b(?:pip\s+install|uv\s+add|uvx\s+|cargo\s+build|cargo\s+run|cmake\s+--build|make\s+-j)\b")
LICENSE_EXTRACT_REGEX = re.compile(r"(?i)\blicen[sc]e\b\s*[:\-]?\s*([A-Za-z0-9.\-]+(?:\s+[0-9.]+)?)\b")


# ============================================================================
# Stage 1 Filter Implementation
# ============================================================================

class Stage1Filter:
    """Deterministic local quality filter for NewsScout."""

    def __init__(
        self,
        db: Optional[Database] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        strict_licenses: bool = False,
    ) -> None:
        self.db = db
        self.http_client = http_client
        self.strict_licenses = strict_licenses
        self._url_probe_cache: dict[str, bool] = {}

    # ------------------------------------------------------------------------
    # Sub-Engine 1: Anti-Hype Blacklist Scanner
    # ------------------------------------------------------------------------
    def check_blacklist(self, raw_item: RawItem) -> tuple[bool, Optional[str], dict[str, Any]]:
        """Scans title, content, and URL against anti-hype and spam blacklists."""
        # 1. Domain check
        try:
            parsed = urlparse(raw_item.url)
            hostname = (parsed.hostname or "").lower()
            for b_domain in BLACKLISTED_DOMAINS:
                if hostname == b_domain or hostname.endswith("." + b_domain):
                    return False, f"Blacklisted domain: {hostname}", {"domain": hostname}
        except Exception:
            pass

        # 2. Text fields scan
        text_fields = {
            "title": raw_item.title,
            "raw_content": raw_item.raw_content or "",
            "description": str(raw_item.metadata.get("description", "")),
        }

        for category, patterns in BLACKLIST_PATTERNS.items():
            for pattern in patterns:
                for field_name, text in text_fields.items():
                    match = pattern.search(text)
                    if match:
                        matched_term = match.group(0)
                        return (
                            False,
                            f"Anti-hype blacklist matched: '{matched_term}' ({category})",
                            {"category": category, "term": matched_term, "field": field_name},
                        )

        return True, None, {"status": "clean"}

    # ------------------------------------------------------------------------
    # Sub-Engine 2: Credibility & Activity Filter
    # ------------------------------------------------------------------------
    def check_credibility_activity(self, raw_item: RawItem) -> tuple[bool, Optional[str], dict[str, Any]]:
        """Checks staleness (>180 days inactive) and artificial star spike metrics."""
        meta = raw_item.metadata

        # 1. Archived check
        if meta.get("archived") is True or meta.get("is_archived") is True:
            return False, "Repository is archived by owner", {"archived": True}

        # 2. Activity / Staleness
        last_activity_raw = (
            meta.get("pushed_at")
            or meta.get("last_commit_at")
            or meta.get("latest_release_at")
            or meta.get("published_at")
        )
        if last_activity_raw:
            try:
                dt_str = str(last_activity_raw).replace("Z", "+00:00")
                activity_dt = datetime.fromisoformat(dt_str)
                if activity_dt.tzinfo is None:
                    activity_dt = activity_dt.replace(tzinfo=timezone.utc)
                now_utc = datetime.now(timezone.utc)
                inactivity_days = (now_utc - activity_dt).days
                if inactivity_days > MAX_INACTIVITY_DAYS:
                    return (
                        False,
                        f"Stale repository: inactive for {inactivity_days} days (threshold: {MAX_INACTIVITY_DAYS} days)",
                        {"inactivity_days": inactivity_days, "last_activity": str(last_activity_raw)},
                    )
            except Exception as ex:
                logger.debug("Could not parse activity timestamp %s: %s", last_activity_raw, ex)

        # 3. Artificial star spike / bot manipulation
        def _safe_int(val: Any) -> int:
            try:
                return int(val or 0)
            except (ValueError, TypeError):
                return 0

        stars = _safe_int(meta.get("stars") or meta.get("github_stars"))
        forks = _safe_int(meta.get("forks"))
        open_issues = _safe_int(meta.get("open_issues"))
        closed_issues = _safe_int(meta.get("closed_issues"))
        total_issues = open_issues + closed_issues

        if stars >= 500 and forks == 0:
            return (
                False,
                f"Suspicious star-to-fork ratio (stars={stars}, forks={forks}) indicating artificial star inflation",
                {"stars": stars, "forks": forks},
            )

        if stars >= 1000 and (forks / stars) < 0.005:
            ratio = forks / stars
            return (
                False,
                f"Suspicious star-to-fork ratio ({forks}/{stars} = {ratio:.4f} < 0.5%) indicating artificial star inflation",
                {"stars": stars, "forks": forks, "ratio": ratio},
            )

        if stars >= 1000 and total_issues == 0 and "issues" in meta:
            return (
                False,
                "Zero issue activity on repository with >1000 stars (artificial engagement anomaly)",
                {"stars": stars, "total_issues": total_issues},
            )

        return True, None, {"stars": stars, "forks": forks}

    # ------------------------------------------------------------------------
    # Sub-Engine 3: License Verification
    # ------------------------------------------------------------------------
    def check_license(self, raw_item: RawItem) -> tuple[bool, Optional[str], Optional[str], dict[str, Any]]:
        """Verifies license against OSI-approved permissive and non-commercial lists."""
        meta = raw_item.metadata
        raw_license = meta.get("spdx_id") or meta.get("license") or meta.get("license_spdx")

        # When license is a dict from GitHub API, extract string
        if isinstance(raw_license, dict):
            raw_license = raw_license.get("spdx_id") or raw_license.get("key") or raw_license.get("name")
        elif raw_license is not None and not isinstance(raw_license, str):
            raw_license = str(raw_license)

        # Fallback to regex scan in raw_content
        if not raw_license and raw_item.raw_content:
            m = LICENSE_EXTRACT_REGEX.search(raw_item.raw_content)
            if m:
                raw_license = m.group(1).strip()

        if not raw_license:
            if self.strict_licenses:
                return False, None, "Missing or unverified open-source license", {"license": None}
            return True, None, None, {"license": "unverified", "status": "deferred_to_stage2"}

        lic_str = str(raw_license).strip()
        normalized = lic_str.lower().replace(" ", "-").replace("_", "-")
        raw_lower = lic_str.lower()

        # Check for non-commercial or proprietary
        if (
            normalized in NON_COMMERCIAL_LICENSES
            or raw_lower in NON_COMMERCIAL_LICENSES
            or "-nc" in normalized
            or "non-commercial" in normalized
            or "noncommercial" in normalized
            or "proprietary" in raw_lower
            or "for-research-only" in normalized
        ):
            return (
                False,
                lic_str,
                f"Non-commercial or proprietary license ({lic_str}) violates open-source requirements",
                {"raw_license": lic_str, "normalized": normalized},
            )

        # Check for OSI approved
        if normalized in OSI_APPROVED_LICENSES:
            return True, lic_str, None, {"raw_license": lic_str, "osi_approved": True}

        # Unknown license
        if self.strict_licenses:
            return False, lic_str, f"Non-OSI or unapproved license: {lic_str}", {"license": lic_str}

        return True, lic_str, None, {"raw_license": lic_str, "osi_approved": False, "status": "flagged"}

    # ------------------------------------------------------------------------
    # Sub-Engine 4: Runnable Artifacts & Docker Detection
    # ------------------------------------------------------------------------
    def check_runnable_artifacts(
        self, raw_item: RawItem
    ) -> tuple[bool, bool, bool, Optional[str], dict[str, Any]]:
        """Verifies presence of runnable code, build files, and Docker assets."""
        meta = raw_item.metadata
        has_docker = False
        has_runnable_code = False
        detected_artifacts: list[str] = []

        # 1. Inspect metadata root_files / filenames
        root_files = [str(f).lower() for f in (meta.get("root_files") or meta.get("files") or []) if f]
        for f in root_files:
            base_name = f.split("/")[-1]
            if base_name in RUNNABLE_FILENAMES:
                has_runnable_code = True
                detected_artifacts.append(base_name)
            if base_name in DOCKER_FILENAMES:
                has_docker = True

        # 2. Inspect release assets
        assets = meta.get("assets", []) or meta.get("release_assets", [])
        for asset in assets:
            asset_name = asset if isinstance(asset, str) else asset.get("name", "")
            lower_asset = asset_name.lower()
            for ext in BINARY_ASSET_EXTENSIONS:
                if lower_asset.endswith(ext):
                    has_runnable_code = True
                    detected_artifacts.append(asset_name)
                    if "docker" in lower_asset:
                        has_docker = True
                    break

        # 3. Inspect raw_content markdown commands
        content = raw_item.raw_content or ""
        if DOCKER_CMD_REGEX.search(content):
            has_docker = True
            has_runnable_code = True
            detected_artifacts.append("command:docker")

        if RUN_CMD_REGEX.search(content):
            has_runnable_code = True
            detected_artifacts.append("command:build/install")

        # 4. Source-specific repository requirement checks
        if raw_item.source == "hacker_news":
            extracted_urls = meta.get("extracted_github_urls", [])
            if not extracted_urls and "github.com" not in raw_item.url:
                return False, False, False, "No code repository linked in Hacker News submission", {}
            if extracted_urls or "github.com" in raw_item.url:
                has_runnable_code = True
                detected_artifacts.append("repo:linked_github")

        if raw_item.source == "hf_daily_papers":
            github_repo = meta.get("github_repo") or meta.get("paper", {}).get("githubRepo")
            if not github_repo:
                return False, False, False, "No linked code repository in Hugging Face Daily Paper", {}
            has_runnable_code = True
            detected_artifacts.append(f"repo:{github_repo}")

        if meta.get("github_repo"):
            has_runnable_code = True
            detected_artifacts.append(f"repo:{meta.get('github_repo')}")

        if meta.get("extracted_github_urls"):
            has_runnable_code = True
            detected_artifacts.append("repo:extracted_github")

        if not has_runnable_code:
            return (
                False,
                has_docker,
                False,
                "No runnable artifacts or release binaries found (missing Dockerfile, pyproject.toml, requirements.txt, Cargo.toml, CMakeLists.txt, or binary assets)",
                {"detected": detected_artifacts},
            )

        return True, has_docker, has_runnable_code, None, {"detected": detected_artifacts}

    # ------------------------------------------------------------------------
    # Public Synchronous Evaluator (Fast Local CPU Path)
    # ------------------------------------------------------------------------
    def evaluate(self, raw_item: RawItem) -> Stage1Evaluation:
        """Evaluates a raw item deterministically without external network requests."""
        heuristics: dict[str, Any] = {}

        # 1. Anti-Hype Blacklist
        bl_pass, bl_reason, bl_info = self.check_blacklist(raw_item)
        heuristics["blacklist"] = bl_info
        if not bl_pass:
            return Stage1Evaluation(
                raw_item_id=raw_item.id or 0,
                passed=False,
                drop_reason=bl_reason,
                detected_license=None,
                has_docker=False,
                has_runnable_code=False,
                heuristics=heuristics,
            )

        # 2. Credibility & Activity
        cred_pass, cred_reason, cred_info = self.check_credibility_activity(raw_item)
        heuristics["credibility"] = cred_info
        if not cred_pass:
            return Stage1Evaluation(
                raw_item_id=raw_item.id or 0,
                passed=False,
                drop_reason=cred_reason,
                detected_license=None,
                has_docker=False,
                has_runnable_code=False,
                heuristics=heuristics,
            )

        # 3. License Check
        lic_pass, lic_name, lic_reason, lic_info = self.check_license(raw_item)
        heuristics["license"] = lic_info
        if not lic_pass:
            return Stage1Evaluation(
                raw_item_id=raw_item.id or 0,
                passed=False,
                drop_reason=lic_reason,
                detected_license=lic_name,
                has_docker=False,
                has_runnable_code=False,
                heuristics=heuristics,
            )

        # 4. Runnable Artifacts Check
        art_pass, has_docker, has_runnable_code, art_reason, art_info = self.check_runnable_artifacts(raw_item)
        heuristics["artifacts"] = art_info
        if not art_pass:
            return Stage1Evaluation(
                raw_item_id=raw_item.id or 0,
                passed=False,
                drop_reason=art_reason,
                detected_license=lic_name,
                has_docker=has_docker,
                has_runnable_code=False,
                heuristics=heuristics,
            )

        # All checks passed cleanly
        return Stage1Evaluation(
            raw_item_id=raw_item.id or 0,
            passed=True,
            drop_reason=None,
            detected_license=lic_name,
            has_docker=has_docker,
            has_runnable_code=has_runnable_code,
            heuristics=heuristics,
        )

    # ------------------------------------------------------------------------
    # Async Network-Aware Evaluator & Probing (Optional)
    # ------------------------------------------------------------------------
    async def evaluate_async(self, raw_item: RawItem, probe_network: bool = False) -> Stage1Evaluation:
        """Asynchronously evaluates raw item, optionally probing GitHub if root_files is missing."""
        evaluation = self.evaluate(raw_item)

        # If dropped because artifacts were missing, and probe_network is enabled, probe raw.githubusercontent.com
        if not evaluation.passed and "No runnable artifacts" in (evaluation.drop_reason or "") and probe_network and self.http_client:
            owner_repo = self._extract_github_owner_repo(raw_item)
            if owner_repo:
                probed = await self._probe_github_artifacts(owner_repo)
                if probed["has_runnable_code"]:
                    evaluation.passed = True
                    evaluation.drop_reason = None
                    evaluation.has_docker = probed["has_docker"]
                    evaluation.has_runnable_code = True
                    evaluation.heuristics["artifacts"]["probed"] = probed

        return evaluation

    # ------------------------------------------------------------------------
    # Pipeline & Database Operations
    # ------------------------------------------------------------------------
    async def save_evaluation(self, eval_result: Stage1Evaluation) -> int:
        """Persists evaluation to stage1_evaluations table using idempotent UPSERT."""
        if not self.db:
            raise RuntimeError("Cannot save Stage1Evaluation without configured Database instance")

        params = eval_result.to_db_params()
        query = """
        INSERT INTO stage1_evaluations (
            raw_item_id, passed, drop_reason, detected_license,
            has_docker, has_runnable_code, heuristics_json, evaluated_at
        ) VALUES (
            :raw_item_id, :passed, :drop_reason, :detected_license,
            :has_docker, :has_runnable_code, :heuristics_json, :evaluated_at
        )
        ON CONFLICT(raw_item_id) DO UPDATE SET
            passed = excluded.passed,
            drop_reason = excluded.drop_reason,
            detected_license = excluded.detected_license,
            has_docker = excluded.has_docker,
            has_runnable_code = excluded.has_runnable_code,
            heuristics_json = excluded.heuristics_json,
            evaluated_at = excluded.evaluated_at;
        """
        return await self.db.execute(query, params)

    async def run_pipeline(self, limit: int = 100) -> list[Stage1Evaluation]:
        """Fetches unevaluated items from raw_items, evaluates them, and persists results."""
        if not self.db:
            raise RuntimeError("Cannot run Stage 1 pipeline without configured Database instance")

        query = """
        SELECT r.*
        FROM raw_items r
        LEFT JOIN stage1_evaluations s ON r.id = s.raw_item_id
        WHERE s.id IS NULL
        ORDER BY r.ingested_at ASC
        LIMIT ?;
        """
        rows = await self.db.fetch_all(query, (limit,))
        results: list[Stage1Evaluation] = []

        for row in rows:
            raw_item = RawItem.from_row(row)
            eval_res = self.evaluate(raw_item)
            eval_res.raw_item_id = raw_item.id or 0
            await self.save_evaluation(eval_res)
            results.append(eval_res)

        return results

    # ------------------------------------------------------------------------
    # Internal Probing Helpers
    # ------------------------------------------------------------------------
    @staticmethod
    def _extract_github_owner_repo(url_or_item: Any) -> Optional[tuple[str, str]]:
        if isinstance(url_or_item, RawItem):
            # 1. Check url
            res = Stage1Filter._extract_github_owner_repo(url_or_item.url)
            if res:
                return res
            # 2. Check metadata github_repo
            gh_repo = url_or_item.metadata.get("github_repo")
            if gh_repo and isinstance(gh_repo, str):
                res = Stage1Filter._extract_github_owner_repo(gh_repo)
                if res:
                    return res
            # 3. Check extracted_github_urls
            for u in url_or_item.metadata.get("extracted_github_urls", []):
                if isinstance(u, str):
                    res = Stage1Filter._extract_github_owner_repo(u)
                    if res:
                        return res
            return None

        if isinstance(url_or_item, str):
            match = re.search(r"github\.com/([^/]+)/([^/\s#?]+)", url_or_item)
            if match:
                owner = match.group(1)
                repo = match.group(2).removesuffix(".git").rstrip("/.")
                return owner, repo
        return None

    async def _probe_github_artifacts(self, owner_repo: tuple[str, str]) -> dict[str, Any]:
        owner, repo = owner_repo
        key = f"{owner}/{repo}"
        if key in self._url_probe_cache:
            return {"has_runnable_code": self._url_probe_cache[key], "has_docker": False}

        has_runnable_code = False
        has_docker = False
        target_probes = ["Dockerfile", "pyproject.toml", "requirements.txt", "Cargo.toml", "CMakeLists.txt"]

        if not self.http_client:
            return {"has_runnable_code": False, "has_docker": False}

        for file in target_probes:
            url = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{file}"
            try:
                resp = await self.http_client.head(url, timeout=2.0)
                if resp.status_code == 200:
                    has_runnable_code = True
                    if "docker" in file.lower():
                        has_docker = True
                    break
            except Exception:
                continue

        self._url_probe_cache[key] = has_runnable_code
        return {"has_runnable_code": has_runnable_code, "has_docker": has_docker}
