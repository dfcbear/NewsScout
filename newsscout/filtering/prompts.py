"""newsscout.filtering.prompts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Prompt templates, system instructions, and schema definitions for Stage 2
Gemini 3.8 Flash evaluation and decision card generation.
"""

from __future__ import annotations

import json
from typing import Any, Final, Optional

from newsscout.storage.models import RawItem, Stage1Evaluation
from newsscout.storage.preferences import ExemplarPair

SYSTEM_INSTRUCTION: Final[str] = """You are NewsScout's Principal AI Systems Architect and Lead Engineering Scout.
Your mission is to rigorously evaluate open-source AI breakthroughs against a senior engineer's exact technical profile.

You are completely immune to marketing hype, VC buzzwords, SaaS wrappers, and inflated star counts.
You evaluate tools based on raw engineering merit, architectural novelty, and practical utility on local hardware.

TARGET PROFILE WEIGHTING:
- 80% CORE FOCUS:
  * Autonomous Agents & Agentic Coding: Spec-Driven harnesses, multi-agent frameworks, MCP (Model Context Protocol), tooling on par with Antigravity / Hermes Agent / Cline, tool-use evaluation.
  * Local Inference Engines & Execution: llama.cpp, vLLM, sglang, custom quantization (NVFP4, EXL2, AWQ), FlashInfer, Triton/CUDA kernels for NVIDIA RTX 4090 (24GB VRAM).
  * Advanced Multimodal RAG & Parsing: Qdrant vector store, high-speed document/table/formula extractors (Docling, marker), late-chunking, hybrid sparse/dense retrieval.
  * Embedded & Low-Latency Systems: C++/C# for microcontrollers (ESP32, Teensy), DirectInput, low-latency telemetry/control.
- 20% INTENTIONAL SERENDIPITY:
  * Robotics & Physical Control Loops: ROS2 + VLA, imitation learning, humanoid/quadruped balance controllers on edge ARM64.
  * Scientific ML & Sensor Processing: Bio/chem ML, real-time sensor processing, edge DSP.
  * Neuromorphic Computing & Edge Accelerators: Spiking neural networks, Hailo-8, Google Coral, Rockchip RK3588 NPU acceleration.
  * Novel Architectures: Mamba, SSMs, Titans, RWKV, test-time compute scaling methods.

HARD REJECTION CRITERIA (0% TOLERANCE -> CATEGORY MUST BE 'discard', SCORES <= 3.0):
- Pure API wrappers around closed SaaS (OpenAI, Claude).
- Low-code / no-code landing page or website builders.
- SEO blog post generators, marketing spam, affiliate content bots.
- Theoretical academic papers without reproducible code, weights, or Dockerfiles.
- Star-manipulated repos or dead repositories.

SCORING CRITERIA:
- breakthrough_score (1.0 - 10.0):
  * 9.0 - 10.0: Revolutionary paradigm shift or major kernel breakthrough.
  * 7.0 - 8.9: High architectural novelty, major performance jump, or robust new framework.
  * 4.0 - 6.9: Incremental improvements, minor features, wrapper utilities.
  * 1.0 - 3.9: Low-effort wrappers, marketing hype, trivial scripts.
- roi_score (1.0 - 10.0):
  * 9.0 - 10.0: Immediate high-value drop-in utility on RTX 4090 (24GB) or Raspberry Pi 5.
  * 7.0 - 8.9: Strong direct utility or high practical benefit with minimal integration friction.
  * 4.0 - 6.9: Niche or requires datacenter infrastructure (8x H100s).
  * 1.0 - 3.9: Zero practical utility for a senior developer/engineer.

QUALIFYING THRESHOLD:
- Core candidates qualify if category == 'core' AND breakthrough_score >= 7.0 AND roi_score >= 7.0.
- Serendipity candidates qualify if category == 'serendipity' AND ((breakthrough_score >= 7.0 AND roi_score >= 7.0) OR (breakthrough_score >= 8.5 AND roi_score >= 6.0)).
- Otherwise, set category to 'discard'.

DECISION CARD REQUIREMENTS (When qualifying):
- tldr: Exactly 1 concise sentence stating what is now possible that was impossible before.
- use_case: Concrete workflow application in the user's stack (RTX 4090, MCP, agent harness, Qdrant).
- comparison: Measurable performance or architectural comparison vs existing baselines (e.g. '3x faster than vLLM on 4090').
- quickstart: 1-line copy-pasteable execution command ('docker run ...' or 'uvx ...').
- hardware_requirements: Exact VRAM, CUDA version, or CPU specs.
- license: Open source license name and commercial status (e.g. 'Apache 2.0 (Permissive)').

Respond strictly in valid JSON conforming to the requested schema.
"""

STAGE2_JSON_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "breakthrough_score": {
            "type": "number",
            "description": "Architectural novelty and technical leap on a 1.0 to 10.0 scale.",
        },
        "roi_score": {
            "type": "number",
            "description": "Practical utility, latency/throughput gains for user stack on a 1.0 to 10.0 scale.",
        },
        "category": {
            "type": "string",
            "enum": ["core", "serendipity", "discard"],
            "description": "Classification: 'core' (80% quota), 'serendipity' (20% quota), or 'discard'.",
        },
        "relevance_justification": {
            "type": "string",
            "description": "Dense 2-3 sentence technical justification referencing user stack or reason for rejection.",
        },
        "decision_card": {
            "type": "object",
            "properties": {
                "tldr": {
                    "type": "string",
                    "description": "Exactly 1 sentence: what is now possible that was previously impossible.",
                },
                "use_case": {
                    "type": "string",
                    "description": "Concrete application mapping to user workflows.",
                },
                "comparison": {
                    "type": "string",
                    "description": "Performance or architectural comparison vs existing baselines.",
                },
                "quickstart": {
                    "type": "string",
                    "description": "Copy-pasteable 1-liner command (docker run... or uvx...).",
                },
                "hardware_requirements": {
                    "type": "string",
                    "description": "Specific VRAM, CUDA version, compute requirements.",
                },
                "license": {
                    "type": "string",
                    "description": "Open source license name and commercial usability.",
                },
            },
            "required": [
                "tldr",
                "use_case",
                "comparison",
                "quickstart",
                "hardware_requirements",
                "license",
            ],
        },
    },
    "required": [
        "breakthrough_score",
        "roi_score",
        "category",
        "relevance_justification",
        "decision_card",
    ],
}


def format_few_shot_calibration(exemplars: ExemplarPair) -> str:
    """Renders user feedback exemplars into markdown for prompt calibration."""
    pos_lines: list[str] = []
    for i, ex in enumerate(exemplars.positive, 1):
        pos_lines.append(
            f"{i}. **{ex.title}** (Rating: {ex.rating.value.upper()}, Category: {ex.category.value}, "
            f"Score: {ex.breakthrough_score:.1f}/10, ROI: {ex.roi_score:.1f}/10)\n"
            f"   - TL;DR: {ex.tldr}\n"
            f"   - User Use Case: {ex.use_case}"
        )

    neg_lines: list[str] = []
    for i, ex in enumerate(exemplars.negative, 1):
        neg_lines.append(
            f"{i}. **{ex.title}** (Rating: {ex.rating.value.upper()}, Category: discard, "
            f"Score: {ex.breakthrough_score:.1f}/10, ROI: {ex.roi_score:.1f}/10)\n"
            f"   - TL;DR: {ex.tldr}\n"
            f"   - Reason for Rejection: {ex.use_case}"
        )

    pos_text = "\n".join(pos_lines) if pos_lines else "None provided yet."
    neg_text = "\n".join(neg_lines) if neg_lines else "None provided yet."

    return (
        "### DYNAMIC FEW-SHOT USER CALIBRATION\n\n"
        "#### RECENT POSITIVE RATINGS (HIT / INSPIRE):\n"
        f"{pos_text}\n\n"
        "#### RECENT NEGATIVE RATINGS (HYPE / BANAL):\n"
        f"{neg_text}\n"
    )


def build_evaluation_user_prompt(
    raw_item: RawItem,
    stage1_eval: Optional[Stage1Evaluation] = None,
    exemplars: Optional[ExemplarPair] = None,
    max_content_chars: int = 12000,
) -> str:
    """Constructs the user message payload for Gemini 3.8 Flash evaluation."""
    few_shot_block = format_few_shot_calibration(exemplars) if exemplars else ""

    # Sanitize and truncate content if necessary (keeping top and tail)
    content = raw_item.raw_content or ""
    if len(content) > max_content_chars:
        head = content[: int(max_content_chars * 0.7)]
        tail = content[-int(max_content_chars * 0.3) :]
        content = f"{head}\n\n... [TRUNCATED FOR LENGTH] ...\n\n{tail}"

    stage1_info = ""
    if stage1_eval:
        stage1_info = (
            f"- Detected License: {stage1_eval.detected_license or 'Unknown'}\n"
            f"- Dockerfile Found: {stage1_eval.has_docker}\n"
            f"- Runnable Code Found: {stage1_eval.has_runnable_code}\n"
        )

    meta_str = json.dumps(raw_item.metadata, indent=2, ensure_ascii=False)

    return (
        f"{few_shot_block}\n"
        "### CANDIDATE FOR EVALUATION\n"
        f"- Title: {raw_item.title}\n"
        f"- Source: {raw_item.source} (ID: {raw_item.source_id})\n"
        f"- URL: {raw_item.url}\n"
        f"{stage1_info}"
        f"- Metadata:\n```json\n{meta_str}\n```\n\n"
        "### REPOSITORY / PAPER CONTENT & RELEASE NOTES:\n"
        f"```markdown\n{content}\n```\n\n"
        "Evaluate this candidate and provide your evaluation strictly in valid JSON matching the schema."
    )
