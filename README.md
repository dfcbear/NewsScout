# NewsScout 🚀

**Autonomous AI Breakthrough Scout & Audio-Digest Pipeline**  
Designed for 24/7 self-hosted operation on Raspberry Pi 5 (8GB, ARM64) and Docker Compose.

📖 **Ausführliches deutsches Handbuch**: Siehe [HANDBUCH.md](file:///c:/Users/Racing_Bear/Documents/Antigravity/agent_harness_test/HANDBUCH.md) für Setup, UX-Mockups, Telegram-Befehle und Troubleshooting.

---

## Architecture Overview

NewsScout continuously monitors high-signal open-source developer channels:
1. **GitHub Releases & Atom feeds**: Tier-1 local AI runtime engines (vLLM, llama.cpp, Ollama, SGLang, Qdrant, Docling)
2. **Hacker News Firehose**: Critical peer review with community sceptic ratio filtering
3. **Hugging Face Daily Papers**: Verified trending research with reproducible GitHub code

### Multi-Stage Processing Pipeline
- **Stage 1**: Sub-millisecond deterministic heuristics (license verification, runnable Docker/code checks, anti-hype blacklist)
- **Stage 2**: Deep technical evaluation using Gemini 3.8 Flash with dynamic few-shot calibration (80% Core Stack / 20% Serendipity Quota)
- **1-Minute Decision Cards**: Standardized TL;DR, concrete use case, performance comparison, copy-pasteable 1-liner quickstart, and hardware/license badges
- **Audio Digest Podcast**: Twice-daily German 2-speaker tech dialogue (Conrad & Katja via `edge-tts`) delivered as segmented tracks with merged playlist
- **Web Dashboard**: Mobile-first single-page application for browsing breakthroughs, managing the topic radar, and reviewing feedback stats

---

## Quickstart

### Environment Configuration
Copy `.env.example` to `.env` and configure your API credentials:

```bash
cp .env.example .env
```

Required keys:
- `GEMINI_API_KEY`: Google Gemini API key for Stage 2 evaluations
- `TELEGRAM_BOT_TOKEN`: Telegram bot token from `@BotFather`
- `TELEGRAM_CHAT_ID`: Your personal Telegram user or group chat ID

### Running with Docker Compose

```bash
docker compose up -d --build
```

Access the dashboard at `http://localhost:8000` (or over Tailscale from your phone).

---

## License
MIT License. Open Source & Self-Hosted.