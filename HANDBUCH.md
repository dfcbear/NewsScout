# NewsScout — Anwender- & Betriebshandbuch 📖

> **Autonome KI-Durchbruchs-Erkennung & Zweistimmiger Audio-Digest**  
> Konzipiert für den 24/7-Dauerbetrieb auf dem **Raspberry Pi 5 (8GB RAM)** mit Fernzugriff über **Tailscale Zero-Trust**.

---

## Inhaltsverzeichnis

1. [Überblick & Systemkonzept](#1-überblick--systemkonzept)
2. [Die User Experience im Alltag (Ein Tag mit NewsScout)](#2-die-user-experience-im-alltag)
   - [A. Der tägliche Audio-Podcast um 07:00 & 16:00 Uhr](#a-der-tägliche-audio-podcast)
   - [B. Die 1-Minute Decision Cards im Telegram-Chat](#b-die-1-minute-decision-cards)
   - [C. Feedback-Schleife & Few-Shot Lernkurve](#c-feedback-schleife--few-shot-lernkurve)
   - [D. Steuerung von unterwegs via Telegram-Befehle](#d-steuerung-von-unterwegs-via-telegram-befehle)
   - [E. Mobiles Web-Dashboard via Tailscale](#e-mobiles-web-dashboard-via-tailscale)
3. [Schritt-für-Schritt Einrichtung & Inbetriebnahme](#3-schritt-für-schritt-einrichtung--inbetriebnahme)
   - [Voraussetzungen](#voraussetzungen)
   - [Schritt 1: SSH-Verbindung aufbauen](#schritt-1-ssh-verbindung-aufbauen)
   - [Schritt 2: Docker & Docker Compose installieren](#schritt-2-docker--docker-compose-installieren)
   - [Schritt 3: Repository klonen](#schritt-3-repository-klonen)
   - [Schritt 4: Telegram Bot & Chat-ID einrichten](#schritt-4-telegram-bot--chat-id-einrichten)
   - [Schritt 5: Gemini API-Key beschaffen](#schritt-5-gemini-api-key-beschaffen)
   - [Schritt 6: Konfiguration (`.env`) erstellen](#schritt-6-konfiguration-env-erstellen)
   - [Schritt 7: Tailscale für Smartphone-Zugriff aktivieren](#schritt-7-tailscale-für-smartphone-zugriff-aktivieren)
   - [Schritt 8: NewsScout starten](#schritt-8-newsscout-starten)
4. [Anwendung & Themen-Steuerung](#4-anwendung--themen-steuerung)
   - [Themenradar im Web-Dashboard](#themenradar-im-web-dashboard)
   - [Chat-Befehle im Telegram-Bot](#chat-befehle-im-telegram-bot)
   - [Quellen-Feinjustierung in der Datenbank](#quellen-feinjustierung-in-der-datenbank)
5. [Wartung, Backup & Troubleshooting](#5-wartung-backup--troubleshooting)
   - [System-Updates & Aktualisieren (Git Pull)](#system-updates--aktualisieren-git-pull)
   - [Logs & Status prüfen](#logs--status-prüfen)
   - [Healthcheck & Monitoring](#healthcheck--monitoring)
   - [Datensicherung (SQLite & Preferences)](#datensicherung-sqlite--preferences)
   - [Fallback-Strategie bei Quota-Limits](#fallback-strategie-bei-quota-limits)
   - [Häufige Probleme & Lösungen](#häufige-probleme--lösungen)

---

## 1. Überblick & Systemkonzept

Das KI-Ökosystem entwickelt sich rasant, doch 99% aller Veröffentlichungen sind Marketing-Hype, reine API-Wrapper oder unfertige theoretische Paper ohne lauffähigen Code. 

**NewsScout** löst dieses Problem autonom direkt auf deinem heimischen Raspberry Pi 5:
* **Kontinuierliche Ingestion**: Überwacht GitHub-Releases (Tier-1 Engines), die Hacker News Firebase API (mit kritischem Peer-Review-Filter) und Hugging Face Daily Papers (nur mit verifizierten Repositories).
* **Zweistufige Filterung**:
  * *Stage 1 (Lokal auf dem Pi)*: Blitzschnelle Heuristik prüft OSI-Lizenzen, lauffähige Docker-/Python-Dateien und filtert Anti-Hype-Muster (No-Code-Builder, SaaS-Boilerplates, SEO-Spam).
  * *Stage 2 (Gemini 3.8 Flash)*: Analysiert die architektonische Neuartigkeit anhand deines spezifischen Tech-Profils (**80% Core**: RTX 4090, vLLM, llama.cpp, Agent Harnesses, Qdrant / **20% Serendipity**: Robotik, Neuromorphic, Mamba, Embedded C++).
* **Lieferung & Audio**: Erstellt kompakte **1-Minute Decision Cards** und produziert zweimal täglich (07:00 & 16:00 Uhr) einen zweistimmigen deutschen Audio-Podcast (Conrad & Katja via `edge-tts`).

```
[ Ingestion Firehose ] 
   (GitHub Releases, Hacker News API, HF Papers)
         │
         ▼
[ Stage 1: Lokale Heuristik ] ──(Fällt durch)──> Discard
   (Docker? OSI-Lizenz? Kein Hype?)
         │
         ▼ (Bestanden)
[ Stage 2: Deep LLM Evaluator ] <─── [ Few-Shot Exemplare ]
   (Gemini 3.8 Flash / Backup)              ▲
         │                                  │ (Feedback)
         ├──────────────────────────────────┤
         ▼                                  ▼
[ 1-Min Decision Cards ]          [ Audio Digest Podcast ]
   - Telegram Push Bot               - 07:00 & 16:00 Uhr
   - Web Dashboard                   - 5 Tracks + Playlist
```

---

## 2. Die User Experience im Alltag

### A. Der tägliche Audio-Podcast

Pünktlich zum Start deines Arbeitswegs morgens um **07:00 Uhr** und nachmittags um **16:00 Uhr** sendet NewsScout eine Benachrichtigung auf dein Smartphone via Telegram:

#### 📱 Ansicht in Telegram:
```text
🎙️ NewsScout Audio-Digest — Morning (18.09.2026)
5 Tracks • Gesamtdauer: 24:15 Min

Track 1: Executive Summary & Tages-Agenda (2:45 Min)
Track 2: Deep Dive: vLLM v0.6.2 Chunked-Prefill Benchmark (7:30 Min)
Track 3: Deep Dive: Docling Multimodal Document Parsing (6:50 Min)
Track 4: Serendipity: Unitree Go2 ROS2 VLA Architecture (5:10 Min)
Track 5: Abend-Aktions-Fazit & Setup-Befehle (2:00 Min)

Wähle die Wiedergabe:
┌──────────────────────────────────────────────┐
│  🎧 Play All (Komplette Playlist abspielen)  │
├──────────────────────┬───────────────────────┤
│  ▶️ Track 1 (Summary) │  ▶️ Track 2 (vLLM)    │
├──────────────────────┼───────────────────────┤
│  ▶️ Track 3 (Docling) │  ▶️ Track 4 (Robotik) │
├──────────────────────┴───────────────────────┤
│  ▶️ Track 5 (Fazit & Setup-Befehle)          │
└──────────────────────────────────────────────┘
```

* **Zwei Sprecher**: **Conrad** moderiert die Fakten, Benchmarks und Befehle; **Katja** analysiert kritisch Architektur, Flaschenhälse und Abgrenzungen zu bestehenden Tools.
* **Flexible Kapitel**: Du kannst die gesamte Playlist nahtlos im Hintergrund abspielen oder per Knopfdruck nur einzelne Themen auswählen.

---

### B. Die 1-Minute Decision Cards

Jeder qualifizierte Durchbruch (Score ≥ 7.0) wird als strukturierte Markdown-/HTML-Karteikarte gesendet. Sie beantwortet in unter einer Minute die Frage: *„Lohnt es sich, das auszuprobieren?“*

#### 📱 Beispiel-Karte im Chat:
```html
🚀 <b>vLLM v0.6.2: Chunked-Prefill mit FP8-KV-Cache für Ada</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<b>Kategorie:</b> Core Focus | <b>Breakthrough:</b> 9.2/10 | <b>ROI:</b> 9.0/10

💡 <b>TL;DR:</b>
Ermöglicht 3.4x höheren Token-Durchsatz durch adaptive Chunked-Prefills bei gleichzeitiger Halbierung des KV-Cache-VRAM-Bedarfs auf RTX 4090.

🎯 <b>Konkreter Use Case:</b>
Direkte Beschleunigung lokaler Multi-Agenten-Harnesses mit großen Kontextfenstern (128k) unter vLLM ohne Out-of-Memory.

⚖️ <b>Vergleich & Baseline:</b>
Übertrifft Standard vLLM v0.5.4 um 240% Prefill-Geschwindigkeit und erreicht Parität mit TensorRT-LLM bei nativer OpenAI-API-Kompatibilität.

⚡ <b>Quickstart (Copy-Paste):</b>
<code>docker run --gpus all -p 8000:8000 vllm/vllm-openai:v0.6.2 --model mistralai/Mistral-7B-Instruct-v0.3 --kv-cache-dtype fp8</code>

📦 <b>Hardware & Lizenz:</b>
<code>24GB VRAM (RTX 4090) | Apache 2.0</code>

🔗 <b>Repository:</b> https://github.com/vllm-project/vllm
```

---

### C. Feedback-Schleife & Few-Shot Lernkurve

Direkt unter jeder Decision Card befinden sich interaktive Bewertungsknöpfe:

```
┌───────────────────────┬────────────────────────┐
│  🎯 Volltreffer       │  💤 Zu banal / Hype   │
├───────────────────────┼────────────────────────┤
│  ✅ Kenne ich schon   │  🚀 Geniale Inspiration │
└───────────────────────┴────────────────────────┘
```

* Wenn du auf einen Button tippst:
  1. Telegram bestätigt sofort mit Popup: `Dankeschön! 🎯 Feedback gespeichert.`
  2. Die Bewertung wird in SQLite (`feedback`) und atomar in `data/preferences.json` synchronisiert.
  3. **Automatischer Lerneffekt**: Beim nächsten Ingestions-Zyklus injiziert NewsScout deine positiv bewerteten Tools als Few-Shot-Vorbilder und deine abgelehnten Tools als Negativ-Beispiele in den Gemini-Prompt. Nach 1–2 Wochen passt sich die Filterung exakt deinen Vorlieben an.

---

### D. Steuerung von unterwegs via Telegram-Befehle

Du musst nicht an den Rechner, um Quellen anzupassen:

* **Neues Repository überwachen**:
  ```text
  /track state-spaces/mamba
  ```
  ➜ *Bot antwortet:* `✅ Repository hinzugefügt: state-spaces/mamba wird jetzt überwacht.`

* **Neues Schlagwort zum Radar hinzufügen**:
  ```text
  /interest neuromorphic computing
  ```
  ➜ *Bot antwortet:* `🎯 Keyword hinzugefügt: 'neuromorphic computing' ist jetzt im Hacker News Radar aktiv.`

* **Aktuellen Status abfragen**:
  ```text
  /radar
  ```
  ➜ *Bot listet alle aktuell aktiven GitHub-Repositories und Keywords auf.*

* **Hilfe & Befehlsübersicht**:
  ```text
  /help
  ```

---

### E. Mobiles Web-Dashboard via Tailscale

Über dein Smartphone erreichst du die Weboberfläche von überall per VPN (Tailscale): `http://raspberrypi:8000`.

Das Dashboard ist dunkel gehalten, lädt extrem schnell (0 externe CDNs) und bietet 5 Tabs:
1. **Breakthroughs**: Durchsuchbares Archiv aller bewerteten Tools mit Filtern nach Kategorie (Core/Serendipity), Score (7+, 8+, 9+) und Volltextsuche.
2. **Statistik**: Auswertung deines Feedback-Verhaltens, Trefferquoten und Verteilung.
3. **Digests**: Historie aller bisherigen Audio-Episoden mit direkter Wiedergabemöglichkeit im Browser.
4. **Watchlist**: Schnellzugriff auf alle von dir mit einem Stern markierten Tools für das Wochenende.
5. **Themenradar**: Übersicht aller überwachten Repositories und Keywords mit Ein-Klick-Löschung und Hinzufügen.

---

## 3. Schritt-für-Schritt Einrichtung & Inbetriebnahme

### Voraussetzungen
* **Hardware**: Raspberry Pi 5 (8GB RAM empfohlen), bootet idealerweise von USB-3-SSD.
* **Betriebssystem**: Raspberry Pi OS (64-Bit) / Debian Bookworm.
* **Software**: Docker & Docker Compose (Installation siehe [Schritt 2](#schritt-2-docker--docker-compose-installieren) – offizieller `get.docker.com` Installer).
* **Netzwerk**: [Tailscale](https://tailscale.com/) auf dem Pi und deinem Smartphone installiert.

---

### Schritt 1: SSH-Verbindung aufbauen

Verbinde dich vom Rechner aus per SSH mit deinem Raspberry Pi:

```bash
ssh pi@newsscout.local
```

> **Hinweis**: Falls `newsscout.local` nicht auflösbar ist, verwende die IP-Adresse deines Pi (z. B. `ssh pi@192.168.1.50`).

---

### Schritt 2: Docker & Docker Compose installieren (1-Befehl-Offizieller Installer)

Installiere Docker inklusive Compose-V2 mit dem offiziellen Installationsskript:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
```

Füge deinen Benutzer der Docker-Gruppe hinzu, damit `docker` ohne `sudo` funktioniert:

```bash
sudo usermod -aG docker $USER
```

Damit die neuen Gruppenrechte aktiv werden, logge dich einmal aus und wieder ein:

```bash
exit
ssh pi@newsscout.local
```

---

### Schritt 3: Repository klonen

Klone das Projekt direkt von GitHub:

```bash
git clone https://github.com/dfcbear/NewsScout.git ~/NewsScout
cd ~/NewsScout
```

---

### Schritt 4: Telegram Bot & Chat-ID einrichten

1. Öffne Telegram und starte einen Chat mit [@BotFather](https://t.me/botfather).
2. Sende `/newbot` und vergib einen Namen und Benutzernamen (z. B. `MeinNewsScoutBot`).
3. Kopiere den erhaltenen **Bot-Token** (Format: `123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ`).
4. Ermittle deine persönliche Chat-ID:
   * Starte deinen neu erstellten Bot mit `/start`.
   * Starte einen Chat mit [@userinfobot](https://t.me/userinfobot), um deine numerische **Chat-ID** zu sehen (z. B. `987654321`).

---

### Schritt 5: Gemini API-Key beschaffen

1. Öffne [Google AI Studio](https://aistudio.google.com/).
2. Melde dich mit deinem Google-Konto an und klicke auf **Get API key** ➜ **Create API key**.
3. Kopiere den Schlüssel (kostenloses Kontingent reicht für hunderte Analysen pro Tag völlig aus).

---

### Schritt 6: Konfiguration (`.env`) erstellen

Klone das Projekt auf deinen Raspberry Pi oder navigiere in das Projektverzeichnis:

```bash
cd ~/newsscout
cp .env.example .env
nano .env
```

Passe die folgenden Pflichtvariablen an:

```ini
# ============================================================================
# API Credentials (PFLICHT)
# ============================================================================
NEWSSCOUT_GEMINI_API_KEY=AIzaSyDeinEchterGeminiSchluesselHier
NEWSSCOUT_TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ
NEWSSCOUT_TELEGRAM_CHAT_ID=987654321

# Optional: GitHub Personal Access Token (erhöht GitHub Rate-Limit von 60 auf 5.000 Req/h)
NEWSSCOUT_GITHUB_TOKEN=

# ============================================================================
# Zeitplan (Podcast-Zeiten)
# ============================================================================
NEWSSCOUT_SCHEDULE_MORNING=07:00
NEWSSCOUT_SCHEDULE_AFTERNOON=16:00
NEWSSCOUT_PIPELINE_INTERVAL_HOURS=2
NEWSSCOUT_TIMEZONE=Europe/Berlin

# ============================================================================
# Dashboard & Netzwerk
# ============================================================================
NEWSSCOUT_HOST=0.0.0.0
NEWSSCOUT_PORT=8000
```

Speichere die Datei mit `Strg + O`, `Enter` und schließe mit `Strg + X`.

---

### Schritt 7: Tailscale für Smartphone-Zugriff aktivieren

Tailscale ermöglicht den sicheren Zugriff auf das Dashboard von unterwegs, ohne Ports am Router öffnen zu müssen:

1. Installiere Tailscale auf dem Pi (falls noch nicht geschehen):
   ```bash
   curl -fsSL https://tailscale.com/install.sh | sh
   sudo tailscale up
   ```
2. Installiere die Tailscale-App auf deinem Smartphone und melde dich mit demselben Konto an.
3. Notiere dir den Tailscale-Namen oder die IP deines Pi (z. B. `100.x.y.z` oder `http://raspberrypi:8000`).

---

### Schritt 8: NewsScout starten

Baue und starte NewsScout inklusive aller Sidecars (SearXNG, Signal):

```bash
docker compose --profile all up -d --build
```

> **Hinweis**: Der Schalter `--build` stellt sicher, dass das Python-Image beim ersten Start (sowie nach jedem Code-Update) frisch aus dem lokalen `Dockerfile` gebaut wird. Ohne `--profile all` startet nur der Kern (`newsscout`) ohne die optionalen Sidecars. Verwende `docker compose --profile all up -d --build`, um alle Dienste (SearXNG, Signal-Bridge) zu bauen und zu aktivieren.

Überprüfe den Status der Container:

```bash
docker compose ps
```

NAME        IMAGE               COMMAND                  SERVICE     CREATED        STATUS                 PORTS
newsscout   newsscout-service   "python -m newsscout.…"  newsscout   1 minute ago   Up 1 minute (healthy)  0.0.0.0:8000->8000/tcp
```

Sobald der Status auf `(healthy)` steht, ist das Dashboard unter `http://<Tailscale-IP>:8000` erreichbar!

---

## 4. Anwendung & Themen-Steuerung

### Themenradar im Web-Dashboard
1. Öffne im Browser `http://<Tailscale-IP>:8000`.
2. Klicke auf den Tab **Themenradar**.
3. **Repo hinzufügen**: Gib im Eingabefeld den Pfad ein (z. B. `karpathy/nanoGPT` oder `state-spaces/mamba`) und klicke auf `+ Repo hinzufügen`.
4. **Keyword hinzufügen**: Gib z. B. `spiking neural networks` ein und klicke auf `+ Keyword hinzufügen`.

### Chat-Befehle im Telegram-Bot
NewsScout hört im Hintergrund auf deine Nachrichten:
* `/track <owner/repo>` ➔ Fügt das Repo live in die Ingestion ein.
* `/interest <keyword>` ➔ Fügt das Keyword zum Hacker News Regex hinzu.
* `/radar` ➔ Zeigt die aktive Liste aller Repos und Keywords an.

---

## 5. Wartung, Backup & Troubleshooting

### System-Updates & Aktualisieren (Git Pull)
Wenn du eine neue Version aus GitHub auf deinen Raspberry Pi übernehmen möchtest:

```bash
cd ~/NewsScout

# 1. Neuesten Code von GitHub abrufen
git pull

# 2. Container mit den neuen Änderungen neu bauen und im Hintergrund starten
docker compose --profile all up -d --build
```

---

### Logs & Status prüfen
Echtzeit-Protokolle einsehen:
```bash
# Alle Logs mitverfolgen
docker compose logs -f

# Nur die letzten 50 Zeilen
docker compose logs --tail=50
```

### Healthcheck & Monitoring
Du kannst den Dienst jederzeit lokal oder remote abfragen:
```bash
curl http://localhost:8000/api/stats
```
Gibt JSON mit aktuellem Feedback-Status und Tool-Zahlen zurück.

### Datensicherung (SQLite & Preferences)
Alle wichtigen Zustände liegen im gemounteten Verzeichnis `./data` auf dem Pi:
* `./data/ai_scout.db`: SQLite-Datenbank (WAL-Modus) mit allen Breakthroughs, Bewertungen und Digest-Tracks.
* `./data/preferences.json`: Gelernte Few-Shot-Exemplare und Vorlieben.
* `./data/audio/`: Generierte MP3-Dateien (werden nach 14 Tagen automatisch bereinigt).

**Backup-Befehl (z. B. wöchentlich per Cronjob):**
```bash
tar -czvf newsscout_backup_$(date +%Y%m%d).tar.gz ./data/*.db* ./data/*.json
```

---

### Fallback-Strategie bei Quota-Limits

Sollte das primäre Gemini-Modell sein Rate-Limit („Quota reached“) erreichen:

1. **Schnellwechsel in `.env`**:
   Passe in `.env` das Modell an:
   ```ini
   NEWSSCOUT_GEMINI_MODEL=gemini-3.8-flash  # oder Fallback auf GLM 5.2 via API-Gateway
   ```
2. **Neustart des Dienstes**:
   ```bash
   docker compose restart
   ```

---

### Häufige Probleme & Lösungen

| Symptom | Mögliche Ursache | Lösung |
| :--- | :--- | :--- |
| **Telegram-Bot antwortet nicht** | Ungültiger Bot-Token oder falsche Chat-ID | Prüfe `.env` mit `docker compose logs \| grep Telegram`. Verifiziere Token via `@BotFather`. |
| **Audio wird nicht generiert** | Fehlende Internetverbindung für `edge-tts` oder FFmpeg fehlt | `edge-tts` benötigt ausgehenden Internetzugriff (Port 443). Im Docker-Container ist FFmpeg bereits integriert. |
| **Keine neuen Breakthroughs** | Quellen liefern keine Treffer mit Score ≥ 7.0 | Normal bei ruhigen Tagen. Prüfe mit `/radar`, ob Repositories aktiv sind, oder senke testweise `min_breakthrough_score` in `.env`. |
| **Container meldet `unhealthy`** | Port 8000 blockiert oder Web-Server abgestürzt | Prüfe `docker compose logs`. Stelle sicher, dass kein anderer Dienst Port 8000 belegt. |
| **Datenbank gesperrt (`database is locked`)** | Gleichzeitige externe Zugriffe ohne WAL | NewsScout nutzt SQLite WAL mit `busy_timeout=5000`. Greife von außen nur lesend zu (`sqlite3 data/ai_scout.db "PRAGMA query..."`). |

---

*NewsScout ist nun vollständig eingerichtet und einsatzbereit. Viel Erfolg bei der Jagd nach echten KI-Durchbrüchen!*