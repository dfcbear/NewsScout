# Dockerfile for NewsScout — ARM64 (Raspberry Pi 5) optimized
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY newsscout/ ./newsscout/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Create data directories
RUN mkdir -p /app/data/audio

# Expose web dashboard port
EXPOSE 8000

# Environment defaults
ENV NEWSSCOUT_HOST=0.0.0.0
ENV NEWSSCOUT_PORT=8000
ENV NEWSSCOUT_DB_PATH=/app/data/ai_scout.db
ENV NEWSSCOUT_AUDIO_OUTPUT_DIR=/app/data/audio
ENV NEWSSCOUT_PREFERENCES_FILE=/app/data/preferences.json

# Run the scheduler (which also serves the dashboard)
CMD ["python", "-m", "newsscout.scheduler"]
