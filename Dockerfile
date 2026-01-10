# Multi-stage build for faster rebuilds
FROM python:3.11-slim AS dependencies

# Install system dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
WORKDIR /app

# Copy and install Python dependencies (separate layer for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM dependencies AS runtime

# Configure Streamlit
RUN mkdir -p /home/app/.streamlit/
RUN echo "[server]\\n\\nheadless = true\\n\\nport = 8501\\n\\nenabledCORS = false\\n\\nallowRunOnSave = true\\n\\n" > /home/app/.streamlit/config.toml

# Copy application code (this layer changes most frequently)
COPY --chown=app:app . .

# Switch to non-root user
USER app

# Streamlit runs on port 8501 by default
EXPOSE 8501

# Start the Streamlit application
CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0"]
