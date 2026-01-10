# Use Alpine for much smaller base image
FROM python:3.11-alpine AS dependencies

# Install minimal system dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    && rm -rf /var/cache/apk/*

WORKDIR /app

# Copy and install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Runtime stage - minimal Alpine
FROM python:3.11-alpine AS runtime

# Copy only installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Create non-root user
RUN adduser -D -s /bin/sh app
WORKDIR /app

# Copy application code only
COPY --chown=app:app *.py ./
COPY --chown=app:app requirements.txt ./

# Switch to non-root user
USER app

# Configure Streamlit for Cloud Run with dynamic port
ENV HOST=0.0.0.0
EXPOSE 8501

# Start the Streamlit application with dynamic port support
CMD streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port ${PORT:-8501} --server.headless true --server.enableCORS false --server.enableXsrfProtection false
