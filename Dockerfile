FROM python:3.11-slim

WORKDIR /app

# Set UTF-8 locale for proper Unicode handling
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (includes football_data.duckdb if present)
COPY . .

# Ensure DuckDB database is present (create empty with schema if not exists)
RUN python -c "import duckdb; import os; \
    if not os.path.exists('football_data.duckdb'): \
        print('Database not found, creating empty football_data.duckdb'); \
        conn = duckdb.connect('football_data.duckdb'); \
        conn.close(); \
    else: \
        print('football_data.duckdb already present')"

# Expose port
EXPOSE 8080

# Set environment variable for port
ENV PORT=8080

# Run the API server
CMD ["python", "api_server.py"]
