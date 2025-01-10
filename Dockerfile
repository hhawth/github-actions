# Use a slim Python image for minimal base image
FROM python:3.11-slim

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Chrome and ChromeDriver
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    libxkbcommon0 \
    libxrandr2 \
    xdg-utils && \
    wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    dpkg -i google-chrome-stable_current_amd64.deb && \
    rm google-chrome-stable_current_amd64.deb && \
    apt-get install -y --no-install-recommends fonts-liberation && \
    wget https://chromedriver.storage.googleapis.com/LATEST_RELEASE && \
    echo "LATEST_RELEASE=$(cat LATEST_RELEASE)" > version && \
    wget https://chromedriver.storage.googleapis.com/$(cat version)/chromedriver_linux64.zip && \
    unzip chromedriver_linux64.zip && \
    rm chromedriver_linux64.zip version && \
    apt-get clean

# Set working directory in container
WORKDIR /python-docker

# Copy and install Python dependencies

# Expose port 5000 for Cloud Run
EXPOSE 5000

# Copy the application code
COPY . .

# Start the application (assuming it's a Flask or similar app)
CMD [ "python3", "app.py"]
