# Use a slim Python image for minimal base image
FROM python:3.11-slim

# Install necessary packages
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    chromium \
    && rm -rf /var/lib/apt/lists/*

# Get the exact version of Chromium and download the matching ChromeDriver
RUN CHROME_VERSION=$(chromium --product-version | cut -d '.' -f 1,2,3) && \
    wget -q -O /tmp/chromedriver.zip "https://chromedriver.storage.googleapis.com/$CHROME_VERSION/chromedriver_linux64.zip" && \
    unzip /tmp/chromedriver.zip -d /usr/bin/ && \
    rm /tmp/chromedriver.zip

ENV CHROME_BIN=/usr/bin/chromium
ENV CHROME_DRIVER=/usr/bin/chromedriver
# Set working directory in container
WORKDIR /python-docker

# Copy and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose port 5000 for Cloud Run
EXPOSE 5000

# Copy the application code
COPY . .

# Start the application (assuming it's a Flask or similar app)
CMD [ "python3", "app.py"]
