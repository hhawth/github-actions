# Use a slim Python image for minimal base image
FROM python:3.11-slim

# Stage 1: Install dependencies
RUN apt-get update && apt-get install -y wget chromium && rm -rf /var/lib/apt/lists/*

# Stage 2: Download ChromeDriver (assuming same context)
RUN CHROME_VERSION=$(chromium --version | awk '{print $2}' | cut -d '.' -f 1-3) 
RUN echo "Detected Chrome version: $CHROME_VERSION"
RUN wget -q -O /tmp/chromedriver.zip "https://chromedriver.storage.googleapis.com/$CHROME_VERSION/chromedriver_linux64.zip"

# Stage 3: Final image (copy and unzip ChromeDriver)
FROM python:3.11-slim  # Switch to the final Python image
COPY --from=0 /tmp/chromedriver.zip /tmp/chromedriver.zip  # Copy from previous stage
RUN unzip /tmp/chromedriver.zip -d /usr/bin/
RUN rm /tmp/chromedriver.zip

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
