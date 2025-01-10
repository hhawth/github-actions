# Use a slim Python image for minimal base image
FROM python:3.11-slim

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Download and install ChromeDriver automatically using webdriver-manager
RUN apt-get update && apt-get install -y unzip xvfb  # Install dependencies for ChromeDriver
RUN webdriver-manager download chromedriver --version=latest --arch=linux64 --destination=/app/chromedriver

# Set working directory in container
WORKDIR /python-docker

# Copy and install Python dependencies

# Expose port 5000 for Cloud Run
EXPOSE 5000

# Copy the application code
COPY . .

# Start the application (assuming it's a Flask or similar app)
CMD [ "python3", "app.py"]
