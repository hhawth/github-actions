# Use a slim Python image for minimal base image
FROM python:3.11-slim

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir flask selenium webdriver-manager
RUN pip3 install --no-cache-dir -r requirements.txt

# Install all dependencies in a single RUN command
RUN pip install --no-cache-dir flask selenium webdriver-manager \
    && apt-get update && apt-get install -y unzip xvfb \
    && webdriver-manager download chromedriver --version=latest --arch=linux64 --destination=/app/chromedriver

# Set working directory in container
WORKDIR /python-docker

# Copy and install Python dependencies

# Expose port 5000 for Cloud Run
EXPOSE 5000

# Copy the application code
COPY . .

# Start the application (assuming it's a Flask or similar app)
CMD [ "python3", "app.py"]
