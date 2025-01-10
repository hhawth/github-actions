# Use a slim Python image for minimal base image
FROM selenium/node-chrome-debug

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/* 

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy and install Python dependencies

# Expose port 5000 for Cloud Run
EXPOSE 5000

# Copy the application code
COPY . .

# Start the application (assuming it's a Flask or similar app)
CMD [ "python3", "app.py"]
