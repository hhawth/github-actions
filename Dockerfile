# Use a slim Python image for minimal base image
FROM selenium/python-detox

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy and install Python dependencies

# Expose port 5000 for Cloud Run
EXPOSE 5000

# Copy the application code
COPY . .

# Start the application (assuming it's a Flask or similar app)
CMD [ "python3", "app.py"]
