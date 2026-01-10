# Use a slim Python image for minimal base image
FROM python:3.11-slim

WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Streamlit runs on port 8501 by default
EXPOSE 8501

COPY . .

# Start the Streamlit application
CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0"]
