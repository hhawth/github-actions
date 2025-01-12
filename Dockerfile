# Use a slim Python image for minimal base image
FROM python:3.11-slim

WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

EXPOSE 5000

COPY . .


# Start the application (assuming it's a Flask or similar app)
CMD [ "python3", "app.py"]
