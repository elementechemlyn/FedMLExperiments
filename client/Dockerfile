# client/Dockerfile
# Use a slim Python image for smaller size
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the client application code
COPY client.py .

# Create a directory for client data
RUN mkdir -p data

# No CMD here, as it will be set dynamically when running the container
# to pass client_id and server_url
