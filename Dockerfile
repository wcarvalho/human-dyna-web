FROM python:3.10-slim

RUN apt update

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -e libraries/nicewebrl -e libraries/housemaze -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV GOOGLE_APPLICATION_CREDENTIALS ./keys/datastore-key.json

# Command to run your application using Uvicorn
CMD ["python", "main.py"]