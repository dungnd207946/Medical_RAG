# Use the Anaconda base image
FROM python:3.10-slim

# Install necessary packages using Conda and Pip
# RUN conda install -c pytorch faiss-cpu -y && \
#     pip install flask numpy pandas
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

RUN pip install --no-cache-dir \
    faiss-cpu \
    flask \
    numpy \
    pandas

# Copy the application files into the container
COPY ./server.py /app/server.py
# COPY ./faiss_indices /app/faiss_indices
# COPY ./PMIDs /app/PMIDs

# Set environment variables
ENV FLASK_APP=server.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Expose port 5000 for communication with the Flask app
EXPOSE 5000

# Define the command that runs when the container starts
CMD ["python", "server.py"]

# docker run -d --name faiss_cpt -p 5000:5000 faiss:latest