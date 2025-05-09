# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for dlib and OpenCV
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgtk2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirments.txt first (for caching efficiency)
COPY requirments.txt .

# Install required Python dependencies
RUN pip install --no-cache-dir -r requirments.txt

# Copy all the application files
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8083

# Run FastAPI using Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8083"]