# Use an official PyTorch base image with CUDA 12.1 support
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /workspace

# Set environment variables
ENV DOCKER_MODE=true
ENV PYTHONUNBUFFERED=1

# Copy requirements first to maximize Docker build caching
COPY requirements.txt /workspace/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the pipeline script into the workspace
COPY cmrminivlm.py /workspace/

# Create standard directories for host volume mounting
RUN mkdir -p /workspace/data /workspace/emidec_data

# Execute the script when the container launches
CMD ["python", "cmrminivlm.py"]
