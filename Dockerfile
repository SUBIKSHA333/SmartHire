# Use Python 3.10 as the base
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (for faster rebuilds via Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]