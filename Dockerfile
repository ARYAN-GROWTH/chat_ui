# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirement file first to install dependencies (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the FastAPI port
EXPOSE 8001

# Default command to run server
CMD ["uvicorn", "src.main:app", "--reload", "--host", "0.0.0.0", "--port", "8001"]
