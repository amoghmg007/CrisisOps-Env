FROM public.ecr.aws/docker/library/python:3.10-slim

WORKDIR /app

# Install system dependencies if any (none needed for now)
# RUN apt-get update && apt-get install -y --no-install-recommends ...

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Ensure server package is discoverable
ENV PYTHONPATH=/app

# Start the FastAPI server using the absolute module path
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]