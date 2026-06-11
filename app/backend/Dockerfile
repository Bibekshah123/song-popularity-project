FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for MariaDB client
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libmariadb-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY models /app/models
COPY data /app/data

EXPOSE 8000
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
