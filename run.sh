#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "============================================================"
echo "  SONG POPULARITY MLOPS — FULL PIPELINE"
echo "============================================================"

# Step 1: Start infrastructure
echo ""
echo "[1/5] Starting infrastructure (MariaDB + Redis + MLflow)..."
docker compose up -d mariadb redis mlflow

# Wait for services to be healthy
echo "  Waiting for MariaDB to be healthy..."
until docker compose exec mariadb healthcheck.sh --connect --innodb_initialized 2>/dev/null; do
  sleep 2
done
echo "  ✓ MariaDB ready"

echo "  Waiting for Redis to be ready..."
until docker compose exec redis redis-cli ping 2>/dev/null | grep -q PONG; do
  sleep 1
done
echo "  ✓ Redis ready"

# Step 2: Run pipeline
echo ""
echo "[2/5] Running pipeline (ingestion → validation → preprocessing → training)..."
docker compose --profile pipeline run --rm pipeline
echo "  ✓ Pipeline complete"

# Step 3: Build and start backend
echo ""
echo "[3/5] Building and starting FastAPI backend..."
docker compose up -d --build backend
echo "  ✓ Backend running on http://localhost:8000"

# Step 4: Start frontend
echo ""
echo "[4/5] Starting frontend..."
docker compose up -d frontend
echo "  ✓ Frontend running on http://localhost:5500"

# Step 5: Show status
echo ""
echo "[5/5] Final status:"
echo "============================================================"
docker compose ps
echo ""
echo "  Frontend:  http://localhost:5500"
echo "  FastAPI:   http://localhost:8000/docs"
echo "  MLflow:    http://localhost:5000"
echo "============================================================"
