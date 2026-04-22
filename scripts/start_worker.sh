#!/bin/bash
# Start the QLib train worker
# Usage: ./scripts/start_worker.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"

echo "=========================================="
echo "  QLib Train Worker"
echo "=========================================="

# Change to backend directory
cd "$BACKEND_DIR"

# Set default environment variables
export DATABASE_URL="${DATABASE_URL:-sqlite:///./qlib_management.db}"
export SECRET_KEY="${SECRET_KEY:-dev-secret-key-change-in-production}"
export TRAINING_SERVER_URL="${TRAINING_SERVER_URL:-http://ddns.hoo.ink:8000}"

echo "Database: $DATABASE_URL"
echo "Training server: $TRAINING_SERVER_URL"
echo ""

# Start the train worker
echo "Starting train worker..."
exec python train_worker.py
