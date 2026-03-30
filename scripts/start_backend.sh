#!/bin/bash
# Start the QLib backend API server
# Usage: ./scripts/start_backend.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"

echo "=========================================="
echo "  QLib Backend Server"
echo "=========================================="

# Change to backend directory
cd "$BACKEND_DIR"

# Set default environment variables if not already set
export DATABASE_URL="${DATABASE_URL:-sqlite:///./qlib_management.db}"
export SECRET_KEY="${SECRET_KEY:-dev-secret-key-change-in-production}"
export ALGORITHM="${ALGORITHM:-HS256}"
export ACCESS_TOKEN_EXPIRE_MINUTES="${ACCESS_TOKEN_EXPIRE_MINUTES:-30}"
export SKIP_EMAIL_VERIFICATION="${SKIP_EMAIL_VERIFICATION:-True}"
export CORS_ORIGINS="${CORS_ORIGINS:-http://localhost:3001,http://localhost:3000,http://localhost:8000,http://127.0.0.1:3001,http://127.0.0.1:3000}"
export TRAINING_SERVER_URL="${TRAINING_SERVER_URL:-http://ddns.hoo.ink:8000}"

echo "Database: $DATABASE_URL"
echo "Skip email verification: $SKIP_EMAIL_VERIFICATION"
echo "Training server: $TRAINING_SERVER_URL"
echo ""

# Initialize database if needed
echo "Initializing database..."
python init_db.py 2>/dev/null || echo "Database initialization skipped (may already exist)"
echo ""

# Start the backend server
echo "Starting backend on http://0.0.0.0:8000"
echo "API docs: http://localhost:8000/docs"
echo ""
exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload
