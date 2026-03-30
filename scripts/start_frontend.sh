#!/bin/bash
# Start the QLib frontend development server
# Usage: ./scripts/start_frontend.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

echo "=========================================="
echo "  QLib Frontend Dev Server"
echo "=========================================="

# Change to frontend directory
cd "$FRONTEND_DIR"

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
    echo ""
fi

# Start the frontend dev server
echo "Starting frontend on http://localhost:3001"
echo "API proxy: /api -> http://localhost:8000"
echo ""
exec npm run dev
