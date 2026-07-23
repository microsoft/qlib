#!/bin/bash
# Run integration tests against the backend API
# Usage: ./scripts/run_tests.sh [--base-url http://localhost:8000]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE_URL="${1:-http://localhost:8000}"

echo "=========================================="
echo "  QLib Integration Tests"
echo "=========================================="
echo "Target: $BASE_URL"
echo ""

# Run integration tests
cd "$PROJECT_ROOT"
python scripts/integration_test.py --base-url "$BASE_URL"
