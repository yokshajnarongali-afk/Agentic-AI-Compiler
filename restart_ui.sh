#!/bin/bash
# =============================================================================
#   AGentic_C — UI Server Restart Script
#   Kills any process on port 5050, clears Python cache, and relaunches fresh.
# =============================================================================

echo ""
echo "========================================"
echo "  AGentic_C — Restarting UI Server"
echo "========================================"
echo ""

# ── Step 1: Kill any process currently using port 5050 ────────────────────
echo "[1/4] Killing any existing process on port 5050..."
PID=$(lsof -ti tcp:5050)
if [ -n "$PID" ]; then
    kill -9 $PID
    echo "      ✓ Killed process PID: $PID"
else
    echo "      ✓ No process found on port 5050 (already clear)"
fi

sleep 1

# ── Step 2: Clear Python __pycache__ to prevent stale bytecode ────────────
echo "[2/4] Clearing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "      ✓ Cache cleared"

# ── Step 3: Confirm the new index.html has the correct version ────────────
echo "[3/4] Verifying new UI file..."
if grep -q "AGentic_C Enterprise" src/web_ui/static/index.html; then
    echo "      ✓ New white UI (index.html) confirmed"
else
    echo "      ⚠ Warning: index.html may not have the new UI. Check the file."
fi

# ── Step 4: Launch the server fresh ───────────────────────────────────────
echo "[4/4] Launching fresh UI server on http://localhost:5050 ..."
echo ""
echo "      Dashboard will open automatically in your browser."
echo "      Press Ctrl+C to stop the server."
echo ""

# Navigate to the project root and start the server
cd "$(dirname "$0")"
python3 src/web_ui/app.py

