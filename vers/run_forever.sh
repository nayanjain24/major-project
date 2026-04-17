#!/bin/bash
# Script to run VERS continuously and automatically restart it if it stops

cd "$(dirname "$0")"

echo "============================================="
echo "Starting VERS in continuous background mode..."
echo "To stop this completely, press Ctrl+C"
echo "============================================="

while true; do
  echo "[$(date)] Starting orchestrator..."
  
  # Run the dashboard mode
  .venv/bin/python src/orchestrate.py --mode dashboard
  
  echo ""
  echo "[$(date)] Application exited or crashed."
  echo "Restarting in 5 seconds... (Press Ctrl+C to stop)"
  sleep 5
done
