#!/usr/bin/env bash
set -e

# AI-Based Medical Image Diagnosis System Startup Script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "================================================================="
echo "   Launching ClinAI Medical Image Diagnosis & Triage System"
echo "================================================================="

# Start Uvicorn FastAPI server
python3 scripts/run_system.py "$@"
