#!/usr/bin/env bash
# StudyBuddy launcher for macOS / Linux.   Usage:  ./start_unix.sh
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "First run: creating virtual environment..."
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python -m ml.train
echo "Starting... open http://127.0.0.1:${STUDYBUDDY_PORT:-5000} in your browser. The first start can take up to a minute; press Ctrl+C once to stop."
exec .venv/bin/python app.py
