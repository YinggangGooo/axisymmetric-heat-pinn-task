#!/bin/bash
set -euo pipefail
cd /app
python -m pip install --disable-pip-version-check --no-deps scipy==1.15.3
python /solution/fix_code.py
