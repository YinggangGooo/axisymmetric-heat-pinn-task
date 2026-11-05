#!/bin/bash
set -Eeuo pipefail
if [ "$(id -u)" -ne 0 ]; then
  echo "INFRASTRUCTURE ERROR: verifier must run as root in the task container" >&2
  exit 2
fi
mkdir -p /logs/verifier
chmod 700 /logs/verifier
chmod -R go-rwx /tests
exec /opt/venv/bin/python -I -B /tests/verify.py
