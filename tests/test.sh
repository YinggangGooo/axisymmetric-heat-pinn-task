#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
chmod 700 /logs/verifier
chmod -R go-rwx /tests
/usr/local/bin/python /tests/verify.py
