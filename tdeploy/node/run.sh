#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "=== tdeploy/node: npm install from registry ==="
rm -rf node_modules package-lock.json
npm install --no-fund --no-audit
npm test
echo "=== tdeploy/node ALL OK ==="
