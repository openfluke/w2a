#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "=== tdeploy/react: npm install from registry ==="
rm -rf node_modules package-lock.json dist public/main.wasm public/wasm_exec.js
npm install --no-fund --no-audit
npm run sync-wasm
npm test
echo "=== tdeploy/react ALL OK ==="
echo "Dev UI: npm run dev → http://localhost:5177/"
