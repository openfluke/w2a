#!/usr/bin/env bash
# Run all tdeploy targets: node + bun + react (registry @openfluke/welvet).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
FAIL=0

run() {
  local name="$1"
  echo ""
  echo "########## $name ##########"
  if bash "$ROOT/$name/run.sh"; then
    echo "########## $name OK ##########"
  else
    echo "########## $name FAIL ##########" >&2
    FAIL=1
  fi
}

run node
run bun
run react

echo ""
if [[ "$FAIL" -ne 0 ]]; then
  echo "=== tdeploy FAILED ===" >&2
  exit 1
fi
echo "=== tdeploy ALL OK (node + bun + react) ==="
