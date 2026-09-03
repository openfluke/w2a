#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

TOOLS="$(pwd)/.tools"
BUN_BIN=""

ensure_bun() {
  if command -v bun >/dev/null 2>&1; then
    BUN_BIN="$(command -v bun)"
    return 0
  fi
  if [[ -x "$HOME/.bun/bin/bun" ]]; then
    export PATH="$HOME/.bun/bin:$PATH"
    BUN_BIN="$HOME/.bun/bin/bun"
    return 0
  fi
  if [[ -x "$TOOLS/bun" ]]; then
    BUN_BIN="$TOOLS/bun"
    return 0
  fi

  echo "Bun not on PATH — downloading linux-x64 binary into $TOOLS …"
  local ver="${BUN_VERSION:-1.2.21}"
  local zip="/tmp/bun-linux-x64-${ver}.zip"
  local url="https://github.com/oven-sh/bun/releases/download/bun-v${ver}/bun-linux-x64.zip"
  mkdir -p "$TOOLS"
  curl -fsSL -o "$zip" "$url"
  rm -rf /tmp/bun-extract && mkdir -p /tmp/bun-extract
  unzip -qo "$zip" -d /tmp/bun-extract
  local found
  found="$(find /tmp/bun-extract -type f -name bun | head -1)"
  [[ -n "$found" ]] || { echo "bun binary missing in zip"; exit 1; }
  install -m 755 "$found" "$TOOLS/bun"
  BUN_BIN="$TOOLS/bun"
  "$BUN_BIN" --version
}

ensure_bun
export PATH="$(dirname "$BUN_BIN"):$PATH"
echo "=== tdeploy/bun: bun install from registry (bun $($BUN_BIN --version)) ==="
rm -rf node_modules bun.lock bun.lockb package-lock.json
"$BUN_BIN" install
"$BUN_BIN" smoke.mjs
"$BUN_BIN" cameral.mjs
echo "=== tdeploy/bun ALL OK ==="
