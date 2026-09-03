#!/usr/bin/env bash
# Build Welvet 1.1.1 WASM for @openfluke/welvet TypeScript package.
# Uses Go -overlay so broken non-amd64/arm64 simd stubs are fixed without
# editing the welvet engine tree.
set -euo pipefail

# Resolve through symlinks so go -overlay keys match the paths the compiler sees.
SCRIPT_DIR="$(cd -P "$(dirname "$0")" && pwd)"
W2A_ROOT="$(cd -P "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$W2A_ROOT/typescript/assets"
WELVET_ROOT="$(cd -P "$W2A_ROOT/../.." && pwd)"
OVERLAY_DIR="$SCRIPT_DIR/overlays"
OVERLAY_JSON="$SCRIPT_DIR/.simd_overlay.json"
SIMD_DIR="$WELVET_ROOT/simd"

mkdir -p "$DIST_DIR"

python3 - <<PY
import json, os
simd = os.path.realpath("$SIMD_DIR")
ov = os.path.realpath("$OVERLAY_DIR")
# Include both realpath and any symlink alias under ~/git/welvet.
aliases = {simd}
link = os.path.expanduser("~/git/welvet/simd")
if os.path.isdir(link):
    aliases.add(os.path.abspath(link))
replace = {}
for base in aliases:
    for name in ("dot_u8_stub.go", "saxpy_shifted_input_stub.go", "stub.go"):
        replace[os.path.join(base, name)] = os.path.join(ov, name)
with open("$OVERLAY_JSON", "w") as f:
    json.dump({"Replace": replace}, f, indent=2)
print("Wrote $OVERLAY_JSON")
PY

echo "Building Welvet WASM (GOOS=js GOARCH=wasm) with simd overlay…"
cd "$W2A_ROOT"
env GOOS=js GOARCH=wasm go build -overlay "$OVERLAY_JSON" -o "$DIST_DIR/main.wasm" ./wasm

echo "Copying wasm_exec.js…"
GOROOT="$(go env GOROOT)"
WASM_EXEC="$GOROOT/lib/wasm/wasm_exec.js"
if [[ ! -f "$WASM_EXEC" ]]; then
  WASM_EXEC="$GOROOT/misc/wasm/wasm_exec.js"
fi
cp "$WASM_EXEC" "$DIST_DIR/wasm_exec.js"
cp "$WASM_EXEC" "$SCRIPT_DIR/wasm_exec.js"

echo "Copying HTML verify pages…"
cp "$SCRIPT_DIR"/*.html "$DIST_DIR/" 2>/dev/null || true

echo "Build complete: $DIST_DIR/main.wasm"
ls -lh "$DIST_DIR/main.wasm"
