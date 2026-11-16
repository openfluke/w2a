#!/usr/bin/env bash
# Pack python/src/welvet/macos_arm64 → dist/welvet-cabi-macos-arm64-<ver>.tar.gz
#
# macOS Apple silicon only (arm64 + Metal/MLX). No universal / x86_64.
#
# Usage:
#   ./scripts/pack_cabi_macos.sh              # pack existing macos_arm64 tree
#   ./scripts/pack_cabi_macos.sh --build      # build arm64 CABI first, then pack
#
# Needs: tar. Optional --build needs Go + Xcode clang (+ Homebrew mlx for MLX).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD=0
for arg in "$@"; do
	case "$arg" in
		--build) BUILD=1 ;;
		-h|--help)
			sed -n '2,14p' "$0"
			exit 0
			;;
		*)
			echo "unknown flag: $arg (arm64-only; no universal/amd64)" >&2
			exit 2
			;;
	esac
done

if [[ "$BUILD" -eq 1 ]]; then
	echo "→ building macOS arm64 CABI (Metal + MLX)…"
	(cd cabi && ./build_macos.sh --test arm64)
fi

PLAT="$ROOT/python/src/welvet/macos_arm64"
if [[ ! -f "$PLAT/welvet.dylib" ]]; then
	echo "ERROR: missing $PLAT/welvet.dylib — run with --build or:" >&2
	echo "  cd cabi && ./build_macos.sh --test arm64" >&2
	exit 1
fi
if [[ ! -f "$PLAT/libwelvet_metal.dylib" ]]; then
	echo "ERROR: missing Metal sibling $PLAT/libwelvet_metal.dylib" >&2
	exit 1
fi
if [[ ! -f "$PLAT/libwelvet_mlx.dylib" ]]; then
	echo "ERROR: missing MLX sibling $PLAT/libwelvet_mlx.dylib" >&2
	exit 1
fi

VER="$(python3 - <<'PY'
from pathlib import Path
import re
text = Path("python/src/welvet/__init__.py").read_text(encoding="utf-8")
m = re.search(r'ENGINE_VERSION\s*=\s*"([^"]+)"', text)
print(m.group(1) if m else "0.0.0")
PY
)"

DIST="$ROOT/dist"
mkdir -p "$DIST"
NAME="welvet-cabi-macos-arm64-${VER}.tar.gz"
OUT="$DIST/$NAME"

echo "→ packing $PLAT → $OUT"
tar -C "$PLAT" -czf "$OUT" .
echo "→ $(du -h "$OUT" | cut -f1)  $NAME"
echo "$OUT"
