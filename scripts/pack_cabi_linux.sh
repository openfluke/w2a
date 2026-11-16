#!/usr/bin/env bash
# Pack python/src/welvet/linux_amd64 → dist/welvet-cabi-linux-amd64-<ver>.tar.gz
#
# Usage:
#   ./scripts/pack_cabi_linux.sh              # pack existing linux_amd64 tree
#   ./scripts/pack_cabi_linux.sh --build      # build CABI first, then pack
#
# Needs: tar. Optional --build needs Go + gcc.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD=0
for arg in "$@"; do
	case "$arg" in
		--build) BUILD=1 ;;
		-h|--help)
			sed -n '2,12p' "$0"
			exit 0
			;;
		*)
			echo "unknown flag: $arg" >&2
			exit 2
			;;
	esac
done

if [[ "$BUILD" -eq 1 ]]; then
	echo "→ building linux amd64 CABI…"
	(cd cabi/internal/build && ./build_linux.sh --test amd64)
fi

PLAT="$ROOT/python/src/welvet/linux_amd64"
if [[ ! -f "$PLAT/welvet.so" ]]; then
	echo "ERROR: missing $PLAT/welvet.so — run with --build or:" >&2
	echo "  cd cabi/internal/build && ./build_linux.sh --test && ./copy_to_python.sh" >&2
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
NAME="welvet-cabi-linux-amd64-${VER}.tar.gz"
OUT="$DIST/$NAME"

echo "→ packing $PLAT → $OUT"
tar -C "$PLAT" -czf "$OUT" .
echo "→ $(du -h "$OUT" | cut -f1)  $NAME"
echo "$OUT"
