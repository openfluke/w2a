#!/usr/bin/env bash
# Build platform wheels for PyPI from python/src/welvet/<plat>/ binaries.
#
# Usage:
#   ./scripts/build_pypi_wheels.sh           # linux_amd64 + macos_arm64 if present
#   ./scripts/build_pypi_wheels.sh linux_amd64 macos_arm64
#
# Output: python/dist/*.whl
# Upload:  twine upload python/dist/*.whl
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY_ROOT="$ROOT/python"
SRC="$PY_ROOT/src/welvet"
DIST="$PY_ROOT/dist"
BUILD="$PY_ROOT/build"
VENV="$PY_ROOT/.venv-release"

cd "$ROOT"

PLATS=("$@")
if [[ ${#PLATS[@]} -eq 0 ]]; then
	PLATS=()
	[[ -f "$SRC/linux_amd64/welvet.so" ]] && PLATS+=(linux_amd64)
	[[ -f "$SRC/macos_arm64/welvet.dylib" ]] && PLATS+=(macos_arm64)
	[[ -f "$SRC/macos_amd64/welvet.dylib" ]] && PLATS+=(macos_amd64)
	[[ -f "$SRC/windows_amd64/welvet.dll" ]] && PLATS+=(windows_amd64)
fi
if [[ ${#PLATS[@]} -eq 0 ]]; then
	echo "ERROR: no platform binaries under $SRC" >&2
	echo "  ./scripts/fetch_cabi.sh cabi-v1.1.2" >&2
	exit 1
fi

if [[ ! -d "$VENV" ]]; then
	python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip -q install -U pip build wheel setuptools twine

wheel_tag_for() {
	case "$1" in
		linux_amd64) echo "py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64" ;;
		linux_arm64) echo "py3-none-manylinux_2_17_aarch64.manylinux2014_aarch64" ;;
		macos_arm64) echo "py3-none-macosx_11_0_arm64" ;;
		macos_amd64) echo "py3-none-macosx_10_14_x86_64" ;;
		windows_amd64) echo "py3-none-win_amd64" ;;
		*) echo "py3-none-any" ;;
	esac
}

rm -rf "$DIST" "$BUILD"
mkdir -p "$DIST"

# Stash all platform dirs, restore one at a time so each wheel only ships that OS.
STASH="$(mktemp -d)"
trap 'for d in "$STASH"/*; do [[ -d "$d" ]] && mv "$d" "$SRC/"; done; rm -rf "$STASH"' EXIT

for d in "$SRC"/linux_* "$SRC"/macos_* "$SRC"/windows_* "$SRC"/android_* "$SRC"/ios_*; do
	[[ -d "$d" ]] || continue
	mv "$d" "$STASH/"
done

VER="$(python3 - <<'PY'
from pathlib import Path
import re
text = Path("python/src/welvet/__init__.py").read_text(encoding="utf-8")
print(re.search(r'__version__\s*=\s*"([^"]+)"', text).group(1))
PY
)"

for plat in "${PLATS[@]}"; do
	if [[ ! -d "$STASH/$plat" ]]; then
		echo "skip missing $plat" >&2
		continue
	fi
	echo "════════════════════════════════════════"
	echo " wheel · $plat"
	echo "════════════════════════════════════════"
	mv "$STASH/$plat" "$SRC/$plat"

	(cd "$PY_ROOT" && python -m build --wheel --outdir "$DIST")

	# Retag py3-none-any → platform tag (ctypes + bundled native libs).
	any="$(ls "$DIST"/welvet-"$VER"-*.whl | head -1)"
	tag="$(wheel_tag_for "$plat")"
	base="welvet-${VER}-${tag}.whl"
	mv "$any" "$DIST/$base"
	echo "→ $base  $(du -h "$DIST/$base" | cut -f1)"

	mv "$SRC/$plat" "$STASH/$plat"
	rm -rf "$BUILD" "$PY_ROOT"/*.egg-info "$PY_ROOT"/src/*.egg-info 2>/dev/null || true
done

echo ""
echo "Wheels:"
ls -lh "$DIST"/*.whl
echo ""
echo "Upload:"
echo "  source $VENV/bin/activate"
echo "  twine upload $DIST/*.whl"
