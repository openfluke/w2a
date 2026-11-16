#!/usr/bin/env bash
# Download prebuilt CABI into python/src/welvet/<platform>/ from a GitHub Release.
#
# Usage:
#   ./scripts/fetch_cabi.sh                  # latest release, detect platform
#   ./scripts/fetch_cabi.sh v1.7.3           # specific w2a / welvet-aligned tag
#   ./scripts/fetch_cabi.sh --version 1.1.2  # match ENGINE_VERSION in asset name
#
# Needs: curl. Optional: gh (preferred).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="openfluke/w2a"
TAG="latest"
ENGINE_VER=""

while [[ $# -gt 0 ]]; do
	case "$1" in
		--version)
			ENGINE_VER="${2:?}"
			shift 2
			;;
		-h|--help)
			sed -n '2,14p' "$0"
			exit 0
			;;
		*)
			TAG="$1"
			shift
			;;
	esac
done

if [[ -z "$ENGINE_VER" ]]; then
	ENGINE_VER="$(python3 - <<'PY'
from pathlib import Path
import re
text = Path("python/src/welvet/__init__.py").read_text(encoding="utf-8")
m = re.search(r'ENGINE_VERSION\s*=\s*"([^"]+)"', text)
print(m.group(1) if m else "1.1.2")
PY
)"
fi
cd "$ROOT"

sys="$(uname -s | tr '[:upper:]' '[:lower:]')"
mach="$(uname -m)"
case "$mach" in
	x86_64|amd64) arch=amd64 ;;
	aarch64|arm64) arch=arm64 ;;
	*) arch="$mach" ;;
esac
case "$sys" in
	linux) plat="linux_${arch}" ;;
	darwin) plat="macos_${arch}" ;;
	*)
		echo "ERROR: unsupported platform $sys/$mach" >&2
		exit 1
		;;
esac

ASSET="welvet-cabi-${plat}-${ENGINE_VER}.tar.gz"
DEST="$ROOT/python/src/welvet/${plat}"
mkdir -p "$DEST"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "→ fetching ${ASSET} (tag=${TAG})…"

download() {
	local url="$1" out="$2"
	curl -fsSL -L -o "$out" "$url"
}

if command -v gh >/dev/null 2>&1; then
	if [[ "$TAG" == "latest" ]]; then
		gh release download --repo "$REPO" -p "$ASSET" -D "$TMP"
	else
		gh release download "$TAG" --repo "$REPO" -p "$ASSET" -D "$TMP"
	fi
else
	if [[ "$TAG" == "latest" ]]; then
		url="https://github.com/${REPO}/releases/latest/download/${ASSET}"
	else
		url="https://github.com/${REPO}/releases/download/${TAG}/${ASSET}"
	fi
	download "$url" "$TMP/$ASSET"
fi

ARCHIVE="$TMP/$ASSET"
if [[ ! -f "$ARCHIVE" ]]; then
	# gh may have downloaded with exact name; pick first tar.gz
	ARCHIVE="$(find "$TMP" -name '*.tar.gz' | head -1)"
fi
if [[ ! -f "$ARCHIVE" ]]; then
	echo "ERROR: asset not found: $ASSET" >&2
	echo "  build locally:  ./scripts/pack_cabi_linux.sh --build" >&2
	exit 1
fi

tar -xzf "$ARCHIVE" -C "$DEST"
echo "→ installed into $DEST"
ls -lh "$DEST"/welvet.so "$DEST"/libwelvet*.so 2>/dev/null || ls -lh "$DEST"
