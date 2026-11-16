#!/usr/bin/env bash
# Download prebuilt CABI into python/src/welvet/<platform>/ from a GitHub Release.
#
# Usage:
#   ./scripts/fetch_cabi.sh                  # latest release, detect platform
#   ./scripts/fetch_cabi.sh cabi-v1.1.2      # specific release tag
#   ./scripts/fetch_cabi.sh --version 1.1.2  # match ENGINE_VERSION in asset name
#   ./scripts/fetch_cabi.sh --plat macos_arm64 cabi-v1.1.2
#   ./scripts/fetch_cabi.sh --all cabi-v1.1.2
#
# Needs: curl. Optional: gh (preferred).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="openfluke/w2a"
TAG="latest"
ENGINE_VER=""
FORCE_PLAT=""
FETCH_ALL=0

while [[ $# -gt 0 ]]; do
	case "$1" in
		--version)
			ENGINE_VER="${2:?}"
			shift 2
			;;
		--plat)
			FORCE_PLAT="${2:?}"
			shift 2
			;;
		--all)
			FETCH_ALL=1
			shift
			;;
		-h|--help)
			sed -n '2,16p' "$0"
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

detect_plat() {
	local sys mach arch
	sys="$(uname -s | tr '[:upper:]' '[:lower:]')"
	mach="$(uname -m)"
	case "$mach" in
		x86_64|amd64) arch=amd64 ;;
		aarch64|arm64) arch=arm64 ;;
		*) arch="$mach" ;;
	esac
	case "$sys" in
		linux) echo "linux_${arch}" ;;
		darwin) echo "macos_${arch}" ;;
		*)
			echo "ERROR: unsupported platform $sys/$mach" >&2
			return 1
			;;
	esac
}

# plat dir uses underscores; release asset uses hyphens (linux-amd64).
asset_for_plat() {
	local plat="$1"
	echo "welvet-cabi-${plat//_/-}-${ENGINE_VER}.tar.gz"
}

fetch_one() {
	local plat="$1"
	local asset dest tmp archive url
	asset="$(asset_for_plat "$plat")"
	dest="$ROOT/python/src/welvet/${plat}"
	mkdir -p "$dest"
	tmp="$(mktemp -d)"
	echo "→ fetching ${asset} (tag=${TAG}) → ${plat}"

	if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
		if [[ "$TAG" == "latest" ]]; then
			gh release download --repo "$REPO" -p "$asset" -D "$tmp" || true
		else
			gh release download "$TAG" --repo "$REPO" -p "$asset" -D "$tmp" || true
		fi
	fi
	archive="$tmp/$asset"
	if [[ ! -f "$archive" ]]; then
		if [[ "$TAG" == "latest" ]]; then
			url="https://github.com/${REPO}/releases/latest/download/${asset}"
		else
			url="https://github.com/${REPO}/releases/download/${TAG}/${asset}"
		fi
		curl -fsSL -L -o "$archive" "$url"
	fi
	if [[ ! -f "$archive" ]]; then
		echo "ERROR: asset not found: $asset" >&2
		rm -rf "$tmp"
		return 1
	fi
	# Clear prior extract (ignore AppleDouble junk)
	find "$dest" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
	tar -xzf "$archive" -C "$dest"
	find "$dest" -name '._*' -delete
	rm -rf "$tmp"
	echo "→ installed into $dest"
	ls -lh "$dest"
}

if [[ "$FETCH_ALL" -eq 1 ]]; then
	fetch_one linux_amd64
	fetch_one macos_arm64
	exit 0
fi

PLAT="${FORCE_PLAT:-$(detect_plat)}"
fetch_one "$PLAT"
