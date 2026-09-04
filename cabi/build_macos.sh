#!/usr/bin/env bash
# macOS dylib — native arch by default; pass "universal" for arm64+amd64 lipo.
#
# Usage:
#   ./build_macos.sh              # native (arm64 or amd64)
#   ./build_macos.sh universal    # both slices + lipo
#   ./build_macos.sh --test
#   ./build_macos.sh --clean universal
set -euo pipefail
cd "$(dirname "$0")/internal/build"

FLAGS=()
ARCH=""
for arg in "$@"; do
	case "$arg" in
		-clean|--clean) FLAGS+=(--clean) ;;
		-test|--test) FLAGS+=(--test) ;;
		-soft|--soft) FLAGS+=(--soft) ;;
		universal|amd64|arm64) ARCH="$arg" ;;
		*)
			echo "usage: $0 [--clean] [--test] [universal|amd64|arm64]" >&2
			exit 1
			;;
	esac
done

if [[ -z "$ARCH" ]]; then
	case "$(uname -m)" in
		arm64|aarch64) ARCH=arm64 ;;
		*) ARCH=amd64 ;;
	esac
fi

./build_unix.sh ${FLAGS[@]+"${FLAGS[@]}"} darwin "$ARCH"
