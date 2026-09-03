#!/usr/bin/env bash
# Welvet C-ABI build — Linux / macOS / cross targets.
set -euo pipefail
cd "$(dirname "$0")"

TARGET_OS=""
TARGET_ARCH=""
EXTRA_FLAGS=()

for arg in "$@"; do
	case "$arg" in
		-clean|--clean) EXTRA_FLAGS+=(-clean) ;;
		-test|--test) EXTRA_FLAGS+=(-test) ;;
		-soft|--soft) EXTRA_FLAGS+=(-soft) ;;
		all) TARGET_OS=all ;;
		linux|darwin|windows|android|ios) TARGET_OS="$arg" ;;
		amd64|arm64|x86_64|universal) TARGET_ARCH="$arg" ;;
		*)
			echo "Unknown argument: $arg" >&2
			exit 1
			;;
	esac
done

if [[ -z "$TARGET_OS" ]]; then
	case "$(uname -s)" in
		Linux) TARGET_OS=linux ;;
		Darwin) TARGET_OS=darwin ;;
		*) echo "Cannot auto-detect OS"; exit 1 ;;
	esac
fi

if [[ -z "$TARGET_ARCH" && "$TARGET_OS" != all ]]; then
	case "$(uname -m)" in
		x86_64|amd64) TARGET_ARCH=amd64 ;;
		aarch64|arm64) TARGET_ARCH=arm64 ;;
		*) echo "Cannot auto-detect arch"; exit 1 ;;
	esac
fi

if [[ "$TARGET_OS" == all ]]; then
	echo "Building all platforms (soft-skip missing toolchains)..."
	go run builder.go -os all -soft "${EXTRA_FLAGS[@]}"
else
	echo "Building $TARGET_OS $TARGET_ARCH..."
	go run builder.go -os "$TARGET_OS" -arch "$TARGET_ARCH" "${EXTRA_FLAGS[@]}"
fi

# Mirror to Python when dist exists
if [[ -d dist ]] && [[ -d ../../../python/src/welvet ]]; then
	echo "Mirroring to Python source..."
	./copy_to_python.sh
fi
