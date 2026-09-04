#!/usr/bin/env bash
# iOS device C-ABI static archive (c-archive) — arm64.
#
# Prereq: Xcode with iPhoneOS SDK
#   xcrun --sdk iphoneos --show-sdk-path
#
# Usage:
#   ./build_ios.sh
#   ./build_ios.sh --clean
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -s)" != Darwin ]]; then
	echo "ERROR: iOS builds require macOS + Xcode" >&2
	exit 1
fi

SDK="$(xcrun --sdk iphoneos --show-sdk-path 2>/dev/null || true)"
if [[ -z "$SDK" ]]; then
	echo "ERROR: iPhoneOS SDK not found. Install Xcode and accept the license." >&2
	exit 1
fi
echo "→ iOS SDK: $SDK"

./internal/build/build_unix.sh "$@" ios arm64
echo "→ output: internal/build/dist/ios_arm64/welvet.a (+ welvet.h)"
