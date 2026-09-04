#!/usr/bin/env bash
# Android C-ABI (.so) via NDK — arm64 + x86_64 (emulator).
#
# Prereq: Android SDK NDK under ANDROID_HOME/ndk or ANDROID_NDK_HOME.
#   Typical macOS: ~/Library/Android/sdk  (ANDROID_HOME)
#
# Usage:
#   ./build_android.sh              # arm64 + x86_64
#   ./build_android.sh arm64        # device only
#   ./build_android.sh --clean
set -euo pipefail
cd "$(dirname "$0")"

FLAGS=()
ARCHES=()
for arg in "$@"; do
	case "$arg" in
		-clean|--clean) FLAGS+=(--clean) ;;
		-soft|--soft) FLAGS+=(--soft) ;;
		arm64|amd64|x86_64) ARCHES+=("$arg") ;;
		all) ARCHES=(arm64 amd64) ;;
		*)
			echo "usage: $0 [--clean] [all|arm64|amd64]..." >&2
			exit 1
			;;
	esac
done
[[ ${#ARCHES[@]} -eq 0 ]] && ARCHES=(arm64 amd64)

if [[ -z "${ANDROID_NDK_HOME:-}${ANDROID_NDK_ROOT:-}" ]]; then
	if [[ -z "${ANDROID_HOME:-}" && -d "$HOME/Library/Android/sdk" ]]; then
		export ANDROID_HOME="$HOME/Library/Android/sdk"
		echo "→ ANDROID_HOME=$ANDROID_HOME"
	fi
fi

first=1
for arch in "${ARCHES[@]}"; do
	[[ "$arch" == x86_64 ]] && arch=amd64
	extra=()
	if [[ $first -eq 1 ]]; then
		extra=(${FLAGS[@]+"${FLAGS[@]}"})
	else
		for f in ${FLAGS[@]+"${FLAGS[@]}"}; do
			[[ "$f" == "--clean" || "$f" == "-clean" ]] || extra+=("$f")
		done
	fi
	./internal/build/build_unix.sh ${extra[@]+"${extra[@]}"} android "$arch"
	first=0
done

echo "→ outputs under internal/build/dist/android_*/"
