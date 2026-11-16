#!/usr/bin/env bash
# Copy C-ABI dist/* into apps/w2a/python/src/welvet/
set -euo pipefail
cd "$(dirname "$0")"
SRC="${1:-./dist}"
DST="$(cd ../../../python/src/welvet && pwd)"
WELVET_ROOT="$(cd ../../../../welvet && pwd)"

if [[ ! -d "$SRC" ]]; then
	echo "Source not found: $SRC — build first" >&2
	exit 1
fi

mkdir -p "$DST"
cp -rv "$SRC"/* "$DST"/

# Sibling accel libs next to welvet.so (Python pins WELVET_*_LIBRARY to these).
sync_native() {
	local plat="$1" src="$2" name="$3"
	local dest_dir="$DST/$plat"
	[[ -d "$dest_dir" ]] || return 0
	[[ -f "$src" ]] || return 0
	cp -v "$src" "$dest_dir/$name"
}
sync_native linux_amd64 "$WELVET_ROOT/native/cuda/linux_x86_64/libwelvet_cuda.so" libwelvet_cuda.so
sync_native linux_amd64 "$WELVET_ROOT/native/rocm/linux_x86_64/libwelvet_rocm.so" libwelvet_rocm.so
sync_native linux_amd64 "$WELVET_ROOT/native/vulkan/linux_x86_64/libwelvet_vulkan.so" libwelvet_vulkan.so
sync_native linux_amd64 "$WELVET_ROOT/native/cpp/linux_amd64/libwelvet.so" libwelvet.so
sync_native windows_amd64 "$WELVET_ROOT/native/cuda/windows_amd64/libwelvet_cuda.dll" libwelvet_cuda.dll
# macOS arm64 only — Metal (dlopen) + MLX/nativecpp (cgo @rpath)
sync_native macos_arm64 "$WELVET_ROOT/native/metal/darwin_arm64/libwelvet_metal.dylib" libwelvet_metal.dylib
sync_native macos_arm64 "$WELVET_ROOT/native/mlx/darwin_arm64/libwelvet_mlx.dylib" libwelvet_mlx.dylib
sync_native macos_arm64 "$WELVET_ROOT/native/cpp/darwin_arm64/libwelvet.dylib" libwelvet.dylib

# Point packaged macos_arm64 welvet.dylib at sibling dylibs (@loader_path), not build-tree rpaths.
fix_macos_arm64_rpaths() {
	local dylib="$DST/macos_arm64/welvet.dylib"
	[[ -f "$dylib" ]] || return 0
	command -v install_name_tool >/dev/null 2>&1 || return 0
	command -v otool >/dev/null 2>&1 || return 0
	# Drop absolute build-tree rpaths; keep / add @loader_path for shipped siblings.
	while IFS= read -r p; do
		[[ -n "$p" ]] || continue
		[[ "$p" == "@loader_path" ]] && continue
		install_name_tool -delete_rpath "$p" "$dylib" 2>/dev/null || true
	done < <(otool -l "$dylib" | awk '/LC_RPATH/{p=1} p&&/path/{sub(/^.*path /,""); sub(/ \(offset.*/,""); print; p=0}')
	install_name_tool -add_rpath "@loader_path" "$dylib" 2>/dev/null || true
	echo "  rpath → @loader_path on $dylib"
	otool -l "$dylib" | awk '/LC_RPATH/{p=1} p&&/path/{print; p=0}'
}
fix_macos_arm64_rpaths

echo "Mirrored $SRC → $DST (+ native accel siblings when present)"
ls -lh "$DST"/*/welvet.* 2>/dev/null | head -20 || true
ls -lh "$DST"/macos_arm64/libwelvet*.dylib 2>/dev/null || true
