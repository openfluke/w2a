package suites

// AffinePackable reports whether a weight matrix of the given shape can be
// packed as FormatAffinePacked. quant.PackAffine4 uses a fixed group size of
// 64 columns (AffineG64Group) with 4-bit codes (8 codes/word): cols must be a
// multiple of both 64 (group) and 8 (codes/word) — the 64 check subsumes the 8
// check, but both are spelled out here to match the shape rule at the source.
func AffinePackable(rows, cols int) bool {
	return cols%64 == 0 && cols%8 == 0
}

// AffineSkipNote explains why a cell was skipped instead of attempting to
// pack AffinePacked weights that quant.PackAffine4 would reject outright.
func AffineSkipNote() string {
	return "AffinePacked requires cols%64==0 (group size); shape not packable"
}

// WebGPUKind reports the honest on-device status/note for a layer's WebGPU
// backend, so suite tables never claim "OK" for a path that still silently
// runs on host. layer is the lowercase layer name (e.g. "rmsnorm", "cnn1").
func WebGPUKind(layer string) (statusIfWorks string, note string) {
	switch layer {
	case "rmsnorm":
		return "OK", "RMSNorm fwd+bwd on-device (webgpu.RMSNorm/RMSNormBackward)"
	case "layernorm":
		return "OK", "LayerNorm fwd on-device; bwd host (webgpu.LayerNorm fwd only)"
	case "softmax":
		return "OK", "Softmax fwd+bwd on-device for standard/grid/hierarchical; other kinds host-only (error)"
	case "swiglu":
		return "OK", "SiLU⊙ device; proj DenseGEMV/GEMVT device (fwd); bwd combine host"
	case "mha":
		return "OK", "attn host; proj DenseGEMV/GEMVT on-device"
	case "cnn1", "cnn2", "cnn3":
		return "OK", "im2col host + on-device DenseGEMV/GEMVT"
	case "dense":
		return "OK", "on-device DenseGEMV/GEMVT"
	default:
		return "GAP", "no WebGPU honesty note registered for layer " + layer
	}
}

// StampWebGPUNote attaches the honest on-device note when a WebGPU cell is OK
// and the caller left note empty (keeps an existing more-specific note).
func StampWebGPUNote(layer string, isWebGPU bool, status, note string) (string, string) {
	return StampBackendNote(layer, false, isWebGPU, status, note)
}

// SIMDKind reports honest SIMD depth for a layer (proj DotTile vs host ALU).
func SIMDKind(layer string) (statusIfWorks string, note string) {
	switch layer {
	case "dense":
		return "OK", "Plan 9 GEMV/saxpy (incl. AffinePacked inflate+DotTile)"
	case "rmsnorm", "layernorm":
		return "OK", "DotTile stats; scale on host"
	case "swiglu":
		return "OK", "proj Dense SIMD; SiLU⊙ host"
	case "mha":
		return "OK", "proj Dense SIMD; attn ALU host"
	case "cnn1", "cnn2", "cnn3":
		return "OK", "im2col host + Dense SIMD GEMV"
	case "softmax", "embedding":
		return "OK", "host ALU (Enabled gate only)"
	default:
		return "OK", "SIMD path (see layer simd.go)"
	}
}

// StampBackendNote stamps SIMD or WebGPU honesty notes on OK cells.
func StampBackendNote(layer string, isSIMD, isWebGPU bool, status, note string) (string, string) {
	if status != "OK" || note != "" {
		return status, note
	}
	if isWebGPU {
		return WebGPUKind(layer)
	}
	if isSIMD {
		return SIMDKind(layer)
	}
	return status, note
}
