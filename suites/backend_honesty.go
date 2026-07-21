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
		return "OK", "LayerNorm fwd+bwd on-device (webgpu.LayerNorm/LayerNormBackward)"
	case "softmax":
		return "OK", "SoftmaxEx fwd+bwd on-device (std/grid/hier/temp + Gumbel/Masked/Sparse/Entmax)"
	case "swiglu":
		return "OK", "SiLU⊙ fuse fwd+bwd on-device; proj DenseGEMV/GEMVT device"
	case "mha":
		return "OK", "attn/RoPE fwd+bwd on-device when GPUAttnSupported; else host attn + proj device"
	case "cnn1", "cnn2", "cnn3":
		return "OK", "FormatNone f32 tiled conv on-device; else im2col host + DenseGEMV/GEMVT"
	case "dense":
		return "OK", "on-device DenseGEMV/GEMVT (AffinePacked resident fwd+bwd)"
	case "embedding":
		return "OK", "gather/scatter on-device (webgpu.EmbeddingGather/Scatter)"
	case "rnn":
		return "OK", "FormatNone f32 fused RNN step fwd+bwd; else Dense recurrence"
	case "lstm":
		return "OK", "FormatNone f32 fused LSTM step fwd+bwd; else Dense recurrence"
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
		return "OK", "DotTile stats + full SIMD scale (RMSNormScaleF32/LayerNormScaleF32)"
	case "swiglu":
		return "OK", "proj Dense SIMD; SiLU⊙ via simd.SiluMul*"
	case "mha":
		return "OK", "proj Dense SIMD; attn ALU host"
	case "cnn1", "cnn2", "cnn3":
		return "OK", "im2col host + Dense SIMD GEMV"
	case "softmax":
		return "OK", "simd.SoftmaxF32/SoftmaxBwdF32 (std/temp/grid/hier)"
	case "embedding":
		return "OK", "host gather (Enabled gate)"
	case "rnn", "lstm":
		return "OK", "Dense SIMD projs; recurrence host ALU"
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
