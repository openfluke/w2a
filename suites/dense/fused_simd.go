package dense

import (
	"fmt"
	"math"
	"runtime"
	"strings"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
)

// kIQAffineSIMDFormats — Dense SIMD fused path (Int8QS + scales; no F32 inflate).
var kIQAffineSIMDFormats = []quant.Format{
	quant.FormatQ2_K, quant.FormatQ3_K, quant.FormatQ4_K, quant.FormatQ5_K, quant.FormatQ6_K,
	quant.FormatIQ1_S, quant.FormatIQ2_XXS, quant.FormatIQ2_XS,
	quant.FormatIQ3_XXS, quant.FormatIQ3_S, quant.FormatIQ4_NL, quant.FormatIQ4_XS,
	quant.FormatAffinePacked,
}

// kSIMDCacheNoF32Inflate — EnsureKSIMDCache projects codes without building F32Cache.
func kSIMDCacheNoF32Inflate() error {
	init := make([]float32, 32*64)
	for i := range init {
		init[i] = float32((i%17)-8) * 0.05
	}
	b, err := quant.Pack(quant.FormatQ4_K, init, 32, 64)
	if err != nil {
		return err
	}
	quant.EnsureKSIMDCache(b)
	if len(b.Int8QS) < len(init) || len(b.Scales) == 0 {
		return fmt.Errorf("cache incomplete qs=%d scales=%d", len(b.Int8QS), len(b.Scales))
	}
	if len(b.F32Cache) != 0 {
		return fmt.Errorf("F32Cache should stay empty, got %d", len(b.F32Cache))
	}
	fmt.Printf("(Q4_K Int8QS=%d scales=%d) ", len(b.Int8QS), len(b.Scales))
	return nil
}

// fusedKIQAffineSIMDParity — CPU packed MatVec vs BackendSIMD fwd for all k/IQ/Affine.
func fusedKIQAffineSIMDParity() error {
	if !simd.Enabled() {
		return fmt.Errorf("Plan 9 SIMD not enabled on %s", runtime.GOARCH)
	}
	const in, out, batch = 64, 32, 2
	const tol = 5e-3
	init := make([]float32, out*in)
	for i := range init {
		init[i] = float32((i%13)-6) * 0.1
	}
	x := core.NewTensor[float32](batch, in)
	for i := range x.Data {
		x.Data[i] = float32(i%5)*0.25 + 0.1
	}

	var fails []string
	for _, f := range kIQAffineSIMDFormats {
		if err := fusedSIMDParityOne(f, init, x, in, out, tol); err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", f, err))
		}
	}
	fmt.Printf("(%d formats) ", len(kIQAffineSIMDFormats))
	if len(fails) > 0 {
		return fmt.Errorf("%s", strings.Join(fails, " | "))
	}
	return nil
}

func fusedSIMDParityOne(f quant.Format, init []float32, x *core.Tensor[float32], in, out int, tol float64) error {
	lCPU, err := dense.NewConfigured(in, out, core.ActivationLinear, core.DTypeFloat32, f, init)
	if err != nil {
		return fmt.Errorf("cpu layer: %w", err)
	}
	lCPU.Exec.Backend = core.BackendCPUTiled
	lSIMD, err := dense.NewConfigured(in, out, core.ActivationLinear, core.DTypeFloat32, f, init)
	if err != nil {
		return fmt.Errorf("simd layer: %w", err)
	}
	lSIMD.Exec.Backend = core.BackendSIMD

	preCPU, _, err := dense.Forward(lCPU, x)
	if err != nil {
		return fmt.Errorf("cpu fwd: %w", err)
	}
	preSIMD, _, err := dense.Forward(lSIMD, x)
	if err != nil {
		return fmt.Errorf("simd fwd: %w", err)
	}
	for i := range preCPU.Data {
		if math.Abs(float64(preCPU.Data[i]-preSIMD.Data[i])) > tol {
			return fmt.Errorf("fwd idx %d: cpu=%v simd=%v", i, preCPU.Data[i], preSIMD.Data[i])
		}
	}
	if lSIMD.Weights.Packed == nil {
		return fmt.Errorf("simd packed missing")
	}
	if len(lSIMD.Weights.Packed.F32Cache) != 0 {
		return fmt.Errorf("SIMD path built F32Cache (%d)", len(lSIMD.Weights.Packed.F32Cache))
	}

	yRef := make([]float32, out)
	ySIMD := make([]float32, out)
	if err := quant.MatVec(lCPU.Weights.Packed, x.Data[:in], yRef); err != nil {
		return fmt.Errorf("cpu MatVec: %w", err)
	}
	if err := dense.MatVecPackedBlob(lSIMD.Weights.Packed, x.Data[:in], ySIMD); err != nil {
		return fmt.Errorf("MatVecPackedBlob: %w", err)
	}
	for i := range yRef {
		if math.Abs(float64(yRef[i]-ySIMD[i])) > tol {
			return fmt.Errorf("MatVec idx %d: ref=%v simd=%v", i, yRef[i], ySIMD[i])
		}
	}
	return nil
}

// affinePackedSIMDParity — AffinePacked CPU vs SIMD (subset of fused suite; kept as named case).
func affinePackedSIMDParity() error {
	if !simd.Enabled() {
		return fmt.Errorf("Plan 9 SIMD not enabled on %s", runtime.GOARCH)
	}
	const in, out, batch = 64, 32, 2
	init := make([]float32, out*in)
	for i := range init {
		init[i] = float32((i%13)-6) * 0.1
	}
	x := core.NewTensor[float32](batch, in)
	for i := range x.Data {
		x.Data[i] = float32(i%5)*0.25 + 0.1
	}
	return fusedSIMDParityOne(quant.FormatAffinePacked, init, x, in, out, 5e-3)
}
