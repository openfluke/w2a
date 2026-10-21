package go_v_cpp

import (
	"fmt"
	"strings"
	"time"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
)

// CasesShadowTrain — Parallel BranchMode Shadow KD + top-level Stack ModeShadow
// (≡ Freeze: Forward → MSE → no ApplyGrad) for every Freeze-signed kind.
func CasesShadowTrain() []Case {
	return []Case{
		// Parallel BranchMode Shadow KD (SHB1) — keep
		{Name: "Parallel Shadow∥NormalBP KD Float32 — exact bits", Run: caseShadowKDF32},
		{Name: "Parallel Shadow∥NormalBP KD coef=0.5 Float32 — exact bits", Run: caseShadowKDCoefF32},
		{Name: "Parallel Shadow∥NormalBP KD × all 34 dtypes — spectrum", Run: caseShadowKDAllDTypes},
		{Name: "Parallel Shadow volume 2×2×2×1 Float32 — exact bits (≡ Freeze)", Run: casePnFreezeVol222F32},

		// Leaf Stack ModeShadow ≡ Freeze (FRB1 C++ wire)
		{Name: "Dense Shadow multi-hop Float32 — exact bits (W/loss)", Run: caseDenseShadowMultiHopF32},
		{Name: "Dense Shadow 1-hop × all 34 dtypes — spectrum + speed", Run: caseDenseShadow1HopAllDTypes},
		{Name: "RMSNorm Shadow 1-hop Float32 — exact bits", Run: caseRMSNormShadowF32},
		{Name: "RMSNorm Shadow 1-hop × all 34 dtypes — spectrum", Run: caseRMSNormShadowAllDTypes},
		{Name: "LayerNorm Shadow 1-hop Float32 — exact bits", Run: caseLayerNormShadowF32},
		{Name: "LayerNorm Shadow 1-hop × all 34 dtypes — spectrum", Run: caseLayerNormShadowAllDTypes},
		{Name: "Softmax Shadow 1-hop Float32 — exact bits (loss)", Run: caseSoftmaxShadowF32},
		{Name: "Softmax Shadow 1-hop × all 34 dtypes — spectrum", Run: caseSoftmaxShadowAllDTypes},
		{Name: "SwiGLU Shadow 1-hop Float32 — exact bits", Run: caseSwiGLUShadowF32},
		{Name: "SwiGLU Shadow 1-hop × all 34 dtypes — spectrum", Run: caseSwiGLUShadowAllDTypes},
		{Name: "Sequential Shadow 1-hop Float32 — exact bits", Run: caseSequentialShadowF32},
		{Name: "Sequential Shadow 1-hop × all 34 dtypes — spectrum", Run: caseSequentialShadowAllDTypes},
		{Name: "Residual Shadow 1-hop Float32 — exact bits", Run: caseResidualShadowF32},
		{Name: "Residual Shadow 1-hop × all 34 dtypes — spectrum", Run: caseResidualShadowAllDTypes},
		{Name: "Mamba Shadow 1-hop Float32 — exact bits", Run: caseMambaShadowF32},
		{Name: "Mamba Shadow 1-hop × all 34 dtypes — spectrum", Run: caseMambaShadowAllDTypes},
		{Name: "Metacognition Shadow 1-hop Float32 — exact bits", Run: caseMetacognitionShadowF32},
		{Name: "Metacognition Shadow 1-hop × all 34 dtypes — spectrum", Run: caseMetacognitionShadowAllDTypes},
		{Name: "KMeans Shadow 1-hop Float32 — exact bits", Run: caseKMeansShadowF32},
		{Name: "KMeans Shadow 1-hop × all 34 dtypes — spectrum", Run: caseKMeansShadowAllDTypes},
		{Name: "GDN Shadow 1-hop Float32 — exact bits", Run: caseGDNShadowF32},
		{Name: "NestedStack Shadow 1-hop Float32 — exact bits", Run: caseNestedStackShadowF32},
		{Name: "NestedStack Shadow 1-hop × all 34 dtypes — spectrum", Run: caseNestedStackShadowAllDTypes},
		{Name: "Embedding Shadow 1-hop Float32 — exact bits", Run: caseEmbeddingShadowF32},
		{Name: "Embedding Shadow 1-hop × all 34 dtypes — spectrum", Run: caseEmbeddingShadowAllDTypes},
		{Name: "MHA Shadow 1-hop Float32 — exact bits", Run: caseMHAShadowF32},
		{Name: "MHA Shadow 1-hop × all 34 dtypes — spectrum", Run: caseMHAShadowAllDTypes},
		{Name: "CNN1 Shadow 1-hop Float32 — exact bits", Run: caseCNN1ShadowF32},
		{Name: "CNN1 Shadow 1-hop × all 34 dtypes — spectrum", Run: caseCNN1ShadowAllDTypes},
		{Name: "ConvT1 Shadow 1-hop Float32 — exact bits", Run: caseConvT1ShadowF32},
		{Name: "ConvT1 Shadow 1-hop × all 34 dtypes — spectrum", Run: caseConvT1ShadowAllDTypes},
		{Name: "RNN Shadow 1-hop Float32 — exact bits", Run: caseRNNShadowF32},
		{Name: "RNN Shadow 1-hop × all 34 dtypes — spectrum", Run: caseRNNShadowAllDTypes},
		{Name: "LSTM Shadow 1-hop Float32 — exact bits", Run: caseLSTMShadowF32},
		{Name: "LSTM Shadow 1-hop × all 34 dtypes — spectrum", Run: caseLSTMShadowAllDTypes},

		// Volume (Freeze FR* magics; MSE-only ≡ ModeShadow no SGD)
		{Name: "Dense Shadow volume 1×1×1×1 Float32 — exact bits", Run: caseDenseFreezeVol111F32},
		{Name: "Dense Shadow volume 2×1×1×1 chain Float32 — exact bits", Run: caseDenseFreezeVol211F32},
		{Name: "Dense Shadow volume 2×2×2×1 Float32 — exact bits", Run: caseDenseFreezeVol222F32},
		{Name: "RMSNorm Shadow volume 2×2×2×1 Float32 — exact bits", Run: caseRMSNormFreezeVol222F32},
		{Name: "LayerNorm Shadow volume 2×2×2×1 Float32 — exact bits", Run: caseLayerNormFreezeVol222F32},
		{Name: "Softmax Shadow volume 2×2×2×1 Float32 — exact bits", Run: caseSoftmaxFreezeVol222F32},
		{Name: "SwiGLU Shadow volume 2×2×2×1 Float32 — exact bits", Run: caseSwiGLUFreezeVol222F32},
		{Name: "Sequential Shadow volume 2×2×2×1 Float32 — exact bits", Run: caseSeqFreezeVol222F32},
		{Name: "Residual Shadow volume 2×2×2×1 Float32 — exact bits", Run: caseReFreezeVol222F32},
		{Name: "Mamba Shadow volume 2×2×2×1 Float32 — exact bits", Run: caseMbFreezeVol222F32},
		{Name: "Metacognition Shadow volume 2×2×2×1 Float32 — exact bits", Run: caseMcFreezeVol222F32},
		{Name: "KMeans Shadow volume 2×2×2×1 Float32 — exact bits", Run: caseKmFreezeVol222F32},
		{Name: "GDN Shadow volume 2×2×2×1 Float32 — exact bits", Run: caseGdFreezeVol222F32},
		{Name: "NestedStack Shadow volume 2×2×2×1 Float32 — exact bits", Run: caseSkFreezeVol222F32},
		{Name: "Embedding Shadow volume 2×2×2×1 Float32 — exact bits", Run: caseEmbeddingFreezeVol222F32},
		{Name: "MHA Shadow volume 2×2×2×1 Float32 — exact bits", Run: caseMHAFreezeVol222F32},
		{Name: "CNN1 Shadow volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNFreezeVol222F32(1) }},
		{Name: "CNN2 Shadow volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNFreezeVol222F32(2) }},
		{Name: "CNN3 Shadow volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNFreezeVol222F32(3) }},
		{Name: "ConvT1 Shadow volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTFreezeVol222F32(1) }},
		{Name: "ConvT2 Shadow volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTFreezeVol222F32(2) }},
		{Name: "ConvT3 Shadow volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTFreezeVol222F32(3) }},
		{Name: "RNN Shadow volume 2×2×2×1 Float32 — exact bits", Run: caseRNNFreezeVol222F32},
		{Name: "LSTM Shadow volume 2×2×2×1 Float32 — exact bits", Run: caseLSTMFreezeVol222F32},
	}
}

func RunShadowTrain() error {
	var fails []string
	cs := CasesShadowTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [shadow %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_shadow_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_shadow_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp shadow_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func caseShadowKDF32() error {
	modes := []parallel.TrainMode{parallel.ModeShadow, parallel.ModeNormalBP}
	kit := parallel.CamKit{ShadowCoef: 1.0}
	return runCameralModesCase("SHB1", "SHBO", modes, kit, core.DTypeFloat32, true, 40)
}

func caseShadowKDCoefF32() error {
	modes := []parallel.TrainMode{parallel.ModeShadow, parallel.ModeNormalBP}
	kit := parallel.CamKit{ShadowCoef: 0.5}
	return runCameralModesCase("SHB1", "SHBO", modes, kit, core.DTypeFloat32, true, 40)
}

func caseShadowKDAllDTypes() error {
	modes := []parallel.TrainMode{parallel.ModeShadow, parallel.ModeNormalBP}
	kit := parallel.CamKit{ShadowCoef: 1.0}
	return runCameralModesSpectrum("Shadow", "SHB1", "SHBO", modes, kit)
}

// --- Dense ModeShadow (clone Freeze builders) ---

func caseDenseShadowMultiHopF32() error {
	const batch = 2
	lr := 1e-2
	children := []denseStepChild{
		{in: 4, out: 3, act: core.ActivationReLU, W: []float32{0.5, -0.2, 0.1, 0.3, -0.4, 0.6, 0.2, -0.1, 0.05, 0.15, -0.25, 0.35}, bias: []float64{0.1, -0.05, 0.02}},
		{in: 3, out: 3, act: core.ActivationLinear, W: []float32{0.2, -0.1, 0.05, 0.1, 0.3, -0.2, -0.15, 0.25, 0.4}, bias: []float64{0.01, -0.02, 0.03}},
		{in: 3, out: 2, act: core.ActivationReLU, W: []float32{0.3, -0.2, 0.1, -0.1, 0.4, 0.2}, bias: []float64{0.05, -0.05}},
	}
	x := []float32{1, 0.5, -0.5, 2, -1, 0.25, 0.75, -0.25}
	tgt := []float32{0.2, 0.4, 0.1, 0.3}
	ticksX := [][]float32{append([]float32(nil), x...)}
	ticksT := [][]float32{append([]float32(nil), tgt...)}
	goO, err := runGoDenseFreeze(core.DTypeFloat32, lr, children, batch, ticksX, ticksT, parallel.ModeShadow)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload := encodeDenseFRB1(core.DTypeFloat32, lr, children, batch, []int32{batch, 4}, []int32{batch, 2}, ticksX, ticksT)
	cppO, _, err := runCppFRB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareFreeze(goO, cppO, true)
}

func caseDenseShadow1HopAllDTypes() error {
	const in, out, batch, benchIters = 6, 4, 2, 20
	W, x, bias := makeSharedInit(out, in, batch)
	tgt := make([]float32, batch*out)
	for i := range tgt {
		tgt[i] = float32(0.1*float64(i+1) + 0.05)
	}
	fmt.Printf("\n      %-12s %-12s %8s %10s %10s %9s\n", "dtype", "band", "exact%W", "goµs", "cppµs", "spd%Δ")
	var fails []string
	for _, dt := range core.AllDTypes {
		ch := []denseStepChild{{in: in, out: out, act: core.ActivationLinear, W: append([]float32(nil), W...), bias: append([]float64(nil), bias...)}}
		tx := [][]float32{append([]float32(nil), x...)}
		tt := [][]float32{append([]float32(nil), tgt...)}
		t0 := time.Now()
		goO, err := runGoDenseFreeze(dt, 1e-2, ch, batch, tx, tt, parallel.ModeShadow)
		goDur := time.Since(t0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go:%v", dt, err))
			continue
		}
		payload := encodeDenseFRB1(dt, 1e-2, ch, batch, []int32{batch, in}, []int32{batch, out}, tx, tt)
		cppO, cppDur, err := runCppFRB1(payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp-bench:%v", dt, err))
			continue
		}
		cppO, _, err = runCppFRB1(payload, 0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp:%v", dt, err))
			continue
		}
		m := suites.CompareF32Spectrum(goO.WAfter[0], cppO.WAfter[0])
		goUS := float64(goDur.Nanoseconds()) / 1000
		cppUS := float64(cppDur.Nanoseconds()) / float64(benchIters) / 1000
		spd := suites.SpeedPctDiff(goUS, cppUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %10.2f %10.2f %8.1f%%\n",
			dt, m.Band, m.ExactPct, goUS, cppUS, spd)
		if err := compareFreeze(goO, cppO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s:%v", dt, err))
		}
	}
	if len(fails) > 0 {
		return fmt.Errorf("dtype fails: %s", strings.Join(fails, "; "))
	}
	return nil
}

// --- leaf ModeShadow via Freeze builders ---

func runShadowLeaf(dtype core.DType, exact bool, build func(core.DType, parallel.TrainMode) (freezeOut, []byte, error)) error {
	goO, payload, err := build(dtype, parallel.ModeShadow)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFRB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareFreeze(goO, cppO, exact)
}

func caseRMSNormShadowF32() error { return runShadowLeaf(core.DTypeFloat32, true, buildRMSNormFreeze) }
func caseRMSNormShadowAllDTypes() error {
	return runFreezeSpectrum("RMSNormShadow", func(dt core.DType) (freezeOut, []byte, error) {
		return buildRMSNormFreeze(dt, parallel.ModeShadow)
	})
}

func caseLayerNormShadowF32() error { return runShadowLeaf(core.DTypeFloat32, true, buildLayerNormFreeze) }
func caseLayerNormShadowAllDTypes() error {
	return runFreezeSpectrum("LayerNormShadow", func(dt core.DType) (freezeOut, []byte, error) {
		return buildLayerNormFreeze(dt, parallel.ModeShadow)
	})
}

func caseSoftmaxShadowF32() error { return runShadowLeaf(core.DTypeFloat32, true, buildSoftmaxFreeze) }
func caseSoftmaxShadowAllDTypes() error {
	return runFreezeSpectrum("SoftmaxShadow", func(dt core.DType) (freezeOut, []byte, error) {
		return buildSoftmaxFreeze(dt, parallel.ModeShadow)
	})
}

func caseSwiGLUShadowF32() error { return runShadowLeaf(core.DTypeFloat32, true, buildSwiGLUFreeze) }
func caseSwiGLUShadowAllDTypes() error {
	return runFreezeSpectrum("SwiGLUShadow", func(dt core.DType) (freezeOut, []byte, error) {
		return buildSwiGLUFreeze(dt, parallel.ModeShadow)
	})
}

func caseSequentialShadowF32() error {
	return runSeqResShadow(true, core.DTypeFloat32, true)
}
func caseSequentialShadowAllDTypes() error {
	return runFreezeSpectrum("SequentialShadow", func(dt core.DType) (freezeOut, []byte, error) {
		return buildSeqResFreeze(true, dt, parallel.ModeShadow)
	})
}
func caseResidualShadowF32() error {
	return runSeqResShadow(false, core.DTypeFloat32, true)
}
func caseResidualShadowAllDTypes() error {
	return runFreezeSpectrum("ResidualShadow", func(dt core.DType) (freezeOut, []byte, error) {
		return buildSeqResFreeze(false, dt, parallel.ModeShadow)
	})
}
func runSeqResShadow(isSeq bool, dtype core.DType, exact bool) error {
	goO, payload, err := buildSeqResFreeze(isSeq, dtype, parallel.ModeShadow)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFRB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareFreeze(goO, cppO, exact)
}

func caseMambaShadowF32() error { return runShadowLeaf(core.DTypeFloat32, true, buildMambaFreeze) }
func caseMambaShadowAllDTypes() error {
	return runFreezeSpectrum("MambaShadow", func(dt core.DType) (freezeOut, []byte, error) {
		return buildMambaFreeze(dt, parallel.ModeShadow)
	})
}

func caseMetacognitionShadowF32() error {
	return runShadowLeaf(core.DTypeFloat32, true, buildMetacognitionFreeze)
}
func caseMetacognitionShadowAllDTypes() error {
	return runFreezeSpectrum("MetacognitionShadow", func(dt core.DType) (freezeOut, []byte, error) {
		return buildMetacognitionFreeze(dt, parallel.ModeShadow)
	})
}

func caseKMeansShadowF32() error { return runShadowLeaf(core.DTypeFloat32, true, buildKMeansFreeze) }
func caseKMeansShadowAllDTypes() error {
	return runFreezeSpectrum("KMeansShadow", func(dt core.DType) (freezeOut, []byte, error) {
		return buildKMeansFreeze(dt, parallel.ModeShadow)
	})
}

func caseGDNShadowF32() error {
	goO, payload, err := buildGDNFreeze(parallel.ModeShadow)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFRB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareFreeze(goO, cppO, true)
}

func caseNestedStackShadowF32() error {
	return runShadowLeaf(core.DTypeFloat32, true, buildNestedStackFreeze)
}
func caseNestedStackShadowAllDTypes() error {
	return runFreezeSpectrum("NestedStackShadow", func(dt core.DType) (freezeOut, []byte, error) {
		return buildNestedStackFreeze(dt, parallel.ModeShadow)
	})
}

func caseEmbeddingShadowF32() error {
	return runShadowLeaf(core.DTypeFloat32, true, buildEmbeddingFreeze)
}
func caseEmbeddingShadowAllDTypes() error {
	return runFreezeSpectrum("EmbeddingShadow", func(dt core.DType) (freezeOut, []byte, error) {
		return buildEmbeddingFreeze(dt, parallel.ModeShadow)
	})
}

func caseMHAShadowF32() error { return runShadowLeaf(core.DTypeFloat32, true, buildMHAFreeze) }
func caseMHAShadowAllDTypes() error {
	return runFreezeSpectrum("MHAShadow", func(dt core.DType) (freezeOut, []byte, error) {
		return buildMHAFreeze(dt, parallel.ModeShadow)
	})
}

func caseCNN1ShadowF32() error { return runShadowLeaf(core.DTypeFloat32, true, buildCNN1Freeze) }
func caseCNN1ShadowAllDTypes() error {
	return runFreezeSpectrum("CNN1Shadow", func(dt core.DType) (freezeOut, []byte, error) {
		return buildCNN1Freeze(dt, parallel.ModeShadow)
	})
}

func caseConvT1ShadowF32() error { return runShadowLeaf(core.DTypeFloat32, true, buildConvT1Freeze) }
func caseConvT1ShadowAllDTypes() error {
	return runFreezeSpectrum("ConvT1Shadow", func(dt core.DType) (freezeOut, []byte, error) {
		return buildConvT1Freeze(dt, parallel.ModeShadow)
	})
}

func caseRNNShadowF32() error { return runShadowLeaf(core.DTypeFloat32, true, buildRNNFreeze) }
func caseRNNShadowAllDTypes() error {
	return runFreezeSpectrum("RNNShadow", func(dt core.DType) (freezeOut, []byte, error) {
		return buildRNNFreeze(dt, parallel.ModeShadow)
	})
}

func caseLSTMShadowF32() error { return runShadowLeaf(core.DTypeFloat32, true, buildLSTMFreeze) }
func caseLSTMShadowAllDTypes() error {
	return runFreezeSpectrum("LSTMShadow", func(dt core.DType) (freezeOut, []byte, error) {
		return buildLSTMFreeze(dt, parallel.ModeShadow)
	})
}
