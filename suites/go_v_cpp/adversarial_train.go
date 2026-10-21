package go_v_cpp

import (
	"fmt"
	"strings"
	"time"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
)

// CasesAdversarialTrain — Parallel BranchMode Adversarial (neg-LR) + top-level Stack
// ModeAdversarial (≡ FamilyBP @ full lr; C++ TCB1 / TweenChain volume wires) for every
// Freeze/Shadow-signed kind.
func CasesAdversarialTrain() []Case {
	return []Case{
		// Parallel BranchMode Adversarial (ADB1) — keep neg-LR
		{Name: "Parallel Adversarial∥NormalBP Float32 — exact bits", Run: caseAdversarialMixedF32},
		{Name: "Parallel Adversarial∥Adversarial Float32 — exact bits", Run: caseAdversarialBothF32},
		{Name: "Parallel Adversarial∥NormalBP × all 34 dtypes — spectrum", Run: caseAdversarialMixedAllDTypes},
		{Name: "Parallel Adversarial volume 2×2×2×1 Float32 — exact bits (≡ FamilyBP)", Run: casePnNBPVol222F32},

		// Leaf Stack ModeAdversarial ≡ FamilyBP (TCB1 C++ wire)
		{Name: "Dense Adversarial multi-hop Float32 — exact bits (W/loss)", Run: caseDenseAdversarialMultiHopF32},
		{Name: "Dense Adversarial 1-hop × all 34 dtypes — spectrum + speed", Run: caseDenseAdversarial1HopAllDTypes},
		{Name: "RMSNorm Adversarial 1-hop Float32 — exact bits", Run: caseRMSNormAdversarialF32},
		{Name: "RMSNorm Adversarial 1-hop × all 34 dtypes — spectrum", Run: caseRMSNormAdversarialAllDTypes},
		{Name: "LayerNorm Adversarial 1-hop Float32 — exact bits", Run: caseLayerNormAdversarialF32},
		{Name: "LayerNorm Adversarial 1-hop × all 34 dtypes — spectrum", Run: caseLayerNormAdversarialAllDTypes},
		{Name: "Softmax Adversarial 1-hop Float32 — exact bits (loss)", Run: caseSoftmaxAdversarialF32},
		{Name: "Softmax Adversarial 1-hop × all 34 dtypes — spectrum", Run: caseSoftmaxAdversarialAllDTypes},
		{Name: "SwiGLU Adversarial 1-hop Float32 — exact bits", Run: caseSwiGLUAdversarialF32},
		{Name: "SwiGLU Adversarial 1-hop × all 34 dtypes — spectrum", Run: caseSwiGLUAdversarialAllDTypes},
		{Name: "Sequential Adversarial 1-hop Float32 — exact bits", Run: caseSequentialAdversarialF32},
		{Name: "Sequential Adversarial 1-hop × all 34 dtypes — spectrum", Run: caseSequentialAdversarialAllDTypes},
		{Name: "Residual Adversarial 1-hop Float32 — exact bits", Run: caseResidualAdversarialF32},
		{Name: "Residual Adversarial 1-hop × all 34 dtypes — spectrum", Run: caseResidualAdversarialAllDTypes},
		{Name: "Mamba Adversarial 1-hop Float32 — exact bits", Run: caseMambaAdversarialF32},
		{Name: "Mamba Adversarial 1-hop × all 34 dtypes — spectrum", Run: caseMambaAdversarialAllDTypes},
		{Name: "Metacognition Adversarial 1-hop Float32 — exact bits", Run: caseMetacognitionAdversarialF32},
		{Name: "Metacognition Adversarial 1-hop × all 34 dtypes — spectrum", Run: caseMetacognitionAdversarialAllDTypes},
		{Name: "KMeans Adversarial 1-hop Float32 — exact bits", Run: caseKMeansAdversarialF32},
		{Name: "KMeans Adversarial 1-hop × all 34 dtypes — spectrum", Run: caseKMeansAdversarialAllDTypes},
		{Name: "GDN Adversarial 1-hop Float32 — exact bits", Run: caseGDNAdversarialF32},
		{Name: "NestedStack Adversarial 1-hop Float32 — exact bits", Run: caseNestedStackAdversarialF32},
		{Name: "NestedStack Adversarial 1-hop × all 34 dtypes — spectrum", Run: caseNestedStackAdversarialAllDTypes},
		{Name: "Embedding Adversarial 1-hop Float32 — exact bits", Run: caseEmbeddingAdversarialF32},
		{Name: "Embedding Adversarial 1-hop × all 34 dtypes — spectrum", Run: caseEmbeddingAdversarialAllDTypes},
		{Name: "MHA Adversarial 1-hop Float32 — exact bits", Run: caseMHAAdversarialF32},
		{Name: "MHA Adversarial 1-hop × all 34 dtypes — spectrum", Run: caseMHAAdversarialAllDTypes},
		{Name: "CNN1 Adversarial 1-hop Float32 — exact bits", Run: caseCNN1AdversarialF32},
		{Name: "CNN1 Adversarial 1-hop × all 34 dtypes — spectrum", Run: caseCNN1AdversarialAllDTypes},
		{Name: "ConvT1 Adversarial 1-hop Float32 — exact bits", Run: caseConvT1AdversarialF32},
		{Name: "ConvT1 Adversarial 1-hop × all 34 dtypes — spectrum", Run: caseConvT1AdversarialAllDTypes},
		{Name: "RNN Adversarial 1-hop Float32 — exact bits", Run: caseRNNAdversarialF32},
		{Name: "RNN Adversarial 1-hop × all 34 dtypes — spectrum", Run: caseRNNAdversarialAllDTypes},
		{Name: "LSTM Adversarial 1-hop Float32 — exact bits", Run: caseLSTMAdversarialF32},
		{Name: "LSTM Adversarial 1-hop × all 34 dtypes — spectrum", Run: caseLSTMAdversarialAllDTypes},

		// Volume (TweenChain TRC1…MBTC / FamilyBP credit ≡ ModeAdversarial)
		{Name: "Dense Adversarial volume 1×1×1×1 Float32 — exact bits", Run: caseDenseTweenChainVol111F32},
		{Name: "Dense Adversarial volume 2×1×1×1 chain Float32 — exact bits", Run: caseDenseTweenChainVol211F32},
		{Name: "Dense Adversarial volume 2×2×2×1 Float32 — exact bits", Run: caseDenseTweenChainVol222F32},
		{Name: "RMSNorm Adversarial volume 2×2×2×1 Float32 — exact bits", Run: caseRMSNormTweenChainVol222F32},
		{Name: "LayerNorm Adversarial volume 2×2×2×1 Float32 — exact bits", Run: caseLayerNormTweenChainVol222F32},
		{Name: "Softmax Adversarial volume 2×2×2×1 Float32 — exact bits", Run: caseSoftmaxTweenChainVol222F32},
		{Name: "SwiGLU Adversarial volume 2×2×2×1 Float32 — exact bits", Run: caseSwiGLUTweenChainVol222F32},
		{Name: "Sequential Adversarial volume 2×2×2×1 Float32 — exact bits", Run: caseSeqTweenChainVol222F32},
		{Name: "Residual Adversarial volume 2×2×2×1 Float32 — exact bits", Run: caseReTweenChainVol222F32},
		{Name: "Mamba Adversarial volume 2×2×2×1 Float32 — exact bits", Run: caseMbTweenChainVol222F32},
		{Name: "Metacognition Adversarial volume 2×2×2×1 Float32 — exact bits", Run: caseMcTweenChainVol222F32},
		{Name: "KMeans Adversarial volume 2×2×2×1 Float32 — exact bits", Run: caseKmTweenChainVol222F32},
		{Name: "GDN Adversarial volume 2×2×2×1 Float32 — exact bits", Run: caseGdTweenChainVol222F32},
		{Name: "NestedStack Adversarial volume 2×2×2×1 Float32 — exact bits", Run: caseSkTweenChainVol222F32},
		{Name: "Embedding Adversarial volume 2×2×2×1 Float32 — exact bits", Run: caseEmbeddingTweenChainVol222F32},
		{Name: "MHA Adversarial volume 2×2×2×1 Float32 — exact bits", Run: caseMHATweenChainVol222F32},
		{Name: "CNN1 Adversarial volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNTweenChainVol222F32(1) }},
		{Name: "CNN2 Adversarial volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNTweenChainVol222F32(2) }},
		{Name: "CNN3 Adversarial volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNTweenChainVol222F32(3) }},
		{Name: "ConvT1 Adversarial volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTTweenChainVol222F32(1) }},
		{Name: "ConvT2 Adversarial volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTTweenChainVol222F32(2) }},
		{Name: "ConvT3 Adversarial volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTTweenChainVol222F32(3) }},
		{Name: "RNN Adversarial volume 2×2×2×1 Float32 — exact bits", Run: caseRNNTweenChainVol222F32},
		{Name: "LSTM Adversarial volume 2×2×2×1 Float32 — exact bits", Run: caseLSTMTweenChainVol222F32},
	}
}

func RunAdversarialTrain() error {
	var fails []string
	cs := CasesAdversarialTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [adversarial %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_adversarial_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_adversarial_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp adversarial_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func caseAdversarialMixedF32() error {
	modes := []parallel.TrainMode{parallel.ModeNormalBP, parallel.ModeAdversarial}
	return runCameralModesCase("ADB1", "ADBO", modes, parallel.CamKit{}, core.DTypeFloat32, true, 40)
}

func caseAdversarialBothF32() error {
	modes := []parallel.TrainMode{parallel.ModeAdversarial, parallel.ModeAdversarial}
	return runCameralModesCase("ADB1", "ADBO", modes, parallel.CamKit{}, core.DTypeFloat32, true, 40)
}

func caseAdversarialMixedAllDTypes() error {
	modes := []parallel.TrainMode{parallel.ModeNormalBP, parallel.ModeAdversarial}
	return runCameralModesSpectrum("Adversarial", "ADB1", "ADBO", modes, parallel.CamKit{})
}

// --- Dense ModeAdversarial (clone TweenChain builders @ FamilyBP) ---

func caseDenseAdversarialMultiHopF32() error {
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
	goO, err := runGoDenseTweenChain(core.DTypeFloat32, lr, children, batch, ticksX, ticksT, parallel.ModeAdversarial)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload := encodeDenseTCB1(core.DTypeFloat32, lr, children, batch, []int32{batch, 4}, []int32{batch, 2}, ticksX, ticksT)
	cppO, _, err := runCppTCB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenChain(goO, cppO, true)
}

func caseDenseAdversarial1HopAllDTypes() error {
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
		goO, err := runGoDenseTweenChain(dt, 1e-2, ch, batch, tx, tt, parallel.ModeAdversarial)
		goDur := time.Since(t0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go:%v", dt, err))
			continue
		}
		payload := encodeDenseTCB1(dt, 1e-2, ch, batch, []int32{batch, in}, []int32{batch, out}, tx, tt)
		cppO, cppDur, err := runCppTCB1(payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp-bench:%v", dt, err))
			continue
		}
		cppO, _, err = runCppTCB1(payload, 0)
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
		if err := compareTweenChain(goO, cppO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s:%v", dt, err))
		}
	}
	if len(fails) > 0 {
		return fmt.Errorf("dtype fails: %s", strings.Join(fails, "; "))
	}
	return nil
}

// --- leaf ModeAdversarial via TweenChain builders (FamilyBP @ full lr) ---

func runAdversarialLeaf(dtype core.DType, exact bool, build func(core.DType, parallel.TrainMode) (tweenChainOut, []byte, error)) error {
	goO, payload, err := build(dtype, parallel.ModeAdversarial)
	if err != nil {
		return err
	}
	cppO, _, err := runCppTCB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenChain(goO, cppO, exact)
}

func caseRMSNormAdversarialF32() error {
	return runAdversarialLeaf(core.DTypeFloat32, true, buildRMSNormTweenChain)
}
func caseRMSNormAdversarialAllDTypes() error {
	return runTweenChainSpectrum("RMSNormAdversarial", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildRMSNormTweenChain(dt, parallel.ModeAdversarial)
	})
}

func caseLayerNormAdversarialF32() error {
	return runAdversarialLeaf(core.DTypeFloat32, true, buildLayerNormTweenChain)
}
func caseLayerNormAdversarialAllDTypes() error {
	return runTweenChainSpectrum("LayerNormAdversarial", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildLayerNormTweenChain(dt, parallel.ModeAdversarial)
	})
}

func caseSoftmaxAdversarialF32() error {
	return runAdversarialLeaf(core.DTypeFloat32, true, buildSoftmaxTweenChain)
}
func caseSoftmaxAdversarialAllDTypes() error {
	return runTweenChainSpectrum("SoftmaxAdversarial", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildSoftmaxTweenChain(dt, parallel.ModeAdversarial)
	})
}

func caseSwiGLUAdversarialF32() error {
	return runAdversarialLeaf(core.DTypeFloat32, true, buildSwiGLUTweenChain)
}
func caseSwiGLUAdversarialAllDTypes() error {
	return runTweenChainSpectrum("SwiGLUAdversarial", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildSwiGLUTweenChain(dt, parallel.ModeAdversarial)
	})
}

func caseSequentialAdversarialF32() error {
	return runSeqResAdversarial(true, core.DTypeFloat32, true)
}
func caseSequentialAdversarialAllDTypes() error {
	return runTweenChainSpectrum("SequentialAdversarial", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildSeqResTweenChain(true, dt, parallel.ModeAdversarial)
	})
}
func caseResidualAdversarialF32() error {
	return runSeqResAdversarial(false, core.DTypeFloat32, true)
}
func caseResidualAdversarialAllDTypes() error {
	return runTweenChainSpectrum("ResidualAdversarial", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildSeqResTweenChain(false, dt, parallel.ModeAdversarial)
	})
}
func runSeqResAdversarial(isSeq bool, dtype core.DType, exact bool) error {
	goO, payload, err := buildSeqResTweenChain(isSeq, dtype, parallel.ModeAdversarial)
	if err != nil {
		return err
	}
	cppO, _, err := runCppTCB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenChain(goO, cppO, exact)
}

func caseMambaAdversarialF32() error {
	return runAdversarialLeaf(core.DTypeFloat32, true, buildMambaTweenChain)
}
func caseMambaAdversarialAllDTypes() error {
	return runTweenChainSpectrum("MambaAdversarial", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildMambaTweenChain(dt, parallel.ModeAdversarial)
	})
}

func caseMetacognitionAdversarialF32() error {
	return runAdversarialLeaf(core.DTypeFloat32, true, buildMetacognitionTweenChain)
}
func caseMetacognitionAdversarialAllDTypes() error {
	return runTweenChainSpectrum("MetacognitionAdversarial", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildMetacognitionTweenChain(dt, parallel.ModeAdversarial)
	})
}

func caseKMeansAdversarialF32() error {
	return runAdversarialLeaf(core.DTypeFloat32, true, buildKMeansTweenChain)
}
func caseKMeansAdversarialAllDTypes() error {
	return runTweenChainSpectrum("KMeansAdversarial", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildKMeansTweenChain(dt, parallel.ModeAdversarial)
	})
}

func caseGDNAdversarialF32() error {
	goO, payload, err := buildGDNTweenChain(parallel.ModeAdversarial)
	if err != nil {
		return err
	}
	cppO, _, err := runCppTCB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenChain(goO, cppO, true)
}

func caseNestedStackAdversarialF32() error {
	return runAdversarialLeaf(core.DTypeFloat32, true, buildNestedStackTweenChain)
}
func caseNestedStackAdversarialAllDTypes() error {
	return runTweenChainSpectrum("NestedStackAdversarial", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildNestedStackTweenChain(dt, parallel.ModeAdversarial)
	})
}

func caseEmbeddingAdversarialF32() error {
	return runAdversarialLeaf(core.DTypeFloat32, true, buildEmbeddingTweenChain)
}
func caseEmbeddingAdversarialAllDTypes() error {
	return runTweenChainSpectrum("EmbeddingAdversarial", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildEmbeddingTweenChain(dt, parallel.ModeAdversarial)
	})
}

func caseMHAAdversarialF32() error {
	return runAdversarialLeaf(core.DTypeFloat32, true, buildMHATweenChain)
}
func caseMHAAdversarialAllDTypes() error {
	return runTweenChainSpectrum("MHAAdversarial", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildMHATweenChain(dt, parallel.ModeAdversarial)
	})
}

func caseCNN1AdversarialF32() error {
	return runAdversarialLeaf(core.DTypeFloat32, true, buildCNN1TweenChain)
}
func caseCNN1AdversarialAllDTypes() error {
	return runTweenChainSpectrum("CNN1Adversarial", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildCNN1TweenChain(dt, parallel.ModeAdversarial)
	})
}

func caseConvT1AdversarialF32() error {
	return runAdversarialLeaf(core.DTypeFloat32, true, buildConvT1TweenChain)
}
func caseConvT1AdversarialAllDTypes() error {
	return runTweenChainSpectrum("ConvT1Adversarial", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildConvT1TweenChain(dt, parallel.ModeAdversarial)
	})
}

func caseRNNAdversarialF32() error {
	return runAdversarialLeaf(core.DTypeFloat32, true, buildRNNTweenChain)
}
func caseRNNAdversarialAllDTypes() error {
	return runTweenChainSpectrum("RNNAdversarial", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildRNNTweenChain(dt, parallel.ModeAdversarial)
	})
}

func caseLSTMAdversarialF32() error {
	return runAdversarialLeaf(core.DTypeFloat32, true, buildLSTMTweenChain)
}
func caseLSTMAdversarialAllDTypes() error {
	return runTweenChainSpectrum("LSTMAdversarial", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildLSTMTweenChain(dt, parallel.ModeAdversarial)
	})
}
