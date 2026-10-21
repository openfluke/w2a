package go_v_cpp

import (
	"fmt"
	"strings"
	"time"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
)

// CasesMemoryTrain — Parallel BranchMode Memory (CamKit SurpriseThresh) + top-level
// Stack ModeMemory (≡ FamilyBP @ full lr; C++ TCB1 / TweenChain volume wires) for every
// Adversarial/Shadow-signed kind.
func CasesMemoryTrain() []Case {
	return []Case{
		// Parallel BranchMode Memory (MMB1) — CamKit surprise gate
		{Name: "Parallel Memory gate open (thresh=0) Float32 — exact bits", Run: caseMemoryAlwaysF32},
		{Name: "Parallel Memory gate closed (high thresh) Float32 — exact bits", Run: caseMemoryBlockedF32},
		{Name: "Parallel Memory gate open × all 34 dtypes — spectrum", Run: caseMemoryAlwaysAllDTypes},

		// Leaf Stack ModeMemory ≡ FamilyBP (TCB1 C++ wire)
		{Name: "Dense Memory multi-hop Float32 — exact bits (W/loss)", Run: caseDenseMemoryMultiHopF32},
		{Name: "Dense Memory 1-hop × all 34 dtypes — spectrum + speed", Run: caseDenseMemory1HopAllDTypes},
		{Name: "RMSNorm Memory 1-hop Float32 — exact bits", Run: caseRMSNormMemoryF32},
		{Name: "RMSNorm Memory 1-hop × all 34 dtypes — spectrum", Run: caseRMSNormMemoryAllDTypes},
		{Name: "LayerNorm Memory 1-hop Float32 — exact bits", Run: caseLayerNormMemoryF32},
		{Name: "LayerNorm Memory 1-hop × all 34 dtypes — spectrum", Run: caseLayerNormMemoryAllDTypes},
		{Name: "Softmax Memory 1-hop Float32 — exact bits (loss)", Run: caseSoftmaxMemoryF32},
		{Name: "Softmax Memory 1-hop × all 34 dtypes — spectrum", Run: caseSoftmaxMemoryAllDTypes},
		{Name: "SwiGLU Memory 1-hop Float32 — exact bits", Run: caseSwiGLUMemoryF32},
		{Name: "SwiGLU Memory 1-hop × all 34 dtypes — spectrum", Run: caseSwiGLUMemoryAllDTypes},
		{Name: "Sequential Memory 1-hop Float32 — exact bits", Run: caseSequentialMemoryF32},
		{Name: "Sequential Memory 1-hop × all 34 dtypes — spectrum", Run: caseSequentialMemoryAllDTypes},
		{Name: "Residual Memory 1-hop Float32 — exact bits", Run: caseResidualMemoryF32},
		{Name: "Residual Memory 1-hop × all 34 dtypes — spectrum", Run: caseResidualMemoryAllDTypes},
		{Name: "Mamba Memory 1-hop Float32 — exact bits", Run: caseMambaMemoryF32},
		{Name: "Mamba Memory 1-hop × all 34 dtypes — spectrum", Run: caseMambaMemoryAllDTypes},
		{Name: "Metacognition Memory 1-hop Float32 — exact bits", Run: caseMetacognitionMemoryF32},
		{Name: "Metacognition Memory 1-hop × all 34 dtypes — spectrum", Run: caseMetacognitionMemoryAllDTypes},
		{Name: "KMeans Memory 1-hop Float32 — exact bits", Run: caseKMeansMemoryF32},
		{Name: "KMeans Memory 1-hop × all 34 dtypes — spectrum", Run: caseKMeansMemoryAllDTypes},
		{Name: "GDN Memory 1-hop Float32 — exact bits", Run: caseGDNMemoryF32},
		{Name: "NestedStack Memory 1-hop Float32 — exact bits", Run: caseNestedStackMemoryF32},
		{Name: "NestedStack Memory 1-hop × all 34 dtypes — spectrum", Run: caseNestedStackMemoryAllDTypes},
		{Name: "Embedding Memory 1-hop Float32 — exact bits", Run: caseEmbeddingMemoryF32},
		{Name: "Embedding Memory 1-hop × all 34 dtypes — spectrum", Run: caseEmbeddingMemoryAllDTypes},
		{Name: "MHA Memory 1-hop Float32 — exact bits", Run: caseMHAMemoryF32},
		{Name: "MHA Memory 1-hop × all 34 dtypes — spectrum", Run: caseMHAMemoryAllDTypes},
		{Name: "CNN1 Memory 1-hop Float32 — exact bits", Run: caseCNN1MemoryF32},
		{Name: "CNN1 Memory 1-hop × all 34 dtypes — spectrum", Run: caseCNN1MemoryAllDTypes},
		{Name: "ConvT1 Memory 1-hop Float32 — exact bits", Run: caseConvT1MemoryF32},
		{Name: "ConvT1 Memory 1-hop × all 34 dtypes — spectrum", Run: caseConvT1MemoryAllDTypes},
		{Name: "RNN Memory 1-hop Float32 — exact bits", Run: caseRNNMemoryF32},
		{Name: "RNN Memory 1-hop × all 34 dtypes — spectrum", Run: caseRNNMemoryAllDTypes},
		{Name: "LSTM Memory 1-hop Float32 — exact bits", Run: caseLSTMMemoryF32},
		{Name: "LSTM Memory 1-hop × all 34 dtypes — spectrum", Run: caseLSTMMemoryAllDTypes},

		// Volume (TweenChain TRC1…MBTC / FamilyBP credit ≡ ModeMemory)
		{Name: "Dense Memory volume 1×1×1×1 Float32 — exact bits", Run: caseDenseTweenChainVol111F32},
		{Name: "Dense Memory volume 2×1×1×1 chain Float32 — exact bits", Run: caseDenseTweenChainVol211F32},
		{Name: "Dense Memory volume 2×2×2×1 Float32 — exact bits", Run: caseDenseTweenChainVol222F32},
		{Name: "RMSNorm Memory volume 2×2×2×1 Float32 — exact bits", Run: caseRMSNormTweenChainVol222F32},
		{Name: "LayerNorm Memory volume 2×2×2×1 Float32 — exact bits", Run: caseLayerNormTweenChainVol222F32},
		{Name: "Softmax Memory volume 2×2×2×1 Float32 — exact bits", Run: caseSoftmaxTweenChainVol222F32},
		{Name: "SwiGLU Memory volume 2×2×2×1 Float32 — exact bits", Run: caseSwiGLUTweenChainVol222F32},
		{Name: "Sequential Memory volume 2×2×2×1 Float32 — exact bits", Run: caseSeqTweenChainVol222F32},
		{Name: "Residual Memory volume 2×2×2×1 Float32 — exact bits", Run: caseReTweenChainVol222F32},
		{Name: "Mamba Memory volume 2×2×2×1 Float32 — exact bits", Run: caseMbTweenChainVol222F32},
		{Name: "Metacognition Memory volume 2×2×2×1 Float32 — exact bits", Run: caseMcTweenChainVol222F32},
		{Name: "KMeans Memory volume 2×2×2×1 Float32 — exact bits", Run: caseKmTweenChainVol222F32},
		{Name: "GDN Memory volume 2×2×2×1 Float32 — exact bits", Run: caseGdTweenChainVol222F32},
		{Name: "NestedStack Memory volume 2×2×2×1 Float32 — exact bits", Run: caseSkTweenChainVol222F32},
		{Name: "Embedding Memory volume 2×2×2×1 Float32 — exact bits", Run: caseEmbeddingTweenChainVol222F32},
		{Name: "MHA Memory volume 2×2×2×1 Float32 — exact bits", Run: caseMHATweenChainVol222F32},
		{Name: "CNN1 Memory volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNTweenChainVol222F32(1) }},
		{Name: "CNN2 Memory volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNTweenChainVol222F32(2) }},
		{Name: "CNN3 Memory volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNTweenChainVol222F32(3) }},
		{Name: "ConvT1 Memory volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTTweenChainVol222F32(1) }},
		{Name: "ConvT2 Memory volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTTweenChainVol222F32(2) }},
		{Name: "ConvT3 Memory volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTTweenChainVol222F32(3) }},
		{Name: "RNN Memory volume 2×2×2×1 Float32 — exact bits", Run: caseRNNTweenChainVol222F32},
		{Name: "LSTM Memory volume 2×2×2×1 Float32 — exact bits", Run: caseLSTMTweenChainVol222F32},
	}
}

func RunMemoryTrain() error {
	var fails []string
	cs := CasesMemoryTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [memory %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_memory_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_memory_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp memory_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func caseMemoryAlwaysF32() error {
	modes := []parallel.TrainMode{parallel.ModeMemory, parallel.ModeNormalBP}
	kit := parallel.CamKit{SurpriseThresh: 0} // always train
	return runCameralModesCase("MMB1", "MMBO", modes, kit, core.DTypeFloat32, true, 40)
}

func caseMemoryBlockedF32() error {
	// Huge thresh ⇒ LastLoss < thresh ⇒ Memory cam skips; NormalBP cam still updates.
	modes := []parallel.TrainMode{parallel.ModeMemory, parallel.ModeNormalBP}
	kit := parallel.CamKit{SurpriseThresh: 1e9}
	return runCameralModesCase("MMB1", "MMBO", modes, kit, core.DTypeFloat32, true, 40)
}

func caseMemoryAlwaysAllDTypes() error {
	modes := []parallel.TrainMode{parallel.ModeMemory, parallel.ModeNormalBP}
	kit := parallel.CamKit{SurpriseThresh: 0}
	return runCameralModesSpectrum("Memory", "MMB1", "MMBO", modes, kit)
}

// --- Dense ModeMemory (clone TweenChain builders @ FamilyBP) ---

func caseDenseMemoryMultiHopF32() error {
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
	goO, err := runGoDenseTweenChain(core.DTypeFloat32, lr, children, batch, ticksX, ticksT, parallel.ModeMemory)
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

func caseDenseMemory1HopAllDTypes() error {
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
		goO, err := runGoDenseTweenChain(dt, 1e-2, ch, batch, tx, tt, parallel.ModeMemory)
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

// --- leaf ModeMemory via TweenChain builders (FamilyBP @ full lr) ---

func runMemoryLeaf(dtype core.DType, exact bool, build func(core.DType, parallel.TrainMode) (tweenChainOut, []byte, error)) error {
	goO, payload, err := build(dtype, parallel.ModeMemory)
	if err != nil {
		return err
	}
	cppO, _, err := runCppTCB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenChain(goO, cppO, exact)
}

func caseRMSNormMemoryF32() error {
	return runMemoryLeaf(core.DTypeFloat32, true, buildRMSNormTweenChain)
}
func caseRMSNormMemoryAllDTypes() error {
	return runTweenChainSpectrum("RMSNormMemory", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildRMSNormTweenChain(dt, parallel.ModeMemory)
	})
}

func caseLayerNormMemoryF32() error {
	return runMemoryLeaf(core.DTypeFloat32, true, buildLayerNormTweenChain)
}
func caseLayerNormMemoryAllDTypes() error {
	return runTweenChainSpectrum("LayerNormMemory", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildLayerNormTweenChain(dt, parallel.ModeMemory)
	})
}

func caseSoftmaxMemoryF32() error {
	return runMemoryLeaf(core.DTypeFloat32, true, buildSoftmaxTweenChain)
}
func caseSoftmaxMemoryAllDTypes() error {
	return runTweenChainSpectrum("SoftmaxMemory", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildSoftmaxTweenChain(dt, parallel.ModeMemory)
	})
}

func caseSwiGLUMemoryF32() error {
	return runMemoryLeaf(core.DTypeFloat32, true, buildSwiGLUTweenChain)
}
func caseSwiGLUMemoryAllDTypes() error {
	return runTweenChainSpectrum("SwiGLUMemory", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildSwiGLUTweenChain(dt, parallel.ModeMemory)
	})
}

func caseSequentialMemoryF32() error {
	return runSeqResMemory(true, core.DTypeFloat32, true)
}
func caseSequentialMemoryAllDTypes() error {
	return runTweenChainSpectrum("SequentialMemory", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildSeqResTweenChain(true, dt, parallel.ModeMemory)
	})
}
func caseResidualMemoryF32() error {
	return runSeqResMemory(false, core.DTypeFloat32, true)
}
func caseResidualMemoryAllDTypes() error {
	return runTweenChainSpectrum("ResidualMemory", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildSeqResTweenChain(false, dt, parallel.ModeMemory)
	})
}
func runSeqResMemory(isSeq bool, dtype core.DType, exact bool) error {
	goO, payload, err := buildSeqResTweenChain(isSeq, dtype, parallel.ModeMemory)
	if err != nil {
		return err
	}
	cppO, _, err := runCppTCB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenChain(goO, cppO, exact)
}

func caseMambaMemoryF32() error {
	return runMemoryLeaf(core.DTypeFloat32, true, buildMambaTweenChain)
}
func caseMambaMemoryAllDTypes() error {
	return runTweenChainSpectrum("MambaMemory", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildMambaTweenChain(dt, parallel.ModeMemory)
	})
}

func caseMetacognitionMemoryF32() error {
	return runMemoryLeaf(core.DTypeFloat32, true, buildMetacognitionTweenChain)
}
func caseMetacognitionMemoryAllDTypes() error {
	return runTweenChainSpectrum("MetacognitionMemory", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildMetacognitionTweenChain(dt, parallel.ModeMemory)
	})
}

func caseKMeansMemoryF32() error {
	return runMemoryLeaf(core.DTypeFloat32, true, buildKMeansTweenChain)
}
func caseKMeansMemoryAllDTypes() error {
	return runTweenChainSpectrum("KMeansMemory", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildKMeansTweenChain(dt, parallel.ModeMemory)
	})
}

func caseGDNMemoryF32() error {
	goO, payload, err := buildGDNTweenChain(parallel.ModeMemory)
	if err != nil {
		return err
	}
	cppO, _, err := runCppTCB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenChain(goO, cppO, true)
}

func caseNestedStackMemoryF32() error {
	return runMemoryLeaf(core.DTypeFloat32, true, buildNestedStackTweenChain)
}
func caseNestedStackMemoryAllDTypes() error {
	return runTweenChainSpectrum("NestedStackMemory", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildNestedStackTweenChain(dt, parallel.ModeMemory)
	})
}

func caseEmbeddingMemoryF32() error {
	return runMemoryLeaf(core.DTypeFloat32, true, buildEmbeddingTweenChain)
}
func caseEmbeddingMemoryAllDTypes() error {
	return runTweenChainSpectrum("EmbeddingMemory", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildEmbeddingTweenChain(dt, parallel.ModeMemory)
	})
}

func caseMHAMemoryF32() error {
	return runMemoryLeaf(core.DTypeFloat32, true, buildMHATweenChain)
}
func caseMHAMemoryAllDTypes() error {
	return runTweenChainSpectrum("MHAMemory", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildMHATweenChain(dt, parallel.ModeMemory)
	})
}

func caseCNN1MemoryF32() error {
	return runMemoryLeaf(core.DTypeFloat32, true, buildCNN1TweenChain)
}
func caseCNN1MemoryAllDTypes() error {
	return runTweenChainSpectrum("CNN1Memory", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildCNN1TweenChain(dt, parallel.ModeMemory)
	})
}

func caseConvT1MemoryF32() error {
	return runMemoryLeaf(core.DTypeFloat32, true, buildConvT1TweenChain)
}
func caseConvT1MemoryAllDTypes() error {
	return runTweenChainSpectrum("ConvT1Memory", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildConvT1TweenChain(dt, parallel.ModeMemory)
	})
}

func caseRNNMemoryF32() error {
	return runMemoryLeaf(core.DTypeFloat32, true, buildRNNTweenChain)
}
func caseRNNMemoryAllDTypes() error {
	return runTweenChainSpectrum("RNNMemory", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildRNNTweenChain(dt, parallel.ModeMemory)
	})
}

func caseLSTMMemoryF32() error {
	return runMemoryLeaf(core.DTypeFloat32, true, buildLSTMTweenChain)
}
func caseLSTMMemoryAllDTypes() error {
	return runTweenChainSpectrum("LSTMMemory", func(dt core.DType) (tweenChainOut, []byte, error) {
		return buildLSTMTweenChain(dt, parallel.ModeMemory)
	})
}
