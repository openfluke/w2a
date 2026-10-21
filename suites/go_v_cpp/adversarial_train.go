package go_v_cpp

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/layers/parallel"
)

// CasesAdversarialTrain — Parallel Dense Add with BranchMode Adversarial (negated LR).
func CasesAdversarialTrain() []Case {
	return []Case{
		{Name: "Parallel Adversarial∥NormalBP Float32 — exact bits", Run: caseAdversarialMixedF32},
		{Name: "Parallel Adversarial∥Adversarial Float32 — exact bits", Run: caseAdversarialBothF32},
		{Name: "Parallel Adversarial∥NormalBP × all 34 dtypes — spectrum", Run: caseAdversarialMixedAllDTypes},
		{Name: "Metacognition Adversarial 1-hop Float32 — exact bits", Run: caseMetacognitionAdversarialF32},
		{Name: "Metacognition Adversarial 1-hop × all 34 dtypes — spectrum", Run: caseMetacognitionAdversarialAllDTypes},
		{Name: "KMeans Adversarial 1-hop Float32 — exact bits", Run: caseKMeansAdversarialF32},
		{Name: "KMeans Adversarial 1-hop × all 34 dtypes — spectrum", Run: caseKMeansAdversarialAllDTypes},
		{Name: "GDN Adversarial 1-hop Float32 — exact bits", Run: caseGDNAdversarialF32},
		{Name: "NestedStack Adversarial 1-hop Float32 — exact bits", Run: caseNestedStackAdversarialF32},
		{Name: "NestedStack Adversarial 1-hop × all 34 dtypes — spectrum", Run: caseNestedStackAdversarialAllDTypes},
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

// --- CASCADE Meta/GDN/NestedStack (Adversarial familyBP via TrainStackMSE) ---

func caseMetacognitionAdversarialF32() error {
	return runMetacognitionAdversarial(core.DTypeFloat32, true)
}
func caseMetacognitionAdversarialAllDTypes() error {
	return runTweenChainSpectrum("MetacognitionAdversarial", buildMetacognitionAdversarial)
}
func buildMetacognitionAdversarial(dtype core.DType) (tweenChainOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeMcModeLeaf(dtype, 0x41445600^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return tweenChainOut{}, nil, err
	}
	st, err := newMcStack(cfg, dtype, packed)
	if err != nil {
		return tweenChainOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, cfg.Dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, cfg.Dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeAdversarial, lr)
	if err != nil {
		return tweenChainOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ml := st.Children[0].(*metacognition.Layer)
	wa, err := packMetacognitionWeightsF32(ml)
	if err != nil {
		return tweenChainOut{}, nil, err
	}
	goO := tweenChainOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTCB1Header(1, 1, batch, lr)
	writeMcChild(buf, cfg.Dim, seq, dtype, packed)
	writeShapesTicks(buf, []int32{int32(batch), int32(seq), int32(cfg.Dim)},
		[]int32{int32(batch), int32(seq), int32(cfg.Dim)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}
func runMetacognitionAdversarial(dtype core.DType, exact bool) error {
	goO, payload, err := buildMetacognitionAdversarial(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppTCB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenChain(goO, cppO, exact)
}

func caseGDNAdversarialF32() error { return runGDNAdversarial(true) }
func buildGDNAdversarial() (tweenChainOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeGdModeLeaf(0x41445600 ^ 0x47444e00)
	if err != nil {
		return tweenChainOut{}, nil, err
	}
	st, err := newGdStack(cfg, packed)
	if err != nil {
		return tweenChainOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, cfg.HiddenSize)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, cfg.HiddenSize)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeAdversarial, lr)
	if err != nil {
		return tweenChainOut{}, nil, fmt.Errorf("go: %w", err)
	}
	gl := st.Children[0].(*gdn.Layer)
	wa, err := packGDNWeightsF32(gl)
	if err != nil {
		return tweenChainOut{}, nil, err
	}
	goO := tweenChainOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTCB1Header(1, 1, batch, lr)
	writeGdChild(buf, cfg, seq, packed)
	writeShapesTicks(buf, []int32{int32(batch), int32(seq), int32(cfg.HiddenSize)},
		[]int32{int32(batch), int32(seq), int32(cfg.HiddenSize)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}
func runGDNAdversarial(exact bool) error {
	goO, payload, err := buildGDNAdversarial()
	if err != nil {
		return err
	}
	cppO, _, err := runCppTCB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenChain(goO, cppO, exact)
}

func caseNestedStackAdversarialF32() error {
	return runNestedStackAdversarial(core.DTypeFloat32, true)
}
func caseNestedStackAdversarialAllDTypes() error {
	return runTweenChainSpectrum("NestedStackAdversarial", buildNestedStackAdversarial)
}
func buildNestedStackAdversarial(dtype core.DType) (tweenChainOut, []byte, error) {
	dim, nChildren, batch, seq, packed, x, tgt, lr, err := makeStModeLeaf(dtype, 0x41445600^uint32(dtype)*0x9e3779b9^0x5354)
	if err != nil {
		return tweenChainOut{}, nil, err
	}
	st, err := buildNestedStackDense(dim, nChildren, dtype, packed)
	if err != nil {
		return tweenChainOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch*seq, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch*seq, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeAdversarial, lr)
	if err != nil {
		return tweenChainOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := packStackWeightsF32(st)
	if err != nil {
		return tweenChainOut{}, nil, err
	}
	goO := tweenChainOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTCB1Header(1, 1, batch, lr)
	if err := writeNestedStackDenseChild(buf, dim, nChildren, dtype, packed); err != nil {
		return tweenChainOut{}, nil, err
	}
	writeShapesTicks(buf, []int32{int32(batch * seq), int32(dim)},
		[]int32{int32(batch * seq), int32(dim)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}
func runNestedStackAdversarial(dtype core.DType, exact bool) error {
	goO, payload, err := buildNestedStackAdversarial(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppTCB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenChain(goO, cppO, exact)
}

// --- CASCADE KMeans leaf (auto) ---

func caseKMeansAdversarialF32() error {
	return runKMeansAdversarial(core.DTypeFloat32, true)
}

func caseKMeansAdversarialAllDTypes() error {
	return runTweenChainSpectrum("KMeans", buildKMeansAdversarial)
}

func buildKMeansAdversarial(dtype core.DType) (tweenChainOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeKmModeLeaf(dtype, 0x4b4d5450^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return tweenChainOut{}, nil, err
	}
	st, err := newKmStack(cfg, dtype, packed)
	if err != nil {
		return tweenChainOut{}, nil, err
	}
	outDim := cfg.OutDim()
	xin := core.NewTensor[float32](batch, seq, cfg.FeatureDim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, outDim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeAdversarial, lr)
	if err != nil {
		return tweenChainOut{}, nil, fmt.Errorf("go: %w", err)
	}
	kl := st.Children[0].(*kmeans.Layer)
	wa, err := packKMeansWeightsF32(kl)
	if err != nil {
		return tweenChainOut{}, nil, err
	}
	goO := tweenChainOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTCB1Header(1, 1, batch, lr)
	writeKmChild(buf, cfg, seq, dtype, packed)
	writeShapesTicks(buf, []int32{int32(batch), int32(seq), int32(cfg.FeatureDim)},
		[]int32{int32(batch), int32(seq), int32(outDim)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runKMeansAdversarial(dtype core.DType, exact bool) error {
	goO, payload, err := buildKMeansAdversarial(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppTCB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenChain(goO, cppO, exact)
}

