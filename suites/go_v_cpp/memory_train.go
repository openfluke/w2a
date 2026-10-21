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

// CasesMemoryTrain — Parallel Dense Add + ModeMemory surprise gate via CamKit.
func CasesMemoryTrain() []Case {
	return []Case{
		{Name: "Parallel Memory gate open (thresh=0) Float32 — exact bits", Run: caseMemoryAlwaysF32},
		{Name: "Parallel Memory gate closed (high thresh) Float32 — exact bits", Run: caseMemoryBlockedF32},
		{Name: "Parallel Memory gate open × all 34 dtypes — spectrum", Run: caseMemoryAlwaysAllDTypes},
		{Name: "Metacognition Memory 1-hop Float32 — exact bits", Run: caseMetacognitionMemoryF32},
		{Name: "Metacognition Memory 1-hop × all 34 dtypes — spectrum", Run: caseMetacognitionMemoryAllDTypes},
		{Name: "KMeans Memory 1-hop Float32 — exact bits", Run: caseKMeansMemoryF32},
		{Name: "KMeans Memory 1-hop × all 34 dtypes — spectrum", Run: caseKMeansMemoryAllDTypes},
		{Name: "GDN Memory 1-hop Float32 — exact bits", Run: caseGDNMemoryF32},
		{Name: "NestedStack Memory 1-hop Float32 — exact bits", Run: caseNestedStackMemoryF32},
		{Name: "NestedStack Memory 1-hop × all 34 dtypes — spectrum", Run: caseNestedStackMemoryAllDTypes},
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

// --- CASCADE Meta/GDN/NestedStack (Memory familyBP via TrainStackMSE) ---

func caseMetacognitionMemoryF32() error {
	return runMetacognitionMemory(core.DTypeFloat32, true)
}
func caseMetacognitionMemoryAllDTypes() error {
	return runTweenChainSpectrum("MetacognitionMemory", buildMetacognitionMemory)
}
func buildMetacognitionMemory(dtype core.DType) (tweenChainOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeMcModeLeaf(dtype, 0x4d4d4500^uint32(dtype)*0x9e3779b9)
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
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeMemory, lr)
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
func runMetacognitionMemory(dtype core.DType, exact bool) error {
	goO, payload, err := buildMetacognitionMemory(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppTCB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenChain(goO, cppO, exact)
}

func caseGDNMemoryF32() error { return runGDNMemory(true) }
func buildGDNMemory() (tweenChainOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeGdModeLeaf(0x4d4d4500 ^ 0x47444e00)
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
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeMemory, lr)
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
func runGDNMemory(exact bool) error {
	goO, payload, err := buildGDNMemory()
	if err != nil {
		return err
	}
	cppO, _, err := runCppTCB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenChain(goO, cppO, exact)
}

func caseNestedStackMemoryF32() error {
	return runNestedStackMemory(core.DTypeFloat32, true)
}
func caseNestedStackMemoryAllDTypes() error {
	return runTweenChainSpectrum("NestedStackMemory", buildNestedStackMemory)
}
func buildNestedStackMemory(dtype core.DType) (tweenChainOut, []byte, error) {
	dim, nChildren, batch, seq, packed, x, tgt, lr, err := makeStModeLeaf(dtype, 0x4d4d4500^uint32(dtype)*0x9e3779b9^0x5354)
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
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeMemory, lr)
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
func runNestedStackMemory(dtype core.DType, exact bool) error {
	goO, payload, err := buildNestedStackMemory(dtype)
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

func caseKMeansMemoryF32() error {
	return runKMeansMemory(core.DTypeFloat32, true)
}

func caseKMeansMemoryAllDTypes() error {
	return runTweenChainSpectrum("KMeans", buildKMeansMemory)
}

func buildKMeansMemory(dtype core.DType) (tweenChainOut, []byte, error) {
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
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeMemory, lr)
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

func runKMeansMemory(dtype core.DType, exact bool) error {
	goO, payload, err := buildKMeansMemory(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppTCB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenChain(goO, cppO, exact)
}

