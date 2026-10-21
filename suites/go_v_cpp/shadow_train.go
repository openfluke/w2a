package go_v_cpp

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/layers/parallel"
)

// CasesShadowTrain — Parallel Dense Add Shadow teacher + NormalBP student + KD.
func CasesShadowTrain() []Case {
	return []Case{
		{Name: "Parallel Shadow∥NormalBP KD Float32 — exact bits", Run: caseShadowKDF32},
		{Name: "Parallel Shadow∥NormalBP KD coef=0.5 Float32 — exact bits", Run: caseShadowKDCoefF32},
		{Name: "Parallel Shadow∥NormalBP KD × all 34 dtypes — spectrum", Run: caseShadowKDAllDTypes},
		{Name: "Metacognition Shadow 1-hop Float32 — exact bits", Run: caseMetacognitionShadowF32},
		{Name: "Metacognition Shadow 1-hop × all 34 dtypes — spectrum", Run: caseMetacognitionShadowAllDTypes},
		{Name: "GDN Shadow 1-hop Float32 — exact bits", Run: caseGDNShadowF32},
		{Name: "NestedStack Shadow 1-hop Float32 — exact bits", Run: caseNestedStackShadowF32},
		{Name: "NestedStack Shadow 1-hop × all 34 dtypes — spectrum", Run: caseNestedStackShadowAllDTypes},
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

// --- CASCADE Meta/GDN/NestedStack (Shadow ≡ Freeze: fwd+mse, no apply) ---

func caseMetacognitionShadowF32() error {
	return runMetacognitionShadow(core.DTypeFloat32, true)
}
func caseMetacognitionShadowAllDTypes() error {
	return runFreezeSpectrum("MetacognitionShadow", buildMetacognitionShadow)
}
func buildMetacognitionShadow(dtype core.DType) (freezeOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeMcModeLeaf(dtype, 0x5348444d^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return freezeOut{}, nil, err
	}
	st, err := newMcStack(cfg, dtype, packed)
	if err != nil {
		return freezeOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, cfg.Dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, cfg.Dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeShadow, lr)
	if err != nil {
		return freezeOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ml := st.Children[0].(*metacognition.Layer)
	wa, err := packMetacognitionWeightsF32(ml)
	if err != nil {
		return freezeOut{}, nil, err
	}
	goO := freezeOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeFRB1Header(1, 1, batch, lr)
	writeMcChild(buf, cfg.Dim, seq, dtype, packed)
	writeShapesTicks(buf, []int32{int32(batch), int32(seq), int32(cfg.Dim)},
		[]int32{int32(batch), int32(seq), int32(cfg.Dim)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}
func runMetacognitionShadow(dtype core.DType, exact bool) error {
	goO, payload, err := buildMetacognitionShadow(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFRB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareFreeze(goO, cppO, exact)
}

func caseGDNShadowF32() error { return runGDNShadow(true) }
func buildGDNShadow() (freezeOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeGdModeLeaf(0x53484744)
	if err != nil {
		return freezeOut{}, nil, err
	}
	st, err := newGdStack(cfg, packed)
	if err != nil {
		return freezeOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, cfg.HiddenSize)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, cfg.HiddenSize)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeShadow, lr)
	if err != nil {
		return freezeOut{}, nil, fmt.Errorf("go: %w", err)
	}
	gl := st.Children[0].(*gdn.Layer)
	wa, err := packGDNWeightsF32(gl)
	if err != nil {
		return freezeOut{}, nil, err
	}
	goO := freezeOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeFRB1Header(1, 1, batch, lr)
	writeGdChild(buf, cfg, seq, packed)
	writeShapesTicks(buf, []int32{int32(batch), int32(seq), int32(cfg.HiddenSize)},
		[]int32{int32(batch), int32(seq), int32(cfg.HiddenSize)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}
func runGDNShadow(exact bool) error {
	goO, payload, err := buildGDNShadow()
	if err != nil {
		return err
	}
	cppO, _, err := runCppFRB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareFreeze(goO, cppO, exact)
}

func caseNestedStackShadowF32() error {
	return runNestedStackShadow(core.DTypeFloat32, true)
}
func caseNestedStackShadowAllDTypes() error {
	return runFreezeSpectrum("NestedStackShadow", buildNestedStackShadow)
}
func buildNestedStackShadow(dtype core.DType) (freezeOut, []byte, error) {
	dim, nChildren, batch, seq, packed, x, tgt, lr, err := makeStModeLeaf(dtype, 0x53485354^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return freezeOut{}, nil, err
	}
	st, err := buildNestedStackDense(dim, nChildren, dtype, packed)
	if err != nil {
		return freezeOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch*seq, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch*seq, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeShadow, lr)
	if err != nil {
		return freezeOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := packStackWeightsF32(st)
	if err != nil {
		return freezeOut{}, nil, err
	}
	goO := freezeOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeFRB1Header(1, 1, batch, lr)
	if err := writeNestedStackDenseChild(buf, dim, nChildren, dtype, packed); err != nil {
		return freezeOut{}, nil, err
	}
	writeShapesTicks(buf, []int32{int32(batch * seq), int32(dim)},
		[]int32{int32(batch * seq), int32(dim)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}
func runNestedStackShadow(dtype core.DType, exact bool) error {
	goO, payload, err := buildNestedStackShadow(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFRB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareFreeze(goO, cppO, exact)
}
