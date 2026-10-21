package go_v_cpp

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

// CasesParallelNBPTrain — Parallel Dense ModeNormalBP (Add + Avg) Inside+Fully + 3D.
// Wire PNB1/PNBO leaf; PNV1/PNVO volume (MeshBP / ModeMeshBP).
func CasesParallelNBPTrain() []Case {
	return []Case{
		{Name: "Dense Add Parallel NormalBP Float32 — exact bits", Run: caseParallelNBPAddF32},
		{Name: "Dense Add Parallel NormalBP × all 34 dtypes — spectrum", Run: caseParallelNBPAddAllDTypes},
		{Name: "Dense Avg Parallel NormalBP Float32 — exact bits", Run: caseParallelNBPAvgF32},
		{Name: "Dense Concat Parallel NormalBP Float32 — exact bits", Run: caseParallelNBPConcatF32},
		{Name: "Dense Max Parallel NormalBP Float32 — exact bits", Run: caseParallelNBPMaxF32},
		{Name: "Dense SparseK Parallel NormalBP Float32 — exact bits", Run: caseParallelNBPSparseKF32},
		{Name: "Dense Disagree Parallel NormalBP Float32 — exact bits", Run: caseParallelNBPDisagreeF32},
		{Name: "Dense Filter Parallel NormalBP Float32 — exact bits", Run: caseParallelNBPFilterF32},
		{Name: "Dense Add Parallel NormalBP volume 2×2×2×1 Float32 — exact bits", Run: casePnNBPVol222F32},
	}
}

func RunParallelNBPTrain() error {
	var fails []string
	cs := CasesParallelNBPTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [parallel-nbp %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_parallel_nbp_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_parallel_nbp_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp parallel_nbp_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func parallelCombineCode(c parallel.CombineMode) int32 {
	switch c {
	case parallel.CombineConcat:
		return 0
	case parallel.CombineAdd:
		return 1
	case parallel.CombineAvg:
		return 2
	case parallel.CombineFilter:
		return 3
	case parallel.CombineMax:
		return 4
	case parallel.CombineSparseK:
		return 5
	case parallel.CombineDisagree:
		return 6
	default:
		return 1
	}
}

func encodeParallelNBP(batch, dim, outFeat, branches int, dtype core.DType,
	packed, x, target []float32, lr float64, modes []parallel.TrainMode,
	combine parallel.CombineMode) []byte {
	var in bytes.Buffer
	_, _ = in.Write([]byte("PNB1"))
	for _, v := range []int32{int32(batch), int32(dim), int32(outFeat), int32(branches), int32(dtype)} {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	// Combine before packed/target so C++ can size Filter gate + Concat out.
	_ = binary.Write(&in, binary.LittleEndian, parallelCombineCode(combine))
	for _, v := range packed {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range target {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	_ = binary.Write(&in, binary.LittleEndian, lr)
	for _, m := range modes {
		_ = binary.Write(&in, binary.LittleEndian, int32(m))
	}
	_ = binary.Write(&in, binary.LittleEndian, float64(0)) // shadow_coef unused
	_ = binary.Write(&in, binary.LittleEndian, float64(0)) // surprise_thresh unused
	return in.Bytes()
}

func runGoParallelNBP(dtype core.DType, batch, dim, outFeat, branches int,
	packed, x, target []float32, lr float64, modes []parallel.TrainMode,
	combine parallel.CombineMode) (cameralModesOut, error) {
	var z cameralModesOut
	cfg := parallel.Config{Dim: dim, OutFeat: outFeat, Branches: branches, Combine: combine}
	branchN := branches * outFeat * dim
	var branchPacked, gateInit []float32
	if combine == parallel.CombineFilter {
		if len(packed) < branchN+branches*dim {
			return z, fmt.Errorf("filter packed short")
		}
		branchPacked = packed[:branchN]
		gateInit = packed[branchN : branchN+branches*dim]
	} else {
		branchPacked = packed
	}
	l, err := parallel.NewConfigured[float32](cfg, dtype, quant.FormatNone, branchPacked, gateInit)
	if err != nil {
		return z, err
	}
	if len(modes) > 0 {
		l.SetBranchModes(modes...)
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	outD := outFeat
	if combine == parallel.CombineConcat {
		outD = branches * outFeat
	}
	tgt := core.NewTensor[float32](batch, outD)
	copy(tgt.Data, target)
	loss, err := parallel.TrainMSE(l, xin, tgt, parallel.ModeNormalBP, lr)
	if err != nil {
		return z, err
	}
	z.loss = loss
	z.WAfter = make([][]float32, branches)
	for i := 0; i < branches; i++ {
		dl := l.Branches[i].(*dense.Layer)
		Wa, err := weights.MatrixF32(dl.Weights)
		if err != nil {
			return z, err
		}
		z.WAfter[i] = append([]float32(nil), Wa...)
	}
	return z, nil
}

func runParallelNBPCase(combine parallel.CombineMode, dtype core.DType, exact bool, benchIters int) error {
	const batch, dim, outFeat, branches = 2, 4, 4, 2
	const lr = 1e-2
	// Inherit BranchModes → unified NormalBP (no mixed Adv/Memory/Shadow path).
	modes := []parallel.TrainMode{parallel.ModeInherit, parallel.ModeInherit}
	wn := branches * outFeat * dim
	if combine == parallel.CombineFilter {
		wn += branches * dim
	}
	packed := randInit(1, wn, 0x504e4201)
	for i := range packed {
		packed[i] *= 0.05
	}
	x := randInit(batch, dim, 0x504e4202)
	for i := range x {
		x[i] *= 0.05
	}
	outD := outFeat
	if combine == parallel.CombineConcat {
		outD = branches * outFeat
	}
	target := make([]float32, batch*outD)
	for i := range target {
		target[i] = float32(0.05 * float64(i%outD+1))
	}
	goO, err := runGoParallelNBP(dtype, batch, dim, outFeat, branches, packed, x, target, lr, modes, combine)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload := encodeParallelNBP(batch, dim, outFeat, branches, dtype, packed, x, target, lr, modes, combine)
	if benchIters > 0 {
		_, _, err = runCppCameralModes(payload, "PNBO", benchIters)
		if err != nil {
			return fmt.Errorf("cpp-bench: %w", err)
		}
	}
	cppO, _, err := runCppCameralModes(payload, "PNBO", 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareCameralModes(goO, cppO, exact)
}

func caseParallelNBPAddF32() error {
	return runParallelNBPCase(parallel.CombineAdd, core.DTypeFloat32, true, defaultGoBenchIters)
}

func caseParallelNBPAvgF32() error {
	return runParallelNBPCase(parallel.CombineAvg, core.DTypeFloat32, true, defaultGoBenchIters)
}

func caseParallelNBPConcatF32() error {
	return runParallelNBPCase(parallel.CombineConcat, core.DTypeFloat32, true, defaultGoBenchIters)
}

func caseParallelNBPMaxF32() error {
	return runParallelNBPCase(parallel.CombineMax, core.DTypeFloat32, true, defaultGoBenchIters)
}

func caseParallelNBPSparseKF32() error {
	return runParallelNBPCase(parallel.CombineSparseK, core.DTypeFloat32, true, defaultGoBenchIters)
}

func caseParallelNBPDisagreeF32() error {
	return runParallelNBPCase(parallel.CombineDisagree, core.DTypeFloat32, true, defaultGoBenchIters)
}

func caseParallelNBPFilterF32() error {
	return runParallelNBPCase(parallel.CombineFilter, core.DTypeFloat32, true, defaultGoBenchIters)
}

func caseParallelNBPAddAllDTypes() error {
	const benchIters = defaultGoBenchIters
	fmt.Printf("\n      [Parallel NormalBP Add ×34]\n")
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s\n", "dtype", "band", "exact%W", "goµs", "cppµs", "spd%Δ")
	var fails []string
	modes := []parallel.TrainMode{parallel.ModeInherit, parallel.ModeInherit}
	const batch, dim, outFeat, branches = 2, 4, 4, 2
	const lr = 1e-2
	for _, dt := range core.AllDTypes {
		err := runParallelNBPCase(parallel.CombineAdd, dt, true, 0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s:%v", dt, err))
			fmt.Printf("      %-12s %-12s %7s %10s %10s %8s  %v\n", dt, suites.DetBroken, "—",
				"—", "—", "—", err)
			continue
		}
		packed := randInit(1, branches*outFeat*dim, 0x504e4201)
		for i := range packed {
			packed[i] *= 0.05
		}
		x := randInit(batch, dim, 0x504e4202)
		for i := range x {
			x[i] *= 0.05
		}
		target := make([]float32, batch*outFeat)
		for i := range target {
			target[i] = float32(0.05 * float64(i%outFeat+1))
		}
		payload := encodeParallelNBP(batch, dim, outFeat, branches, dt, packed, x, target, lr, modes, parallel.CombineAdd)
		_, cppDur, err := runCppCameralModes(payload, "PNBO", benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s bench:%v", dt, err))
			continue
		}
		goO, _ := runGoParallelNBP(dt, batch, dim, outFeat, branches, packed, x, target, lr, modes, parallel.CombineAdd)
		cppO, _, _ := runCppCameralModes(payload, "PNBO", 0)
		m := suites.CompareF32Spectrum(goO.WAfter[0], cppO.WAfter[0])
		goUS, berr := benchGoUS(func() error {
			_, err := runGoParallelNBP(dt, batch, dim, outFeat, branches, packed, x, target, lr, modes, parallel.CombineAdd)
			return err
		}, benchIters)
		if berr != nil {
			fails = append(fails, fmt.Sprintf("%s go-bench:%v", dt, berr))
			continue
		}
		cppUS := perIterUS(cppDur, benchIters)
		spd := suites.SpeedPctDiff(goUS, cppUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %8.1f%%\n",
			dt, m.Band, m.ExactPct, FormatUS(goUS), FormatUS(cppUS), spd)
	}
	if len(fails) > 0 {
		return fmt.Errorf("dtype fails: %s", joinFails(fails))
	}
	return nil
}
