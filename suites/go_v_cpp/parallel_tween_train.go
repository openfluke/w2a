package go_v_cpp

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"strings"
	"time"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

// CasesParallelTweenTrain — Parallel Dense Tween-family parents (Phase C).
// Wires: PTW1 Tween, PTC1 TweenChain≡BP, PTS1 TweenSplit, PTA1 TweenAlt.
// Step* Parallel-as-Line share Family credit with non-Step sibling.
func CasesParallelTweenTrain() []Case {
	return []Case{
		{Name: "Dense Add Parallel Tween Float32 — exact bits", Run: caseParallelTweenAddF32},
		{Name: "Dense Add Parallel Tween × all 34 dtypes — spectrum", Run: caseParallelTweenAddAllDTypes},
		{Name: "Dense Avg Parallel Tween Float32 — exact bits", Run: caseParallelTweenAvgF32},
		{Name: "Dense Add Parallel Tween volume 2×2×2×1 Float32 — exact bits", Run: casePnTweenVol222F32},

		{Name: "Dense Add Parallel TweenChain Float32 — exact bits", Run: caseParallelTweenChainAddF32},
		{Name: "Dense Add Parallel TweenChain × all 34 dtypes — spectrum", Run: caseParallelTweenChainAddAllDTypes},
		{Name: "Dense Avg Parallel TweenChain Float32 — exact bits", Run: caseParallelTweenChainAvgF32},
		{Name: "Dense Add Parallel TweenChain volume 2×2×2×1 Float32 — exact bits", Run: casePnTweenChainVol222F32},

		{Name: "Dense Add Parallel TweenSplit Float32 — exact bits", Run: caseParallelTweenSplitAddF32},
		{Name: "Dense Add Parallel TweenSplit × all 34 dtypes — spectrum", Run: caseParallelTweenSplitAddAllDTypes},
		{Name: "Dense Avg Parallel TweenSplit Float32 — exact bits", Run: caseParallelTweenSplitAvgF32},
		{Name: "Dense Add Parallel TweenSplit volume 2×2×2×1 Float32 — exact bits", Run: casePnTweenSplitVol222F32},

		{Name: "Dense Add Parallel TweenAlt Float32 — exact bits", Run: caseParallelTweenAltAddF32},
		{Name: "Dense Add Parallel TweenAlt × all 34 dtypes — spectrum", Run: caseParallelTweenAltAddAllDTypes},
		{Name: "Dense Avg Parallel TweenAlt Float32 — exact bits", Run: caseParallelTweenAltAvgF32},
		{Name: "Dense Add Parallel TweenAlt volume 2×2×2×1 Float32 — exact bits", Run: casePnTweenAltVol222F32},
	}
}

func RunParallelTweenTrain() error {
	var fails []string
	cs := CasesParallelTweenTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [parallel-tween %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_parallel_tween_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_parallel_tween_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp parallel_tween_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func encodeParallelTweenFamily(magic string, batch, dim, outFeat, branches int, dtype core.DType,
	packed, x, target []float32, lr float64, modes []parallel.TrainMode,
	combine parallel.CombineMode) []byte {
	var in bytes.Buffer
	_, _ = in.Write([]byte(magic))
	for _, v := range []int32{int32(batch), int32(dim), int32(outFeat), int32(branches), int32(dtype)} {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	// Combine before packed (matches PNB1 / C++ run_parallel_cameral_train).
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

func runGoParallelTweenFamily(dtype core.DType, batch, dim, outFeat, branches int,
	packed, x, target []float32, lr float64, modes []parallel.TrainMode,
	combine parallel.CombineMode, parent parallel.TrainMode) (cameralModesOut, error) {
	var z cameralModesOut
	cfg := parallel.Config{Dim: dim, OutFeat: outFeat, Branches: branches, Combine: combine}
	l, err := parallel.NewConfigured[float32](cfg, dtype, quant.FormatNone, packed, nil)
	if err != nil {
		return z, err
	}
	if len(modes) > 0 {
		l.SetBranchModes(modes...)
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	tgt := core.NewTensor[float32](batch, outFeat)
	copy(tgt.Data, target)
	loss, err := parallel.TrainMSE(l, xin, tgt, parent, lr)
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

func runParallelTweenFamilyCase(magic, outMagic string, parent parallel.TrainMode,
	combine parallel.CombineMode, dtype core.DType, exact bool, benchIters int, seedBase uint32) error {
	const batch, dim, outFeat, branches = 2, 4, 4, 2
	const lr = 1e-2
	modes := []parallel.TrainMode{parallel.ModeInherit, parallel.ModeInherit}
	packed := randInit(1, branches*outFeat*dim, seedBase)
	for i := range packed {
		packed[i] *= 0.05
	}
	x := randInit(batch, dim, seedBase+1)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, batch*outFeat)
	for i := range target {
		target[i] = float32(0.05 * float64(i%outFeat+1))
	}
	goO, err := runGoParallelTweenFamily(dtype, batch, dim, outFeat, branches, packed, x, target, lr, modes, combine, parent)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload := encodeParallelTweenFamily(magic, batch, dim, outFeat, branches, dtype, packed, x, target, lr, modes, combine)
	if benchIters > 0 {
		_, _, err = runCppCameralModes(payload, outMagic, benchIters)
		if err != nil {
			return fmt.Errorf("cpp-bench: %w", err)
		}
	}
	cppO, _, err := runCppCameralModes(payload, outMagic, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareCameralModes(goO, cppO, exact)
}

func runParallelTweenFamilySpectrum(label, magic, outMagic string, parent parallel.TrainMode, seedBase uint32) error {
	const benchIters = 20
	fmt.Printf("\n      [Parallel %s Add ×34]\n", label)
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s\n", "dtype", "band", "exact%W", "goµs", "cppµs", "spd%Δ")
	var fails []string
	modes := []parallel.TrainMode{parallel.ModeInherit, parallel.ModeInherit}
	const batch, dim, outFeat, branches = 2, 4, 4, 2
	const lr = 1e-2
	for _, dt := range core.AllDTypes {
		t0 := time.Now()
		err := runParallelTweenFamilyCase(magic, outMagic, parent, parallel.CombineAdd, dt, true, 0, seedBase)
		goDur := time.Since(t0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s:%v", dt, err))
			fmt.Printf("      %-12s %-12s %7s %10.2f %10s %8s  %v\n", dt, suites.DetBroken, "—",
				float64(goDur.Nanoseconds())/1000, "—", "—", err)
			continue
		}
		packed := randInit(1, branches*outFeat*dim, seedBase)
		for i := range packed {
			packed[i] *= 0.05
		}
		x := randInit(batch, dim, seedBase+1)
		for i := range x {
			x[i] *= 0.05
		}
		target := make([]float32, batch*outFeat)
		for i := range target {
			target[i] = float32(0.05 * float64(i%outFeat+1))
		}
		payload := encodeParallelTweenFamily(magic, batch, dim, outFeat, branches, dt, packed, x, target, lr, modes, parallel.CombineAdd)
		_, cppDur, err := runCppCameralModes(payload, outMagic, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s bench:%v", dt, err))
			continue
		}
		goO, _ := runGoParallelTweenFamily(dt, batch, dim, outFeat, branches, packed, x, target, lr, modes, parallel.CombineAdd, parent)
		cppO, _, _ := runCppCameralModes(payload, outMagic, 0)
		m := suites.CompareF32Spectrum(goO.WAfter[0], cppO.WAfter[0])
		goUS := float64(goDur.Nanoseconds()) / 1000
		cppUS := float64(cppDur.Nanoseconds()) / float64(benchIters) / 1000
		spd := suites.SpeedPctDiff(goUS, cppUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %10.2f %10.2f %8.1f%%\n",
			dt, m.Band, m.ExactPct, goUS, cppUS, spd)
	}
	if len(fails) > 0 {
		return fmt.Errorf("dtype fails: %s", joinFails(fails))
	}
	return nil
}

func caseParallelTweenAddF32() error {
	return runParallelTweenFamilyCase("PTW1", "PTWO", parallel.ModeTween, parallel.CombineAdd, core.DTypeFloat32, true, 40, 0x50545701)
}
func caseParallelTweenAvgF32() error {
	return runParallelTweenFamilyCase("PTW1", "PTWO", parallel.ModeTween, parallel.CombineAvg, core.DTypeFloat32, true, 40, 0x50545701)
}
func caseParallelTweenAddAllDTypes() error {
	return runParallelTweenFamilySpectrum("Tween", "PTW1", "PTWO", parallel.ModeTween, 0x50545701)
}

func caseParallelTweenChainAddF32() error {
	return runParallelTweenFamilyCase("PTC1", "PTCO", parallel.ModeTweenChain, parallel.CombineAdd, core.DTypeFloat32, true, 40, 0x50544301)
}
func caseParallelTweenChainAvgF32() error {
	return runParallelTweenFamilyCase("PTC1", "PTCO", parallel.ModeTweenChain, parallel.CombineAvg, core.DTypeFloat32, true, 40, 0x50544301)
}
func caseParallelTweenChainAddAllDTypes() error {
	return runParallelTweenFamilySpectrum("TweenChain", "PTC1", "PTCO", parallel.ModeTweenChain, 0x50544301)
}

func caseParallelTweenSplitAddF32() error {
	return runParallelTweenFamilyCase("PTS1", "PTSO", parallel.ModeTweenSplit, parallel.CombineAdd, core.DTypeFloat32, true, 40, 0x50545301)
}
func caseParallelTweenSplitAvgF32() error {
	return runParallelTweenFamilyCase("PTS1", "PTSO", parallel.ModeTweenSplit, parallel.CombineAvg, core.DTypeFloat32, true, 40, 0x50545301)
}
func caseParallelTweenSplitAddAllDTypes() error {
	return runParallelTweenFamilySpectrum("TweenSplit", "PTS1", "PTSO", parallel.ModeTweenSplit, 0x50545301)
}

func caseParallelTweenAltAddF32() error {
	return runParallelTweenFamilyCase("PTA1", "PTAO", parallel.ModeTweenAlt, parallel.CombineAdd, core.DTypeFloat32, true, 40, 0x50544101)
}
func caseParallelTweenAltAvgF32() error {
	return runParallelTweenFamilyCase("PTA1", "PTAO", parallel.ModeTweenAlt, parallel.CombineAvg, core.DTypeFloat32, true, 40, 0x50544101)
}
func caseParallelTweenAltAddAllDTypes() error {
	return runParallelTweenFamilySpectrum("TweenAlt", "PTA1", "PTAO", parallel.ModeTweenAlt, 0x50544101)
}
