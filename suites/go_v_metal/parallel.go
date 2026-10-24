package go_v_metal

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"time"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
)

// Wire combine codes (must match C++/Metal welvet::parallel::Combine).
const (
	plConcat   = 0
	plAdd      = 1
	plAvg      = 2
	plFilter   = 3
	plMax      = 4
	plSparseK  = 5
	plDisagree = 6
)

func plCombineMode(code int) parallel.CombineMode {
	switch code {
	case plAdd:
		return parallel.CombineAdd
	case plAvg:
		return parallel.CombineAvg
	case plFilter:
		return parallel.CombineFilter
	case plMax:
		return parallel.CombineMax
	case plSparseK:
		return parallel.CombineSparseK
	case plDisagree:
		return parallel.CombineDisagree
	default:
		return parallel.CombineConcat
	}
}

func CasesParallel() []Case {
	return []Case{
		{Name: "Locate welvet.metal tool", Run: caseLocateMetal},
		{Name: "Locate welvet.cpp tool (optional speed)", Run: caseLocateCppOptional},
		{Name: "Parallel Add Float32 [B,D] — exact bits vs Go", Run: casePlAdd2D},
		{Name: "Parallel Add Float32 [B,T,D] — exact bits vs Go", Run: casePlAdd3D},
		{Name: "Parallel Concat Float32 — exact bits vs Go", Run: casePlConcat},
		{Name: "Parallel Avg Float32 — exact bits vs Go", Run: casePlAvg},
		{Name: "Parallel Filter Float32 — exact bits vs Go", Run: casePlFilter},
		{Name: "Parallel Max Float32 — exact bits vs Go", Run: casePlMax},
		{Name: "Parallel SparseK Float32 — exact bits vs Go", Run: casePlSparseK},
		{Name: "Parallel Disagree Float32 (2+3 branches) — exact bits vs Go", Run: casePlDisagree},
		{Name: "Parallel Add ×34 — ≥99% + Go/C++/Metal timings", Run: casePlAddAllDTypes},
	}
}

func RunParallel() error {
	var fails []string
	cs := CasesParallel()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [pl %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_metal_parallel", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_metal_parallel", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_metal parallel: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func plWeightCount(cfg parallel.Config) int {
	n := cfg.Branches * cfg.OutFeat * cfg.Dim
	if cfg.Combine == parallel.CombineFilter {
		n += cfg.Branches * cfg.Dim
	}
	return n
}

type plCase struct {
	cfg         parallel.Config
	combineCode int
	dtype       core.DType
	backend     core.Backend
	batch       int
	seq         int
	packed      []float32 // branches (+ gate if filter)
	x           []float32
}

func runGoParallel(fc plCase) (pre, post []float32, err error) {
	cfg := fc.cfg
	cfg.Combine = plCombineMode(fc.combineCode)
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	childN := cfg.OutFeat * cfg.Dim
	branchW := cfg.Branches * childN
	var packed, gateInit []float32
	if len(fc.packed) < branchW {
		return nil, nil, fmt.Errorf("packed short")
	}
	packed = fc.packed[:branchW]
	if cfg.Combine == parallel.CombineFilter {
		if len(fc.packed) < branchW+cfg.Branches*cfg.Dim {
			return nil, nil, fmt.Errorf("gate packed short")
		}
		gateInit = fc.packed[branchW : branchW+cfg.Branches*cfg.Dim]
	}
	l, err := parallel.NewConfigured(cfg, fc.dtype, quant.FormatNone, packed, gateInit)
	if err != nil {
		return nil, nil, err
	}
	l.Exec.Backend = fc.backend
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	l.SyncBranchExec()
	for _, ch := range l.Branches {
		if d, ok := ch.(*dense.Layer); ok && d != nil {
			d.Exec.MultiCore = false
			d.Core.MultiCore = false
			d.Exec.Backend = fc.backend
		}
	}
	if l.Gate != nil {
		l.Gate.Exec.MultiCore = false
		l.Gate.Core.MultiCore = false
		l.Gate.Exec.Backend = fc.backend
	}
	var t *core.Tensor[float32]
	if fc.seq <= 1 {
		t = core.NewTensor[float32](fc.batch, cfg.Dim)
	} else {
		t = core.NewTensor[float32](fc.batch, fc.seq, cfg.Dim)
	}
	copy(t.Data, fc.x)
	preT, postT, err := parallel.Forward(l, t)
	if err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), preT.Data...), append([]float32(nil), postT.Data...), nil
}

func encodeParallel(fc plCase) ([]byte, error) {
	cfg := fc.cfg
	cfg.Combine = plCombineMode(fc.combineCode)
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	seq := fc.seq
	if seq <= 0 {
		seq = 1
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("WPL1"))
	for _, v := range []int32{
		int32(fc.batch), int32(seq), int32(cfg.Dim), int32(cfg.OutFeat), int32(cfg.Branches),
		int32(fc.combineCode), int32(cfg.SparseK), int32(fc.dtype), int32(fc.backend),
	} {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	_ = binary.Write(&in, binary.LittleEndian, cfg.DisagreeBeta)
	for _, v := range fc.packed {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	return in.Bytes(), nil
}

func decodeParallel(raw []byte) (pre, post []float32, bench time.Duration, err error) {
	if len(raw) >= 4+8+4 && string(raw[:4]) == "WVB1" {
		r := bytes.NewReader(raw[4:])
		var ns int64
		var iters int32
		if err := binary.Read(r, binary.LittleEndian, &ns); err != nil {
			return nil, nil, 0, err
		}
		if err := binary.Read(r, binary.LittleEndian, &iters); err != nil {
			return nil, nil, 0, err
		}
		rest, err := io.ReadAll(r)
		if err != nil {
			return nil, nil, 0, err
		}
		raw = rest
		bench = time.Duration(ns)
	}
	if len(raw) < 8 || string(raw[:4]) != "WPLO" {
		return nil, nil, bench, fmt.Errorf("parallel bad magic: %q", string(raw[:min(4, len(raw))]))
	}
	r := bytes.NewReader(raw[4:])
	var preN, postN int32
	if err := binary.Read(r, binary.LittleEndian, &preN); err != nil {
		return nil, nil, bench, err
	}
	pre = make([]float32, preN)
	for i := range pre {
		if err := binary.Read(r, binary.LittleEndian, &pre[i]); err != nil {
			return nil, nil, bench, err
		}
	}
	if err := binary.Read(r, binary.LittleEndian, &postN); err != nil {
		return nil, nil, bench, err
	}
	post = make([]float32, postN)
	for i := range post {
		if err := binary.Read(r, binary.LittleEndian, &post[i]); err != nil {
			return nil, nil, bench, err
		}
	}
	return pre, post, bench, nil
}

func runToolParallel(tool string, fc plCase, benchIters int) (pre, post []float32, dur time.Duration, err error) {
	payload, err := encodeParallel(fc)
	if err != nil {
		return nil, nil, 0, err
	}
	var cmd *exec.Cmd
	if benchIters > 0 {
		cmd = exec.Command(tool, fmt.Sprintf("--bench=%d", benchIters))
	} else {
		cmd = exec.Command(tool)
	}
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, nil, 0, fmt.Errorf("%v: %s", err, stderr.String())
	}
	return decodeParallel(stdout.Bytes())
}

func runMetalParallel(fc plCase) (pre, post []float32, err error) {
	tool, err := findMetalFwd()
	if err != nil {
		return nil, nil, err
	}
	pre, post, _, err = runToolParallel(tool, fc, 0)
	return pre, post, err
}

func requireParallelExact(fc plCase) error {
	goPre, goPost, err := runGoParallel(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	mPre, mPost, err := runMetalParallel(fc)
	if err != nil {
		return fmt.Errorf("metal: %w", err)
	}
	preM := suites.CompareF32Spectrum(goPre, mPre)
	postM := suites.CompareF32Spectrum(goPost, mPost)
	fmt.Printf("\n      pre  %s", suites.FormatSpectrum(preM))
	fmt.Printf("\n      post %s", suites.FormatSpectrum(postM))
	if preM.Band != suites.DetExactBit {
		return fmt.Errorf("pre %s", suites.FormatSpectrum(preM))
	}
	if postM.Band != suites.DetExactBit {
		return fmt.Errorf("post %s", suites.FormatSpectrum(postM))
	}
	return nil
}

func casePlAdd2D() error {
	cfg := parallel.Config{Dim: 8, OutFeat: 8, Branches: 2, Combine: parallel.CombineAdd}
	return requireParallelExact(plCase{
		cfg: cfg, combineCode: plAdd, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch: 2, seq: 1,
		packed: randInit(1, plWeightCount(cfg), 0x504c4101),
		x:      randInit(2, 8, 0x504c4102),
	})
}

func casePlAdd3D() error {
	cfg := parallel.Config{Dim: 6, OutFeat: 6, Branches: 2, Combine: parallel.CombineAdd, SeqLen: 4}
	return requireParallelExact(plCase{
		cfg: cfg, combineCode: plAdd, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch: 2, seq: 4,
		packed: randInit(1, plWeightCount(cfg), 0x504c4103),
		x:      randInit(2*4, 6, 0x504c4104),
	})
}

func casePlConcat() error {
	cfg := parallel.Config{Dim: 8, OutFeat: 4, Branches: 3, Combine: parallel.CombineConcat}
	return requireParallelExact(plCase{
		cfg: cfg, combineCode: plConcat, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch: 2, seq: 1,
		packed: randInit(1, plWeightCount(cfg), 0x504c4301),
		x:      randInit(2, 8, 0x504c4302),
	})
}

func casePlAvg() error {
	cfg := parallel.Config{Dim: 8, OutFeat: 8, Branches: 3, Combine: parallel.CombineAvg}
	return requireParallelExact(plCase{
		cfg: cfg, combineCode: plAvg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch: 2, seq: 2,
		packed: randInit(1, plWeightCount(cfg), 0x504c5601),
		x:      randInit(2*2, 8, 0x504c5602),
	})
}

func casePlFilter() error {
	cfg := parallel.Config{Dim: 8, OutFeat: 8, Branches: 2, Combine: parallel.CombineFilter}
	return requireParallelExact(plCase{
		cfg: cfg, combineCode: plFilter, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch: 2, seq: 1,
		packed: randInit(1, plWeightCount(cfg), 0x504c4601),
		x:      randInit(2, 8, 0x504c4602),
	})
}

func casePlMax() error {
	cfg := parallel.Config{Dim: 8, OutFeat: 8, Branches: 3, Combine: parallel.CombineMax}
	return requireParallelExact(plCase{
		cfg: cfg, combineCode: plMax, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch: 2, seq: 1,
		packed: randInit(1, plWeightCount(cfg), 0x504c4d01),
		x:      randInit(2, 8, 0x504c4d02),
	})
}

func casePlSparseK() error {
	cfg := parallel.Config{Dim: 8, OutFeat: 8, Branches: 3, Combine: parallel.CombineSparseK, SparseK: 2}
	packed := randInit(1, plWeightCount(cfg), 0x504c5301)
	childN := cfg.OutFeat * cfg.Dim
	for b := 0; b < cfg.Branches; b++ {
		scale := float32(1 + b)
		for i := 0; i < childN; i++ {
			packed[b*childN+i] *= scale
		}
	}
	return requireParallelExact(plCase{
		cfg: cfg, combineCode: plSparseK, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch: 2, seq: 1, packed: packed, x: randInit(2, 8, 0x504c5302),
	})
}

func casePlDisagree() error {
	cfg2 := parallel.Config{Dim: 8, OutFeat: 8, Branches: 2, Combine: parallel.CombineDisagree, DisagreeBeta: 0.5}
	if err := requireParallelExact(plCase{
		cfg: cfg2, combineCode: plDisagree, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch: 2, seq: 1,
		packed: randInit(1, plWeightCount(cfg2), 0x504c4401),
		x:      randInit(2, 8, 0x504c4402),
	}); err != nil {
		return fmt.Errorf("disagree nb=2: %w", err)
	}
	cfg3 := parallel.Config{Dim: 8, OutFeat: 8, Branches: 3, Combine: parallel.CombineDisagree, DisagreeBeta: 0.75}
	if err := requireParallelExact(plCase{
		cfg: cfg3, combineCode: plDisagree, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch: 2, seq: 1,
		packed: randInit(1, plWeightCount(cfg3), 0x504c4403),
		x:      randInit(2, 8, 0x504c4404),
	}); err != nil {
		return fmt.Errorf("disagree nb=3: %w", err)
	}
	return nil
}

func casePlAddAllDTypes() error {
	// Mid GEMV so Metal soft-double Dense branches can win; host combine is exact.
	const batch, seq, benchIters = 4, 8, 80
	cfg := parallel.Config{Dim: 64, OutFeat: 64, Branches: 2, Combine: parallel.CombineAdd, SeqLen: seq}
	if err := cfg.Validate(); err != nil {
		return err
	}
	if len(core.AllDTypes) != 34 {
		return fmt.Errorf("expected 34 dtypes, got %d", len(core.AllDTypes))
	}
	metalTool, err := findMetalFwd()
	if err != nil {
		return err
	}
	cppTool, cppErr := findCppFwd()
	haveCpp := cppErr == nil

	packed := randInit(1, plWeightCount(cfg), 0xd7a4e101)
	x := randInit(batch*seq, cfg.Dim, 0xd7a4e102)

	fmt.Printf("\n      Go vs Metal Parallel Add FormatNone×34 — exact_bit preferred (≥99%% ok)")
	fmt.Printf("\n      shape B=%d T=%d D=%d Out=%d Branches=%d  iters=%d",
		batch, seq, cfg.Dim, cfg.OutFeat, cfg.Branches, benchIters)
	if haveCpp {
		fmt.Printf("\n      speed: Go / C++ / Metal")
	}
	fmt.Printf("\n      %-12s %-12s %7s %9s %9s %9s %8s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "mtl_µs", "winner", "vsGo")
	fmt.Printf("      %s\n", strings.Repeat("-", 100))

	var fails, soft []string
	var exactN, totalN int
	winCount := map[string]int{}

	for _, dt := range core.AllDTypes {
		fc := plCase{
			cfg: cfg, combineCode: plAdd, dtype: dt, backend: core.BackendCPUTiled,
			batch: batch, seq: seq, packed: packed, x: x,
		}
		goPre, goPost, err := runGoParallel(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		mPre, mPost, err := runMetalParallel(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s metal: %v", dt, err))
			fmt.Printf("      %-12s FAIL metal %v\n", dt, err)
			continue
		}
		preM := suites.CompareF32Spectrum(goPre, mPre)
		postM := suites.CompareF32Spectrum(goPost, mPost)
		m := postM
		if preM.Band < m.Band {
			m = preM
		}
		exactN += m.ExactBits
		totalN += m.N

		goUS, err := benchGoUS(func() error {
			_, _, e := runGoParallel(fc)
			return e
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}

		cppUS := 0.0
		cppStr := "—"
		if haveCpp {
			_, _, cppDur, err := runToolParallel(cppTool, fc, benchIters)
			if err != nil {
				cppStr = "err"
			} else {
				cppUS = perIterUS(cppDur, benchIters)
				cppStr = FormatUS(cppUS)
			}
		}

		_, _, metalDur, err := runToolParallel(metalTool, fc, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s metal bench: %v", dt, err))
			fmt.Printf("      %-12s %-12s %6.2f%% %9s %9s %9s %8s %s\n",
				dt, m.Band, m.ExactPct, FormatUS(goUS), cppStr, "?", "?", err)
			continue
		}
		metalUS := perIterUS(metalDur, benchIters)

		names := []string{"Go", "Metal"}
		uss := []float64{goUS, metalUS}
		if haveCpp && cppUS > 0 {
			names = []string{"Go", "C++", "Metal"}
			uss = []float64{goUS, cppUS, metalUS}
		}
		winner, _, vsGo := pickFastest(names, uss)
		winCount[winner]++
		vsGoStr := "baseline"
		if winner != "Go" {
			vsGoStr = fmt.Sprintf("%s %+0.1f%%", winner, vsGo)
		}
		fmt.Printf("      %-12s %-12s %6.2f%% %9s %9s %9s %8s %s\n",
			dt, m.Band, m.ExactPct, FormatUS(goUS), cppStr, FormatUS(metalUS), winner, vsGoStr)
		if m.Band != suites.DetExactBit {
			soft = append(soft, fmt.Sprintf("%s: %s", dt, suites.FormatSpectrum(m)))
		}
	}
	overallExact := 0.0
	if totalN > 0 {
		overallExact = 100.0 * float64(exactN) / float64(totalN)
	}
	fmt.Printf("      %s\n", strings.Repeat("-", 100))
	fmt.Printf("      overall exact_bits=%.2f%%  (aim≥99%% vs Go; soft-double Dense + host combine)\n", overallExact)
	parts := []string{}
	for _, n := range []string{"Go", "C++", "Metal"} {
		if winCount[n] > 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", n, winCount[n]))
		}
	}
	fmt.Printf("      wins: %s\n", strings.Join(parts, " "))
	if len(soft) > 0 {
		fmt.Printf("      soft (non-exact_bit, allowed if overall≥99%%): %s\n", strings.Join(soft, " | "))
	}
	fmt.Printf("      note: Parallel = Dense soft-double branches + host combine (Add×34 mid)")
	if len(fails) > 0 {
		return fmt.Errorf("%d parallel hard fails: %s", len(fails), strings.Join(fails, " | "))
	}
	if overallExact < 99.0 {
		return fmt.Errorf("overall exact_bits=%.2f%% < 99%%", overallExact)
	}
	return nil
}
