package go_v_cpp

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

// Wire combine codes (must match C++ welvet::parallel::Combine).
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
		{Name: "Parallel Add Float32 [B,D] — exact bits", Run: casePlAdd2D},
		{Name: "Parallel Add Float32 [B,T,D] — exact bits", Run: casePlAdd3D},
		{Name: "Parallel Concat Float32 — exact bits", Run: casePlConcat},
		{Name: "Parallel Avg Float32 — exact bits", Run: casePlAvg},
		{Name: "Parallel Filter Float32 — exact bits", Run: casePlFilter},
		{Name: "Parallel Max Float32 — exact bits", Run: casePlMax},
		{Name: "Parallel SparseK Float32 — exact bits", Run: casePlSparseK},
		{Name: "Parallel Disagree Float32 (2+3 branches) — exact bits", Run: casePlDisagree},
		{Name: "Parallel Add × all 34 dtypes — spectrum + speed", Run: casePlAddAllDTypes},
	}
}

func RunParallel() error {
	var fails []string
	cs := CasesParallel()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [pl %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_parallel", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_parallel", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp parallel: %d failed: %s", len(fails), strings.Join(fails, ", "))
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
	cfg           parallel.Config
	combineCode   int
	dtype         core.DType
	backend       core.Backend
	batch         int
	seq           int
	packed        []float32 // branches (+ gate if filter)
	x             []float32
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
		return nil, nil, bench, fmt.Errorf("cpp parallel bad magic")
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

func runCppParallel(payload []byte, benchIters int) (pre, post []float32, dur time.Duration, err error) {
	tool, err := findDenseFwd()
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
		return nil, nil, 0, fmt.Errorf("cpp parallel: %v: %s", err, stderr.String())
	}
	return decodeParallel(stdout.Bytes())
}

func runParallelExact(fc plCase) error {
	goPre, goPost, err := runGoParallel(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeParallel(fc)
	if err != nil {
		return err
	}
	cppPre, cppPost, _, err := runCppParallel(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	preM := suites.CompareF32Spectrum(goPre, cppPre)
	postM := suites.CompareF32Spectrum(goPost, cppPost)
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
	fc := plCase{
		cfg: cfg, combineCode: plAdd, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch: 2, seq: 1,
		packed: randInit(1, plWeightCount(cfg), 0x504c4101),
		x:      randInit(2, 8, 0x504c4102),
	}
	return runParallelExact(fc)
}

func casePlAdd3D() error {
	cfg := parallel.Config{Dim: 6, OutFeat: 6, Branches: 2, Combine: parallel.CombineAdd, SeqLen: 4}
	fc := plCase{
		cfg: cfg, combineCode: plAdd, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch: 2, seq: 4,
		packed: randInit(1, plWeightCount(cfg), 0x504c4103),
		x:      randInit(2*4, 6, 0x504c4104),
	}
	return runParallelExact(fc)
}

func casePlConcat() error {
	cfg := parallel.Config{Dim: 8, OutFeat: 4, Branches: 3, Combine: parallel.CombineConcat}
	fc := plCase{
		cfg: cfg, combineCode: plConcat, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch: 2, seq: 1,
		packed: randInit(1, plWeightCount(cfg), 0x504c4301),
		x:      randInit(2, 8, 0x504c4302),
	}
	return runParallelExact(fc)
}

func casePlAvg() error {
	cfg := parallel.Config{Dim: 8, OutFeat: 8, Branches: 3, Combine: parallel.CombineAvg}
	fc := plCase{
		cfg: cfg, combineCode: plAvg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch: 2, seq: 2,
		packed: randInit(1, plWeightCount(cfg), 0x504c5601),
		x:      randInit(2*2, 8, 0x504c5602),
	}
	return runParallelExact(fc)
}

func casePlFilter() error {
	cfg := parallel.Config{Dim: 8, OutFeat: 8, Branches: 2, Combine: parallel.CombineFilter}
	fc := plCase{
		cfg: cfg, combineCode: plFilter, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch: 2, seq: 1,
		packed: randInit(1, plWeightCount(cfg), 0x504c4601),
		x:      randInit(2, 8, 0x504c4602),
	}
	return runParallelExact(fc)
}

func casePlMax() error {
	cfg := parallel.Config{Dim: 8, OutFeat: 8, Branches: 3, Combine: parallel.CombineMax}
	fc := plCase{
		cfg: cfg, combineCode: plMax, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch: 2, seq: 1,
		packed: randInit(1, plWeightCount(cfg), 0x504c4d01),
		x:      randInit(2, 8, 0x504c4d02),
	}
	return runParallelExact(fc)
}

func casePlSparseK() error {
	// Scale branches differently so L2 scores are distinct (avoid sort ties).
	cfg := parallel.Config{Dim: 8, OutFeat: 8, Branches: 3, Combine: parallel.CombineSparseK, SparseK: 2}
	packed := randInit(1, plWeightCount(cfg), 0x504c5301)
	childN := cfg.OutFeat * cfg.Dim
	for b := 0; b < cfg.Branches; b++ {
		scale := float32(1 + b) // 1×, 2×, 3×
		for i := 0; i < childN; i++ {
			packed[b*childN+i] *= scale
		}
	}
	fc := plCase{
		cfg: cfg, combineCode: plSparseK, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch: 2, seq: 1, packed: packed, x: randInit(2, 8, 0x504c5302),
	}
	return runParallelExact(fc)
}

func casePlDisagree() error {
	cfg2 := parallel.Config{Dim: 8, OutFeat: 8, Branches: 2, Combine: parallel.CombineDisagree, DisagreeBeta: 0.5}
	fc2 := plCase{
		cfg: cfg2, combineCode: plDisagree, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch: 2, seq: 1,
		packed: randInit(1, plWeightCount(cfg2), 0x504c4401),
		x:      randInit(2, 8, 0x504c4402),
	}
	if err := runParallelExact(fc2); err != nil {
		return fmt.Errorf("disagree nb=2: %w", err)
	}
	cfg3 := parallel.Config{Dim: 8, OutFeat: 8, Branches: 3, Combine: parallel.CombineDisagree, DisagreeBeta: 0.75}
	fc3 := plCase{
		cfg: cfg3, combineCode: plDisagree, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch: 2, seq: 1,
		packed: randInit(1, plWeightCount(cfg3), 0x504c4403),
		x:      randInit(2, 8, 0x504c4404),
	}
	if err := runParallelExact(fc3); err != nil {
		return fmt.Errorf("disagree nb=3: %w", err)
	}
	return nil
}

func casePlAddAllDTypes() error {
	const batch, seq, benchIters = 2, 3, 20
	cfg := parallel.Config{Dim: 8, OutFeat: 8, Branches: 2, Combine: parallel.CombineAdd, SeqLen: seq}
	if err := cfg.Validate(); err != nil {
		return err
	}
	packed := randInit(1, plWeightCount(cfg), 0xd7a4e101)
	x := randInit(batch*seq, cfg.Dim, 0xd7a4e102)
	fmt.Printf("\n      Go CPU vs C++ CPU — Parallel Add×34\n")
	fmt.Printf("      %-12s %-12s %8s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "spect%", "go_µs", "cpp_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 88))
	var exactN, totalN int
	worstBand := suites.DetExactBit
	var fails []string
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
		payload, err := encodeParallel(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s enc: %v", dt, err))
			continue
		}
		cppPre, cppPost, _, err := runCppParallel(payload, 0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp: %v", dt, err))
			fmt.Printf("      %-12s FAIL cpp %v\n", dt, err)
			continue
		}
		preM := suites.CompareF32Spectrum(goPre, cppPre)
		postM := suites.CompareF32Spectrum(goPost, cppPost)
		m := postM
		if preM.Band < m.Band {
			m = preM
		}
		if m.Band < worstBand {
			worstBand = m.Band
		}
		exactN += m.ExactBits
		totalN += m.N
		for i := 0; i < 2; i++ {
			if _, _, err := runGoParallel(fc); err != nil {
				break
			}
		}
		t0 := time.Now()
		for i := 0; i < benchIters; i++ {
			if _, _, err := runGoParallel(fc); err != nil {
				fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
				break
			}
		}
		goDur := time.Since(t0)
		_, _, cppDur, err := runCppParallel(payload, benchIters)
		goUS := float64(goDur.Nanoseconds()) / float64(benchIters) / 1000
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp bench: %v", dt, err))
			fmt.Printf("      %-12s %-12s %7.2f%% %7.0f%% %10.2f %10s %9s %s\n",
				dt, m.Band, m.ExactPct, m.Band.Percent(), goUS, "?", "?", fmt.Sprintf("%.3g", m.MaxAbs))
			continue
		}
		cppUS := float64(cppDur.Nanoseconds()) / float64(benchIters) / 1000
		spdPct := suites.SpeedPctDiff(goUS, cppUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %7.0f%% %10.2f %10.2f %+8.1f%% %s\n",
			dt, m.Band, m.ExactPct, m.Band.Percent(), goUS, cppUS, spdPct, fmt.Sprintf("%.3g", m.MaxAbs))
		if m.Band != suites.DetExactBit {
			fails = append(fails, fmt.Sprintf("%s: %s", dt, suites.FormatSpectrum(m)))
		}
	}
	overallExact := 0.0
	if totalN > 0 {
		overallExact = 100.0 * float64(exactN) / float64(totalN)
	}
	fmt.Printf("      %s\n", strings.Repeat("-", 88))
	fmt.Printf("      overall exact_bits=%.2f%%  worst_band=%s (%.0f%%)  aim=exact_bit (100%%)\n",
		overallExact, worstBand, worstBand.Percent())
	fmt.Printf("      spd%%Δ = (cpp−go)/go×100  (negative ⇒ C++ faster); poly/cameral Parallel still gap")
	if len(fails) > 0 {
		return fmt.Errorf("%d parallel-dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}
