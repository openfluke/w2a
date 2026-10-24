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
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/quant"
)

func CasesMetacognition() []Case {
	return []Case{
		{Name: "Locate welvet.metal tool", Run: caseLocateMetal},
		{Name: "Locate welvet.cpp tool (optional speed)", Run: caseLocateCppOptional},
		{Name: "Metacognition Float32 [B,D] — exact bits vs Go", Run: caseMeta2D},
		{Name: "Metacognition Float32 [B,T,D] — exact bits vs Go", Run: caseMeta3D},
		{Name: "Metacognition ×34 — ≥99% + Go/C++/Metal timings", Run: caseMetaAllDTypes},
	}
}

func RunMetacognition() error {
	var fails []string
	cs := CasesMetacognition()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [mc %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_metal_metacognition", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_metal_metacognition", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_metal metacognition: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type mcCase struct {
	cfg     metacognition.Config
	dtype   core.DType
	backend core.Backend
	batch   int
	seq     int // 1 → [B,D]
	packed  []float32
	x       []float32
}

func runGoMeta(fc mcCase) (pre, post []float32, err error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	l, err := metacognition.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.packed)
	if err != nil {
		return nil, nil, err
	}
	l.Exec.Backend = fc.backend
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	l.Observed.Exec.MultiCore = false
	l.Observed.Core.MultiCore = false
	l.Observed.Exec.Backend = fc.backend
	var t *core.Tensor[float32]
	if fc.seq <= 1 {
		t = core.NewTensor[float32](fc.batch, cfg.Dim)
	} else {
		t = core.NewTensor[float32](fc.batch, fc.seq, cfg.Dim)
	}
	copy(t.Data, fc.x)
	preT, postT, err := metacognition.Forward(l, t)
	if err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), preT.Data...), append([]float32(nil), postT.Data...), nil
}

func encodeMeta(fc mcCase) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	seq := fc.seq
	if seq <= 0 {
		seq = 1
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("WMC1"))
	for _, v := range []int32{
		int32(fc.batch), int32(seq), int32(cfg.Dim), int32(fc.dtype), int32(fc.backend),
	} {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.packed {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	return in.Bytes(), nil
}

func decodeMeta(raw []byte) (pre, post []float32, bench time.Duration, err error) {
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
	if len(raw) < 8 || string(raw[:4]) != "WMCO" {
		return nil, nil, bench, fmt.Errorf("metacognition bad magic: %q", string(raw[:min(4, len(raw))]))
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

func runToolMeta(tool string, fc mcCase, benchIters int) (pre, post []float32, dur time.Duration, err error) {
	payload, err := encodeMeta(fc)
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
	return decodeMeta(stdout.Bytes())
}

func runMetalMeta(fc mcCase) (pre, post []float32, err error) {
	tool, err := findMetalFwd()
	if err != nil {
		return nil, nil, err
	}
	pre, post, _, err = runToolMeta(tool, fc, 0)
	return pre, post, err
}

func requireMetaExact(fc mcCase) error {
	goPre, goPost, err := runGoMeta(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	mPre, mPost, err := runMetalMeta(fc)
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

func caseMeta2D() error {
	cfg := metacognition.Config{Dim: 8, SeqLen: 0}
	return requireMetaExact(mcCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch: 2, seq: 1,
		packed: randInit(1, cfg.Dim*cfg.Dim, 0x4d455401),
		x:      randInit(2, 8, 0x4d455402),
	})
}

func caseMeta3D() error {
	cfg := metacognition.Config{Dim: 6, SeqLen: 4}
	return requireMetaExact(mcCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch: 2, seq: 4,
		packed: randInit(1, cfg.Dim*cfg.Dim, 0x4d455403),
		x:      randInit(2*4, 6, 0x4d455404),
	})
}

func caseMetaAllDTypes() error {
	// Mid GEMV so Metal soft-double Observed can win; host rules are exact.
	const batch, seq, benchIters = 4, 8, 80
	cfg := metacognition.Config{Dim: 64, SeqLen: seq}
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

	packed := randInit(1, cfg.Dim*cfg.Dim, 0xd7a4e0e1)
	x := randInit(batch*seq, cfg.Dim, 0xd7a4e0e2)

	fmt.Printf("\n      Go vs Metal Metacognition FormatNone×34 — exact_bit preferred (≥99%% ok)")
	fmt.Printf("\n      shape B=%d T=%d D=%d  iters=%d", batch, seq, cfg.Dim, benchIters)
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
		fc := mcCase{
			cfg: cfg, dtype: dt, backend: core.BackendCPUTiled,
			batch: batch, seq: seq, packed: packed, x: x,
		}
		goPre, goPost, err := runGoMeta(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		mPre, mPost, err := runMetalMeta(fc)
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
			_, _, e := runGoMeta(fc)
			return e
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}

		cppUS := 0.0
		cppStr := "—"
		if haveCpp {
			_, _, cppDur, err := runToolMeta(cppTool, fc, benchIters)
			if err != nil {
				cppStr = "err"
			} else {
				cppUS = perIterUS(cppDur, benchIters)
				cppStr = FormatUS(cppUS)
			}
		}

		_, _, metalDur, err := runToolMeta(metalTool, fc, benchIters)
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
	fmt.Printf("      overall exact_bits=%.2f%%  (aim≥99%% vs Go; soft-double Observed + host rules)\n", overallExact)
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
	fmt.Printf("      note: Metacognition = Observed Dense soft-double + DefaultStabilityRules")
	if len(fails) > 0 {
		return fmt.Errorf("%d metacognition hard fails: %s", len(fails), strings.Join(fails, " | "))
	}
	if overallExact < 99.0 {
		return fmt.Errorf("overall exact_bits=%.2f%% < 99%%", overallExact)
	}
	return nil
}
