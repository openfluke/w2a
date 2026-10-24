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
	"github.com/openfluke/welvet/layers/mamba"
	"github.com/openfluke/welvet/quant"
)

func CasesMamba() []Case {
	return []Case{
		{Name: "Locate welvet.metal tool", Run: caseLocateMetal},
		{Name: "Locate welvet.cpp tool (optional speed)", Run: caseLocateCppOptional},
		{Name: "Mamba Float32 [B,T,D] — exact bits vs Go", Run: caseMambaF32},
		{Name: "Mamba Float32 alt shape — exact bits vs Go", Run: caseMambaF32Alt},
		{Name: "Mamba FormatNone ×34 — ≥99% + Go/C++/Metal timings", Run: caseMambaAllDTypes},
	}
}

func RunMamba() error {
	var fails []string
	cs := CasesMamba()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [mb %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_metal_mamba", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_metal_mamba", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_metal mamba: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func mambaWeightCount(cfg mamba.Config) int {
	e := cfg.Expand
	if e <= 0 {
		e = 2
	}
	inner := e * cfg.DModel
	return inner * (3*cfg.DModel + 2)
}

type mbCase struct {
	cfg     mamba.Config
	dtype   core.DType
	backend core.Backend
	batch   int
	packed  []float32 // In|Out|ALog|DSkip
	x       []float32
}

func splitMambaPacked(cfg mamba.Config, packed []float32) (inInit, outInit []float32, aLog, dSkip []float32, err error) {
	if err := cfg.Validate(); err != nil {
		return nil, nil, nil, nil, err
	}
	inner := cfg.InnerDim()
	inN := 2 * inner * cfg.DModel
	outN := cfg.DModel * inner
	need := inN + outN + 2*inner
	if len(packed) < need {
		return nil, nil, nil, nil, fmt.Errorf("mamba packed short: %d < %d", len(packed), need)
	}
	inInit = packed[:inN]
	outInit = packed[inN : inN+outN]
	aLog = packed[inN+outN : inN+outN+inner]
	dSkip = packed[inN+outN+inner : inN+outN+2*inner]
	return inInit, outInit, aLog, dSkip, nil
}

func runGoMamba(fc mbCase) (pre, post []float32, err error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	inInit, outInit, aLog, dSkip, err := splitMambaPacked(cfg, fc.packed)
	if err != nil {
		return nil, nil, err
	}
	l, err := mamba.NewConfigured(cfg, fc.dtype, quant.FormatNone, inInit, outInit, aLog, dSkip)
	if err != nil {
		return nil, nil, err
	}
	l.Exec.Backend = fc.backend
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	l.InProj.Exec.MultiCore = false
	l.InProj.Core.MultiCore = false
	l.InProj.Exec.Backend = fc.backend
	l.OutProj.Exec.MultiCore = false
	l.OutProj.Core.MultiCore = false
	l.OutProj.Exec.Backend = fc.backend
	t := core.NewTensor[float32](fc.batch, cfg.SeqLen, cfg.DModel)
	copy(t.Data, fc.x)
	preT, postT, err := mamba.Forward(l, t)
	if err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), preT.Data...), append([]float32(nil), postT.Data...), nil
}

func encodeMamba(fc mbCase) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("WMB1"))
	for _, v := range []int32{
		int32(fc.batch), int32(cfg.SeqLen), int32(cfg.DModel), int32(cfg.DState),
		int32(cfg.Expand), int32(fc.dtype), int32(fc.backend),
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

func decodeMamba(raw []byte) (pre, post []float32, bench time.Duration, err error) {
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
	if len(raw) < 8 || string(raw[:4]) != "WMBO" {
		return nil, nil, bench, fmt.Errorf("mamba bad magic: %q", string(raw[:min(4, len(raw))]))
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

func runToolMamba(tool string, fc mbCase, benchIters int) (pre, post []float32, dur time.Duration, err error) {
	payload, err := encodeMamba(fc)
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
	return decodeMamba(stdout.Bytes())
}

func runMetalMamba(fc mbCase) (pre, post []float32, err error) {
	tool, err := findMetalFwd()
	if err != nil {
		return nil, nil, err
	}
	pre, post, _, err = runToolMamba(tool, fc, 0)
	return pre, post, err
}

func requireMambaExact(fc mbCase) error {
	goPre, goPost, err := runGoMamba(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	mPre, mPost, err := runMetalMamba(fc)
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

func caseMambaF32() error {
	cfg := mamba.Config{DModel: 8, DState: 4, Expand: 2, SeqLen: 3}
	return requireMambaExact(mbCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch:  2,
		packed: randInit(1, mambaWeightCount(cfg), 0x4d414d01),
		x:      randInit(2*3, 8, 0x4d414d02),
	})
}

func caseMambaF32Alt() error {
	cfg := mamba.Config{DModel: 6, DState: 4, Expand: 2, SeqLen: 4}
	return requireMambaExact(mbCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch:  2,
		packed: randInit(1, mambaWeightCount(cfg), 0x4d414d03),
		x:      randInit(2*4, 6, 0x4d414d04),
	})
}

func caseMambaAllDTypes() error {
	// Mid GEMV so Metal soft-double In/Out can win; scan may soft ~99.9%.
	const batch, benchIters = 4, 80
	cfg := mamba.Config{DModel: 64, DState: 16, Expand: 2, SeqLen: 16}
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

	packed := randInit(1, mambaWeightCount(cfg), 0xd7a4e0d1)
	x := randInit(batch*cfg.SeqLen, cfg.DModel, 0xd7a4e0d2)

	fmt.Printf("\n      Go vs Metal Mamba FormatNone×34 — exact_bit preferred (≥99%% ok)")
	fmt.Printf("\n      shape B=%d T=%d D=%d expand=%d inner=%d  iters=%d",
		batch, cfg.SeqLen, cfg.DModel, cfg.Expand, cfg.InnerDim(), benchIters)
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
		fc := mbCase{
			cfg: cfg, dtype: dt, backend: core.BackendCPUTiled,
			batch: batch, packed: packed, x: x,
		}
		goPre, goPost, err := runGoMamba(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		mPre, mPost, err := runMetalMamba(fc)
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
			_, _, e := runGoMamba(fc)
			return e
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}

		cppUS := 0.0
		cppStr := "—"
		if haveCpp {
			_, _, cppDur, err := runToolMamba(cppTool, fc, benchIters)
			if err != nil {
				cppStr = "err"
			} else {
				cppUS = perIterUS(cppDur, benchIters)
				cppStr = FormatUS(cppUS)
			}
		}

		_, _, metalDur, err := runToolMamba(metalTool, fc, benchIters)
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
	fmt.Printf("      overall exact_bits=%.2f%%  (aim≥99%% vs Go; soft-double Dense + f64 scan)\n", overallExact)
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
	fmt.Printf("      note: Mamba = Dense In/Out soft-double + host f64 selective scan")
	if len(fails) > 0 {
		return fmt.Errorf("%d mamba hard fails: %s", len(fails), strings.Join(fails, " | "))
	}
	if overallExact < 99.0 {
		return fmt.Errorf("overall exact_bits=%.2f%% < 99%%", overallExact)
	}
	return nil
}
