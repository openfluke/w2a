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
	"github.com/openfluke/welvet/layers/mamba"
	"github.com/openfluke/welvet/quant"
)

func CasesMamba() []Case {
	return []Case{
		{Name: "Mamba Float32 [B,T,D] — exact bits", Run: caseMambaF32},
		{Name: "Mamba Float32 alt shape — exact bits", Run: caseMambaF32Alt},
		{Name: "Mamba × all 34 dtypes — spectrum + speed", Run: caseMambaAllDTypes},
	}
}

func RunMamba() error {
	var fails []string
	cs := CasesMamba()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [mb %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_mamba", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_mamba", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp mamba: %d failed: %s", len(fails), strings.Join(fails, ", "))
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
		return nil, nil, bench, fmt.Errorf("cpp mamba bad magic")
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

func runCppMamba(payload []byte, benchIters int) (pre, post []float32, dur time.Duration, err error) {
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
		return nil, nil, 0, fmt.Errorf("cpp mamba: %v: %s", err, stderr.String())
	}
	return decodeMamba(stdout.Bytes())
}

func requireMambaExact(goPre, goPost, cppPre, cppPost []float32) error {
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

func runMambaExact(fc mbCase) error {
	goPre, goPost, err := runGoMamba(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeMamba(fc)
	if err != nil {
		return err
	}
	cppPre, cppPost, _, err := runCppMamba(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return requireMambaExact(goPre, goPost, cppPre, cppPost)
}

func caseMambaF32() error {
	cfg := mamba.Config{DModel: 8, DState: 4, Expand: 2, SeqLen: 3}
	fc := mbCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch:  2,
		packed: randInit(1, mambaWeightCount(cfg), 0x4d414d01),
		x:      randInit(2*3, 8, 0x4d414d02),
	}
	return runMambaExact(fc)
}

func caseMambaF32Alt() error {
	cfg := mamba.Config{DModel: 6, DState: 4, Expand: 2, SeqLen: 4}
	fc := mbCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch:  2,
		packed: randInit(1, mambaWeightCount(cfg), 0x4d414d03),
		x:      randInit(2*4, 6, 0x4d414d04),
	}
	return runMambaExact(fc)
}

func caseMambaAllDTypes() error {
	const batch, benchIters = 2, 20
	cfg := mamba.Config{DModel: 8, DState: 4, Expand: 2, SeqLen: 3}
	if err := cfg.Validate(); err != nil {
		return err
	}
	packed := randInit(1, mambaWeightCount(cfg), 0xd7a4e0d1)
	x := randInit(batch*cfg.SeqLen, cfg.DModel, 0xd7a4e0d2)
	fmt.Printf("\n      Go CPU vs C++ CPU — Mamba×34\n")
	fmt.Printf("      %-12s %-12s %8s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "spect%", "go_µs", "cpp_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 88))
	var exactN, totalN int
	worstBand := suites.DetExactBit
	var fails []string
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
		payload, err := encodeMamba(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s enc: %v", dt, err))
			continue
		}
		cppPre, cppPost, _, err := runCppMamba(payload, 0)
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
			if _, _, err := runGoMamba(fc); err != nil {
				break
			}
		}
		t0 := time.Now()
		for i := 0; i < benchIters; i++ {
			if _, _, err := runGoMamba(fc); err != nil {
				fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
				break
			}
		}
		goDur := time.Since(t0)
		_, _, cppDur, err := runCppMamba(payload, benchIters)
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
	fmt.Printf("      spd%%Δ = (cpp−go)/go×100  (negative ⇒ C++ faster)")
	if len(fails) > 0 {
		return fmt.Errorf("%d mamba-dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}
