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
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
)

func CasesSwiGLU() []Case {
	return []Case{
		{Name: "SwiGLU Float32 [batch,in] smoke — exact bits", Run: caseSwiGLUSmoke2D},
		{Name: "SwiGLU Float32 [batch,seq,in] — exact bits", Run: caseSwiGLUSeq3D},
		{Name: "SwiGLU DefaultFFN Float32 — exact bits", Run: caseSwiGLUDefaultFFN},
		{Name: "SwiGLU FormatNone × all 34 dtypes — spectrum + speed", Run: caseSwiGLUAllDTypes},
	}
}

func RunSwiGLU() error {
	var fails []string
	cs := CasesSwiGLU()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [swiglu %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_swiglu", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_swiglu", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp swiglu: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type swigluCase struct {
	batch, seq int
	cfg        swiglu.Config
	dtype      core.DType
	backend    core.Backend
	gate, up, down []float32
	x          []float32
}

func runGoSwiGLU(fc swigluCase) (pre, post []float32, err error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	l, err := swiglu.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.gate, fc.up, fc.down)
	if err != nil {
		return nil, nil, err
	}
	l.Exec.Backend = fc.backend
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	l.Gate.Exec.MultiCore = false
	l.Up.Exec.MultiCore = false
	l.Down.Exec.MultiCore = false

	var t *core.Tensor[float32]
	if fc.seq == 1 {
		t = core.NewTensor[float32](fc.batch, cfg.InputDim)
	} else {
		t = core.NewTensor[float32](fc.batch, fc.seq, cfg.InputDim)
	}
	copy(t.Data, fc.x)
	preT, postT, err := swiglu.Forward(l, t)
	if err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), preT.Data...), append([]float32(nil), postT.Data...), nil
}

func encodeSwiGLU(fc swigluCase) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("WVS1"))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.batch))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.seq))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.InputDim))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.IntermediateDim))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.dtype))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.backend))
	for _, v := range fc.gate {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.up {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.down {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	return in.Bytes(), nil
}

func decodeSwiGLU(raw []byte) (pre, post []float32, bench time.Duration, err error) {
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
	if len(raw) < 8 || string(raw[:4]) != "WVSO" {
		return nil, nil, bench, fmt.Errorf("cpp swiglu bad magic")
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

func runCppSwiGLU(fc swigluCase) (pre, post []float32, err error) {
	tool, err := findDenseFwd()
	if err != nil {
		return nil, nil, err
	}
	payload, err := encodeSwiGLU(fc)
	if err != nil {
		return nil, nil, err
	}
	cmd := exec.Command(tool)
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, nil, fmt.Errorf("cpp swiglu: %v: %s", err, stderr.String())
	}
	pre, post, _, err = decodeSwiGLU(stdout.Bytes())
	return pre, post, err
}

func runCppSwiGLUBench(fc swigluCase, iters int) (pre, post []float32, dur time.Duration, err error) {
	tool, err := findDenseFwd()
	if err != nil {
		return nil, nil, 0, err
	}
	payload, err := encodeSwiGLU(fc)
	if err != nil {
		return nil, nil, 0, err
	}
	cmd := exec.Command(tool, fmt.Sprintf("--bench=%d", iters))
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, nil, 0, fmt.Errorf("cpp swiglu bench: %v: %s", err, stderr.String())
	}
	return decodeSwiGLU(stdout.Bytes())
}

func requireSwiGLUExact(fc swigluCase) error {
	goPre, goPost, err := runGoSwiGLU(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppPre, cppPost, err := runCppSwiGLU(fc)
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

func makeSwiGLUCase(cfg swiglu.Config, batch, seq int, dt core.DType, seed uint32) (swigluCase, error) {
	if err := cfg.Validate(); err != nil {
		return swigluCase{}, err
	}
	in, inter := cfg.InputDim, cfg.IntermediateDim
	fc := swigluCase{
		batch: batch, seq: seq, cfg: cfg, dtype: dt, backend: core.BackendCPUTiled,
		gate: randInit(inter, in, seed),
		up:   randInit(inter, in, seed+1),
		down: randInit(in, inter, seed+2),
		x:    randInit(batch*seq, in, seed+3),
	}
	return fc, nil
}

func caseSwiGLUSmoke2D() error {
	cfg := swiglu.Config{InputDim: 16, IntermediateDim: 32}
	fc, err := makeSwiGLUCase(cfg, 2, 1, core.DTypeFloat32, 0x53574901)
	if err != nil {
		return err
	}
	return requireSwiGLUExact(fc)
}

func caseSwiGLUSeq3D() error {
	cfg := swiglu.Config{InputDim: 16, IntermediateDim: 24}
	fc, err := makeSwiGLUCase(cfg, 2, 3, core.DTypeFloat32, 0x53574902)
	if err != nil {
		return err
	}
	return requireSwiGLUExact(fc)
}

func caseSwiGLUDefaultFFN() error {
	cfg := swiglu.DefaultFFN(32)
	fc, err := makeSwiGLUCase(cfg, 1, 4, core.DTypeFloat32, 0x53574903)
	if err != nil {
		return err
	}
	return requireSwiGLUExact(fc)
}

func caseSwiGLUAllDTypes() error {
	const batch, seq, benchIters = 1, 2, defaultGoBenchIters
	cfg := swiglu.Config{InputDim: 16, IntermediateDim: 32}
	if err := cfg.Validate(); err != nil {
		return err
	}
	if len(core.AllDTypes) != 34 {
		return fmt.Errorf("expected 34 dtypes, got %d", len(core.AllDTypes))
	}

	fmt.Printf("\n      Go CPU vs C++ CPU — SwiGLU FormatNone×34\n")
	fmt.Printf("      %-12s %-12s %8s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "spect%", "go_µs", "cpp_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 88))

	var (
		exactN, totalN int
		worstBand      = suites.DetExactBit
		fails          []string
	)

	for _, dt := range core.AllDTypes {
		fc, err := makeSwiGLUCase(cfg, batch, seq, dt, 0xd7a4e035)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			fmt.Printf("      %-12s FAIL %v\n", dt, err)
			continue
		}
		goPre, goPost, err := runGoSwiGLU(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		cppPre, cppPost, err := runCppSwiGLU(fc)
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

		goUS, err := benchGoUS(func() error {
			_, _, err := runGoSwiGLU(fc)
			return err
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		_, _, cppDur, err := runCppSwiGLUBench(fc, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp bench: %v", dt, err))
			fmt.Printf("      %-12s %-12s %7.2f%% %7.0f%% %10s %10s %9s %s\n",
				dt, m.Band, m.ExactPct, m.Band.Percent(), FormatUS(goUS), "?", "?",
				fmt.Sprintf("%.3g", m.MaxAbs))
			continue
		}
		cppUS := perIterUS(cppDur, benchIters)
		spdPct := suites.SpeedPctDiff(goUS, cppUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %7.0f%% %10s %10s %+8.1f%% %s\n",
			dt, m.Band, m.ExactPct, m.Band.Percent(), FormatUS(goUS), FormatUS(cppUS), spdPct,
			fmt.Sprintf("%.3g", m.MaxAbs))
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
		return fmt.Errorf("%d swiglu-dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}
