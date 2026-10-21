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
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/quant"
)

func CasesRNN() []Case {
	return []Case{
		{Name: "RNN Float32 — exact bits", Run: caseRNNSmoke},
		{Name: "RNN FormatNone × all 34 dtypes — spectrum + speed", Run: caseRNNAllDTypes},
	}
}

func RunRNN() error {
	var fails []string
	cs := CasesRNN()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [rnn %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_rnn", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_rnn", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp rnn: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type rnnCase struct {
	batch   int
	cfg     rnn.Config
	dtype   core.DType
	backend core.Backend
	packed  []float32
	x       []float32
}

func runGoRNN(fc rnnCase) (pre, post []float32, err error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	l, err := rnn.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.packed)
	if err != nil {
		return nil, nil, err
	}
	l.Exec.Backend = fc.backend
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	if l.IH != nil {
		l.IH.Exec.MultiCore = false
		l.IH.Core.MultiCore = false
		l.IH.Exec.Backend = fc.backend
	}
	if l.HH != nil {
		l.HH.Exec.MultiCore = false
		l.HH.Core.MultiCore = false
		l.HH.Exec.Backend = fc.backend
	}
	t := core.NewTensor[float32](fc.batch, cfg.SeqLen, cfg.InputSize)
	copy(t.Data, fc.x)
	preT, postT, err := rnn.Forward(l, t)
	if err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), preT.Data...), append([]float32(nil), postT.Data...), nil
}

func encodeRNN(fc rnnCase) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("WRN1"))
	for _, v := range []int32{
		int32(fc.batch), int32(cfg.InputSize), int32(cfg.HiddenSize), int32(cfg.SeqLen),
		int32(fc.dtype), int32(fc.backend),
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

func decodeRNN(raw []byte) (pre, post []float32, bench time.Duration, err error) {
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
	if len(raw) < 8 || string(raw[:4]) != "WRNO" {
		return nil, nil, bench, fmt.Errorf("cpp rnn bad magic")
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

func runCppRNN(payload []byte, benchIters int) (pre, post []float32, dur time.Duration, err error) {
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
		return nil, nil, 0, fmt.Errorf("cpp rnn: %v: %s", err, stderr.String())
	}
	return decodeRNN(stdout.Bytes())
}

func requireRNNExact(goPre, goPost, cppPre, cppPost []float32) error {
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

func caseRNNSmoke() error {
	cfg := rnn.Config{InputSize: 4, HiddenSize: 3, SeqLen: 5}
	fc := rnnCase{
		batch: 2, cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		packed: randInit(1, cfg.WeightCount(), 0x52524e01),
		x:      randInit(2*cfg.SeqLen, cfg.InputSize, 0x52524e02),
	}
	goPre, goPost, err := runGoRNN(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeRNN(fc)
	if err != nil {
		return err
	}
	cppPre, cppPost, _, err := runCppRNN(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return requireRNNExact(goPre, goPost, cppPre, cppPost)
}

func caseRNNAllDTypes() error {
	const batch, benchIters = 1, defaultGoBenchIters
	cfg := rnn.Config{InputSize: 4, HiddenSize: 3, SeqLen: 6}
	if err := cfg.Validate(); err != nil {
		return err
	}
	fmt.Printf("\n      Go CPU vs C++ CPU — RNN FormatNone×34\n")
	fmt.Printf("      %-12s %-12s %8s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "spect%", "go_µs", "cpp_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 88))
	var exactN, totalN int
	worstBand := suites.DetExactBit
	var fails []string
	for _, dt := range core.AllDTypes {
		fc := rnnCase{
			batch: batch, cfg: cfg, dtype: dt, backend: core.BackendCPUTiled,
			packed: randInit(1, cfg.WeightCount(), 0xd7a4e061),
			x:      randInit(batch*cfg.SeqLen, cfg.InputSize, 0xd7a4e062),
		}
		goPre, goPost, err := runGoRNN(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		payload, err := encodeRNN(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s enc: %v", dt, err))
			continue
		}
		cppPre, cppPost, _, err := runCppRNN(payload, 0)
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
			_, _, err := runGoRNN(fc)
			return err
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		_, _, cppDur, err := runCppRNN(payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp bench: %v", dt, err))
			fmt.Printf("      %-12s %-12s %7.2f%% %7.0f%% %10s %10s %9s %s\n",
				dt, m.Band, m.ExactPct, m.Band.Percent(), FormatUS(goUS), "?", "?", fmt.Sprintf("%.3g", m.MaxAbs))
			continue
		}
		cppUS := perIterUS(cppDur, benchIters)
		spdPct := suites.SpeedPctDiff(goUS, cppUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %7.0f%% %10s %10s %+8.1f%% %s\n",
			dt, m.Band, m.ExactPct, m.Band.Percent(), FormatUS(goUS), FormatUS(cppUS), spdPct, fmt.Sprintf("%.3g", m.MaxAbs))
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
		return fmt.Errorf("%d rnn-dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}
