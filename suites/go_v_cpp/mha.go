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
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/quant"
)

func CasesMHA() []Case {
	return []Case{
		{Name: "MHA DecoderCausal Float32 smoke — exact bits", Run: caseMHACausalSmoke},
		{Name: "MHA GQA 4h/2kv Float32 — exact bits", Run: caseMHAGQA},
		{Name: "MHA Encoder bidirectional Float32 — exact bits", Run: caseMHAEncoder},
		{Name: "MHA DecoderCausal FormatNone × all 34 dtypes — spectrum + speed", Run: caseMHAAllDTypes},
	}
}

func RunMHA() error {
	var fails []string
	cs := CasesMHA()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [mha %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_mha", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_mha", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp mha: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func eyeInit(rows, cols int) []float32 {
	w := make([]float32, rows*cols)
	n := rows
	if cols < n {
		n = cols
	}
	for i := 0; i < n; i++ {
		w[i*cols+i] = 1
	}
	return w
}

func randInit(rows, cols int, seed uint32) []float32 {
	w := make([]float32, rows*cols)
	s := seed
	for i := range w {
		s = s*1664525 + 1013904223
		w[i] = float32(int32(s%401)-200) / 200.0
	}
	return w
}

type mhaCase struct {
	batch, seq     int
	cfg            mha.Config
	dtype          core.DType
	backend        core.Backend
	q, k, v, o     []float32
	x              []float32
}

func runGoMHA(fc mhaCase) (pre, post []float32, err error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	l, err := mha.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.q, fc.k, fc.v, fc.o)
	if err != nil {
		return nil, nil, err
	}
	l.Exec.Backend = fc.backend
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	l.Q.Exec.MultiCore = false
	l.K.Exec.MultiCore = false
	l.V.Exec.MultiCore = false
	l.O.Exec.MultiCore = false
	t := core.NewTensor[float32](fc.batch, fc.seq, cfg.DModel)
	copy(t.Data, fc.x)
	preT, postT, err := mha.Forward(l, t)
	if err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), preT.Data...), append([]float32(nil), postT.Data...), nil
}

func encodeMHA(fc mhaCase) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("WVM1"))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.batch))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.seq))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.DModel))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.NumHeads))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.NumKVHeads))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.HeadDim))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.MaxSeqLen))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.Mask))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.Pos))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.dtype))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.backend))
	for _, v := range fc.q {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.k {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.v {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.o {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	return in.Bytes(), nil
}

func decodeMHA(raw []byte) (pre, post []float32, bench time.Duration, err error) {
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
	if len(raw) < 8 || string(raw[:4]) != "WVMO" {
		return nil, nil, bench, fmt.Errorf("cpp mha bad magic")
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

func runCppMHA(fc mhaCase) (pre, post []float32, err error) {
	tool, err := findDenseFwd()
	if err != nil {
		return nil, nil, err
	}
	payload, err := encodeMHA(fc)
	if err != nil {
		return nil, nil, err
	}
	cmd := exec.Command(tool)
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, nil, fmt.Errorf("cpp mha: %v: %s", err, stderr.String())
	}
	pre, post, _, err = decodeMHA(stdout.Bytes())
	return pre, post, err
}

func runCppMHABench(fc mhaCase, iters int) (pre, post []float32, dur time.Duration, err error) {
	tool, err := findDenseFwd()
	if err != nil {
		return nil, nil, 0, err
	}
	payload, err := encodeMHA(fc)
	if err != nil {
		return nil, nil, 0, err
	}
	cmd := exec.Command(tool, fmt.Sprintf("--bench=%d", iters))
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, nil, 0, fmt.Errorf("cpp mha bench: %v: %s", err, stderr.String())
	}
	return decodeMHA(stdout.Bytes())
}

func requireMHAExact(fc mhaCase) error {
	goPre, goPost, err := runGoMHA(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppPre, cppPost, err := runCppMHA(fc)
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

func makeMHACase(cfg mha.Config, batch, seq int, dt core.DType, seed uint32) (mhaCase, error) {
	if err := cfg.Validate(); err != nil {
		return mhaCase{}, err
	}
	qDim, kvDim, d := cfg.QDim(), cfg.KVDim(), cfg.DModel
	fc := mhaCase{
		batch: batch, seq: seq, cfg: cfg, dtype: dt, backend: core.BackendCPUTiled,
		q: randInit(qDim, d, seed),
		k: randInit(kvDim, d, seed+1),
		v: randInit(kvDim, d, seed+2),
		o: randInit(d, qDim, seed+3),
		x: randInit(batch*seq, d, seed+4),
	}
	// flatten x: randInit(batch*seq, d) already length batch*seq*d
	return fc, nil
}

func caseMHACausalSmoke() error {
	cfg := mha.DecoderCausal(32, 4, 4)
	cfg.MaxSeqLen = 16
	fc, err := makeMHACase(cfg, 1, 4, core.DTypeFloat32, 0x4d484101)
	if err != nil {
		return err
	}
	// Use eye for stable smoke too — mix: random is fine for exact_bit aim
	return requireMHAExact(fc)
}

func caseMHAGQA() error {
	cfg := mha.DecoderCausal(32, 4, 2)
	cfg.MaxSeqLen = 16
	fc, err := makeMHACase(cfg, 1, 3, core.DTypeFloat32, 0x47514102)
	if err != nil {
		return err
	}
	return requireMHAExact(fc)
}

func caseMHAEncoder() error {
	cfg := mha.EncoderBidirectional(32, 4)
	cfg.MaxSeqLen = 16
	fc, err := makeMHACase(cfg, 2, 3, core.DTypeFloat32, 0x454e4303)
	if err != nil {
		return err
	}
	return requireMHAExact(fc)
}

func caseMHAAllDTypes() error {
	const batch, seq, benchIters = 1, 4, defaultGoBenchIters
	cfg := mha.DecoderCausal(32, 4, 4)
	cfg.MaxSeqLen = 16
	if err := cfg.Validate(); err != nil {
		return err
	}
	if len(core.AllDTypes) != 34 {
		return fmt.Errorf("expected 34 dtypes, got %d", len(core.AllDTypes))
	}

	fmt.Printf("\n      Go CPU vs C++ CPU — MHA DecoderCausal FormatNone×34\n")
	fmt.Printf("      %-12s %-12s %8s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "spect%", "go_µs", "cpp_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 88))

	var (
		exactN, totalN int
		worstBand      = suites.DetExactBit
		fails          []string
	)

	for _, dt := range core.AllDTypes {
		fc, err := makeMHACase(cfg, batch, seq, dt, 0xd7a4e034)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			fmt.Printf("      %-12s FAIL %v\n", dt, err)
			continue
		}
		goPre, goPost, err := runGoMHA(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		cppPre, cppPost, err := runCppMHA(fc)
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
			_, _, err := runGoMHA(fc)
			return err
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		_, _, cppDur, err := runCppMHABench(fc, benchIters)
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
		return fmt.Errorf("%d mha-dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}
