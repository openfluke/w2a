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
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/quant"
)

func CasesLayerNorm() []Case {
	return []Case{
		{Name: "LayerNorm Float32 [batch,dim] γ=1 β=0 — exact bits", Run: caseLayerNormOnes2D},
		{Name: "LayerNorm Float32 [batch,seq,dim] random affine — exact bits", Run: caseLayerNormSeq3D},
		{Name: "LayerNorm Float32 custom eps — exact bits", Run: caseLayerNormCustomEps},
		{Name: "LayerNorm FormatNone × all 34 dtypes — spectrum + speed", Run: caseLayerNormAllDTypes},
	}
}

func RunLayerNorm() error {
	var fails []string
	cs := CasesLayerNorm()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [layernorm %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_layernorm", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_layernorm", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp layernorm: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type layernormCase struct {
	batch, seq     int
	cfg            layernorm.Config
	dtype          core.DType
	backend        core.Backend
	gamma, beta, x []float32
}

func runGoLayerNorm(fc layernormCase) (pre, post []float32, err error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	l, err := layernorm.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.gamma, fc.beta)
	if err != nil {
		return nil, nil, err
	}
	l.Exec.Backend = fc.backend
	l.Exec.MultiCore = false
	l.Core.MultiCore = false

	var t *core.Tensor[float32]
	if fc.seq == 1 {
		t = core.NewTensor[float32](fc.batch, cfg.Dim)
	} else {
		t = core.NewTensor[float32](fc.batch, fc.seq, cfg.Dim)
	}
	copy(t.Data, fc.x)
	preT, postT, err := layernorm.Forward(l, t)
	if err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), preT.Data...), append([]float32(nil), postT.Data...), nil
}

func encodeLayerNorm(fc layernormCase) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("WVL1"))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.batch))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.seq))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.Dim))
	_ = binary.Write(&in, binary.LittleEndian, cfg.Eps)
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.dtype))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.backend))
	for _, v := range fc.gamma {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.beta {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	return in.Bytes(), nil
}

func decodeLayerNorm(raw []byte) (pre, post []float32, bench time.Duration, err error) {
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
	if len(raw) < 8 || string(raw[:4]) != "WVLO" {
		return nil, nil, bench, fmt.Errorf("cpp layernorm bad magic")
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

func runCppLayerNorm(fc layernormCase) (pre, post []float32, err error) {
	tool, err := findDenseFwd()
	if err != nil {
		return nil, nil, err
	}
	payload, err := encodeLayerNorm(fc)
	if err != nil {
		return nil, nil, err
	}
	cmd := exec.Command(tool)
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, nil, fmt.Errorf("cpp layernorm: %v: %s", err, stderr.String())
	}
	pre, post, _, err = decodeLayerNorm(stdout.Bytes())
	return pre, post, err
}

func runCppLayerNormBench(fc layernormCase, iters int) (pre, post []float32, dur time.Duration, err error) {
	tool, err := findDenseFwd()
	if err != nil {
		return nil, nil, 0, err
	}
	payload, err := encodeLayerNorm(fc)
	if err != nil {
		return nil, nil, 0, err
	}
	cmd := exec.Command(tool, fmt.Sprintf("--bench=%d", iters))
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, nil, 0, fmt.Errorf("cpp layernorm bench: %v: %s", err, stderr.String())
	}
	return decodeLayerNorm(stdout.Bytes())
}

func requireLayerNormExact(fc layernormCase) error {
	goPre, goPost, err := runGoLayerNorm(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppPre, cppPost, err := runCppLayerNorm(fc)
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

func zerosBeta(dim int) []float32 {
	return make([]float32, dim)
}

func makeLayerNormCase(cfg layernorm.Config, batch, seq int, dt core.DType, seed uint32, identity bool) (layernormCase, error) {
	if err := cfg.Validate(); err != nil {
		return layernormCase{}, err
	}
	fc := layernormCase{
		batch: batch, seq: seq, cfg: cfg, dtype: dt, backend: core.BackendCPUTiled,
		x: randInit(batch*seq, cfg.Dim, seed+2),
	}
	if identity {
		fc.gamma = onesGamma(cfg.Dim)
		fc.beta = zerosBeta(cfg.Dim)
	} else {
		fc.gamma = randInit(1, cfg.Dim, seed)
		for i := range fc.gamma {
			fc.gamma[i] = 0.5 + 0.5*fc.gamma[i]
		}
		fc.beta = randInit(1, cfg.Dim, seed+1)
		for i := range fc.beta {
			fc.beta[i] *= 0.25
		}
	}
	return fc, nil
}

func caseLayerNormOnes2D() error {
	cfg := layernorm.Config{Dim: 32}
	fc, err := makeLayerNormCase(cfg, 2, 1, core.DTypeFloat32, 0x4c4e5201, true)
	if err != nil {
		return err
	}
	return requireLayerNormExact(fc)
}

func caseLayerNormSeq3D() error {
	cfg := layernorm.Config{Dim: 24}
	fc, err := makeLayerNormCase(cfg, 2, 3, core.DTypeFloat32, 0x4c4e5202, false)
	if err != nil {
		return err
	}
	return requireLayerNormExact(fc)
}

func caseLayerNormCustomEps() error {
	cfg := layernorm.Config{Dim: 16, Eps: 1e-6}
	fc, err := makeLayerNormCase(cfg, 1, 4, core.DTypeFloat32, 0x4c4e5203, false)
	if err != nil {
		return err
	}
	return requireLayerNormExact(fc)
}

func caseLayerNormAllDTypes() error {
	const batch, seq, benchIters = 1, 4, 20
	cfg := layernorm.Config{Dim: 32}
	if err := cfg.Validate(); err != nil {
		return err
	}
	if len(core.AllDTypes) != 34 {
		return fmt.Errorf("expected 34 dtypes, got %d", len(core.AllDTypes))
	}

	fmt.Printf("\n      Go CPU vs C++ CPU — LayerNorm FormatNone×34\n")
	fmt.Printf("      %-12s %-12s %8s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "spect%", "go_µs", "cpp_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 88))

	var (
		exactN, totalN int
		worstBand      = suites.DetExactBit
		fails          []string
	)

	for _, dt := range core.AllDTypes {
		fc, err := makeLayerNormCase(cfg, batch, seq, dt, 0xd7a4e037, false)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			fmt.Printf("      %-12s FAIL %v\n", dt, err)
			continue
		}
		goPre, goPost, err := runGoLayerNorm(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		cppPre, cppPost, err := runCppLayerNorm(fc)
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
			if _, _, err := runGoLayerNorm(fc); err != nil {
				fails = append(fails, fmt.Sprintf("%s warm: %v", dt, err))
				break
			}
		}
		t0 := time.Now()
		for i := 0; i < benchIters; i++ {
			if _, _, err := runGoLayerNorm(fc); err != nil {
				fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
				break
			}
		}
		goDur := time.Since(t0)
		_, _, cppDur, err := runCppLayerNormBench(fc, benchIters)
		goUS := float64(goDur.Nanoseconds()) / float64(benchIters) / 1000
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp bench: %v", dt, err))
			fmt.Printf("      %-12s %-12s %7.2f%% %7.0f%% %10.2f %10s %9s %s\n",
				dt, m.Band, m.ExactPct, m.Band.Percent(), goUS, "?", "?",
				fmt.Sprintf("%.3g", m.MaxAbs))
			continue
		}
		cppUS := float64(cppDur.Nanoseconds()) / float64(benchIters) / 1000
		spdPct := suites.SpeedPctDiff(goUS, cppUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %7.0f%% %10.2f %10.2f %+8.1f%% %s\n",
			dt, m.Band, m.ExactPct, m.Band.Percent(), goUS, cppUS, spdPct,
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
		return fmt.Errorf("%d layernorm-dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}
