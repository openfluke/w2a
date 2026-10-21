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
	"github.com/openfluke/welvet/layers/convt1"
	"github.com/openfluke/welvet/layers/convt2"
	"github.com/openfluke/welvet/layers/convt3"
	"github.com/openfluke/welvet/quant"
)

func CasesConvT() []Case {
	return []Case{
		{Name: "ConvT1 Float32 Linear pad — exact bits", Run: caseConvT1Smoke},
		{Name: "ConvT2 Float32 ReLU — exact bits", Run: caseConvT2Smoke},
		{Name: "ConvT3 Float32 Linear out_pad — exact bits", Run: caseConvT3Smoke},
		{Name: "ConvT1 FormatNone × all 34 dtypes — spectrum + speed", Run: caseConvT1AllDTypes},
		{Name: "ConvT2 FormatNone × all 34 dtypes — spectrum + speed", Run: caseConvT2AllDTypes},
		{Name: "ConvT3 FormatNone × all 34 dtypes — spectrum + speed", Run: caseConvT3AllDTypes},
	}
}

func RunConvT() error {
	var fails []string
	cs := CasesConvT()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [convt %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_convt", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_convt", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp convt: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func disableConvT1(l *convt1.Layer) {
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	l.Proj.Exec.MultiCore = false
	l.Proj.Core.MultiCore = false
}

func disableConvT2(l *convt2.Layer) {
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	l.Proj.Exec.MultiCore = false
	l.Proj.Core.MultiCore = false
}

func disableConvT3(l *convt3.Layer) {
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	l.Proj.Exec.MultiCore = false
	l.Proj.Core.MultiCore = false
}

type convt1Case struct {
	batch   int
	cfg     convt1.Config
	dtype   core.DType
	backend core.Backend
	W, x    []float32
}

type convt2Case struct {
	batch   int
	cfg     convt2.Config
	dtype   core.DType
	backend core.Backend
	W, x    []float32
}

type convt3Case struct {
	batch   int
	cfg     convt3.Config
	dtype   core.DType
	backend core.Backend
	W, x    []float32
}

func runGoConvT1(fc convt1Case) (pre, post []float32, err error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	l, err := convt1.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.W)
	if err != nil {
		return nil, nil, err
	}
	l.Exec.Backend = fc.backend
	disableConvT1(l)
	t := core.NewTensor[float32](fc.batch, cfg.InChannels, cfg.SeqLen)
	copy(t.Data, fc.x)
	preT, postT, err := convt1.Forward(l, t)
	if err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), preT.Data...), append([]float32(nil), postT.Data...), nil
}

func runGoConvT2(fc convt2Case) (pre, post []float32, err error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	l, err := convt2.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.W)
	if err != nil {
		return nil, nil, err
	}
	l.Exec.Backend = fc.backend
	disableConvT2(l)
	t := core.NewTensor[float32](fc.batch, cfg.InChannels, cfg.Height, cfg.Width)
	copy(t.Data, fc.x)
	preT, postT, err := convt2.Forward(l, t)
	if err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), preT.Data...), append([]float32(nil), postT.Data...), nil
}

func runGoConvT3(fc convt3Case) (pre, post []float32, err error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	l, err := convt3.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.W)
	if err != nil {
		return nil, nil, err
	}
	l.Exec.Backend = fc.backend
	disableConvT3(l)
	t := core.NewTensor[float32](fc.batch, cfg.InChannels, cfg.Depth, cfg.Height, cfg.Width)
	copy(t.Data, fc.x)
	preT, postT, err := convt3.Forward(l, t)
	if err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), preT.Data...), append([]float32(nil), postT.Data...), nil
}

func encodeConvT1(fc convt1Case) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("WVT1"))
	for _, v := range []int32{
		int32(fc.batch), int32(cfg.InChannels), int32(cfg.Filters), int32(cfg.SeqLen),
		int32(cfg.Kernel), int32(cfg.Stride), int32(cfg.Padding), int32(cfg.OutputPadding),
		int32(cfg.Activation), int32(fc.dtype), int32(fc.backend),
	} {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.W {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	return in.Bytes(), nil
}

func encodeConvT2(fc convt2Case) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("WVT2"))
	for _, v := range []int32{
		int32(fc.batch), int32(cfg.InChannels), int32(cfg.Filters), int32(cfg.Height),
		int32(cfg.Width), int32(cfg.Kernel), int32(cfg.Stride), int32(cfg.Padding),
		int32(cfg.OutputPadding), int32(cfg.Activation), int32(fc.dtype), int32(fc.backend),
	} {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.W {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	return in.Bytes(), nil
}

func encodeConvT3(fc convt3Case) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("WVT3"))
	for _, v := range []int32{
		int32(fc.batch), int32(cfg.InChannels), int32(cfg.Filters), int32(cfg.Depth),
		int32(cfg.Height), int32(cfg.Width), int32(cfg.Kernel), int32(cfg.Stride),
		int32(cfg.Padding), int32(cfg.OutputPadding), int32(cfg.Activation), int32(fc.dtype),
		int32(fc.backend),
	} {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.W {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	return in.Bytes(), nil
}

func decodeConvT(raw []byte) (pre, post []float32, bench time.Duration, err error) {
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
	if len(raw) < 8 || string(raw[:4]) != "WVTO" {
		return nil, nil, bench, fmt.Errorf("cpp convt bad magic")
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

func runCppConvT(payload []byte, benchIters int) (pre, post []float32, dur time.Duration, err error) {
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
		return nil, nil, 0, fmt.Errorf("cpp convt: %v: %s", err, stderr.String())
	}
	return decodeConvT(stdout.Bytes())
}

func requireConvTExact(goPre, goPost, cppPre, cppPost []float32) error {
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

func caseConvT1Smoke() error {
	cfg := convt1.Config{
		InChannels: 2, Filters: 3, SeqLen: 5, Kernel: 3, Stride: 2, Padding: 1, OutputPadding: 1,
		Activation: core.ActivationLinear,
	}
	fc := convt1Case{
		batch: 2, cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		W: randInit(cfg.Filters, cfg.PatchDim(), 0x43543101),
		x: randInit(2*cfg.InChannels, cfg.SeqLen, 0x43543102),
	}
	goPre, goPost, err := runGoConvT1(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeConvT1(fc)
	if err != nil {
		return err
	}
	cppPre, cppPost, _, err := runCppConvT(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return requireConvTExact(goPre, goPost, cppPre, cppPost)
}

func caseConvT2Smoke() error {
	cfg := convt2.Config{
		InChannels: 2, Filters: 2, Height: 4, Width: 4, Kernel: 3, Stride: 1, Padding: 1,
		Activation: core.ActivationReLU,
	}
	fc := convt2Case{
		batch: 1, cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		W: randInit(cfg.Filters, cfg.PatchDim(), 0x43543201),
		x: randInit(cfg.InChannels*cfg.Height, cfg.Width, 0x43543202),
	}
	goPre, goPost, err := runGoConvT2(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeConvT2(fc)
	if err != nil {
		return err
	}
	cppPre, cppPost, _, err := runCppConvT(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return requireConvTExact(goPre, goPost, cppPre, cppPost)
}

func caseConvT3Smoke() error {
	cfg := convt3.Config{
		InChannels: 1, Filters: 2, Depth: 3, Height: 3, Width: 3, Kernel: 2, Stride: 2, Padding: 0,
		OutputPadding: 1, Activation: core.ActivationLinear,
	}
	fc := convt3Case{
		batch: 1, cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		W: randInit(cfg.Filters, cfg.PatchDim(), 0x43543301),
		x: randInit(cfg.InChannels*cfg.Depth*cfg.Height, cfg.Width, 0x43543302),
	}
	goPre, goPost, err := runGoConvT3(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeConvT3(fc)
	if err != nil {
		return err
	}
	cppPre, cppPost, _, err := runCppConvT(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return requireConvTExact(goPre, goPost, cppPre, cppPost)
}

func caseConvT1AllDTypes() error {
	const batch, benchIters = 1, defaultGoBenchIters
	cfg := convt1.Config{
		InChannels: 2, Filters: 3, SeqLen: 6, Kernel: 3, Stride: 1, Padding: 1,
		Activation: core.ActivationLinear,
	}
	if err := cfg.Validate(); err != nil {
		return err
	}
	fmt.Printf("\n      Go CPU vs C++ CPU — ConvT1 FormatNone×34\n")
	fmt.Printf("      %-12s %-12s %8s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "spect%", "go_µs", "cpp_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 88))
	var exactN, totalN int
	worstBand := suites.DetExactBit
	var fails []string
	for _, dt := range core.AllDTypes {
		fc := convt1Case{
			batch: batch, cfg: cfg, dtype: dt, backend: core.BackendCPUTiled,
			W: randInit(cfg.Filters, cfg.PatchDim(), 0xd7a4e051),
			x: randInit(batch*cfg.InChannels, cfg.SeqLen, 0xd7a4e052),
		}
		goPre, goPost, err := runGoConvT1(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		payload, err := encodeConvT1(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s enc: %v", dt, err))
			continue
		}
		cppPre, cppPost, _, err := runCppConvT(payload, 0)
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
			_, _, err := runGoConvT1(fc)
			return err
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		_, _, cppDur, err := runCppConvT(payload, benchIters)
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
		return fmt.Errorf("%d convt1-dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

func caseConvT2AllDTypes() error {
	const batch, benchIters = 1, defaultGoBenchIters
	cfg := convt2.Config{
		InChannels: 2, Filters: 2, Height: 4, Width: 4, Kernel: 3, Stride: 1, Padding: 0,
		Activation: core.ActivationLinear,
	}
	if err := cfg.Validate(); err != nil {
		return err
	}
	fmt.Printf("\n      Go CPU vs C++ CPU — ConvT2 FormatNone×34\n")
	fmt.Printf("      %-12s %-12s %8s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "spect%", "go_µs", "cpp_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 88))
	var exactN, totalN int
	worstBand := suites.DetExactBit
	var fails []string
	for _, dt := range core.AllDTypes {
		fc := convt2Case{
			batch: batch, cfg: cfg, dtype: dt, backend: core.BackendCPUTiled,
			W: randInit(cfg.Filters, cfg.PatchDim(), 0xd7a4e053),
			x: randInit(cfg.InChannels*cfg.Height, cfg.Width, 0xd7a4e054),
		}
		goPre, goPost, err := runGoConvT2(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		payload, err := encodeConvT2(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s enc: %v", dt, err))
			continue
		}
		cppPre, cppPost, _, err := runCppConvT(payload, 0)
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
			_, _, err := runGoConvT2(fc)
			return err
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		_, _, cppDur, err := runCppConvT(payload, benchIters)
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
		return fmt.Errorf("%d convt2-dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

func caseConvT3AllDTypes() error {
	const batch, benchIters = 1, defaultGoBenchIters
	cfg := convt3.Config{
		InChannels: 1, Filters: 2, Depth: 3, Height: 3, Width: 3, Kernel: 2, Stride: 1, Padding: 0,
		Activation: core.ActivationLinear,
	}
	if err := cfg.Validate(); err != nil {
		return err
	}
	fmt.Printf("\n      Go CPU vs C++ CPU — ConvT3 FormatNone×34\n")
	fmt.Printf("      %-12s %-12s %8s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "spect%", "go_µs", "cpp_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 88))
	var exactN, totalN int
	worstBand := suites.DetExactBit
	var fails []string
	for _, dt := range core.AllDTypes {
		fc := convt3Case{
			batch: batch, cfg: cfg, dtype: dt, backend: core.BackendCPUTiled,
			W: randInit(cfg.Filters, cfg.PatchDim(), 0xd7a4e055),
			x: randInit(cfg.InChannels*cfg.Depth*cfg.Height, cfg.Width, 0xd7a4e056),
		}
		goPre, goPost, err := runGoConvT3(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		payload, err := encodeConvT3(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s enc: %v", dt, err))
			continue
		}
		cppPre, cppPost, _, err := runCppConvT(payload, 0)
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
			_, _, err := runGoConvT3(fc)
			return err
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		_, _, cppDur, err := runCppConvT(payload, benchIters)
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
		return fmt.Errorf("%d convt3-dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}
