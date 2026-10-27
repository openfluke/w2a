package go_v_cuda

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
		{Name: "Locate welvet.cuda tool", Run: caseLocateCUDA},
		{Name: "Locate welvet.cpp tool (optional speed)", Run: caseLocateCppOptional},
		{Name: "ConvT1 Float32 Linear pad — exact bits vs Go", Run: caseConvT1Smoke},
		{Name: "ConvT2 Float32 ReLU — exact bits vs Go", Run: caseConvT2Smoke},
		{Name: "ConvT3 Float32 Linear out_pad — exact bits vs Go", Run: caseConvT3Smoke},
		{Name: "ConvT1 FormatNone ×34 — exact_bit + Go/C++/CUDA timings", Run: caseConvT1AllDTypes},
		{Name: "ConvT2 FormatNone ×34 — exact_bit + Go/C++/CUDA timings", Run: caseConvT2AllDTypes},
		{Name: "ConvT3 FormatNone ×34 — exact_bit + Go/C++/CUDA timings", Run: caseConvT3AllDTypes},
	}
}

func RunConvT() error {
	var fails []string
	cs := CasesConvT()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [convt %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cuda_convt", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cuda_convt", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cuda convt: %d failed: %s", len(fails), strings.Join(fails, ", "))
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
		return nil, nil, bench, fmt.Errorf("convt bad magic")
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

func runToolConvT(tool string, payload []byte, benchIters int) (pre, post []float32, dur time.Duration, err error) {
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
	return decodeConvT(stdout.Bytes())
}

func runCUDAConvT(payload []byte) (pre, post []float32, err error) {
	tool, err := findCUDAFwd()
	if err != nil {
		return nil, nil, err
	}
	pre, post, _, err = runToolConvT(tool, payload, 0)
	return pre, post, err
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
	mPre, mPost, err := runCUDAConvT(payload)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	return requireConvTExact(goPre, goPost, mPre, mPost)
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
	mPre, mPost, err := runCUDAConvT(payload)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	return requireConvTExact(goPre, goPost, mPre, mPost)
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
	mPre, mPost, err := runCUDAConvT(payload)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	return requireConvTExact(goPre, goPost, mPre, mPost)
}


func caseConvT1AllDTypes() error {
	const batch, benchIters = 2, 120
	cfg := convt1.Config{
		InChannels: 4, Filters: 8, SeqLen: 32, Kernel: 5, Stride: 1, Padding: 2,
		Activation: core.ActivationLinear,
	}
	if err := cfg.Validate(); err != nil {
		return err
	}
	if len(core.AllDTypes) != 34 {
		return fmt.Errorf("expected 34 dtypes, got %d", len(core.AllDTypes))
	}
	cudaTool, err := findCUDAFwd()
	if err != nil {
		return err
	}
	cppTool, cppErr := findCppFwd()
	haveCpp := cppErr == nil
	fmt.Printf("\n      Go vs CUDA ConvT1 FormatNone×34 — exact_bit required")
	if haveCpp {
		fmt.Printf("\n      speed: Go / C++ / CUDA (CUDA f64 scatter)")
	}
	fmt.Printf("\n      %-12s %-12s %7s %9s %9s %9s %8s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "cuda_µs", "winner", "vsGo")
	fmt.Printf("      %s\n", strings.Repeat("-", 100))
	var fails []string
	var exactN, totalN int
	winCount := map[string]int{}
	W := randInit(cfg.Filters, cfg.PatchDim(), 0xd7a4e051)
	x := randInit(batch*cfg.InChannels, cfg.SeqLen, 0xd7a4e052)
	for _, dt := range core.AllDTypes {
		fc := convt1Case{batch: batch, cfg: cfg, dtype: dt, backend: core.BackendCPUTiled, W: W, x: x}
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
		mPre, mPost, err := runCUDAConvT(payload)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cuda: %v", dt, err))
			fmt.Printf("      %-12s FAIL cuda %v\n", dt, err)
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
		goUS, err := benchGoUS(func() error { _, _, e := runGoConvT1(fc); return e }, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		cppUS := 0.0
		cppStr := "—"
		if haveCpp {
			_, _, cppDur, err := runToolConvT(cppTool, payload, benchIters)
			if err != nil {
				cppStr = "err"
			} else {
				cppUS = perIterUS(cppDur, benchIters)
				cppStr = FormatUS(cppUS)
			}
		}
		_, _, cudaDur, err := runToolConvT(cudaTool, payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cuda bench: %v", dt, err))
			continue
		}
		cudaUS := perIterUS(cudaDur, benchIters)
		names := []string{"Go", "CUDA"}
		uss := []float64{goUS, cudaUS}
		if haveCpp && cppUS > 0 {
			names = []string{"Go", "C++", "CUDA"}
			uss = []float64{goUS, cppUS, cudaUS}
		}
		winner, _, vsGo := pickFastest(names, uss)
		winCount[winner]++
		vsGoStr := "baseline"
		if winner != "Go" {
			vsGoStr = fmt.Sprintf("%s %+0.1f%%", winner, vsGo)
		}
		fmt.Printf("      %-12s %-12s %6.2f%% %9s %9s %9s %8s %s\n",
			dt, m.Band, m.ExactPct, FormatUS(goUS), cppStr, FormatUS(cudaUS), winner, vsGoStr)
		if m.Band != suites.DetExactBit {
			fails = append(fails, fmt.Sprintf("%s: %s", dt, suites.FormatSpectrum(m)))
		}
	}
	overallExact := 0.0
	if totalN > 0 {
		overallExact = 100.0 * float64(exactN) / float64(totalN)
	}
	fmt.Printf("      %s\n", strings.Repeat("-", 100))
	fmt.Printf("      overall exact_bits=%.2f%%  (aim=100%% vs Go)\n", overallExact)
	parts := []string{}
	for _, n := range []string{"Go", "C++", "CUDA"} {
		if winCount[n] > 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", n, winCount[n]))
		}
	}
	fmt.Printf("      wins: %s\n", strings.Join(parts, " "))
	fmt.Printf("      note: ConvT1 CUDA f64 scatter FormatNone×34")
	if len(fails) > 0 {
		return fmt.Errorf("%d ConvT1-dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

func caseConvT2AllDTypes() error {
	const batch, benchIters = 2, 120
	cfg := convt2.Config{
		InChannels: 4, Filters: 8, Height: 16, Width: 16, Kernel: 3, Stride: 1, Padding: 1,
		Activation: core.ActivationLinear,
	}
	if err := cfg.Validate(); err != nil {
		return err
	}
	if len(core.AllDTypes) != 34 {
		return fmt.Errorf("expected 34 dtypes, got %d", len(core.AllDTypes))
	}
	cudaTool, err := findCUDAFwd()
	if err != nil {
		return err
	}
	cppTool, cppErr := findCppFwd()
	haveCpp := cppErr == nil
	fmt.Printf("\n      Go vs CUDA ConvT2 FormatNone×34 — exact_bit required")
	if haveCpp {
		fmt.Printf("\n      speed: Go / C++ / CUDA (CUDA f64 scatter)")
	}
	fmt.Printf("\n      %-12s %-12s %7s %9s %9s %9s %8s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "cuda_µs", "winner", "vsGo")
	fmt.Printf("      %s\n", strings.Repeat("-", 100))
	var fails []string
	var exactN, totalN int
	winCount := map[string]int{}
	W := randInit(cfg.Filters, cfg.PatchDim(), 0xd7a4e053)
	x := randInit(batch*cfg.InChannels*cfg.Height, cfg.Width, 0xd7a4e054)
	for _, dt := range core.AllDTypes {
		fc := convt2Case{batch: batch, cfg: cfg, dtype: dt, backend: core.BackendCPUTiled, W: W, x: x}
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
		mPre, mPost, err := runCUDAConvT(payload)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cuda: %v", dt, err))
			fmt.Printf("      %-12s FAIL cuda %v\n", dt, err)
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
		goUS, err := benchGoUS(func() error { _, _, e := runGoConvT2(fc); return e }, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		cppUS := 0.0
		cppStr := "—"
		if haveCpp {
			_, _, cppDur, err := runToolConvT(cppTool, payload, benchIters)
			if err != nil {
				cppStr = "err"
			} else {
				cppUS = perIterUS(cppDur, benchIters)
				cppStr = FormatUS(cppUS)
			}
		}
		_, _, cudaDur, err := runToolConvT(cudaTool, payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cuda bench: %v", dt, err))
			continue
		}
		cudaUS := perIterUS(cudaDur, benchIters)
		names := []string{"Go", "CUDA"}
		uss := []float64{goUS, cudaUS}
		if haveCpp && cppUS > 0 {
			names = []string{"Go", "C++", "CUDA"}
			uss = []float64{goUS, cppUS, cudaUS}
		}
		winner, _, vsGo := pickFastest(names, uss)
		winCount[winner]++
		vsGoStr := "baseline"
		if winner != "Go" {
			vsGoStr = fmt.Sprintf("%s %+0.1f%%", winner, vsGo)
		}
		fmt.Printf("      %-12s %-12s %6.2f%% %9s %9s %9s %8s %s\n",
			dt, m.Band, m.ExactPct, FormatUS(goUS), cppStr, FormatUS(cudaUS), winner, vsGoStr)
		if m.Band != suites.DetExactBit {
			fails = append(fails, fmt.Sprintf("%s: %s", dt, suites.FormatSpectrum(m)))
		}
	}
	overallExact := 0.0
	if totalN > 0 {
		overallExact = 100.0 * float64(exactN) / float64(totalN)
	}
	fmt.Printf("      %s\n", strings.Repeat("-", 100))
	fmt.Printf("      overall exact_bits=%.2f%%  (aim=100%% vs Go)\n", overallExact)
	parts := []string{}
	for _, n := range []string{"Go", "C++", "CUDA"} {
		if winCount[n] > 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", n, winCount[n]))
		}
	}
	fmt.Printf("      wins: %s\n", strings.Join(parts, " "))
	fmt.Printf("      note: ConvT2 CUDA f64 scatter FormatNone×34")
	if len(fails) > 0 {
		return fmt.Errorf("%d ConvT2-dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

func caseConvT3AllDTypes() error {
	const batch, benchIters = 2, 120
	cfg := convt3.Config{
		InChannels: 2, Filters: 4, Depth: 8, Height: 8, Width: 8, Kernel: 3, Stride: 1, Padding: 1,
		Activation: core.ActivationLinear,
	}
	if err := cfg.Validate(); err != nil {
		return err
	}
	if len(core.AllDTypes) != 34 {
		return fmt.Errorf("expected 34 dtypes, got %d", len(core.AllDTypes))
	}
	cudaTool, err := findCUDAFwd()
	if err != nil {
		return err
	}
	cppTool, cppErr := findCppFwd()
	haveCpp := cppErr == nil
	fmt.Printf("\n      Go vs CUDA ConvT3 FormatNone×34 — exact_bit required")
	if haveCpp {
		fmt.Printf("\n      speed: Go / C++ / CUDA (CUDA f64 scatter)")
	}
	fmt.Printf("\n      %-12s %-12s %7s %9s %9s %9s %8s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "cuda_µs", "winner", "vsGo")
	fmt.Printf("      %s\n", strings.Repeat("-", 100))
	var fails []string
	var exactN, totalN int
	winCount := map[string]int{}
	W := randInit(cfg.Filters, cfg.PatchDim(), 0xd7a4e055)
	x := randInit(batch*cfg.InChannels*cfg.Depth*cfg.Height, cfg.Width, 0xd7a4e056)
	for _, dt := range core.AllDTypes {
		fc := convt3Case{batch: batch, cfg: cfg, dtype: dt, backend: core.BackendCPUTiled, W: W, x: x}
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
		mPre, mPost, err := runCUDAConvT(payload)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cuda: %v", dt, err))
			fmt.Printf("      %-12s FAIL cuda %v\n", dt, err)
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
		goUS, err := benchGoUS(func() error { _, _, e := runGoConvT3(fc); return e }, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		cppUS := 0.0
		cppStr := "—"
		if haveCpp {
			_, _, cppDur, err := runToolConvT(cppTool, payload, benchIters)
			if err != nil {
				cppStr = "err"
			} else {
				cppUS = perIterUS(cppDur, benchIters)
				cppStr = FormatUS(cppUS)
			}
		}
		_, _, cudaDur, err := runToolConvT(cudaTool, payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cuda bench: %v", dt, err))
			continue
		}
		cudaUS := perIterUS(cudaDur, benchIters)
		names := []string{"Go", "CUDA"}
		uss := []float64{goUS, cudaUS}
		if haveCpp && cppUS > 0 {
			names = []string{"Go", "C++", "CUDA"}
			uss = []float64{goUS, cppUS, cudaUS}
		}
		winner, _, vsGo := pickFastest(names, uss)
		winCount[winner]++
		vsGoStr := "baseline"
		if winner != "Go" {
			vsGoStr = fmt.Sprintf("%s %+0.1f%%", winner, vsGo)
		}
		fmt.Printf("      %-12s %-12s %6.2f%% %9s %9s %9s %8s %s\n",
			dt, m.Band, m.ExactPct, FormatUS(goUS), cppStr, FormatUS(cudaUS), winner, vsGoStr)
		if m.Band != suites.DetExactBit {
			fails = append(fails, fmt.Sprintf("%s: %s", dt, suites.FormatSpectrum(m)))
		}
	}
	overallExact := 0.0
	if totalN > 0 {
		overallExact = 100.0 * float64(exactN) / float64(totalN)
	}
	fmt.Printf("      %s\n", strings.Repeat("-", 100))
	fmt.Printf("      overall exact_bits=%.2f%%  (aim=100%% vs Go)\n", overallExact)
	parts := []string{}
	for _, n := range []string{"Go", "C++", "CUDA"} {
		if winCount[n] > 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", n, winCount[n]))
		}
	}
	fmt.Printf("      wins: %s\n", strings.Join(parts, " "))
	fmt.Printf("      note: ConvT3 CUDA f64 scatter FormatNone×34")
	if len(fails) > 0 {
		return fmt.Errorf("%d ConvT3-dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}
