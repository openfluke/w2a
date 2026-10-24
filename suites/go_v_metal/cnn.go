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
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/quant"
)

func CasesCNN() []Case {
	return []Case{
		{Name: "Locate welvet.metal tool", Run: caseLocateMetal},
		{Name: "Locate welvet.cpp tool (optional speed)", Run: caseLocateCppOptional},
		{Name: "CNN1 Float32 Linear pad — exact bits vs Go", Run: caseCNN1Smoke},
		{Name: "CNN2 Float32 ReLU — exact bits vs Go", Run: caseCNN2Smoke},
		{Name: "CNN3 Float32 Linear — exact bits vs Go", Run: caseCNN3Smoke},
		{Name: "CNN1 FormatNone ×34 — exact_bit + Go/C++/Metal timings", Run: caseCNN1AllDTypes},
		{Name: "CNN2 FormatNone ×34 — exact_bit + Go/C++/Metal timings", Run: caseCNN2AllDTypes},
		{Name: "CNN3 FormatNone ×34 — exact_bit + Go/C++/Metal timings", Run: caseCNN3AllDTypes},
	}
}

func RunCNN() error {
	var fails []string
	cs := CasesCNN()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [cnn %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_metal_cnn", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_metal_cnn", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_metal cnn: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type cnn1Case struct {
	batch          int
	cfg            cnn1.Config
	dtype          core.DType
	backend        core.Backend
	W              []float32
	bias           []float64
	x              []float32
}

type cnn2Case struct {
	batch          int
	cfg            cnn2.Config
	dtype          core.DType
	backend        core.Backend
	W              []float32
	bias           []float64
	x              []float32
}

type cnn3Case struct {
	batch          int
	cfg            cnn3.Config
	dtype          core.DType
	backend        core.Backend
	W              []float32
	bias           []float64
	x              []float32
}

func disableCNN1(l *cnn1.Layer) {
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	l.Proj.Exec.MultiCore = false
	l.Proj.Core.MultiCore = false
}
func disableCNN2(l *cnn2.Layer) {
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	l.Proj.Exec.MultiCore = false
	l.Proj.Core.MultiCore = false
}
func disableCNN3(l *cnn3.Layer) {
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	l.Proj.Exec.MultiCore = false
	l.Proj.Core.MultiCore = false
}

func runGoCNN1(fc cnn1Case) (pre, post []float32, err error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	l, err := cnn1.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.W)
	if err != nil {
		return nil, nil, err
	}
	disableCNN1(l)
	l.Exec.Backend = fc.backend
	l.Proj.Exec.Backend = fc.backend
	if len(fc.bias) > 0 {
		l.Proj.Weights.Bias = append([]float64(nil), fc.bias...)
	}
	t := core.NewTensor[float32](fc.batch, cfg.InChannels, cfg.SeqLen)
	copy(t.Data, fc.x)
	preT, postT, err := cnn1.Forward(l, t)
	if err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), preT.Data...), append([]float32(nil), postT.Data...), nil
}

func runGoCNN2(fc cnn2Case) (pre, post []float32, err error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	l, err := cnn2.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.W)
	if err != nil {
		return nil, nil, err
	}
	disableCNN2(l)
	l.Exec.Backend = fc.backend
	l.Proj.Exec.Backend = fc.backend
	if len(fc.bias) > 0 {
		l.Proj.Weights.Bias = append([]float64(nil), fc.bias...)
	}
	t := core.NewTensor[float32](fc.batch, cfg.InChannels, cfg.Height, cfg.Width)
	copy(t.Data, fc.x)
	preT, postT, err := cnn2.Forward(l, t)
	if err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), preT.Data...), append([]float32(nil), postT.Data...), nil
}

func runGoCNN3(fc cnn3Case) (pre, post []float32, err error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	l, err := cnn3.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.W)
	if err != nil {
		return nil, nil, err
	}
	disableCNN3(l)
	l.Exec.Backend = fc.backend
	l.Proj.Exec.Backend = fc.backend
	if len(fc.bias) > 0 {
		l.Proj.Weights.Bias = append([]float64(nil), fc.bias...)
	}
	t := core.NewTensor[float32](fc.batch, cfg.InChannels, cfg.Depth, cfg.Height, cfg.Width)
	copy(t.Data, fc.x)
	preT, postT, err := cnn3.Forward(l, t)
	if err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), preT.Data...), append([]float32(nil), postT.Data...), nil
}

func writeBias(buf *bytes.Buffer, bias []float64) {
	_ = binary.Write(buf, binary.LittleEndian, uint32(len(bias)))
	for _, v := range bias {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
}

func encodeCNN1(fc cnn1Case) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("WVC1"))
	for _, v := range []int32{int32(fc.batch), int32(cfg.InChannels), int32(cfg.Filters), int32(cfg.SeqLen),
		int32(cfg.Kernel), int32(cfg.Stride), int32(cfg.Padding), int32(cfg.Activation), int32(fc.dtype), int32(fc.backend)} {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.W {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	writeBias(&in, fc.bias)
	for _, v := range fc.x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	return in.Bytes(), nil
}

func encodeCNN2(fc cnn2Case) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("WVC2"))
	for _, v := range []int32{int32(fc.batch), int32(cfg.InChannels), int32(cfg.Filters), int32(cfg.Height),
		int32(cfg.Width), int32(cfg.Kernel), int32(cfg.Stride), int32(cfg.Padding), int32(cfg.Activation),
		int32(fc.dtype), int32(fc.backend)} {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.W {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	writeBias(&in, fc.bias)
	for _, v := range fc.x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	return in.Bytes(), nil
}

func encodeCNN3(fc cnn3Case) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("WVC3"))
	for _, v := range []int32{int32(fc.batch), int32(cfg.InChannels), int32(cfg.Filters), int32(cfg.Depth),
		int32(cfg.Height), int32(cfg.Width), int32(cfg.Kernel), int32(cfg.Stride), int32(cfg.Padding),
		int32(cfg.Activation), int32(fc.dtype), int32(fc.backend)} {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.W {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	writeBias(&in, fc.bias)
	for _, v := range fc.x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	return in.Bytes(), nil
}

func decodeCNN(raw []byte) (pre, post []float32, bench time.Duration, err error) {
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
	if len(raw) < 8 || string(raw[:4]) != "WVCO" {
		return nil, nil, bench, fmt.Errorf("cnn bad magic")
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

func runToolCNN(tool string, payload []byte, benchIters int) (pre, post []float32, dur time.Duration, err error) {
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
	return decodeCNN(stdout.Bytes())
}

func runMetalCNN(payload []byte) (pre, post []float32, err error) {
	tool, err := findMetalFwd()
	if err != nil {
		return nil, nil, err
	}
	pre, post, _, err = runToolCNN(tool, payload, 0)
	return pre, post, err
}

func requireCNNExact(goPre, goPost, cppPre, cppPost []float32) error {
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

func caseCNN1Smoke() error {
	cfg := cnn1.Config{
		InChannels: 2, Filters: 3, SeqLen: 8, Kernel: 3, Stride: 1, Padding: 1,
		Activation: core.ActivationLinear,
	}
	fc := cnn1Case{
		batch: 2, cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		W:    randInit(cfg.Filters, cfg.PatchDim(), 0x434e3101),
		bias: []float64{0.1, -0.2, 0.05},
		x:    randInit(2*cfg.InChannels, cfg.SeqLen, 0x434e3102),
	}
	goPre, goPost, err := runGoCNN1(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeCNN1(fc)
	if err != nil {
		return err
	}
	mPre, mPost, err := runMetalCNN(payload)
	if err != nil {
		return fmt.Errorf("metal: %w", err)
	}
	return requireCNNExact(goPre, goPost, mPre, mPost)
}

func caseCNN2Smoke() error {
	cfg := cnn2.Config{
		InChannels: 2, Filters: 3, Height: 6, Width: 6, Kernel: 3, Stride: 1, Padding: 1,
		Activation: core.ActivationReLU,
	}
	fc := cnn2Case{
		batch: 1, cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		W: randInit(cfg.Filters, cfg.PatchDim(), 0x434e3201),
		x: randInit(cfg.InChannels*cfg.Height, cfg.Width, 0x434e3202),
	}
	goPre, goPost, err := runGoCNN2(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeCNN2(fc)
	if err != nil {
		return err
	}
	mPre, mPost, err := runMetalCNN(payload)
	if err != nil {
		return fmt.Errorf("metal: %w", err)
	}
	return requireCNNExact(goPre, goPost, mPre, mPost)
}

func caseCNN3Smoke() error {
	cfg := cnn3.Config{
		InChannels: 1, Filters: 2, Depth: 4, Height: 4, Width: 4, Kernel: 2, Stride: 1, Padding: 0,
		Activation: core.ActivationLinear,
	}
	fc := cnn3Case{
		batch: 1, cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		W:    randInit(cfg.Filters, cfg.PatchDim(), 0x434e3301),
		bias: []float64{0.25, -0.1},
		x:    randInit(cfg.InChannels*cfg.Depth*cfg.Height, cfg.Width, 0x434e3302),
	}
	goPre, goPost, err := runGoCNN3(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeCNN3(fc)
	if err != nil {
		return err
	}
	mPre, mPost, err := runMetalCNN(payload)
	if err != nil {
		return fmt.Errorf("metal: %w", err)
	}
	return requireCNNExact(goPre, goPost, mPre, mPost)
}


func cnnAllDTypesReport(name string, exactN, totalN int, winCount map[string]int, soft, fails []string) error {
	overallExact := 0.0
	if totalN > 0 {
		overallExact = 100.0 * float64(exactN) / float64(totalN)
	}
	fmt.Printf("      %s\n", strings.Repeat("-", 100))
	fmt.Printf("      overall exact_bits=%.2f%%  (aim≥99%% vs Go)\n", overallExact)
	parts := []string{}
	for _, n := range []string{"Go", "C++", "Metal"} {
		if winCount[n] > 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", n, winCount[n]))
		}
	}
	fmt.Printf("      wins: %s\n", strings.Join(parts, " "))
	if len(soft) > 0 {
		fmt.Printf("      soft: %s\n", strings.Join(soft, " | "))
	}
	fmt.Printf("      note: %s im2col host + Metal soft-double Dense", name)
	if len(fails) > 0 {
		return fmt.Errorf("%d %s hard fails: %s", len(fails), name, strings.Join(fails, " | "))
	}
	if overallExact < 99.0 {
		return fmt.Errorf("%s overall exact_bits=%.2f%% < 99%%", name, overallExact)
	}
	return nil
}

func caseCNN1AllDTypes() error {
	const batch, benchIters = 2, 80
	cfg := cnn1.Config{
		InChannels: 4, Filters: 8, SeqLen: 48, Kernel: 5, Stride: 1, Padding: 2,
		Activation: core.ActivationLinear,
	}
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
	fmt.Printf("\n      Go vs Metal CNN1 FormatNone×34 — exact_bit preferred (≥99%% ok)")
	fmt.Printf("\n      shape B=%d inC=%d F=%d seq=%d k=%d  iters=%d", batch, cfg.InChannels, cfg.Filters, cfg.SeqLen, cfg.Kernel, benchIters)
	if haveCpp {
		fmt.Printf("\n      speed: Go / C++ / Metal")
	}
	fmt.Printf("\n      %-12s %-12s %7s %9s %9s %9s %8s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "mtl_µs", "winner", "vsGo")
	fmt.Printf("      %s\n", strings.Repeat("-", 100))
	var fails, soft []string
	var exactN, totalN int
	winCount := map[string]int{}
	W := randInit(cfg.Filters, cfg.PatchDim(), 0xd7a4e041)
	x := randInit(batch*cfg.InChannels, cfg.SeqLen, 0xd7a4e042)
	for _, dt := range core.AllDTypes {
		fc := cnn1Case{batch: batch, cfg: cfg, dtype: dt, backend: core.BackendCPUTiled, W: W, x: x}
		goPre, goPost, err := runGoCNN1(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		payload, err := encodeCNN1(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s enc: %v", dt, err))
			continue
		}
		mPre, mPost, err := runMetalCNN(payload)
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
		goUS, err := benchGoUS(func() error { _, _, e := runGoCNN1(fc); return e }, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		cppUS := 0.0
		cppStr := "—"
		if haveCpp {
			_, _, cppDur, err := runToolCNN(cppTool, payload, benchIters)
			if err != nil {
				cppStr = "err"
			} else {
				cppUS = perIterUS(cppDur, benchIters)
				cppStr = FormatUS(cppUS)
			}
		}
		_, _, metalDur, err := runToolCNN(metalTool, payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s metal bench: %v", dt, err))
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
	return cnnAllDTypesReport("CNN1", exactN, totalN, winCount, soft, fails)
}

func caseCNN2AllDTypes() error {
	const batch, benchIters = 2, 80
	cfg := cnn2.Config{
		InChannels: 4, Filters: 8, Height: 16, Width: 16, Kernel: 3, Stride: 1, Padding: 1,
		Activation: core.ActivationLinear,
	}
	if err := cfg.Validate(); err != nil {
		return err
	}
	metalTool, err := findMetalFwd()
	if err != nil {
		return err
	}
	cppTool, cppErr := findCppFwd()
	haveCpp := cppErr == nil
	fmt.Printf("\n      Go vs Metal CNN2 FormatNone×34 — exact_bit preferred (≥99%% ok)")
	fmt.Printf("\n      shape B=%d inC=%d F=%d %dx%d k=%d  iters=%d", batch, cfg.InChannels, cfg.Filters, cfg.Height, cfg.Width, cfg.Kernel, benchIters)
	if haveCpp {
		fmt.Printf("\n      speed: Go / C++ / Metal")
	}
	fmt.Printf("\n      %-12s %-12s %7s %9s %9s %9s %8s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "mtl_µs", "winner", "vsGo")
	fmt.Printf("      %s\n", strings.Repeat("-", 100))
	var fails, soft []string
	var exactN, totalN int
	winCount := map[string]int{}
	W := randInit(cfg.Filters, cfg.PatchDim(), 0xd7a4e043)
	x := randInit(batch*cfg.InChannels*cfg.Height, cfg.Width, 0xd7a4e044)
	for _, dt := range core.AllDTypes {
		fc := cnn2Case{batch: batch, cfg: cfg, dtype: dt, backend: core.BackendCPUTiled, W: W, x: x}
		goPre, goPost, err := runGoCNN2(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		payload, err := encodeCNN2(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s enc: %v", dt, err))
			continue
		}
		mPre, mPost, err := runMetalCNN(payload)
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
		goUS, err := benchGoUS(func() error { _, _, e := runGoCNN2(fc); return e }, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		cppUS := 0.0
		cppStr := "—"
		if haveCpp {
			_, _, cppDur, err := runToolCNN(cppTool, payload, benchIters)
			if err != nil {
				cppStr = "err"
			} else {
				cppUS = perIterUS(cppDur, benchIters)
				cppStr = FormatUS(cppUS)
			}
		}
		_, _, metalDur, err := runToolCNN(metalTool, payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s metal bench: %v", dt, err))
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
	return cnnAllDTypesReport("CNN2", exactN, totalN, winCount, soft, fails)
}

func caseCNN3AllDTypes() error {
	const batch, benchIters = 1, 80
	cfg := cnn3.Config{
		InChannels: 2, Filters: 4, Depth: 8, Height: 8, Width: 8, Kernel: 3, Stride: 1, Padding: 1,
		Activation: core.ActivationLinear,
	}
	if err := cfg.Validate(); err != nil {
		return err
	}
	metalTool, err := findMetalFwd()
	if err != nil {
		return err
	}
	cppTool, cppErr := findCppFwd()
	haveCpp := cppErr == nil
	fmt.Printf("\n      Go vs Metal CNN3 FormatNone×34 — exact_bit preferred (≥99%% ok)")
	fmt.Printf("\n      shape B=%d inC=%d F=%d %dx%dx%d k=%d  iters=%d", batch, cfg.InChannels, cfg.Filters, cfg.Depth, cfg.Height, cfg.Width, cfg.Kernel, benchIters)
	if haveCpp {
		fmt.Printf("\n      speed: Go / C++ / Metal")
	}
	fmt.Printf("\n      %-12s %-12s %7s %9s %9s %9s %8s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "mtl_µs", "winner", "vsGo")
	fmt.Printf("      %s\n", strings.Repeat("-", 100))
	var fails, soft []string
	var exactN, totalN int
	winCount := map[string]int{}
	W := randInit(cfg.Filters, cfg.PatchDim(), 0xd7a4e045)
	x := randInit(batch*cfg.InChannels*cfg.Depth*cfg.Height, cfg.Width, 0xd7a4e046)
	for _, dt := range core.AllDTypes {
		fc := cnn3Case{batch: batch, cfg: cfg, dtype: dt, backend: core.BackendCPUTiled, W: W, x: x}
		goPre, goPost, err := runGoCNN3(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		payload, err := encodeCNN3(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s enc: %v", dt, err))
			continue
		}
		mPre, mPost, err := runMetalCNN(payload)
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
		goUS, err := benchGoUS(func() error { _, _, e := runGoCNN3(fc); return e }, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		cppUS := 0.0
		cppStr := "—"
		if haveCpp {
			_, _, cppDur, err := runToolCNN(cppTool, payload, benchIters)
			if err != nil {
				cppStr = "err"
			} else {
				cppUS = perIterUS(cppDur, benchIters)
				cppStr = FormatUS(cppUS)
			}
		}
		_, _, metalDur, err := runToolCNN(metalTool, payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s metal bench: %v", dt, err))
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
	return cnnAllDTypesReport("CNN3", exactN, totalN, winCount, soft, fails)
}
