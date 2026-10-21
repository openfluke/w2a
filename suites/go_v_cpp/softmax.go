package go_v_cpp

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os/exec"
	"strings"
	"time"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/softmax"
)

func CasesSoftmax() []Case {
	return []Case{
		{Name: "Softmax Standard Float32 — exact bits", Run: caseSoftmaxStandard},
		{Name: "Softmax Temperature Float32 — exact bits", Run: caseSoftmaxTemperature},
		{Name: "Softmax Masked Float32 — exact bits", Run: caseSoftmaxMasked},
		{Name: "Softmax Sparse Float32 — exact bits", Run: caseSoftmaxSparse},
		{Name: "Softmax Entmax Float32 — exact bits", Run: caseSoftmaxEntmax},
		{Name: "Softmax Grid Float32 — exact bits", Run: caseSoftmaxGrid},
		{Name: "Softmax Hierarchical Float32 — exact bits", Run: caseSoftmaxHierarchical},
		{Name: "Softmax Gumbel Float32 — smoke (non-deterministic)", Run: caseSoftmaxGumbelSmoke},
		{Name: "Softmax all Kind × all 34 dtypes — spectrum + speed", Run: caseSoftmaxAllKindsAllDTypes},
	}
}

func RunSoftmax() error {
	var fails []string
	cs := CasesSoftmax()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [sm %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_softmax", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_softmax", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp softmax: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type smCase struct {
	cfg     softmax.Config
	dtype   core.DType
	backend core.Backend
	shape   []int
	x       []float32
}

func runGoSoftmax(fc smCase) (pre, post []float32, err error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	l, err := softmax.New(cfg)
	if err != nil {
		return nil, nil, err
	}
	l.Exec.Backend = fc.backend
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	_ = l.SetDType(fc.dtype)
	t := core.NewTensor[float32](fc.shape...)
	copy(t.Data, fc.x)
	preT, postT, err := softmax.Forward(l, t)
	if err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), preT.Data...), append([]float32(nil), postT.Data...), nil
}

func encodeSoftmax(fc smCase) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	n := len(fc.x)
	var in bytes.Buffer
	_, _ = in.Write([]byte("WSM1"))
	_ = binary.Write(&in, binary.LittleEndian, int32(n))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.Dim))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.SeqLen))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.Kind))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.Rows))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.Cols))
	_ = binary.Write(&in, binary.LittleEndian, cfg.Temperature)
	_ = binary.Write(&in, binary.LittleEndian, cfg.EntmaxAlpha)
	_ = binary.Write(&in, binary.LittleEndian, int32(0)) // gumbel_seed
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.dtype))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.backend))
	_ = binary.Write(&in, binary.LittleEndian, int32(len(fc.shape)))
	for _, d := range fc.shape {
		_ = binary.Write(&in, binary.LittleEndian, int32(d))
	}
	_ = binary.Write(&in, binary.LittleEndian, int32(len(cfg.Mask)))
	for _, m := range cfg.Mask {
		var b uint8
		if m {
			b = 1
		}
		_ = binary.Write(&in, binary.LittleEndian, b)
	}
	_ = binary.Write(&in, binary.LittleEndian, int32(len(cfg.HierarchyLevels)))
	for _, h := range cfg.HierarchyLevels {
		_ = binary.Write(&in, binary.LittleEndian, int32(h))
	}
	for _, v := range fc.x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	return in.Bytes(), nil
}

func decodeSoftmax(raw []byte) (pre, post []float32, bench time.Duration, err error) {
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
	if len(raw) < 8 || string(raw[:4]) != "WSMO" {
		return nil, nil, bench, fmt.Errorf("cpp softmax bad magic")
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

func runCppSoftmax(payload []byte, benchIters int) (pre, post []float32, dur time.Duration, err error) {
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
		return nil, nil, 0, fmt.Errorf("cpp softmax: %v: %s", err, stderr.String())
	}
	return decodeSoftmax(stdout.Bytes())
}

func requireSoftmaxExact(goPre, goPost, cppPre, cppPost []float32) error {
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

func checkProbRow(y []float32, cols int) error {
	for r := 0; r*cols < len(y); r++ {
		sum := 0.0
		for i := 0; i < cols && r*cols+i < len(y); i++ {
			v := y[r*cols+i]
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("non-finite at %d", r*cols+i)
			}
			if v < -1e-6 {
				return fmt.Errorf("negative prob %g", v)
			}
			sum += float64(v)
		}
		if math.Abs(sum-1) > 1e-4 && sum > 1e-6 {
			// sparsemax rows sum to 1; allow small tol
			if math.Abs(sum-1) > 1e-3 {
				return fmt.Errorf("row %d sum=%g", r, sum)
			}
		}
	}
	return nil
}

func runExact(fc smCase) error {
	goPre, goPost, err := runGoSoftmax(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeSoftmax(fc)
	if err != nil {
		return err
	}
	cppPre, cppPost, _, err := runCppSoftmax(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return requireSoftmaxExact(goPre, goPost, cppPre, cppPost)
}

func caseSoftmaxStandard() error {
	cfg := softmax.Config{Dim: 8, SeqLen: 3, Kind: softmax.KindStandard, Temperature: 1}
	fc := smCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		shape: []int{2, 3, 8},
		x:     randInit(2*3, 8, 0x534d5301),
	}
	return runExact(fc)
}

func caseSoftmaxTemperature() error {
	cfg := softmax.Config{Dim: 6, SeqLen: 2, Kind: softmax.KindTemperature, Temperature: 0.7}
	fc := smCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		shape: []int{2, 2, 6},
		x:     randInit(2*2, 6, 0x534d5401),
	}
	return runExact(fc)
}

func caseSoftmaxMasked() error {
	n := 2 * 4
	mask := make([]bool, n)
	for i := range mask {
		mask[i] = i%3 != 0 // mask every 3rd
	}
	cfg := softmax.Config{Dim: 4, SeqLen: 2, Kind: softmax.KindMasked, Temperature: 1, Mask: mask}
	fc := smCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		shape: []int{2, 4},
		x:     randInit(2, 4, 0x534d4d01),
	}
	return runExact(fc)
}

func caseSoftmaxSparse() error {
	cfg := softmax.Config{Dim: 8, SeqLen: 2, Kind: softmax.KindSparse, Temperature: 1}
	fc := smCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		shape: []int{2, 8},
		x:     randInit(2, 8, 0x534d5001),
	}
	return runExact(fc)
}

func caseSoftmaxEntmax() error {
	cfg := softmax.Config{Dim: 8, SeqLen: 2, Kind: softmax.KindEntmax, Temperature: 1, EntmaxAlpha: 1.5}
	fc := smCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		shape: []int{2, 8},
		x:     randInit(2, 8, 0x534d4501),
	}
	return runExact(fc)
}

func caseSoftmaxGrid() error {
	cfg := softmax.Config{Dim: 5, SeqLen: 3, Kind: softmax.KindGrid, Rows: 3, Cols: 5, Temperature: 1}
	fc := smCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		shape: []int{3, 5},
		x:     randInit(3, 5, 0x534d4701),
	}
	return runExact(fc)
}

func caseSoftmaxHierarchical() error {
	cfg := softmax.Config{
		Dim: 4, SeqLen: 2, Kind: softmax.KindHierarchical, Temperature: 1,
		HierarchyLevels: []int{2, 4},
	}
	fc := smCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		shape: []int{2, 4},
		x:     randInit(2, 4, 0x534d4801),
	}
	return runExact(fc)
}

func caseSoftmaxGumbelSmoke() error {
	cfg := softmax.Config{Dim: 6, SeqLen: 2, Kind: softmax.KindGumbel, Temperature: 1}
	fc := smCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		shape: []int{2, 6},
		x:     randInit(2, 6, 0x534d5501),
	}
	goPre, _, err := runGoSoftmax(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	if err := checkProbRow(goPre, cfg.Dim); err != nil {
		return fmt.Errorf("go probs: %w", err)
	}
	payload, err := encodeSoftmax(fc)
	if err != nil {
		return err
	}
	cppPre, _, _, err := runCppSoftmax(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := checkProbRow(cppPre, cfg.Dim); err != nil {
		return fmt.Errorf("cpp probs: %w", err)
	}
	fmt.Printf("\n      gumbel smoke: go+cpp finite, row sums≈1 (no exact_bit)")
	return nil
}

type smKindFixture struct {
	name  string
	cfg   softmax.Config
	shape []int
	x     []float32
}

func softmaxKindFixtures() []smKindFixture {
	mask := make([]bool, 2*8)
	for i := range mask {
		mask[i] = i%3 != 0
	}
	return []smKindFixture{
		{
			name:  "standard",
			cfg:   softmax.Config{Dim: 8, SeqLen: 4, Kind: softmax.KindStandard, Temperature: 1},
			shape: []int{2, 4, 8},
			x:     randInit(2*4, 8, 0xd7a4e091),
		},
		{
			name:  "temperature",
			cfg:   softmax.Config{Dim: 8, SeqLen: 4, Kind: softmax.KindTemperature, Temperature: 0.7},
			shape: []int{2, 4, 8},
			x:     randInit(2*4, 8, 0xd7a4e092),
		},
		{
			name: "masked",
			cfg: softmax.Config{
				Dim: 8, SeqLen: 2, Kind: softmax.KindMasked, Temperature: 1, Mask: mask,
			},
			shape: []int{2, 8},
			x:     randInit(2, 8, 0xd7a4e093),
		},
		{
			name:  "sparse",
			cfg:   softmax.Config{Dim: 8, SeqLen: 2, Kind: softmax.KindSparse, Temperature: 1},
			shape: []int{2, 8},
			x:     randInit(2, 8, 0xd7a4e094),
		},
		{
			name: "entmax",
			cfg: softmax.Config{
				Dim: 8, SeqLen: 2, Kind: softmax.KindEntmax, Temperature: 1, EntmaxAlpha: 1.5,
			},
			shape: []int{2, 8},
			x:     randInit(2, 8, 0xd7a4e095),
		},
		{
			name: "grid",
			cfg: softmax.Config{
				Dim: 5, SeqLen: 3, Kind: softmax.KindGrid, Rows: 3, Cols: 5, Temperature: 1,
			},
			shape: []int{3, 5},
			x:     randInit(3, 5, 0xd7a4e096),
		},
		{
			name: "hierarchical",
			cfg: softmax.Config{
				Dim: 4, SeqLen: 2, Kind: softmax.KindHierarchical, Temperature: 1,
				HierarchyLevels: []int{2, 4},
			},
			shape: []int{2, 4},
			x:     randInit(2, 4, 0xd7a4e097),
		},
	}
}

func runSoftmaxDtypeTable(fx smKindFixture, benchIters int) (exactN, totalN int, worst suites.DetBand, fails []string) {
	worst = suites.DetExactBit
	fmt.Printf("\n      Go CPU vs C++ CPU — Softmax %s ×34 dtypes\n", fx.name)
	fmt.Printf("      %-12s %-12s %8s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "spect%", "go_µs", "cpp_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 88))
	for _, dt := range core.AllDTypes {
		fc := smCase{
			cfg: fx.cfg, dtype: dt, backend: core.BackendCPUTiled,
			shape: fx.shape, x: fx.x,
		}
		goPre, goPost, err := runGoSoftmax(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s/%s go: %v", fx.name, dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		payload, err := encodeSoftmax(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s/%s enc: %v", fx.name, dt, err))
			continue
		}
		cppPre, cppPost, _, err := runCppSoftmax(payload, 0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s/%s cpp: %v", fx.name, dt, err))
			fmt.Printf("      %-12s FAIL cpp %v\n", dt, err)
			continue
		}
		preM := suites.CompareF32Spectrum(goPre, cppPre)
		postM := suites.CompareF32Spectrum(goPost, cppPost)
		m := postM
		if preM.Band < m.Band {
			m = preM
		}
		if m.Band < worst {
			worst = m.Band
		}
		exactN += m.ExactBits
		totalN += m.N
		goUS, err := benchGoUS(func() error {
			_, _, err := runGoSoftmax(fc)
			return err
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s/%s go bench: %v", fx.name, dt, err))
			continue
		}
		_, _, cppDur, err := runCppSoftmax(payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s/%s cpp bench: %v", fx.name, dt, err))
			fmt.Printf("      %-12s %-12s %7.2f%% %7.0f%% %10s %10s %9s %s\n",
				dt, m.Band, m.ExactPct, m.Band.Percent(), FormatUS(goUS), "?", "?", fmt.Sprintf("%.3g", m.MaxAbs))
			continue
		}
		cppUS := perIterUS(cppDur, benchIters)
		spdPct := suites.SpeedPctDiff(goUS, cppUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %7.0f%% %10s %10s %+8.1f%% %s\n",
			dt, m.Band, m.ExactPct, m.Band.Percent(), FormatUS(goUS), FormatUS(cppUS), spdPct, fmt.Sprintf("%.3g", m.MaxAbs))
		if m.Band != suites.DetExactBit {
			fails = append(fails, fmt.Sprintf("%s/%s: %s", fx.name, dt, suites.FormatSpectrum(m)))
		}
	}
	overallExact := 0.0
	if totalN > 0 {
		overallExact = 100.0 * float64(exactN) / float64(totalN)
	}
	fmt.Printf("      %s\n", strings.Repeat("-", 88))
	fmt.Printf("      %s overall exact_bits=%.2f%%  worst_band=%s (%.0f%%)  aim=exact_bit (100%%)\n",
		fx.name, overallExact, worst, worst.Percent())
	fmt.Printf("      spd%%Δ = (cpp−go)/go×100  (negative ⇒ C++ faster)")
	return exactN, totalN, worst, fails
}

func caseSoftmaxAllKindsAllDTypes() error {
	const benchIters = defaultGoBenchIters
	var allFails []string
	var exactN, totalN int
	worstBand := suites.DetExactBit
	for _, fx := range softmaxKindFixtures() {
		en, tn, w, fails := runSoftmaxDtypeTable(fx, benchIters)
		exactN += en
		totalN += tn
		if w < worstBand {
			worstBand = w
		}
		allFails = append(allFails, fails...)
	}
	overallExact := 0.0
	if totalN > 0 {
		overallExact = 100.0 * float64(exactN) / float64(totalN)
	}
	fmt.Printf("\n      ==== Softmax all Kind×34 summary ====\n")
	fmt.Printf("      kinds=%d (excl. gumbel)  overall exact_bits=%.2f%%  worst_band=%s (%.0f%%)\n",
		len(softmaxKindFixtures()), overallExact, worstBand, worstBand.Percent())
	if len(allFails) > 0 {
		return fmt.Errorf("%d softmax kind×dtype issues: %s", len(allFails), strings.Join(allFails, " | "))
	}
	return nil
}
