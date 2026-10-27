package go_v_cuda

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
		{Name: "Locate welvet.cuda tool", Run: caseLocateCUDA},
		{Name: "Locate welvet.cpp tool (optional speed)", Run: caseLocateCppOptional},
		{Name: "Softmax Standard Float32 — exact bits vs Go", Run: caseSoftmaxStandard},
		{Name: "Softmax Temperature Float32 — exact bits vs Go", Run: caseSoftmaxTemperature},
		{Name: "Softmax Masked Float32 — exact bits vs Go", Run: caseSoftmaxMasked},
		{Name: "Softmax Sparse Float32 — exact bits vs Go", Run: caseSoftmaxSparse},
		{Name: "Softmax Entmax Float32 — exact bits vs Go", Run: caseSoftmaxEntmax},
		{Name: "Softmax Grid Float32 — exact bits vs Go", Run: caseSoftmaxGrid},
		{Name: "Softmax Hierarchical Float32 — exact bits vs Go", Run: caseSoftmaxHierarchical},
		{Name: "Softmax Gumbel Float32 — smoke (non-deterministic)", Run: caseSoftmaxGumbelSmoke},
		{Name: "Softmax all Kind ×34 — exact_bit + Go/C++/CUDA timings", Run: caseSoftmaxAllKindsAllDTypes},
	}
}

func RunSoftmax() error {
	var fails []string
	cs := CasesSoftmax()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [sm %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cuda_softmax", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cuda_softmax", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cuda softmax: %d failed: %s", len(fails), strings.Join(fails, ", "))
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
		return nil, nil, bench, fmt.Errorf("softmax bad magic: %q", string(raw[:min(4, len(raw))]))
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

func runToolSoftmax(tool string, fc smCase, benchIters int) (pre, post []float32, dur time.Duration, err error) {
	payload, err := encodeSoftmax(fc)
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
	return decodeSoftmax(stdout.Bytes())
}

func runCUDASoftmax(fc smCase) (pre, post []float32, err error) {
	tool, err := findCUDAFwd()
	if err != nil {
		return nil, nil, err
	}
	pre, post, _, err = runToolSoftmax(tool, fc, 0)
	return pre, post, err
}

func requireSoftmaxExact(fc smCase) error {
	goPre, goPost, err := runGoSoftmax(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	mPre, mPost, err := runCUDASoftmax(fc)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
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
		if math.Abs(sum-1) > 1e-3 && sum > 1e-6 {
			return fmt.Errorf("row %d sum=%g", r, sum)
		}
	}
	return nil
}

func caseSoftmaxStandard() error {
	cfg := softmax.Config{Dim: 8, SeqLen: 3, Kind: softmax.KindStandard, Temperature: 1}
	return requireSoftmaxExact(smCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		shape: []int{2, 3, 8}, x: randInit(2*3, 8, 0x534d5301),
	})
}

func caseSoftmaxTemperature() error {
	cfg := softmax.Config{Dim: 6, SeqLen: 2, Kind: softmax.KindTemperature, Temperature: 0.7}
	return requireSoftmaxExact(smCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		shape: []int{2, 2, 6}, x: randInit(2*2, 6, 0x534d5401),
	})
}

func caseSoftmaxMasked() error {
	n := 2 * 4
	mask := make([]bool, n)
	for i := range mask {
		mask[i] = i%3 != 0
	}
	cfg := softmax.Config{Dim: 4, SeqLen: 2, Kind: softmax.KindMasked, Temperature: 1, Mask: mask}
	return requireSoftmaxExact(smCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		shape: []int{2, 4}, x: randInit(2, 4, 0x534d4d01),
	})
}

func caseSoftmaxSparse() error {
	cfg := softmax.Config{Dim: 8, SeqLen: 2, Kind: softmax.KindSparse, Temperature: 1}
	return requireSoftmaxExact(smCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		shape: []int{2, 8}, x: randInit(2, 8, 0x534d5001),
	})
}

func caseSoftmaxEntmax() error {
	cfg := softmax.Config{Dim: 8, SeqLen: 2, Kind: softmax.KindEntmax, Temperature: 1, EntmaxAlpha: 1.5}
	return requireSoftmaxExact(smCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		shape: []int{2, 8}, x: randInit(2, 8, 0x534d4501),
	})
}

func caseSoftmaxGrid() error {
	cfg := softmax.Config{Dim: 5, SeqLen: 3, Kind: softmax.KindGrid, Rows: 3, Cols: 5, Temperature: 1}
	return requireSoftmaxExact(smCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		shape: []int{3, 5}, x: randInit(3, 5, 0x534d4701),
	})
}

func caseSoftmaxHierarchical() error {
	cfg := softmax.Config{
		Dim: 4, SeqLen: 2, Kind: softmax.KindHierarchical, Temperature: 1,
		HierarchyLevels: []int{2, 4},
	}
	return requireSoftmaxExact(smCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		shape: []int{2, 4}, x: randInit(2, 4, 0x534d4801),
	})
}

func caseSoftmaxGumbelSmoke() error {
	cfg := softmax.Config{Dim: 6, SeqLen: 2, Kind: softmax.KindGumbel, Temperature: 1}
	fc := smCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		shape: []int{2, 6}, x: randInit(2, 6, 0x534d5501),
	}
	goPre, _, err := runGoSoftmax(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	if err := checkProbRow(goPre, cfg.Dim); err != nil {
		return fmt.Errorf("go probs: %w", err)
	}
	mPre, _, err := runCUDASoftmax(fc)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	if err := checkProbRow(mPre, cfg.Dim); err != nil {
		return fmt.Errorf("cuda probs: %w", err)
	}
	fmt.Printf("\n      gumbel smoke: go+cuda finite, row sums≈1 (no exact_bit)")
	return nil
}

type smKindFixture struct {
	name  string
	cfg   softmax.Config
	shape []int
	x     []float32
}

func softmaxKindFixtures() []smKindFixture {
	// Large enough for measurable warm forward (excl. gumbel).
	mask := make([]bool, 8*256)
	for i := range mask {
		mask[i] = i%3 != 0
	}
	return []smKindFixture{
		{
			name:  "standard",
			cfg:   softmax.Config{Dim: 256, SeqLen: 32, Kind: softmax.KindStandard, Temperature: 1},
			shape: []int{8, 32, 256},
			x:     randInit(8*32, 256, 0xd7a4e091),
		},
		{
			name:  "temperature",
			cfg:   softmax.Config{Dim: 256, SeqLen: 32, Kind: softmax.KindTemperature, Temperature: 0.7},
			shape: []int{8, 32, 256},
			x:     randInit(8*32, 256, 0xd7a4e092),
		},
		{
			name: "masked",
			cfg: softmax.Config{
				Dim: 256, SeqLen: 8, Kind: softmax.KindMasked, Temperature: 1, Mask: mask,
			},
			shape: []int{8, 256},
			x:     randInit(8, 256, 0xd7a4e093),
		},
		{
			name:  "sparse",
			cfg:   softmax.Config{Dim: 128, SeqLen: 16, Kind: softmax.KindSparse, Temperature: 1},
			shape: []int{16, 128},
			x:     randInit(16, 128, 0xd7a4e094),
		},
		{
			name: "entmax",
			cfg: softmax.Config{
				Dim: 128, SeqLen: 16, Kind: softmax.KindEntmax, Temperature: 1, EntmaxAlpha: 1.5,
			},
			shape: []int{16, 128},
			x:     randInit(16, 128, 0xd7a4e095),
		},
		{
			name: "grid",
			cfg: softmax.Config{
				Dim: 64, SeqLen: 64, Kind: softmax.KindGrid, Rows: 64, Cols: 64, Temperature: 1,
			},
			shape: []int{64, 64},
			x:     randInit(64, 64, 0xd7a4e096),
		},
		{
			name: "hierarchical",
			cfg: softmax.Config{
				Dim: 64, SeqLen: 32, Kind: softmax.KindHierarchical, Temperature: 1,
				HierarchyLevels: []int{8, 64},
			},
			shape: []int{32, 64},
			x:     randInit(32, 64, 0xd7a4e097),
		},
	}
}

func runSoftmaxDtypeTable(fx smKindFixture, cudaTool, cppTool string, haveCpp bool, benchIters int) (exactN, totalN int, fails []string) {
	fmt.Printf("\n      Go vs CUDA Softmax %s FormatNone×34 — exact_bit required", fx.name)
	fmt.Printf("\n      %-12s %-12s %7s %9s %9s %9s %8s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "cuda_µs", "winner", "vsGo")
	fmt.Printf("      %s\n", strings.Repeat("-", 100))
	winCount := map[string]int{}

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
		mPre, mPost, err := runCUDASoftmax(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s/%s cuda: %v", fx.name, dt, err))
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

		goUS, err := benchGoUS(func() error {
			_, _, e := runGoSoftmax(fc)
			return e
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s/%s go bench: %v", fx.name, dt, err))
			continue
		}

		cppUS := 0.0
		cppStr := "—"
		if haveCpp {
			_, _, cppDur, err := runToolSoftmax(cppTool, fc, benchIters)
			if err != nil {
				cppStr = "err"
			} else {
				cppUS = perIterUS(cppDur, benchIters)
				cppStr = FormatUS(cppUS)
			}
		}

		_, _, cudaDur, err := runToolSoftmax(cudaTool, fc, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s/%s cuda bench: %v", fx.name, dt, err))
			fmt.Printf("      %-12s %-12s %6.2f%% %9s %9s %9s %8s %s\n",
				dt, m.Band, m.ExactPct, FormatUS(goUS), cppStr, "?", "?", err)
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
			fails = append(fails, fmt.Sprintf("%s/%s: %s", fx.name, dt, suites.FormatSpectrum(m)))
		}
	}

	overallExact := 0.0
	if totalN > 0 {
		overallExact = 100.0 * float64(exactN) / float64(totalN)
	}
	parts := []string{}
	for _, n := range []string{"Go", "C++", "CUDA"} {
		if winCount[n] > 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", n, winCount[n]))
		}
	}
	fmt.Printf("      %s\n", strings.Repeat("-", 100))
	fmt.Printf("      %s overall exact_bits=%.2f%%  wins: %s\n", fx.name, overallExact, strings.Join(parts, " "))
	return exactN, totalN, fails
}

func caseSoftmaxAllKindsAllDTypes() error {
	const benchIters = 40
	if len(core.AllDTypes) != 34 {
		return fmt.Errorf("expected 34 dtypes, got %d", len(core.AllDTypes))
	}
	cudaTool, err := findCUDAFwd()
	if err != nil {
		return err
	}
	cppTool, cppErr := findCppFwd()
	haveCpp := cppErr == nil

	var allFails []string
	var exactN, totalN int
	for _, fx := range softmaxKindFixtures() {
		en, tn, fails := runSoftmaxDtypeTable(fx, cudaTool, cppTool, haveCpp, benchIters)
		exactN += en
		totalN += tn
		allFails = append(allFails, fails...)
	}
	overallExact := 0.0
	if totalN > 0 {
		overallExact = 100.0 * float64(exactN) / float64(totalN)
	}
	fmt.Printf("\n      ==== Softmax all Kind×34 summary ====\n")
	fmt.Printf("      kinds=%d (excl. gumbel)  overall exact_bits=%.2f%%  (aim=100%% vs Go)\n",
		len(softmaxKindFixtures()), overallExact)
	fmt.Printf("      note: CUDA one-row Softmax/sparsemax/entmax (f64 exp→f32 exact_bit vs Go)")

	if len(allFails) > 0 {
		return fmt.Errorf("%d softmax kind×dtype issues: %s", len(allFails), strings.Join(allFails, " | "))
	}
	return nil
}
