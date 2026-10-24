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
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/quant"
)

func CasesKMeans() []Case {
	return []Case{
		{Name: "Locate welvet.metal tool", Run: caseLocateMetal},
		{Name: "Locate welvet.cpp tool (optional speed)", Run: caseLocateCppOptional},
		{Name: "KMeans probabilities Float32 Linear — exact bits vs Go", Run: caseKMeansProbs},
		{Name: "KMeans features Float32 Linear — exact bits vs Go", Run: caseKMeansFeatures},
		{Name: "KMeans both modes ×34 — exact_bit + Go/C++/Metal timings", Run: caseKMeansAllModesAllDTypes},
	}
}

func RunKMeans() error {
	var fails []string
	cs := CasesKMeans()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [km %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_metal_kmeans", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_metal_kmeans", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_metal kmeans: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type kmCase struct {
	cfg     kmeans.Config
	dtype   core.DType
	backend core.Backend
	shape   []int
	W       []float32
	x       []float32
}

func outputModeInt(m kmeans.OutputMode) int32 {
	if m == kmeans.OutputFeatures {
		return 1
	}
	return 0
}

func runGoKMeans(fc kmCase) (pre, post []float32, err error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	l, err := kmeans.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.W)
	if err != nil {
		return nil, nil, err
	}
	l.Exec.Backend = fc.backend
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	if l.Centers != nil {
		l.Centers.Exec.MultiCore = false
		l.Centers.Core.MultiCore = false
		l.Centers.Exec.Backend = fc.backend
	}
	t := core.NewTensor[float32](fc.shape...)
	copy(t.Data, fc.x)
	preT, postT, err := kmeans.Forward(l, t)
	if err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), preT.Data...), append([]float32(nil), postT.Data...), nil
}

func encodeKMeans(fc kmCase) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	n := len(fc.x)
	var in bytes.Buffer
	_, _ = in.Write([]byte("WKM1"))
	_ = binary.Write(&in, binary.LittleEndian, int32(n))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.FeatureDim))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.NumClusters))
	_ = binary.Write(&in, binary.LittleEndian, outputModeInt(cfg.OutputMode))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.Activation))
	_ = binary.Write(&in, binary.LittleEndian, cfg.Temperature)
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.dtype))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.backend))
	_ = binary.Write(&in, binary.LittleEndian, int32(len(fc.shape)))
	for _, d := range fc.shape {
		_ = binary.Write(&in, binary.LittleEndian, int32(d))
	}
	for _, v := range fc.W {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	return in.Bytes(), nil
}

func decodeKMeans(raw []byte) (pre, post []float32, bench time.Duration, err error) {
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
	if len(raw) < 8 || string(raw[:4]) != "WKMO" {
		return nil, nil, bench, fmt.Errorf("kmeans bad magic: %q", string(raw[:min(4, len(raw))]))
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

func runToolKMeans(tool string, fc kmCase, benchIters int) (pre, post []float32, dur time.Duration, err error) {
	payload, err := encodeKMeans(fc)
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
	return decodeKMeans(stdout.Bytes())
}

func runMetalKMeans(fc kmCase) (pre, post []float32, err error) {
	tool, err := findMetalFwd()
	if err != nil {
		return nil, nil, err
	}
	pre, post, _, err = runToolKMeans(tool, fc, 0)
	return pre, post, err
}

func requireKMeansExact(fc kmCase) error {
	goPre, goPost, err := runGoKMeans(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	mPre, mPost, err := runMetalKMeans(fc)
	if err != nil {
		return fmt.Errorf("metal: %w", err)
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

func caseKMeansProbs() error {
	cfg := kmeans.Config{
		NumClusters: 4, FeatureDim: 6, Temperature: 1,
		OutputMode: kmeans.OutputProbabilities, Activation: core.ActivationLinear,
	}
	return requireKMeansExact(kmCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		shape: []int{3, 6},
		W:     randInit(cfg.NumClusters, cfg.FeatureDim, 0x4b4d4501),
		x:     randInit(3, 6, 0x4b4d4502),
	})
}

func caseKMeansFeatures() error {
	cfg := kmeans.Config{
		NumClusters: 4, FeatureDim: 6, Temperature: 0.8,
		OutputMode: kmeans.OutputFeatures, Activation: core.ActivationLinear,
	}
	return requireKMeansExact(kmCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		shape: []int{3, 6},
		W:     randInit(cfg.NumClusters, cfg.FeatureDim, 0x4b4d4601),
		x:     randInit(3, 6, 0x4b4d4602),
	})
}

type kmModeFixture struct {
	name  string
	cfg   kmeans.Config
	shape []int
	W     []float32
	x     []float32
}

func kmeansModeFixtures() []kmModeFixture {
	// Mid shapes for 3-way timing (host f64 — Metal≈C++ expected).
	const batch, k, d = 32, 16, 64
	return []kmModeFixture{
		{
			name: "probabilities",
			cfg: kmeans.Config{
				NumClusters: k, FeatureDim: d, Temperature: 1,
				OutputMode: kmeans.OutputProbabilities, Activation: core.ActivationLinear,
			},
			shape: []int{batch, d},
			W:     randInit(k, d, 0xd7a4e0a1),
			x:     randInit(batch, d, 0xd7a4e0a2),
		},
		{
			name: "features",
			cfg: kmeans.Config{
				NumClusters: k, FeatureDim: d, Temperature: 0.9,
				OutputMode: kmeans.OutputFeatures, Activation: core.ActivationLinear,
			},
			shape: []int{batch, d},
			W:     randInit(k, d, 0xd7a4e0a3),
			x:     randInit(batch, d, 0xd7a4e0a4),
		},
	}
}

func runKMeansDtypeTable(fx kmModeFixture, metalTool, cppTool string, haveCpp bool, benchIters int) (exactN, totalN int, winCount map[string]int, fails []string) {
	winCount = map[string]int{}
	fmt.Printf("\n      Go vs Metal KMeans %s FormatNone×34 — exact_bit required", fx.name)
	fmt.Printf("\n      shape %v  K=%d D=%d temp=%g  iters=%d", fx.shape, fx.cfg.NumClusters, fx.cfg.FeatureDim, fx.cfg.Temperature, benchIters)
	if haveCpp {
		fmt.Printf("\n      speed: Go / C++ / Metal (host f64 soft assign)")
	}
	fmt.Printf("\n      %-12s %-12s %7s %9s %9s %9s %8s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "mtl_µs", "winner", "vsGo")
	fmt.Printf("      %s\n", strings.Repeat("-", 100))

	for _, dt := range core.AllDTypes {
		fc := kmCase{
			cfg: fx.cfg, dtype: dt, backend: core.BackendCPUTiled,
			shape: fx.shape, W: fx.W, x: fx.x,
		}
		goPre, goPost, err := runGoKMeans(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s/%s go: %v", fx.name, dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		mPre, mPost, err := runMetalKMeans(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s/%s metal: %v", fx.name, dt, err))
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

		goUS, err := benchGoUS(func() error {
			_, _, e := runGoKMeans(fc)
			return e
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s/%s go bench: %v", fx.name, dt, err))
			continue
		}

		cppUS := 0.0
		cppStr := "—"
		if haveCpp {
			_, _, cppDur, err := runToolKMeans(cppTool, fc, benchIters)
			if err != nil {
				cppStr = "err"
			} else {
				cppUS = perIterUS(cppDur, benchIters)
				cppStr = FormatUS(cppUS)
			}
		}

		_, _, metalDur, err := runToolKMeans(metalTool, fc, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s/%s metal bench: %v", fx.name, dt, err))
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
			fails = append(fails, fmt.Sprintf("%s/%s: %s", fx.name, dt, suites.FormatSpectrum(m)))
		}
	}
	overallExact := 0.0
	if totalN > 0 {
		overallExact = 100.0 * float64(exactN) / float64(totalN)
	}
	fmt.Printf("      %s\n", strings.Repeat("-", 100))
	fmt.Printf("      %s overall exact_bits=%.2f%%  (aim=100%% vs Go)\n", fx.name, overallExact)
	parts := []string{}
	for _, n := range []string{"Go", "C++", "Metal"} {
		if winCount[n] > 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", n, winCount[n]))
		}
	}
	fmt.Printf("      wins: %s\n", strings.Join(parts, " "))
	return exactN, totalN, winCount, fails
}

func caseKMeansAllModesAllDTypes() error {
	const benchIters = 80
	if len(core.AllDTypes) != 34 {
		return fmt.Errorf("expected 34 dtypes, got %d", len(core.AllDTypes))
	}
	metalTool, err := findMetalFwd()
	if err != nil {
		return err
	}
	cppTool, cppErr := findCppFwd()
	haveCpp := cppErr == nil

	var allFails []string
	var exactN, totalN int
	totalWins := map[string]int{}
	for _, fx := range kmeansModeFixtures() {
		en, tn, wins, fails := runKMeansDtypeTable(fx, metalTool, cppTool, haveCpp, benchIters)
		exactN += en
		totalN += tn
		allFails = append(allFails, fails...)
		for k, v := range wins {
			totalWins[k] += v
		}
	}
	overallExact := 0.0
	if totalN > 0 {
		overallExact = 100.0 * float64(exactN) / float64(totalN)
	}
	fmt.Printf("\n      ==== KMeans modes×34 summary ====\n")
	fmt.Printf("      modes=%d  overall exact_bits=%.2f%%  (aim=100%% vs Go)\n",
		len(kmeansModeFixtures()), overallExact)
	parts := []string{}
	for _, n := range []string{"Go", "C++", "Metal"} {
		if totalWins[n] > 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", n, totalWins[n]))
		}
	}
	fmt.Printf("      wins: %s\n", strings.Join(parts, " "))
	fmt.Printf("      note: KMeans = FormatNone×34 centers decode_row_f64 + host f64 softMax")
	if len(allFails) > 0 {
		return fmt.Errorf("%d kmeans mode×dtype issues: %s", len(allFails), strings.Join(allFails, " | "))
	}
	return nil
}
