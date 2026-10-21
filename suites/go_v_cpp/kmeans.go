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
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/quant"
)

func CasesKMeans() []Case {
	return []Case{
		{Name: "KMeans probabilities Float32 Linear — exact bits", Run: caseKMeansProbs},
		{Name: "KMeans features Float32 Linear — exact bits", Run: caseKMeansFeatures},
		{Name: "KMeans both modes × all 34 dtypes — spectrum + speed", Run: caseKMeansAllModesAllDTypes},
	}
}

func RunKMeans() error {
	var fails []string
	cs := CasesKMeans()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [km %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_kmeans", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_kmeans", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp kmeans: %d failed: %s", len(fails), strings.Join(fails, ", "))
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
		return nil, nil, bench, fmt.Errorf("cpp kmeans bad magic")
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

func runCppKMeans(payload []byte, benchIters int) (pre, post []float32, dur time.Duration, err error) {
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
		return nil, nil, 0, fmt.Errorf("cpp kmeans: %v: %s", err, stderr.String())
	}
	return decodeKMeans(stdout.Bytes())
}

func requireKMeansExact(goPre, goPost, cppPre, cppPost []float32) error {
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

func runKMeansExact(fc kmCase) error {
	goPre, goPost, err := runGoKMeans(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeKMeans(fc)
	if err != nil {
		return err
	}
	cppPre, cppPost, _, err := runCppKMeans(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return requireKMeansExact(goPre, goPost, cppPre, cppPost)
}

func caseKMeansProbs() error {
	cfg := kmeans.Config{
		NumClusters: 4, FeatureDim: 6, Temperature: 1,
		OutputMode: kmeans.OutputProbabilities, Activation: core.ActivationLinear,
	}
	fc := kmCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		shape: []int{3, 6},
		W:     randInit(cfg.NumClusters, cfg.FeatureDim, 0x4b4d4501),
		x:     randInit(3, 6, 0x4b4d4502),
	}
	return runKMeansExact(fc)
}

func caseKMeansFeatures() error {
	cfg := kmeans.Config{
		NumClusters: 4, FeatureDim: 6, Temperature: 0.8,
		OutputMode: kmeans.OutputFeatures, Activation: core.ActivationLinear,
	}
	fc := kmCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		shape: []int{3, 6},
		W:     randInit(cfg.NumClusters, cfg.FeatureDim, 0x4b4d4601),
		x:     randInit(3, 6, 0x4b4d4602),
	}
	return runKMeansExact(fc)
}

type kmModeFixture struct {
	name  string
	cfg   kmeans.Config
	shape []int
	W     []float32
	x     []float32
}

func kmeansModeFixtures() []kmModeFixture {
	return []kmModeFixture{
		{
			name: "probabilities",
			cfg: kmeans.Config{
				NumClusters: 5, FeatureDim: 8, Temperature: 1,
				OutputMode: kmeans.OutputProbabilities, Activation: core.ActivationLinear,
			},
			shape: []int{2, 8},
			W:     randInit(5, 8, 0xd7a4e0a1),
			x:     randInit(2, 8, 0xd7a4e0a2),
		},
		{
			name: "features",
			cfg: kmeans.Config{
				NumClusters: 5, FeatureDim: 8, Temperature: 0.9,
				OutputMode: kmeans.OutputFeatures, Activation: core.ActivationLinear,
			},
			shape: []int{2, 8},
			W:     randInit(5, 8, 0xd7a4e0a3),
			x:     randInit(2, 8, 0xd7a4e0a4),
		},
	}
}

func runKMeansDtypeTable(fx kmModeFixture, benchIters int) (exactN, totalN int, worst suites.DetBand, fails []string) {
	worst = suites.DetExactBit
	fmt.Printf("\n      Go CPU vs C++ CPU — KMeans %s ×34 dtypes\n", fx.name)
	fmt.Printf("      %-12s %-12s %8s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "spect%", "go_µs", "cpp_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 88))
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
		payload, err := encodeKMeans(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s/%s enc: %v", fx.name, dt, err))
			continue
		}
		cppPre, cppPost, _, err := runCppKMeans(payload, 0)
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
			_, _, err := runGoKMeans(fc)
			return err
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s/%s go bench: %v", fx.name, dt, err))
			continue
		}
		_, _, cppDur, err := runCppKMeans(payload, benchIters)
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

func caseKMeansAllModesAllDTypes() error {
	const benchIters = defaultGoBenchIters
	var allFails []string
	var exactN, totalN int
	worstBand := suites.DetExactBit
	for _, fx := range kmeansModeFixtures() {
		en, tn, w, fails := runKMeansDtypeTable(fx, benchIters)
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
	fmt.Printf("\n      ==== KMeans modes×34 summary ====\n")
	fmt.Printf("      modes=%d  overall exact_bits=%.2f%%  worst_band=%s (%.0f%%)\n",
		len(kmeansModeFixtures()), overallExact, worstBand, worstBand.Percent())
	if len(allFails) > 0 {
		return fmt.Errorf("%d kmeans mode×dtype issues: %s", len(allFails), strings.Join(allFails, " | "))
	}
	return nil
}
