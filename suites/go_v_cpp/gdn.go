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
	"github.com/openfluke/welvet/layers/gdn"
)

func CasesGDN() []Case {
	return []Case{
		{Name: "GDN Float32 primary — exact bits", Run: caseGdnPrimary},
		{Name: "GDN Float32 alt GQA — exact bits", Run: caseGdnAlt},
		{Name: "GDN Float32 spectrum + speed", Run: caseGdnSpeed},
	}
}

func RunGDN() error {
	var fails []string
	cs := CasesGDN()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [gd %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_gdn", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_gdn", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp gdn: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func gdnWeightCount(cfg gdn.Config) int {
	if err := cfg.Validate(); err != nil {
		return 0
	}
	cd := cfg.NumKeyHeads*cfg.KeyHeadDim*2 + cfg.NumValueHeads*cfg.ValueHeadDim
	vd := cfg.NumValueHeads * cfg.ValueHeadDim
	h, nh, k := cfg.HiddenSize, cfg.NumValueHeads, cfg.ConvKernel
	return cd*h + vd*h + nh*h + nh*h + h*vd + cd*k + nh + nh + cfg.ValueHeadDim
}

type gdCase struct {
	cfg     gdn.Config
	backend core.Backend
	batch   int
	packed  []float32
	x       []float32
}

func splitGdnPacked(cfg gdn.Config, packed []float32) (inQKV, inZ, inB, inA, outW, convW, aLog, dtBias, gamma []float32, err error) {
	if err := cfg.Validate(); err != nil {
		return nil, nil, nil, nil, nil, nil, nil, nil, nil, err
	}
	h := cfg.HiddenSize
	cd := cfg.NumKeyHeads*cfg.KeyHeadDim*2 + cfg.NumValueHeads*cfg.ValueHeadDim
	vd := cfg.NumValueHeads * cfg.ValueHeadDim
	nh := cfg.NumValueHeads
	k := cfg.ConvKernel
	nQKV, nZ, nB, nA := cd*h, vd*h, nh*h, nh*h
	nOut, nConv := h*vd, cd*k
	need := nQKV + nZ + nB + nA + nOut + nConv + nh + nh + cfg.ValueHeadDim
	if len(packed) < need {
		return nil, nil, nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("gdn packed short: %d < %d", len(packed), need)
	}
	o := 0
	take := func(n int) []float32 {
		s := packed[o : o+n]
		o += n
		return s
	}
	inQKV = take(nQKV)
	inZ = take(nZ)
	inB = take(nB)
	inA = take(nA)
	outW = take(nOut)
	convW = take(nConv)
	aLog = take(nh)
	dtBias = take(nh)
	gamma = take(cfg.ValueHeadDim)
	return
}

func runGoGdn(fc gdCase) (pre, post []float32, err error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	inQKV, inZ, inB, inA, outW, convW, aLog, dtBias, gamma, err := splitGdnPacked(cfg, fc.packed)
	if err != nil {
		return nil, nil, err
	}
	l, err := gdn.NewConfigured(cfg, inQKV, inZ, inB, inA, outW, convW, aLog, dtBias, gamma)
	if err != nil {
		return nil, nil, err
	}
	l.Exec.Backend = fc.backend
	l.Exec.MultiCore = false
	seq := len(fc.x) / (fc.batch * cfg.HiddenSize)
	t := core.NewTensor[float32](fc.batch, seq, cfg.HiddenSize)
	copy(t.Data, fc.x)
	preT, postT, err := gdn.Forward(l, t)
	if err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), preT.Data...), append([]float32(nil), postT.Data...), nil
}

func encodeGdn(fc gdCase, seq int) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("WGD1"))
	for _, v := range []int32{
		int32(fc.batch), int32(seq), int32(cfg.HiddenSize),
		int32(cfg.NumKeyHeads), int32(cfg.NumValueHeads),
		int32(cfg.KeyHeadDim), int32(cfg.ValueHeadDim), int32(cfg.ConvKernel),
		int32(core.DTypeFloat32), int32(fc.backend),
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

func decodeGdn(raw []byte) (pre, post []float32, bench time.Duration, err error) {
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
	if len(raw) < 8 || string(raw[:4]) != "WGDO" {
		return nil, nil, bench, fmt.Errorf("cpp gdn bad magic")
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

func runCppGdn(payload []byte, benchIters int) (pre, post []float32, dur time.Duration, err error) {
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
		return nil, nil, 0, fmt.Errorf("cpp gdn: %v: %s", err, stderr.String())
	}
	return decodeGdn(stdout.Bytes())
}

func runGdnExact(fc gdCase, seq int) error {
	goPre, goPost, err := runGoGdn(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeGdn(fc, seq)
	if err != nil {
		return err
	}
	cppPre, cppPost, _, err := runCppGdn(payload, 0)
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

func caseGdnPrimary() error {
	cfg := gdn.Config{
		HiddenSize: 8, NumKeyHeads: 2, NumValueHeads: 4,
		KeyHeadDim: 4, ValueHeadDim: 4, ConvKernel: 4, Eps: 1e-6,
	}
	const batch, seq = 2, 3
	fc := gdCase{
		cfg: cfg, backend: core.BackendCPUTiled, batch: batch,
		packed: randInit(1, gdnWeightCount(cfg), 0x47444e01),
		x:      randInit(batch*seq, cfg.HiddenSize, 0x47444e02),
	}
	return runGdnExact(fc, seq)
}

func caseGdnAlt() error {
	cfg := gdn.Config{
		HiddenSize: 6, NumKeyHeads: 1, NumValueHeads: 2,
		KeyHeadDim: 4, ValueHeadDim: 4, ConvKernel: 4, Eps: 1e-6,
	}
	const batch, seq = 2, 4
	fc := gdCase{
		cfg: cfg, backend: core.BackendCPUTiled, batch: batch,
		packed: randInit(1, gdnWeightCount(cfg), 0x47444e03),
		x:      randInit(batch*seq, cfg.HiddenSize, 0x47444e04),
	}
	return runGdnExact(fc, seq)
}

func caseGdnSpeed() error {
	const batch, seq, benchIters = 2, 3, 20
	cfg := gdn.Config{
		HiddenSize: 8, NumKeyHeads: 2, NumValueHeads: 4,
		KeyHeadDim: 4, ValueHeadDim: 4, ConvKernel: 4, Eps: 1e-6,
	}
	if err := cfg.Validate(); err != nil {
		return err
	}
	fc := gdCase{
		cfg: cfg, backend: core.BackendCPUTiled, batch: batch,
		packed: randInit(1, gdnWeightCount(cfg), 0xd7a4e0f1),
		x:      randInit(batch*seq, cfg.HiddenSize, 0xd7a4e0f2),
	}
	goPre, goPost, err := runGoGdn(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeGdn(fc, seq)
	if err != nil {
		return err
	}
	cppPre, cppPost, _, err := runCppGdn(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	preM := suites.CompareF32Spectrum(goPre, cppPre)
	postM := suites.CompareF32Spectrum(goPost, cppPost)
	m := postM
	if preM.Band < m.Band {
		m = preM
	}
	fmt.Printf("\n      Go CPU vs C++ CPU — GDN Float32\n")
	fmt.Printf("      %-12s %-12s %8s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "spect%", "go_µs", "cpp_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 88))
	for i := 0; i < 2; i++ {
		if _, _, err := runGoGdn(fc); err != nil {
			break
		}
	}
	t0 := time.Now()
	for i := 0; i < benchIters; i++ {
		if _, _, err := runGoGdn(fc); err != nil {
			return err
		}
	}
	goDur := time.Since(t0)
	_, _, cppDur, err := runCppGdn(payload, benchIters)
	if err != nil {
		return err
	}
	goUS := float64(goDur.Nanoseconds()) / float64(benchIters) / 1000
	cppUS := float64(cppDur.Nanoseconds()) / float64(benchIters) / 1000
	spdPct := suites.SpeedPctDiff(goUS, cppUS)
	fmt.Printf("      %-12s %-12s %7.2f%% %7.0f%% %10.2f %10.2f %+8.1f%% %s\n",
		"float32", m.Band, m.ExactPct, m.Band.Percent(), goUS, cppUS, spdPct, fmt.Sprintf("%.3g", m.MaxAbs))
	fmt.Printf("      %s\n", strings.Repeat("-", 88))
	fmt.Printf("      aim=exact_bit (100%%); Go GDN is Float32 FormatNone only (not ×34)")
	if m.Band != suites.DetExactBit {
		return fmt.Errorf("gdn: %s", suites.FormatSpectrum(m))
	}
	return nil
}
