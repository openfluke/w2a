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
	"github.com/openfluke/welvet/layers/gdn"
)

func CasesGDN() []Case {
	return []Case{
		{Name: "Locate welvet.metal tool", Run: caseLocateMetal},
		{Name: "Locate welvet.cpp tool (optional speed)", Run: caseLocateCppOptional},
		{Name: "GDN Float32 primary — ≥industry vs Go, exact vs C++", Run: caseGdnPrimary},
		{Name: "GDN Float32 alt GQA — ≥industry vs Go, exact vs C++", Run: caseGdnAlt},
		{Name: "GDN Float32 mid — parity + Go/C++/Metal timings", Run: caseGdnSpeed},
	}
}

func RunGDN() error {
	var fails []string
	cs := CasesGDN()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [gd %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_metal_gdn", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_metal_gdn", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_metal gdn: %d failed: %s", len(fails), strings.Join(fails, ", "))
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
	seq     int
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
	t := core.NewTensor[float32](fc.batch, fc.seq, cfg.HiddenSize)
	copy(t.Data, fc.x)
	preT, postT, err := gdn.Forward(l, t)
	if err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), preT.Data...), append([]float32(nil), postT.Data...), nil
}

func encodeGdn(fc gdCase) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("WGD1"))
	for _, v := range []int32{
		int32(fc.batch), int32(fc.seq), int32(cfg.HiddenSize),
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
		return nil, nil, bench, fmt.Errorf("gdn bad magic: %q", string(raw[:min(4, len(raw))]))
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

func runToolGdn(tool string, fc gdCase, benchIters int) (pre, post []float32, dur time.Duration, err error) {
	payload, err := encodeGdn(fc)
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
	return decodeGdn(stdout.Bytes())
}

func runMetalGdn(fc gdCase) (pre, post []float32, err error) {
	tool, err := findMetalFwd()
	if err != nil {
		return nil, nil, err
	}
	pre, post, _, err = runToolGdn(tool, fc, 0)
	return pre, post, err
}

func requireGdnParity(fc gdCase) error {
	goPre, goPost, err := runGoGdn(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	mPre, mPost, err := runMetalGdn(fc)
	if err != nil {
		return fmt.Errorf("metal: %w", err)
	}
	preM := suites.CompareF32Spectrum(goPre, mPre)
	postM := suites.CompareF32Spectrum(goPost, mPost)
	fmt.Printf("\n      vsGo  pre  %s", suites.FormatSpectrum(preM))
	fmt.Printf("\n      vsGo  post %s", suites.FormatSpectrum(postM))
	// Go↔C++ GDN is industry/strict today (host f32 scan); Metal mirrors C++.
	if preM.Band < suites.DetIndustry {
		return fmt.Errorf("pre below industry: %s", suites.FormatSpectrum(preM))
	}
	if postM.Band < suites.DetIndustry {
		return fmt.Errorf("post below industry: %s", suites.FormatSpectrum(postM))
	}
	if cppTool, err := findCppFwd(); err == nil {
		cPre, cPost, _, err := runToolGdn(cppTool, fc, 0)
		if err != nil {
			return fmt.Errorf("cpp: %w", err)
		}
		cPreM := suites.CompareF32Spectrum(cPre, mPre)
		cPostM := suites.CompareF32Spectrum(cPost, mPost)
		fmt.Printf("\n      vsCpp pre  %s", suites.FormatSpectrum(cPreM))
		fmt.Printf("\n      vsCpp post %s", suites.FormatSpectrum(cPostM))
		if cPreM.Band != suites.DetExactBit || cPostM.Band != suites.DetExactBit {
			return fmt.Errorf("Metal≠C++ (port bug): pre %s post %s",
				suites.FormatSpectrum(cPreM), suites.FormatSpectrum(cPostM))
		}
	}
	return nil
}

func caseGdnPrimary() error {
	cfg := gdn.Config{
		HiddenSize: 8, NumKeyHeads: 2, NumValueHeads: 4,
		KeyHeadDim: 4, ValueHeadDim: 4, ConvKernel: 4, Eps: 1e-6,
	}
	const batch, seq = 2, 3
	return requireGdnParity(gdCase{
		cfg: cfg, backend: core.BackendCPUTiled, batch: batch, seq: seq,
		packed: randInit(1, gdnWeightCount(cfg), 0x47444e01),
		x:      randInit(batch*seq, cfg.HiddenSize, 0x47444e02),
	})
}

func caseGdnAlt() error {
	cfg := gdn.Config{
		HiddenSize: 6, NumKeyHeads: 1, NumValueHeads: 2,
		KeyHeadDim: 4, ValueHeadDim: 4, ConvKernel: 4, Eps: 1e-6,
	}
	const batch, seq = 2, 4
	return requireGdnParity(gdCase{
		cfg: cfg, backend: core.BackendCPUTiled, batch: batch, seq: seq,
		packed: randInit(1, gdnWeightCount(cfg), 0x47444e03),
		x:      randInit(batch*seq, cfg.HiddenSize, 0x47444e04),
	})
}

func caseGdnSpeed() error {
	// Mid shape for 3-way timing; keep short enough that vsGo stays ≥industry.
	const batch, seq, benchIters = 2, 8, 120
	cfg := gdn.Config{
		HiddenSize: 48, NumKeyHeads: 2, NumValueHeads: 4,
		KeyHeadDim: 16, ValueHeadDim: 16, ConvKernel: 4, Eps: 1e-6,
	}
	if err := cfg.Validate(); err != nil {
		return err
	}
	fc := gdCase{
		cfg: cfg, backend: core.BackendCPUTiled, batch: batch, seq: seq,
		packed: randInit(1, gdnWeightCount(cfg), 0xd7a4e0f1),
		x:      randInit(batch*seq, cfg.HiddenSize, 0xd7a4e0f2),
	}
	metalTool, err := findMetalFwd()
	if err != nil {
		return err
	}
	cppTool, cppErr := findCppFwd()
	haveCpp := cppErr == nil

	goPre, goPost, err := runGoGdn(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	mPre, mPost, err := runMetalGdn(fc)
	if err != nil {
		return fmt.Errorf("metal: %w", err)
	}
	preM := suites.CompareF32Spectrum(goPre, mPre)
	postM := suites.CompareF32Spectrum(goPost, mPost)
	m := postM
	if preM.Band < m.Band {
		m = preM
	}

	fmt.Printf("\n      Go vs Metal GDN Float32 — ≥industry vs Go; exact_bit vs C++ (not ×34)")
	fmt.Printf("\n      shape B=%d T=%d H=%d nk=%d nv=%d hdK=%d hdV=%d  iters=%d",
		batch, seq, cfg.HiddenSize, cfg.NumKeyHeads, cfg.NumValueHeads, cfg.KeyHeadDim, cfg.ValueHeadDim, benchIters)
	if haveCpp {
		fmt.Printf("\n      speed: Go / C++ / Metal (host f32 matvec)")
	}
	fmt.Printf("\n      %-12s %-12s %7s %9s %9s %9s %8s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "mtl_µs", "winner", "vsGo")
	fmt.Printf("      %s\n", strings.Repeat("-", 100))

	goUS, err := benchGoUS(func() error {
		_, _, e := runGoGdn(fc)
		return e
	}, benchIters)
	if err != nil {
		return fmt.Errorf("go bench: %w", err)
	}

	cppUS := 0.0
	cppStr := "—"
	if haveCpp {
		_, _, cppDur, err := runToolGdn(cppTool, fc, benchIters)
		if err != nil {
			cppStr = "err"
		} else {
			cppUS = perIterUS(cppDur, benchIters)
			cppStr = FormatUS(cppUS)
		}
	}

	_, _, metalDur, err := runToolGdn(metalTool, fc, benchIters)
	if err != nil {
		return fmt.Errorf("metal bench: %w", err)
	}
	metalUS := perIterUS(metalDur, benchIters)

	names := []string{"Go", "Metal"}
	uss := []float64{goUS, metalUS}
	if haveCpp && cppUS > 0 {
		names = []string{"Go", "C++", "Metal"}
		uss = []float64{goUS, cppUS, metalUS}
	}
	winner, _, vsGo := pickFastest(names, uss)
	vsGoStr := "baseline"
	if winner != "Go" {
		vsGoStr = fmt.Sprintf("%s %+0.1f%%", winner, vsGo)
	}
	fmt.Printf("      %-12s %-12s %6.2f%% %9s %9s %9s %8s %s\n",
		"float32", m.Band, m.ExactPct, FormatUS(goUS), cppStr, FormatUS(metalUS), winner, vsGoStr)
	fmt.Printf("      %s\n", strings.Repeat("-", 100))
	fmt.Printf("      note: GDN host f32; Metal≡C++; Go↔C++ historically industry/strict (scan FP)")
	if m.Band < suites.DetIndustry {
		return fmt.Errorf("vsGo below industry: %s", suites.FormatSpectrum(m))
	}
	if haveCpp {
		_, cPost, _, err := runToolGdn(cppTool, fc, 0)
		if err != nil {
			return fmt.Errorf("cpp parity: %w", err)
		}
		cM := suites.CompareF32Spectrum(cPost, mPost)
		fmt.Printf("\n      vsCpp post %s", suites.FormatSpectrum(cM))
		if cM.Band != suites.DetExactBit {
			return fmt.Errorf("Metal≠C++: %s", suites.FormatSpectrum(cM))
		}
	}
	return nil
}
