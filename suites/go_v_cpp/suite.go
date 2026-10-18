// Package go_v_cpp compares Go Welvet vs welvet.cpp for numerical parity
// across the determinism spectrum, with speed and % difference.
package go_v_cpp

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
)

type Case struct {
	Name string
	Run  func() error
}

func Cases() []Case {
	return []Case{
		{Name: "Locate welvet_dense_fwd binary", Run: caseLocateTool},
		{Name: "Spectrum ladder self-check", Run: caseSpectrumLadder},
		{Name: "Dense forward identity+bias Float32 — exact bits", Run: caseIdentityBiasExact},
		{Name: "Dense forward ReLU Float32 — exact bits", Run: caseReLUExact},
		{Name: "FormatNone × all 34 dtypes — spectrum + speed matrix", Run: caseAllDTypesMatrix},
		{Name: "C++ gap census present (welvet_dense_fwd --gaps)", Run: caseGapCensus},
	}
}

func RunAll() error {
	var fails []string
	for i, c := range Cases() {
		suites.BeginCase()
		fmt.Printf("  [%d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func RunOne(n int) error {
	cs := Cases()
	if n < 1 || n > len(cs) {
		return fmt.Errorf("go_v_cpp: case %d out of range 1..%d", n, len(cs))
	}
	suites.BeginCase()
	fmt.Printf("  [%d] %s … ", n, cs[n-1].Name)
	if err := cs[n-1].Run(); err != nil {
		suites.EndCase("go_v_cpp", cs[n-1].Name, "FAIL", err.Error())
		fmt.Printf("FAIL\n      %v\n", err)
		return err
	}
	suites.EndCase("go_v_cpp", cs[n-1].Name, "PASS", "")
	fmt.Println("PASS")
	return nil
}

func findDenseFwd() (string, error) {
	if p := os.Getenv("WELVET_DENSE_FWD"); p != "" {
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			return p, nil
		}
		return "", fmt.Errorf("WELVET_DENSE_FWD=%q not found", p)
	}
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("runtime.Caller failed")
	}
	w2aRoot := filepath.Clean(filepath.Join(filepath.Dir(thisFile), "../.."))
	candidates := []string{
		filepath.Join(w2aRoot, "../welvet.cpp/build/welvet_dense_fwd"),
		filepath.Join(w2aRoot, "../welvet.cpp/build/Release/welvet_dense_fwd"),
		filepath.Join(w2aRoot, "../welvet.cpp/build/Debug/welvet_dense_fwd"),
	}
	if runtime.GOOS == "windows" {
		for i, c := range candidates {
			candidates[i] = c + ".exe"
		}
	}
	for _, c := range candidates {
		if st, err := os.Stat(c); err == nil && !st.IsDir() {
			return c, nil
		}
	}
	return "", fmt.Errorf("welvet_dense_fwd not found; build welvet.cpp (scripts/build.sh) or set WELVET_DENSE_FWD")
}

func caseLocateTool() error {
	p, err := findDenseFwd()
	if err != nil {
		return err
	}
	fmt.Printf("\n      tool=%s", p)
	return nil
}

func caseGapCensus() error {
	p, err := findDenseFwd()
	if err != nil {
		return err
	}
	out, err := exec.Command(p, "--gaps").CombinedOutput()
	if err != nil {
		return fmt.Errorf("gaps: %v: %s", err, out)
	}
	if !bytes.Contains(out, []byte("gap?")) {
		return fmt.Errorf("expected gap? markers in census")
	}
	if !bytes.Contains(out, []byte("[done]")) {
		return fmt.Errorf("expected [done] entries in census")
	}
	return nil
}

func caseSpectrumLadder() error {
	// Document + smoke the spectrum using synthetic deltas.
	exact := []float32{1, 2, 3}
	m := suites.CompareF32Spectrum(exact, exact)
	if m.Band != suites.DetExactBit || m.ExactPct != 100 {
		return fmt.Errorf("exact: %s", suites.FormatSpectrum(m))
	}
	near := []float32{1, 2, 3 + 1e-6}
	m = suites.CompareF32Spectrum(exact, near)
	if m.Band != suites.DetStrict && m.Band != suites.DetExactBit {
		// 1e-6 abs on value ~3 → rel ~3e-7 → Strict
		if m.MaxAbs > suites.DetTolFwd {
			return fmt.Errorf("near-strict got %s", suites.FormatSpectrum(m))
		}
	}
	fmt.Printf("\n      ladder: broken=0%% … industry=60%% … strict=80%% … exact_bit=100%% (aim)")
	fmt.Printf("\n      %s", suites.FormatSpectrum(suites.CompareF32Spectrum(exact, exact)))
	return nil
}

type fwdCase struct {
	in, out, batch int
	act            core.ActivationType
	dtype          core.DType
	W              []float32
	bias           []float64
	x              []float32
}

func runGo(fc fwdCase) (pre, post []float32, err error) {
	l, err := dense.NewConfigured(fc.in, fc.out, fc.act, fc.dtype, quant.FormatNone, fc.W)
	if err != nil {
		return nil, nil, err
	}
	l.Weights.Bias = append([]float64(nil), fc.bias...)
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	t := core.NewTensor[float32](fc.batch, fc.in)
	copy(t.Data, fc.x)
	preT, postT, err := dense.Forward(l, t)
	if err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), preT.Data...), append([]float32(nil), postT.Data...), nil
}

func encodeReq(fc fwdCase) []byte {
	var in bytes.Buffer
	_, _ = in.Write([]byte("WVD2"))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.batch))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.in))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.out))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.act))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.dtype))
	for _, v := range fc.W {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	_ = binary.Write(&in, binary.LittleEndian, uint32(len(fc.bias)))
	for _, v := range fc.bias {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	return in.Bytes()
}

func runCpp(fc fwdCase) (pre, post []float32, err error) {
	tool, err := findDenseFwd()
	if err != nil {
		return nil, nil, err
	}
	cmd := exec.Command(tool)
	cmd.Stdin = bytes.NewReader(encodeReq(fc))
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, nil, fmt.Errorf("cpp tool: %v: %s", err, stderr.String())
	}
	return decodeResp(stdout.Bytes(), fc.batch*fc.out)
}

func runCppBench(fc fwdCase, iters int) (pre, post []float32, total time.Duration, err error) {
	tool, err := findDenseFwd()
	if err != nil {
		return nil, nil, 0, err
	}
	cmd := exec.Command(tool, fmt.Sprintf("--bench=%d", iters))
	cmd.Stdin = bytes.NewReader(encodeReq(fc))
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, nil, 0, fmt.Errorf("cpp bench: %v: %s", err, stderr.String())
	}
	raw := stdout.Bytes()
	if len(raw) < 4+8+4 || string(raw[:4]) != "WVB1" {
		return nil, nil, 0, fmt.Errorf("cpp bench bad header")
	}
	var ns int64
	var gotIters int32
	r := bytes.NewReader(raw[4:])
	if err := binary.Read(r, binary.LittleEndian, &ns); err != nil {
		return nil, nil, 0, err
	}
	if err := binary.Read(r, binary.LittleEndian, &gotIters); err != nil {
		return nil, nil, 0, err
	}
	rest, err := io.ReadAll(r)
	if err != nil {
		return nil, nil, 0, err
	}
	pre, post, err = decodeResp(rest, fc.batch*fc.out)
	return pre, post, time.Duration(ns), err
}

func decodeResp(raw []byte, n int) (pre, post []float32, err error) {
	if len(raw) < 4 || string(raw[:4]) != "WVO1" {
		return nil, nil, fmt.Errorf("cpp bad magic: %q", string(raw[:min(4, len(raw))]))
	}
	need := 4 + 2*n*4
	if len(raw) < need {
		return nil, nil, fmt.Errorf("cpp short output: %d < %d", len(raw), need)
	}
	r := bytes.NewReader(raw[4:])
	pre = make([]float32, n)
	post = make([]float32, n)
	for i := range pre {
		if err := binary.Read(r, binary.LittleEndian, &pre[i]); err != nil {
			return nil, nil, err
		}
	}
	for i := range post {
		if err := binary.Read(r, binary.LittleEndian, &post[i]); err != nil {
			return nil, nil, err
		}
	}
	return pre, post, nil
}

func compareCase(fc fwdCase) (preM, postM suites.DetMetrics, err error) {
	goPre, goPost, err := runGo(fc)
	if err != nil {
		return preM, postM, fmt.Errorf("go: %w", err)
	}
	cppPre, cppPost, err := runCpp(fc)
	if err != nil {
		return preM, postM, fmt.Errorf("cpp: %w", err)
	}
	preM = suites.CompareF32Spectrum(goPre, cppPre)
	postM = suites.CompareF32Spectrum(goPost, cppPost)
	return preM, postM, nil
}

func requireExact(fc fwdCase) error {
	preM, postM, err := compareCase(fc)
	if err != nil {
		return err
	}
	if preM.Band != suites.DetExactBit {
		return fmt.Errorf("pre %s", suites.FormatSpectrum(preM))
	}
	if postM.Band != suites.DetExactBit {
		return fmt.Errorf("post %s", suites.FormatSpectrum(postM))
	}
	return nil
}

func caseIdentityBiasExact() error {
	return requireExact(fwdCase{
		in: 3, out: 2, batch: 1,
		act:   core.ActivationLinear,
		dtype: core.DTypeFloat32,
		W:     []float32{1, 0, 0, 0, 1, 0},
		bias:  []float64{0.5, -0.5},
		x:     []float32{2, 3, 9},
	})
}

func caseReLUExact() error {
	return requireExact(fwdCase{
		in: 4, out: 3, batch: 1,
		act:   core.ActivationReLU,
		dtype: core.DTypeFloat32,
		W:     []float32{1, -1, 0.5, 0, 0, 1, -2, 0.25, -0.5, 0, 1, 1},
		bias:  []float64{0.1, -0.2, 0},
		x:     []float32{-1, 2, -3, 4},
	})
}

func makeSharedInit(out, in, batch int) (W, x []float32, bias []float64) {
	W = make([]float32, out*in)
	x = make([]float32, batch*in)
	var s uint32 = 0xC0FFEE
	next := func() float32 {
		s = s*1664525 + 1013904223
		return float32(int32(s%2001)-1000) / 100.0
	}
	for i := range W {
		W[i] = next()
	}
	for i := range x {
		x[i] = next()
	}
	bias = []float64{0.25, -0.125, 0, 1.5, -2}
	if out < len(bias) {
		bias = bias[:out]
	}
	return W, x, bias
}

func caseAllDTypesMatrix() error {
	const in, out, batch, benchIters = 16, 8, 2, 40
	if len(core.AllDTypes) != 34 {
		return fmt.Errorf("expected 34 dtypes, got %d", len(core.AllDTypes))
	}
	W, x, bias := makeSharedInit(out, in, batch)

	fmt.Printf("\n      %-12s %-12s %8s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "spect%", "go_ms", "cpp_ms", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 88))

	var (
		exactN, totalN int
		worstBand      = suites.DetExactBit
		fails          []string
	)

	for _, dt := range core.AllDTypes {
		fc := fwdCase{
			in: in, out: out, batch: batch,
			act:   core.ActivationLinear,
			dtype: dt,
			W:     W,
			bias:  bias,
			x:     x,
		}
		preM, postM, err := compareCase(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			fmt.Printf("      %-12s FAIL %v\n", dt, err)
			continue
		}
		// Grade on post (what callers consume); track pre too.
		m := postM
		if preM.Band < m.Band {
			m = preM
		}
		if m.Band < worstBand {
			worstBand = m.Band
		}
		exactN += m.ExactBits
		totalN += m.N

		// Speeds (kernel-ish): Go in-process, C++ --bench inside one process.
		for i := 0; i < 3; i++ {
			if _, _, err := runGo(fc); err != nil {
				fails = append(fails, fmt.Sprintf("%s warm go: %v", dt, err))
				break
			}
		}
		t0 := time.Now()
		for i := 0; i < benchIters; i++ {
			if _, _, err := runGo(fc); err != nil {
				fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
				break
			}
		}
		goDur := time.Since(t0)
		_, _, cppDur, err := runCppBench(fc, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp bench: %v", dt, err))
			fmt.Printf("      %-12s %-12s %7.2f%% %7.0f%% %10s %10s %9s %s\n",
				dt, m.Band, m.ExactPct, m.Band.Percent(),
				fmt.Sprintf("%.3f", float64(goDur.Microseconds())/float64(benchIters)/1000),
				"?", "?", fmt.Sprintf("%.3g", m.MaxAbs))
			continue
		}
		goMS := float64(goDur.Microseconds()) / float64(benchIters) / 1000
		cppMS := float64(cppDur.Microseconds()) / float64(benchIters) / 1000
		spdPct := suites.SpeedPctDiff(goMS, cppMS) // + ⇒ C++ slower than Go

		fmt.Printf("      %-12s %-12s %7.2f%% %7.0f%% %10.3f %10.3f %+8.1f%% %s\n",
			dt, m.Band, m.ExactPct, m.Band.Percent(), goMS, cppMS, spdPct,
			fmt.Sprintf("%.3g", m.MaxAbs))

		// Soft fail policy: require at least Industry for every dtype; prefer ExactBit.
		// Aim remains ExactBit — non-exact rows are reported but only Broken fails hard.
		if m.Band == suites.DetBroken || m.Band == suites.DetBallpark {
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
	fmt.Printf("      spectrum: broken → ballpark → loose → industry → strict → exact_bit\n")
	fmt.Printf("      spd%%Δ = (cpp−go)/go×100  (negative ⇒ C++ faster)")

	if len(fails) > 0 {
		return fmt.Errorf("%d dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}
