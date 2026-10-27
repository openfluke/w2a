// Package go_v_cuda compares Go Welvet vs welvet.cuda for numerical parity
// (Go is the determinism baseline) and times Go / C++ / CUDA to pick a winner.
//
// How the experiment runs today:
//   w2a spawns welvet.cuda's welvet_dense_fwd as a subprocess over the WVD3/WVO1
//   binary wire (same pattern as go_v_cpp ↔ welvet.cpp). CUDA is not linked into
//   Welvet yet — eventually the CUDA engine will ship as a compiled binary inside
//   Welvet the same way welvet.cpp does; packaging is unchanged for now.
package go_v_cuda

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
		{Name: "Locate welvet.cuda welvet_dense_fwd", Run: caseLocateCUDA},
		{Name: "Locate welvet.cpp welvet_dense_fwd (optional speed)", Run: caseLocateCppOptional},
		{Name: "CUDA device identity", Run: caseCUDAIdentity},
		{Name: "Spectrum ladder self-check", Run: caseSpectrumLadder},
		{Name: "Dense forward identity+bias Float32 CUDA — exact bits vs Go", Run: caseIdentityBiasExact},
		{Name: "Dense forward ReLU Float32 CUDA — exact bits vs Go", Run: caseReLUExact},
		{Name: "CUDA FormatNone ×34 — exact_bit + Go/C++/CUDA timings", Run: caseAllDTypesExact},
	}
}

func RunAll() error {
	var fails []string
	for i, c := range Cases() {
		suites.BeginCase()
		fmt.Printf("  [%d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cuda", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cuda", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cuda: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func RunDenseForward() error { return RunAll() }

func findCUDAFwd() (string, error) {
	if p := os.Getenv("WELVET_CUDA_DENSE_FWD"); p != "" {
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			return p, nil
		}
		return "", fmt.Errorf("WELVET_CUDA_DENSE_FWD=%q not found", p)
	}
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("runtime.Caller failed")
	}
	w2aRoot := filepath.Clean(filepath.Join(filepath.Dir(thisFile), "../.."))
	candidates := []string{
		filepath.Join(w2aRoot, "../welvet.cuda/build/welvet_dense_fwd"),
		filepath.Join(w2aRoot, "../welvet.cuda/build/Release/welvet_dense_fwd"),
	}
	for _, c := range candidates {
		if st, err := os.Stat(c); err == nil && !st.IsDir() {
			return c, nil
		}
	}
	return "", fmt.Errorf("welvet.cuda welvet_dense_fwd not found; build welvet.cuda or set WELVET_CUDA_DENSE_FWD")
}

func findCppFwd() (string, error) {
	if p := os.Getenv("WELVET_DENSE_FWD"); p != "" {
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			return p, nil
		}
	}
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("runtime.Caller failed")
	}
	w2aRoot := filepath.Clean(filepath.Join(filepath.Dir(thisFile), "../.."))
	candidates := []string{
		filepath.Join(w2aRoot, "../welvet.cpp/build/welvet_dense_fwd"),
		filepath.Join(w2aRoot, "../welvet.cpp/build/Release/welvet_dense_fwd"),
	}
	for _, c := range candidates {
		if st, err := os.Stat(c); err == nil && !st.IsDir() {
			return c, nil
		}
	}
	return "", fmt.Errorf("welvet.cpp tool not found")
}

func caseLocateCUDA() error {
	p, err := findCUDAFwd()
	if err != nil {
		return err
	}
	out, _ := exec.Command(p, "--version").CombinedOutput()
	fmt.Printf("\n      cuda_tool=%s version=%s", p, strings.TrimSpace(string(out)))
	return nil
}

func caseLocateCppOptional() error {
	p, err := findCppFwd()
	if err != nil {
		fmt.Printf("\n      cpp_tool=missing (%v) — speed table will omit C++", err)
		return nil
	}
	out, _ := exec.Command(p, "--version").CombinedOutput()
	fmt.Printf("\n      cpp_tool=%s version=%s", p, strings.TrimSpace(string(out)))
	return nil
}

func caseCUDAIdentity() error {
	p, err := findCUDAFwd()
	if err != nil {
		return err
	}
	out, err := exec.Command(p, "--identity").CombinedOutput()
	if err != nil {
		return fmt.Errorf("identity: %v: %s", err, out)
	}
	fmt.Printf("\n      %s", strings.TrimSpace(string(out)))
	return nil
}

func caseSpectrumLadder() error {
	exact := []float32{1, 2, 3}
	m := suites.CompareF32Spectrum(exact, exact)
	if m.Band != suites.DetExactBit || m.ExactPct != 100 {
		return fmt.Errorf("exact: %s", suites.FormatSpectrum(m))
	}
	fmt.Printf("\n      aim=exact_bit vs Go; spd on large GEMV picks fastest warm-exec among Go/C++/CUDA")
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

func mountGo(fc fwdCase) (*dense.Layer, error) {
	l, err := dense.NewConfigured(fc.in, fc.out, fc.act, fc.dtype, quant.FormatNone, fc.W)
	if err != nil {
		return nil, err
	}
	l.Weights.Bias = append([]float64(nil), fc.bias...)
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	return l, nil
}

func execGo(l *dense.Layer, fc fwdCase) (pre, post []float32, err error) {
	t := core.NewTensor[float32](fc.batch, fc.in)
	copy(t.Data, fc.x)
	preT, postT, err := dense.Forward(l, t)
	if err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), preT.Data...), append([]float32(nil), postT.Data...), nil
}

func runGo(fc fwdCase) (pre, post []float32, err error) {
	l, err := mountGo(fc)
	if err != nil {
		return nil, nil, err
	}
	return execGo(l, fc)
}

func encodeReq(fc fwdCase) []byte {
	var in bytes.Buffer
	_, _ = in.Write([]byte("WVD3"))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.batch))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.in))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.out))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.act))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.dtype))
	_ = binary.Write(&in, binary.LittleEndian, int32(0)) // backend ignored on CUDA
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

func decodeResp(raw []byte, n int) (pre, post []float32, err error) {
	if len(raw) < 4 || string(raw[:4]) != "WVO1" {
		return nil, nil, fmt.Errorf("bad magic: %q", string(raw[:min(4, len(raw))]))
	}
	need := 4 + 2*n*4
	if len(raw) < need {
		return nil, nil, fmt.Errorf("short output: %d < %d", len(raw), need)
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

func runTool(tool string, fc fwdCase) (pre, post []float32, err error) {
	cmd := exec.Command(tool)
	cmd.Stdin = bytes.NewReader(encodeReq(fc))
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, nil, fmt.Errorf("%v: %s", err, stderr.String())
	}
	return decodeResp(stdout.Bytes(), fc.batch*fc.out)
}

// runCUDABench — WVB2: mount once, N warm executes. Returns per-iter mount/exec µs.
func runCUDABench(tool string, fc fwdCase, iters int) (pre, post []float32, mountUS, execUS float64, err error) {
	cmd := exec.Command(tool, fmt.Sprintf("--bench=%d", iters))
	cmd.Stdin = bytes.NewReader(encodeReq(fc))
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, nil, 0, 0, fmt.Errorf("%v: %s", err, stderr.String())
	}
	raw := stdout.Bytes()
	if len(raw) < 4+8+8+4 || string(raw[:4]) != "WVB2" {
		return nil, nil, 0, 0, fmt.Errorf("bench bad header (want WVB2): %q", string(raw[:min(4, len(raw))]))
	}
	var mountNS, execNS int64
	var gotIters int32
	r := bytes.NewReader(raw[4:])
	if err := binary.Read(r, binary.LittleEndian, &mountNS); err != nil {
		return nil, nil, 0, 0, err
	}
	if err := binary.Read(r, binary.LittleEndian, &execNS); err != nil {
		return nil, nil, 0, 0, err
	}
	if err := binary.Read(r, binary.LittleEndian, &gotIters); err != nil {
		return nil, nil, 0, 0, err
	}
	rest, err := io.ReadAll(r)
	if err != nil {
		return nil, nil, 0, 0, err
	}
	pre, post, err = decodeResp(rest, fc.batch*fc.out)
	mountUS = float64(mountNS) / 1000.0 // one-shot mount
	execUS = perIterUS(time.Duration(execNS), int(gotIters))
	return pre, post, mountUS, execUS, err
}

// runCppBench — welvet.cpp still emits WVB1 (fused). Treated as exec_µs only.
func runCppBench(tool string, fc fwdCase, iters int) (pre, post []float32, execUS float64, err error) {
	cmd := exec.Command(tool, fmt.Sprintf("--bench=%d", iters))
	cmd.Stdin = bytes.NewReader(encodeReq(fc))
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, nil, 0, fmt.Errorf("%v: %s", err, stderr.String())
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
	return pre, post, perIterUS(time.Duration(ns), int(gotIters)), err
}

func runCUDA(fc fwdCase) (pre, post []float32, err error) {
	tool, err := findCUDAFwd()
	if err != nil {
		return nil, nil, err
	}
	return runTool(tool, fc)
}

func requireExactVsGo(fc fwdCase) error {
	goPre, goPost, err := runGo(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	mPre, mPost, err := runCUDA(fc)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	preM := suites.CompareF32Spectrum(goPre, mPre)
	postM := suites.CompareF32Spectrum(goPost, mPost)
	if preM.Band != suites.DetExactBit {
		return fmt.Errorf("pre %s", suites.FormatSpectrum(preM))
	}
	if postM.Band != suites.DetExactBit {
		return fmt.Errorf("post %s", suites.FormatSpectrum(postM))
	}
	return nil
}

func caseIdentityBiasExact() error {
	return requireExactVsGo(fwdCase{
		in: 3, out: 2, batch: 1,
		act: core.ActivationLinear, dtype: core.DTypeFloat32,
		W: []float32{1, 0, 0, 0, 1, 0}, bias: []float64{0.5, -0.5}, x: []float32{2, 3, 9},
	})
}

func caseReLUExact() error {
	return requireExactVsGo(fwdCase{
		in: 4, out: 3, batch: 1,
		act: core.ActivationReLU, dtype: core.DTypeFloat32,
		W:    []float32{1, -1, 0.5, 0, 0, 1, -2, 0.25, -0.5, 0, 1, 1},
		bias: []float64{0.1, -0.2, 0}, x: []float32{-1, 2, -3, 4},
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
	bias = make([]float64, out)
	for i := range bias {
		bias[i] = float64(next()) * 0.01
	}
	return W, x, bias
}

func pickFastest(names []string, us []float64) (winner string, best float64, vsGoPct float64) {
	best = us[0]
	winner = names[0]
	for i := 1; i < len(us); i++ {
		if us[i] > 0 && (best <= 0 || us[i] < best) {
			best = us[i]
			winner = names[i]
		}
	}
	goUS := us[0]
	if goUS > 1e-12 && best > 0 {
		vsGoPct = (goUS - best) / goUS * 100
	}
	return winner, best, vsGoPct
}

// Large-enough GEMV so CUDA warm-exec beats Go/C++ (dispatch amortized).
// Native f64 MatVec matches Go f64 accumulate across all 34 FormatNone dtypes.
func caseAllDTypesExact() error {
	const in, out, batch, benchIters = 256, 256, 8, 300
	if len(core.AllDTypes) != 34 {
		return fmt.Errorf("expected 34 dtypes, got %d", len(core.AllDTypes))
	}
	W, x, bias := makeSharedInit(out, in, batch)
	cudaTool, err := findCUDAFwd()
	if err != nil {
		return err
	}
	cppTool, cppErr := findCppFwd()
	haveCpp := cppErr == nil

	fmt.Printf("\n      Go (baseline) vs CUDA FormatNone×34 — exact_bit required")
	if haveCpp {
		fmt.Printf("\n      speed: Go / C++ / CUDA — winner on warm cuda_exec (not mount)")
	} else {
		fmt.Printf("\n      speed: Go / CUDA (C++ tool missing)")
	}
	fmt.Printf("\n      shape batch=%d in=%d out=%d  MACs/fwd≈%d  iters=%d\n",
		batch, in, out, batch*in*out, benchIters)
	fmt.Printf("      %-12s %-12s %7s %9s %9s %10s %10s %8s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "cuda_mount", "cuda_exec", "winner", "vsGo")
	fmt.Printf("      %s\n", strings.Repeat("-", 110))

	var fails []string
	var exactN, totalN int
	winCount := map[string]int{}

	for _, dt := range core.AllDTypes {
		fc := fwdCase{
			in: in, out: out, batch: batch,
			act: core.ActivationLinear, dtype: dt,
			W: W, bias: bias, x: x,
		}
		goPre, goPost, err := runGo(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		mPre, mPost, err := runCUDA(fc)
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

		goLayer, err := mountGo(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go mount: %v", dt, err))
			continue
		}
		goUS, err := benchGoUS(func() error {
			_, _, e := execGo(goLayer, fc)
			return e
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}

		cppUS := 0.0
		cppStr := "—"
		if haveCpp {
			_, _, cppDur, err := runCppBench(cppTool, fc, benchIters)
			if err != nil {
				cppStr = "err"
			} else {
				cppUS = cppDur
				cppStr = FormatUS(cppUS)
			}
		}

		_, _, cudaMountUS, cudaExecUS, err := runCUDABench(cudaTool, fc, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cuda bench: %v", dt, err))
			fmt.Printf("      %-12s %-12s %6.2f%% %9s %9s %10s %10s %8s %s\n",
				dt, m.Band, m.ExactPct, FormatUS(goUS), cppStr, "?", "?", "?", err)
			continue
		}

		names := []string{"Go", "CUDA"}
		uss := []float64{goUS, cudaExecUS}
		if haveCpp && cppUS > 0 {
			names = []string{"Go", "C++", "CUDA"}
			uss = []float64{goUS, cppUS, cudaExecUS}
		}
		winner, _, vsGo := pickFastest(names, uss)
		winCount[winner]++

		vsGoStr := "baseline"
		if winner != "Go" {
			vsGoStr = fmt.Sprintf("%s %+0.1f%%", winner, vsGo)
		}

		fmt.Printf("      %-12s %-12s %6.2f%% %9s %9s %10s %10s %8s %s\n",
			dt, m.Band, m.ExactPct, FormatUS(goUS), cppStr,
			FormatUS(cudaMountUS), FormatUS(cudaExecUS), winner, vsGoStr)

		if m.Band != suites.DetExactBit {
			fails = append(fails, fmt.Sprintf("%s: want exact_bit got %s", dt, suites.FormatSpectrum(m)))
		}
	}

	overallExact := 0.0
	if totalN > 0 {
		overallExact = 100.0 * float64(exactN) / float64(totalN)
	}
	fmt.Printf("      %s\n", strings.Repeat("-", 110))
	fmt.Printf("      overall exact_bits=%.2f%%  (aim=100%% vs Go)\n", overallExact)
	parts := []string{}
	for _, n := range []string{"Go", "C++", "CUDA"} {
		if winCount[n] > 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", n, winCount[n]))
		}
	}
	fmt.Printf("      wins(exec): %s\n", strings.Join(parts, " "))
	fmt.Printf("      note: cuda_mount = decode+upload once; cuda_exec = warm resident forward (winner uses exec)")
	fmt.Printf("\n      note: NVIDIA CUDA uses HW float64 MatVec to match Go f64 accumulate")

	if len(fails) > 0 {
		return fmt.Errorf("%d dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}
