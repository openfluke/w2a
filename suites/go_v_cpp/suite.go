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
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/simd"
)

type Case struct {
	Name string
	Run  func() error
}

func CasesCore() []Case {
	return []Case{
		{Name: "Locate welvet_dense_fwd binary", Run: caseLocateTool},
		{Name: "Spectrum ladder self-check", Run: caseSpectrumLadder},
		{Name: "Dense forward identity+bias Float32 CPU — exact bits", Run: caseIdentityBiasExact},
		{Name: "Dense forward ReLU Float32 CPU — exact bits", Run: caseReLUExact},
		{Name: "CPU FormatNone × all 34 dtypes — spectrum + speed", Run: caseAllDTypesCPU},
		{Name: "SIMD FormatNone × all 34 dtypes — Go SIMD vs C++ SIMD", Run: caseAllDTypesSIMD},
		{Name: "Cross: Go CPU vs C++ SIMD Float32 — spectrum report", Run: caseCrossCPUVsSIMD},
		{Name: "Architecture topology 2×3×4×2 + remote link — Go vs C++", Run: caseArchTopology},
		{Name: "Volume Dense 1×1×1×2 chain — exact bits", Run: caseVolumeDenseChain},
		{Name: "Volume Dense 2×1×1×1 remote hop — exact bits", Run: caseVolumeRemoteHop},
		{Name: "Volume Dense 3×3×3×1 × all 34 dtypes — spectrum + speed", Run: caseVolumeDense3x3x3AllDTypes},
		{Name: "C++ gap census present (welvet_dense_fwd --gaps)", Run: caseGapCensus},
	}
}

func Cases() []Case {
	out := append(CasesCore(), CasesMHA()...)
	out = append(out, CasesSwiGLU()...)
	out = append(out, CasesRMSNorm()...)
	out = append(out, CasesLayerNorm()...)
	out = append(out, CasesCNN()...)
	out = append(out, CasesConvT()...)
	out = append(out, CasesRNN()...)
	out = append(out, CasesLSTM()...)
	out = append(out, CasesEmbedding()...)
	out = append(out, CasesSoftmax()...)
	out = append(out, CasesKMeans()...)
	out = append(out, CasesSequential()...)
	out = append(out, CasesResidual()...)
	out = append(out, CasesMamba()...)
	return append(out, CasesMetacognition()...)
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
	out, _ := exec.Command(p, "--version").CombinedOutput()
	fmt.Printf("\n      tool=%s version=%s", p, strings.TrimSpace(string(out)))
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
	exact := []float32{1, 2, 3}
	m := suites.CompareF32Spectrum(exact, exact)
	if m.Band != suites.DetExactBit || m.ExactPct != 100 {
		return fmt.Errorf("exact: %s", suites.FormatSpectrum(m))
	}
	fmt.Printf("\n      ladder: broken=0%% … industry=60%% … strict=80%% … exact_bit=100%% (aim)")
	fmt.Printf("\n      %s", suites.FormatSpectrum(m))
	return nil
}

type fwdCase struct {
	in, out, batch int
	act            core.ActivationType
	dtype          core.DType
	backend        core.Backend // Go + C++ (0=cpu, 1=simd)
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
	l.Exec.Backend = fc.backend
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
	_, _ = in.Write([]byte("WVD3"))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.batch))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.in))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.out))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.act))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.dtype))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.backend))
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
		act: core.ActivationLinear, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		W: []float32{1, 0, 0, 0, 1, 0}, bias: []float64{0.5, -0.5}, x: []float32{2, 3, 9},
	})
}

func caseReLUExact() error {
	return requireExact(fwdCase{
		in: 4, out: 3, batch: 1,
		act: core.ActivationReLU, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		W: []float32{1, -1, 0.5, 0, 0, 1, -2, 0.25, -0.5, 0, 1, 1},
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
	bias = []float64{0.25, -0.125, 0, 1.5, -2}
	if out < len(bias) {
		bias = bias[:out]
	}
	return W, x, bias
}

func runDtypeMatrix(be core.Backend, title string, requireExactBits bool) error {
	const in, out, batch, benchIters = 16, 8, 2, 40
	if len(core.AllDTypes) != 34 {
		return fmt.Errorf("expected 34 dtypes, got %d", len(core.AllDTypes))
	}
	W, x, bias := makeSharedInit(out, in, batch)

	fmt.Printf("\n      %s\n", title)
	fmt.Printf("      %-12s %-12s %8s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "spect%", "go_µs", "cpp_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 88))

	var (
		exactN, totalN int
		worstBand      = suites.DetExactBit
		fails          []string
	)

	for _, dt := range core.AllDTypes {
		fc := fwdCase{
			in: in, out: out, batch: batch,
			act: core.ActivationLinear, dtype: dt, backend: be,
			W: W, bias: bias, x: x,
		}
		preM, postM, err := compareCase(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			fmt.Printf("      %-12s FAIL %v\n", dt, err)
			continue
		}
		m := postM
		if preM.Band < m.Band {
			m = preM
		}
		if m.Band < worstBand {
			worstBand = m.Band
		}
		exactN += m.ExactBits
		totalN += m.N

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
				fmt.Sprintf("%.2f", float64(goDur.Nanoseconds())/float64(benchIters)/1000),
				"?", "?", fmt.Sprintf("%.3g", m.MaxAbs))
			continue
		}
		// Per-iter µs from ns total (ms is too coarse for these kernels).
		goUS := float64(goDur.Nanoseconds()) / float64(benchIters) / 1000
		cppUS := float64(cppDur.Nanoseconds()) / float64(benchIters) / 1000
		spdPct := suites.SpeedPctDiff(goUS, cppUS)

		fmt.Printf("      %-12s %-12s %7.2f%% %7.0f%% %10.2f %10.2f %+8.1f%% %s\n",
			dt, m.Band, m.ExactPct, m.Band.Percent(), goUS, cppUS, spdPct,
			fmt.Sprintf("%.3g", m.MaxAbs))

		if requireExactBits {
			if m.Band != suites.DetExactBit {
				fails = append(fails, fmt.Sprintf("%s: want exact_bit got %s", dt, suites.FormatSpectrum(m)))
			}
		} else if m.Band == suites.DetBroken || m.Band == suites.DetBallpark {
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

func caseAllDTypesCPU() error {
	return runDtypeMatrix(core.BackendCPUTiled, "Go CPU vs C++ CPU (FormatNone×34)", true)
}

func caseAllDTypesSIMD() error {
	if !simd.Enabled() {
		return fmt.Errorf("Go SIMD not enabled on %s/%s — cannot compare SIMD paths", runtime.GOOS, runtime.GOARCH)
	}
	// SIMD vs SIMD: aim exact_bit; if imported kernels diverge, report spectrum (fail below industry).
	return runDtypeMatrix(core.BackendSIMD, "Go SIMD vs C++ imported SIMD (FormatNone×34)", false)
}

func caseCrossCPUVsSIMD() error {
	const in, out, batch = 32, 16, 2
	W, x, bias := makeSharedInit(out, in, batch)
	fcCPU := fwdCase{
		in: in, out: out, batch: batch,
		act: core.ActivationLinear, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		W: W, bias: bias, x: x,
	}
	fcSIMD := fcCPU
	fcSIMD.backend = core.BackendSIMD

	goPre, goPost, err := runGo(fcCPU)
	if err != nil {
		return fmt.Errorf("go cpu: %w", err)
	}
	cppPre, cppPost, err := runCpp(fcSIMD)
	if err != nil {
		return fmt.Errorf("cpp simd: %w", err)
	}
	preM := suites.CompareF32Spectrum(goPre, cppPre)
	postM := suites.CompareF32Spectrum(goPost, cppPost)
	fmt.Printf("\n      Go-CPU vs C++-SIMD Float32: pre %s", suites.FormatSpectrum(preM))
	fmt.Printf("\n      Go-CPU vs C++-SIMD Float32: post %s", suites.FormatSpectrum(postM))
	// Informational cross-path — fail only if broken.
	if postM.Band == suites.DetBroken {
		return fmt.Errorf("cross path broken: %s", suites.FormatSpectrum(postM))
	}
	return nil
}

func runCppArch(depth, rows, cols, lpc int, links [][8]int32) ([][10]int32, error) {
	tool, err := findDenseFwd()
	if err != nil {
		return nil, err
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("WVA1"))
	_ = binary.Write(&in, binary.LittleEndian, int32(depth))
	_ = binary.Write(&in, binary.LittleEndian, int32(rows))
	_ = binary.Write(&in, binary.LittleEndian, int32(cols))
	_ = binary.Write(&in, binary.LittleEndian, int32(lpc))
	_ = binary.Write(&in, binary.LittleEndian, int32(len(links)))
	for _, L := range links {
		for _, v := range L {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
	}
	cmd := exec.Command(tool)
	cmd.Stdin = bytes.NewReader(in.Bytes())
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("cpp arch: %v: %s", err, stderr.String())
	}
	raw := stdout.Bytes()
	if len(raw) < 8 || string(raw[:4]) != "WVAO" {
		return nil, fmt.Errorf("cpp arch bad magic")
	}
	r := bytes.NewReader(raw[4:])
	var n int32
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return nil, err
	}
	out := make([][10]int32, n)
	for i := range out {
		for j := 0; j < 10; j++ {
			if err := binary.Read(r, binary.LittleEndian, &out[i][j]); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}

func caseArchTopology() error {
	const depth, rows, cols, lpc = 2, 3, 4, 2
	links := [][8]int32{{1, 0, 0, 0, 0, 0, 0, 0}}
	g := architecture.NewGrid(depth, rows, cols, lpc)
	if err := g.SetRemoteLink(1, 0, 0, 0, 0, 0, 0, 0); err != nil {
		return err
	}
	cppRows, err := runCppArch(depth, rows, cols, lpc, links)
	if err != nil {
		return err
	}
	hops := g.HopOrder()
	if len(cppRows) != len(hops) {
		return fmt.Errorf("n_cells go=%d cpp=%d", len(hops), len(cppRows))
	}
	for i, c := range hops {
		wantIdx := int32(g.Index(c.Z, c.Y, c.X, c.L))
		row := cppRows[i]
		if row[0] != int32(c.Z) || row[1] != int32(c.Y) || row[2] != int32(c.X) || row[3] != int32(c.L) {
			return fmt.Errorf("hop %d coord go=%v cpp=%v", i, c, row[:4])
		}
		if row[4] != wantIdx {
			return fmt.Errorf("hop %d index go=%d cpp=%d", i, wantIdx, row[4])
		}
		cell := g.At(c.Z, c.Y, c.X, c.L)
		wantRemote := int32(0)
		var tz, ty, tx, tl int32
		if cell != nil && cell.IsRemoteLink {
			wantRemote = 1
			tz, ty, tx, tl = int32(cell.TargetZ), int32(cell.TargetY), int32(cell.TargetX), int32(cell.TargetL)
		}
		if row[5] != wantRemote || row[6] != tz || row[7] != ty || row[8] != tx || row[9] != tl {
			return fmt.Errorf("hop %d remote go=(%d %d %d %d %d) cpp=%v", i, wantRemote, tz, ty, tx, tl, row[5:])
		}
	}
	fmt.Printf("\n      cells=%d remote@z1 exact match", len(hops))
	return nil
}

type volCell struct {
	enabled bool
	in, out int
	act     core.ActivationType
	dtype   core.DType
	W       []float32
	bias    []float64
}

func encodeVolumeReq(depth, rows, cols, lpc, batch int, backend core.Backend, links [][8]int32, cells []volCell, x []float32) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("WVF1"))
	_ = binary.Write(&in, binary.LittleEndian, int32(depth))
	_ = binary.Write(&in, binary.LittleEndian, int32(rows))
	_ = binary.Write(&in, binary.LittleEndian, int32(cols))
	_ = binary.Write(&in, binary.LittleEndian, int32(lpc))
	_ = binary.Write(&in, binary.LittleEndian, int32(batch))
	_ = binary.Write(&in, binary.LittleEndian, int32(backend))
	_ = binary.Write(&in, binary.LittleEndian, int32(len(links)))
	for _, L := range links {
		for _, v := range L {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
	}
	if len(cells) != depth*rows*cols*lpc {
		return nil, fmt.Errorf("cells len %d != stack %d", len(cells), depth*rows*cols*lpc)
	}
	for _, c := range cells {
		en := int32(0)
		if c.enabled {
			en = 1
		}
		_ = binary.Write(&in, binary.LittleEndian, en)
		if !c.enabled {
			continue
		}
		_ = binary.Write(&in, binary.LittleEndian, int32(c.in))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.out))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.act))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.dtype))
		for _, v := range c.W {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
		_ = binary.Write(&in, binary.LittleEndian, uint32(len(c.bias)))
		for _, v := range c.bias {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
	}
	for _, v := range x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	return in.Bytes(), nil
}

func decodeVolumeResp(raw []byte) (out []float32, bench time.Duration, err error) {
	if len(raw) >= 4+8+4 && string(raw[:4]) == "WVB1" {
		r := bytes.NewReader(raw[4:])
		var ns int64
		var iters int32
		if err := binary.Read(r, binary.LittleEndian, &ns); err != nil {
			return nil, 0, err
		}
		if err := binary.Read(r, binary.LittleEndian, &iters); err != nil {
			return nil, 0, err
		}
		rest, err := io.ReadAll(r)
		if err != nil {
			return nil, 0, err
		}
		raw = rest
		bench = time.Duration(ns)
	}
	if len(raw) < 8 || string(raw[:4]) != "WVFO" {
		return nil, bench, fmt.Errorf("cpp volume bad magic")
	}
	r := bytes.NewReader(raw[4:])
	var n int32
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return nil, bench, err
	}
	out = make([]float32, n)
	for i := range out {
		if err := binary.Read(r, binary.LittleEndian, &out[i]); err != nil {
			return nil, bench, err
		}
	}
	return out, bench, nil
}

func runCppVolume(depth, rows, cols, lpc, batch int, backend core.Backend, links [][8]int32, cells []volCell, x []float32) ([]float32, error) {
	tool, err := findDenseFwd()
	if err != nil {
		return nil, err
	}
	payload, err := encodeVolumeReq(depth, rows, cols, lpc, batch, backend, links, cells, x)
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(tool)
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("cpp volume: %v: %s", err, stderr.String())
	}
	out, _, err := decodeVolumeResp(stdout.Bytes())
	return out, err
}

func runCppVolumeBench(depth, rows, cols, lpc, batch int, backend core.Backend, links [][8]int32, cells []volCell, x []float32, iters int) ([]float32, time.Duration, error) {
	tool, err := findDenseFwd()
	if err != nil {
		return nil, 0, err
	}
	payload, err := encodeVolumeReq(depth, rows, cols, lpc, batch, backend, links, cells, x)
	if err != nil {
		return nil, 0, err
	}
	cmd := exec.Command(tool, fmt.Sprintf("--bench=%d", iters))
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, 0, fmt.Errorf("cpp volume bench: %v: %s", err, stderr.String())
	}
	return decodeVolumeResp(stdout.Bytes())
}

func caseVolumeDenseChain() error {
	const batch = 1
	// 1×1×1×2: Dense 4→3 then 3→2
	W0 := []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}
	W1 := []float32{1, 0, 0, 0, 1, 0}
	bias0 := []float64{0.1, -0.2, 0.05}
	bias1 := []float64{0.5, -0.25}
	x := []float32{1, 2, 3, 4}

	g := architecture.NewGrid(1, 1, 1, 2)
	d0, err := dense.NewConfigured(4, 3, core.ActivationReLU, core.DTypeFloat32, quant.FormatNone, W0)
	if err != nil {
		return err
	}
	d0.Weights.Bias = append([]float64(nil), bias0...)
	d0.Exec.Backend = core.BackendCPUTiled
	d0.Exec.MultiCore = false
	d1, err := dense.NewConfigured(3, 2, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, W1)
	if err != nil {
		return err
	}
	d1.Weights.Bias = append([]float64(nil), bias1...)
	d1.Exec.Backend = core.BackendCPUTiled
	d1.Exec.MultiCore = false
	if err := g.BindOp(0, 0, 0, 0, core.Layer{
		Type: core.LayerDense, DType: core.DTypeFloat32, Activation: core.ActivationReLU,
		InputHeight: 4, OutputHeight: 3,
	}, d0); err != nil {
		return err
	}
	if err := g.BindOp(0, 0, 0, 1, core.Layer{
		Type: core.LayerDense, DType: core.DTypeFloat32, Activation: core.ActivationLinear,
		InputHeight: 3, OutputHeight: 2,
	}, d1); err != nil {
		return err
	}

	tin := core.NewTensor[float32](batch, 4)
	copy(tin.Data, x)
	res, err := forward.Forward(g, tin)
	if err != nil {
		return fmt.Errorf("go volume: %w", err)
	}

	cells := []volCell{
		{enabled: true, in: 4, out: 3, act: core.ActivationReLU, dtype: core.DTypeFloat32, W: W0, bias: bias0},
		{enabled: true, in: 3, out: 2, act: core.ActivationLinear, dtype: core.DTypeFloat32, W: W1, bias: bias1},
	}
	cppOut, err := runCppVolume(1, 1, 1, 2, batch, core.BackendCPUTiled, nil, cells, x)
	if err != nil {
		return err
	}
	m := suites.CompareF32Spectrum(res.Output.Data, cppOut)
	fmt.Printf("\n      %s", suites.FormatSpectrum(m))
	if m.Band != suites.DetExactBit {
		return fmt.Errorf("volume chain: %s", suites.FormatSpectrum(m))
	}
	return nil
}

func caseVolumeRemoteHop() error {
	const batch = 1
	// 2×1×1×1: cell z0 Dense 3→2; cell z1 remote←z0 then Dense 2→2
	W0 := []float32{1, 0, 0, 0, 1, 0}
	W1 := []float32{0.5, 0, 0, 0.5}
	bias0 := []float64{0.25, -0.25}
	bias1 := []float64{1, 2}
	x := []float32{2, 3, 4}

	g := architecture.NewGrid(2, 1, 1, 1)
	d0, err := dense.NewConfigured(3, 2, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, W0)
	if err != nil {
		return err
	}
	d0.Weights.Bias = append([]float64(nil), bias0...)
	d0.Exec.Backend = core.BackendCPUTiled
	d0.Exec.MultiCore = false
	d1, err := dense.NewConfigured(2, 2, core.ActivationReLU, core.DTypeFloat32, quant.FormatNone, W1)
	if err != nil {
		return err
	}
	d1.Weights.Bias = append([]float64(nil), bias1...)
	d1.Exec.Backend = core.BackendCPUTiled
	d1.Exec.MultiCore = false
	if err := g.BindOp(0, 0, 0, 0, core.Layer{
		Type: core.LayerDense, DType: core.DTypeFloat32, Activation: core.ActivationLinear,
		InputHeight: 3, OutputHeight: 2,
	}, d0); err != nil {
		return err
	}
	if err := g.BindOp(1, 0, 0, 0, core.Layer{
		Type: core.LayerDense, DType: core.DTypeFloat32, Activation: core.ActivationReLU,
		InputHeight: 2, OutputHeight: 2,
	}, d1); err != nil {
		return err
	}
	if err := g.SetRemoteLink(1, 0, 0, 0, 0, 0, 0, 0); err != nil {
		return err
	}

	tin := core.NewTensor[float32](batch, 3)
	copy(tin.Data, x)
	res, err := forward.Forward(g, tin)
	if err != nil {
		return fmt.Errorf("go volume: %w", err)
	}

	links := [][8]int32{{1, 0, 0, 0, 0, 0, 0, 0}}
	cells := []volCell{
		{enabled: true, in: 3, out: 2, act: core.ActivationLinear, dtype: core.DTypeFloat32, W: W0, bias: bias0},
		{enabled: true, in: 2, out: 2, act: core.ActivationReLU, dtype: core.DTypeFloat32, W: W1, bias: bias1},
	}
	cppOut, err := runCppVolume(2, 1, 1, 1, batch, core.BackendCPUTiled, links, cells, x)
	if err != nil {
		return err
	}
	m := suites.CompareF32Spectrum(res.Output.Data, cppOut)
	fmt.Printf("\n      %s", suites.FormatSpectrum(m))
	if m.Band != suites.DetExactBit {
		return fmt.Errorf("volume remote: %s", suites.FormatSpectrum(m))
	}
	return nil
}

// buildVolume3x3x3 makes a shared 27-cell Dense chain (same W/bias/x) for dtype dt.
func buildVolume3x3x3(dt core.DType, be core.Backend) (g *architecture.Grid, cells []volCell, x []float32, err error) {
	const depth, rows, cols, lpc, dim = 3, 3, 3, 1, 4
	g = architecture.NewGrid(depth, rows, cols, lpc)
	var s uint32 = 0x33333333
	next := func() float32 {
		s = s*1664525 + 1013904223
		return float32(int32(s%401)-200) / 100.0
	}
	x = make([]float32, dim)
	for i := range x {
		x[i] = next()
	}
	cells = make([]volCell, 0, depth*rows*cols*lpc)
	for _, coord := range g.HopOrder() {
		W := make([]float32, dim*dim)
		for i := range W {
			W[i] = next()
		}
		bias := []float64{float64(next()), float64(next()), float64(next()), float64(next())}
		act := core.ActivationLinear
		if (coord.Z+coord.Y+coord.X)%2 == 1 {
			act = core.ActivationReLU
		}
		dl, err := dense.NewConfigured(dim, dim, act, dt, quant.FormatNone, W)
		if err != nil {
			return nil, nil, nil, err
		}
		dl.Weights.Bias = append([]float64(nil), bias...)
		dl.Exec.Backend = be
		dl.Exec.MultiCore = false
		if err := g.BindOp(coord.Z, coord.Y, coord.X, coord.L, core.Layer{
			Type: core.LayerDense, DType: dt, Activation: act,
			InputHeight: dim, OutputHeight: dim,
		}, dl); err != nil {
			return nil, nil, nil, err
		}
		cells = append(cells, volCell{
			enabled: true, in: dim, out: dim, act: act, dtype: dt,
			W: W, bias: bias,
		})
	}
	return g, cells, x, nil
}

func caseVolumeDense3x3x3AllDTypes() error {
	const (
		depth, rows, cols, lpc = 3, 3, 3, 1
		batch, benchIters      = 1, 20
	)
	if len(core.AllDTypes) != 34 {
		return fmt.Errorf("expected 34 dtypes, got %d", len(core.AllDTypes))
	}
	be := core.BackendCPUTiled
	nCells := depth * rows * cols * lpc

	fmt.Printf("\n      Go CPU vs C++ CPU — volume Dense 3×3×3×1 (cells=%d) FormatNone×34\n", nCells)
	fmt.Printf("      %-12s %-12s %8s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "spect%", "go_µs", "cpp_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 88))

	var (
		exactN, totalN int
		worstBand      = suites.DetExactBit
		fails          []string
	)

	for _, dt := range core.AllDTypes {
		g, cells, x, err := buildVolume3x3x3(dt, be)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s build: %v", dt, err))
			fmt.Printf("      %-12s FAIL %v\n", dt, err)
			continue
		}
		tin := core.NewTensor[float32](batch, len(x))
		copy(tin.Data, x)
		res, err := forward.Forward(g, tin)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		cppOut, err := runCppVolume(depth, rows, cols, lpc, batch, be, nil, cells, x)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp: %v", dt, err))
			fmt.Printf("      %-12s FAIL cpp %v\n", dt, err)
			continue
		}
		m := suites.CompareF32Spectrum(res.Output.Data, cppOut)
		if m.Band < worstBand {
			worstBand = m.Band
		}
		exactN += m.ExactBits
		totalN += m.N

		for i := 0; i < 2; i++ {
			if _, err := forward.Forward(g, tin); err != nil {
				fails = append(fails, fmt.Sprintf("%s warm: %v", dt, err))
				break
			}
		}
		t0 := time.Now()
		for i := 0; i < benchIters; i++ {
			if _, err := forward.Forward(g, tin); err != nil {
				fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
				break
			}
		}
		goDur := time.Since(t0)
		_, cppDur, err := runCppVolumeBench(depth, rows, cols, lpc, batch, be, nil, cells, x, benchIters)
		goUS := float64(goDur.Nanoseconds()) / float64(benchIters) / 1000
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp bench: %v", dt, err))
			fmt.Printf("      %-12s %-12s %7.2f%% %7.0f%% %10.2f %10s %9s %s\n",
				dt, m.Band, m.ExactPct, m.Band.Percent(), goUS, "?", "?",
				fmt.Sprintf("%.3g", m.MaxAbs))
			continue
		}
		cppUS := float64(cppDur.Nanoseconds()) / float64(benchIters) / 1000
		spdPct := suites.SpeedPctDiff(goUS, cppUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %7.0f%% %10.2f %10.2f %+8.1f%% %s\n",
			dt, m.Band, m.ExactPct, m.Band.Percent(), goUS, cppUS, spdPct,
			fmt.Sprintf("%.3g", m.MaxAbs))

		if m.Band != suites.DetExactBit {
			fails = append(fails, fmt.Sprintf("%s: want exact_bit got %s", dt, suites.FormatSpectrum(m)))
		}
	}

	overallExact := 0.0
	if totalN > 0 {
		overallExact = 100.0 * float64(exactN) / float64(totalN)
	}
	fmt.Printf("      %s\n", strings.Repeat("-", 88))
	fmt.Printf("      overall exact_bits=%.2f%%  worst_band=%s (%.0f%%)  aim=exact_bit (100%%)\n",
		overallExact, worstBand, worstBand.Percent())
	fmt.Printf("      volume=3×3×3×1 hops=%d  spd%%Δ = (cpp−go)/go×100  (negative ⇒ C++ faster)", nCells)

	if len(fails) > 0 {
		return fmt.Errorf("%d volume-dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}
