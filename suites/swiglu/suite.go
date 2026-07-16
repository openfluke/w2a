package swiglu

import (
	"fmt"
	"math"
	"runtime"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/swiglu"
	"github.com/openfluke/welvet/webgpu"
)

type Case struct {
	Name string
	Run  func() error
}

func Cases() []Case {
	return []Case{
		{Name: "Forward SiLU-gate smoke (identity-ish)", Run: forwardSmoke},
		{Name: "Seq layout [batch,seq,in] smoke", Run: seqLayoutSmoke},
		{Name: "Backward finite-diff dWg spot-check", Run: backwardFiniteDiff},
		{Name: "Grad verify — CPU vs SIMD agreement", Run: gradVerifyBackends},
		{Name: "WebGPU hard-errors without device (no host fake)", Run: webGPUNoDevice},
		{Name: "CPU tiled FormatNone × all 34 dtypes (fwd+bwd)", Run: cpuTiledFormatNoneAll},
		{Name: "ACTIVATION sweep — all core.Numeric Tensor[T] × CPU/SIMD/WebGPU", Run: ActNumericSweep},
		{Name: "TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU", Run: TimedMatrix},
		{Name: "TIMED matrix — all quants × CPU/SIMD/WebGPU (Float32)", Run: TimedQuantMatrix},
		{Name: "SIMD FormatNone × all 34 dtypes (fwd+bwd)", Run: simdFormatNoneAll},
		{Name: "SIMD+WebGPU all quant formats (fwd+bwd, Float32)", Run: simdWebGPUAllQuants},
		{Name: "GAP CENSUS — full matrix (prints gaps; always PASS until v1 gate)", Run: fullMatrixGaps},
		{Name: "TRAIN volumetric — FormatNone × ALL 34 dtypes × CPU/SIMD/WebGPU × 1³/2³/3³", Run: TimedTrainGridsFormatNone},
		{Name: "TRAIN volumetric — ALL 20 quants × CPU/SIMD/WebGPU × 1³/2³/3³", Run: TimedTrainGridsQuant},
	}
}

func RunAll() error {
	var fails []string
	for i, c := range Cases() {
		suites.BeginCase()
		fmt.Printf("  [%d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("swiglu", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("swiglu", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("swiglu: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func RunOne(n int) error {
	cs := Cases()
	if n < 1 || n > len(cs) {
		return fmt.Errorf("swiglu: case %d out of range 1..%d", n, len(cs))
	}
	suites.BeginCase()
	fmt.Printf("  [%d] %s … ", n, cs[n-1].Name)
	if err := cs[n-1].Run(); err != nil {
		suites.EndCase("swiglu", cs[n-1].Name, "FAIL", err.Error())
		fmt.Printf("FAIL\n      %v\n", err)
		return err
	}
	suites.EndCase("swiglu", cs[n-1].Name, "PASS", "")
	fmt.Println("PASS")
	return nil
}

func tinyCfg() swiglu.Config { return swiglu.Config{InputDim: 8, IntermediateDim: 16} }
func defaultCfg() swiglu.Config {
	return swiglu.Config{InputDim: 32, IntermediateDim: 64}
}

func eyeInit(rows, cols int) []float32 {
	w := make([]float32, rows*cols)
	n := rows
	if cols < n {
		n = cols
	}
	for i := 0; i < n; i++ {
		w[i*cols+i] = 1
	}
	return w
}

func newLayer(cfg swiglu.Config, dt core.DType, format quant.Format, be core.Backend) (*swiglu.Layer, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	in, inter := cfg.InputDim, cfg.IntermediateDim
	l, err := swiglu.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone,
		eyeInit(inter, in), eyeInit(inter, in), eyeInit(in, inter))
	if err != nil {
		return nil, err
	}
	if dt != core.DTypeFloat32 {
		if err := l.SetDType(dt); err != nil {
			return nil, err
		}
	}
	if format != quant.FormatNone {
		if err := l.Pack(format); err != nil {
			return nil, err
		}
	}
	l.Exec.Backend = be
	l.Exec.MultiCore = false
	return l, nil
}

func makeInput(batch, in int) *core.Tensor[float32] {
	x := core.NewTensor[float32](batch, in)
	for i := range x.Data {
		x.Data[i] = float32((i%7)-3) * 0.1
	}
	return x
}

func smoke(dt core.DType, format quant.Format, be core.Backend) error {
	l, err := newLayer(tinyCfg(), dt, format, be)
	if err != nil {
		return err
	}
	x := makeInput(2, tinyCfg().InputDim)
	pre, post, err := swiglu.Forward(l, x)
	if err != nil {
		return fmt.Errorf("fwd: %w", err)
	}
	g := core.NewTensor[float32](post.Shape...)
	for i := range g.Data {
		g.Data[i] = 1
	}
	_, _, err = swiglu.Backward(l, g, x, pre)
	if err != nil {
		return fmt.Errorf("bwd: %w", err)
	}
	return nil
}

func smokeForwardOnly(dt core.DType, format quant.Format, be core.Backend) error {
	l, err := newLayer(tinyCfg(), dt, format, be)
	if err != nil {
		return err
	}
	_, _, err = swiglu.Forward(l, makeInput(2, tinyCfg().InputDim))
	return err
}

func forwardSmoke() error {
	l, err := newLayer(tinyCfg(), core.DTypeFloat32, quant.FormatNone, core.BackendCPUTiled)
	if err != nil {
		return err
	}
	x := makeInput(1, 8)
	_, post, err := swiglu.Forward(l, x)
	if err != nil {
		return err
	}
	var sum float64
	for _, v := range post.Data {
		sum += math.Abs(float64(v))
	}
	if sum == 0 {
		return fmt.Errorf("all-zero output")
	}
	fmt.Printf("(in=8 inter=16) ")
	return nil
}

func seqLayoutSmoke() error {
	l, err := newLayer(tinyCfg(), core.DTypeFloat32, quant.FormatNone, core.BackendCPUTiled)
	if err != nil {
		return err
	}
	x := core.NewTensor[float32](2, 3, 8)
	for i := range x.Data {
		x.Data[i] = float32((i%5)-2) * 0.1
	}
	_, post, err := swiglu.Forward(l, x)
	if err != nil {
		return err
	}
	if len(post.Shape) != 3 || post.Shape[0] != 2 || post.Shape[1] != 3 || post.Shape[2] != 8 {
		return fmt.Errorf("post shape %v", post.Shape)
	}
	fmt.Printf("([2,3,8]) ")
	return nil
}

func backwardFiniteDiff() error {
	l, err := newLayer(tinyCfg(), core.DTypeFloat32, quant.FormatNone, core.BackendCPUTiled)
	if err != nil {
		return err
	}
	x := makeInput(1, 8)
	pre, post, err := swiglu.Forward(l, x)
	if err != nil {
		return err
	}
	g := core.NewTensor[float32](post.Shape...)
	for i := range g.Data {
		g.Data[i] = 2 * post.Data[i]
	}
	_, dW, err := swiglu.Backward(l, g, x, pre)
	if err != nil {
		return err
	}
	master, ok := l.Gate.Weights.MasterF32()
	if !ok || len(master) == 0 {
		return fmt.Errorf("no MasterF32")
	}
	eps := float32(1e-3)
	idx := 0
	orig := master[idx]
	lossAt := func() (float64, error) {
		_, p, err := swiglu.Forward(l, x)
		if err != nil {
			return 0, err
		}
		var loss float64
		for _, v := range p.Data {
			loss += float64(v) * float64(v)
		}
		return loss, nil
	}
	master[idx] = orig + eps
	lossP, err := lossAt()
	if err != nil {
		return err
	}
	master[idx] = orig - eps
	lossM, err := lossAt()
	if err != nil {
		return err
	}
	master[idx] = orig
	fd := (lossP - lossM) / float64(2*eps)
	an := float64(dW.Data[idx])
	if fd*an < 0 && math.Abs(fd) > 1e-3 && math.Abs(an) > 1e-3 {
		return fmt.Errorf("finite-diff sign mismatch fd=%v analytic=%v", fd, an)
	}
	fmt.Printf("(fd≈%.4g an≈%.4g) ", fd, an)
	return nil
}

func gradVerifyBackends() error {
	if !simd.Enabled() {
		fmt.Printf("(SIMD off on %s — CPU-only) ", runtime.GOARCH)
		return nil
	}
	cfg := tinyCfg()
	x := makeInput(2, cfg.InputDim)
	run := func(be core.Backend) (*core.Tensor[float32], error) {
		l, err := newLayer(cfg, core.DTypeFloat32, quant.FormatNone, be)
		if err != nil {
			return nil, err
		}
		pre, post, err := swiglu.Forward(l, x)
		if err != nil {
			return nil, err
		}
		g := core.NewTensor[float32](post.Shape...)
		for i := range g.Data {
			g.Data[i] = 1
		}
		_, dW, err := swiglu.Backward(l, g, x, pre)
		return dW, err
	}
	dCPU, err := run(core.BackendCPUTiled)
	if err != nil {
		return err
	}
	dSIMD, err := run(core.BackendSIMD)
	if err != nil {
		return err
	}
	var max float64
	for i := range dCPU.Data {
		e := math.Abs(float64(dCPU.Data[i] - dSIMD.Data[i]))
		if e > max {
			max = e
		}
	}
	if max > 1e-2 {
		return fmt.Errorf("CPU↔SIMD dW max err %v", max)
	}
	fmt.Printf("(maxΔ=%.3g) ", max)
	return nil
}

func webGPUNoDevice() error {
	if webgpu.Available() {
		fmt.Printf("(device present — skip negative check) ")
		return nil
	}
	l, err := newLayer(tinyCfg(), core.DTypeFloat32, quant.FormatNone, core.BackendWebGPU)
	if err != nil {
		return err
	}
	_, _, err = swiglu.Forward(l, makeInput(1, 8))
	if err == nil {
		return fmt.Errorf("expected hard error without WebGPU device")
	}
	fmt.Printf("(got error as required) ")
	return nil
}

func cpuTiledFormatNoneAll() error {
	var fails []string
	for _, dt := range core.AllDTypes {
		if err := smoke(dt, quant.FormatNone, core.BackendCPUTiled); err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
		}
	}
	fmt.Printf("(%d FormatNone) ", len(core.AllDTypes))
	if len(fails) > 0 {
		return fmt.Errorf("%d failed: %s", len(fails), strings.Join(fails[:min(6, len(fails))], " | "))
	}
	return nil
}

func simdFormatNoneAll() error {
	if !simd.Enabled() {
		return fmt.Errorf("Plan 9 SIMD not enabled on %s", runtime.GOARCH)
	}
	var fails []string
	for _, dt := range core.AllDTypes {
		if err := smoke(dt, quant.FormatNone, core.BackendSIMD); err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
		}
	}
	fmt.Printf("(%d FormatNone SIMD) ", len(core.AllDTypes))
	if len(fails) > 0 {
		return fmt.Errorf("%d failed: %s", len(fails), strings.Join(fails[:min(6, len(fails))], " | "))
	}
	return nil
}

func simdWebGPUAllQuants() error {
	var fails []string
	backends := []core.Backend{core.BackendSIMD, core.BackendWebGPU}
	for _, f := range quant.AllFormats {
		if f == quant.FormatNone {
			continue
		}
		for _, be := range backends {
			if be == core.BackendSIMD && !simd.Enabled() {
				fails = append(fails, fmt.Sprintf("%s/%s: SIMD not enabled", f, be))
				continue
			}
			if be == core.BackendWebGPU && !webgpu.Available() {
				continue
			}
			if err := smoke(core.DTypeFloat32, f, be); err != nil {
				fails = append(fails, fmt.Sprintf("%s/%s: %v", f, be, err))
			}
		}
	}
	fmt.Printf("(quant×backend) ")
	if len(fails) > 0 {
		return fmt.Errorf("%d failed: %s", len(fails), strings.Join(fails[:min(8, len(fails))], " | "))
	}
	return nil
}

func fullMatrixGaps() error {
	perms := swiglu.AllPermutations()
	var failN int
	var samples []string
	for _, p := range perms {
		var err error
		if p.Backend == core.BackendWebGPU {
			err = smokeForwardOnly(p.DType, p.Format, p.Backend)
		} else {
			err = smoke(p.DType, p.Format, p.Backend)
		}
		status := "OK"
		note := ""
		if err != nil {
			failN++
			status = "GAP"
			note = err.Error()
			if len(samples) < 6 {
				samples = append(samples, fmt.Sprintf("%s/%s/%s", p.DType, p.Format, p.Backend))
			}
		}
		rec("census", p.DType.String(), p.Format.String(), p.Backend.String(), "-", status, note)
	}
	ok := len(perms) - failN
	fmt.Printf("(%d cells, %d ok, %d gaps; e.g. %s) ", len(perms), ok, failN, strings.Join(samples, ", "))
	return nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func failOrGap(be core.Backend) string {
	if be == core.BackendCPUTiled {
		return "FAIL"
	}
	return "GAP"
}

func fmtNs(ns int64) string {
	if ns <= 0 {
		return "-"
	}
	if ns < 1e3 {
		return fmt.Sprintf("%dns", ns)
	}
	if ns < 1e6 {
		return fmt.Sprintf("%.1fµs", float64(ns)/1e3)
	}
	if ns < 1e9 {
		return fmt.Sprintf("%.2fms", float64(ns)/1e6)
	}
	return fmt.Sprintf("%.2fs", float64(ns)/1e9)
}
