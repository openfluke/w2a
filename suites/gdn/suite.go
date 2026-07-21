package gdn

import (
	"fmt"
	"math"
	"runtime"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/layers/seqmix"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/webgpu"
)

type Case struct {
	Name string
	Run  func() error
}

func Cases() []Case {
	return []Case{
		{Name: "ForwardDecode smoke", Run: decodeSmoke},
		{Name: "Forward [B,T,H] smoke", Run: seqSmoke},
		{Name: "seqmix KindLinearAttn", Run: kindSmoke},
		{Name: "Backward shapes + GradWSize", Run: bwdSmoke},
		{Name: "Backward matches finite-diff (Out proj, exact)", Run: bwdGradCheckOut},
		{Name: "ApplyGradSGD reduces loss", Run: sgdReducesLoss},
		{Name: "Grad verify — CPU vs SIMD agreement", Run: gradVerifyBackends},
		{Name: "WebGPU ForwardWebGPU/BackwardWebGPU gate on device", Run: webGPUNote},
		{Name: "CPU FormatNone × all dtypes (Float32-only; gaps recorded)", Run: cpuFormatNoneAll},
		{Name: "ACTIVATION sweep — all core.Numeric Tensor[T] × CPU/SIMD/WebGPU", Run: ActNumericSweep},
		{Name: "TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU", Run: TimedMatrix},
		{Name: "TIMED matrix — None/BinaryPacked × CPU/SIMD/WebGPU", Run: TimedQuantMatrix},
		{Name: "SIMD FormatNone × all dtypes (Float32-only; gaps recorded)", Run: simdFormatNoneAll},
		{Name: "SIMD+WebGPU None/BinaryPacked (fwd+bwd)", Run: simdWebGPUQuants},
		{Name: "GAP CENSUS — supported GDN permutations", Run: fullMatrixGaps},
		{Name: "TRAIN volumetric — FormatNone × dtypes × backends × 1³/2³/3³", Run: TimedTrainGridsFormatNone},
		{Name: "TRAIN volumetric — None/BinaryPacked × backends × 1³/2³/3³", Run: TimedTrainGridsQuant},
	}
}

func RunAll() error {
	var fails []string
	for i, c := range Cases() {
		suites.BeginCase()
		fmt.Printf("  [%d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("gdn", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("gdn", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("gdn: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func RunOne(n int) error {
	cs := Cases()
	if n < 1 || n > len(cs) {
		return fmt.Errorf("gdn: case %d out of range 1..%d", n, len(cs))
	}
	suites.BeginCase()
	fmt.Printf("  [%d] %s … ", n, cs[n-1].Name)
	if err := cs[n-1].Run(); err != nil {
		suites.EndCase("gdn", cs[n-1].Name, "FAIL", err.Error())
		fmt.Printf("FAIL\n      %v\n", err)
		return err
	}
	suites.EndCase("gdn", cs[n-1].Name, "PASS", "")
	fmt.Println("PASS")
	return nil
}

func cfg() gdn.Config {
	return gdn.Config{HiddenSize: 16, NumKeyHeads: 2, NumValueHeads: 2, KeyHeadDim: 4, ValueHeadDim: 4, ConvKernel: 2, Eps: 1e-6}
}

func matrixCfg() gdn.Config {
	return gdn.Config{HiddenSize: 32, NumKeyHeads: 2, NumValueHeads: 2, KeyHeadDim: 8, ValueHeadDim: 8, ConvKernel: 2, Eps: 1e-6}
}

func newLayer(c gdn.Config, dt core.DType, format quant.Format, be core.Backend) (*gdn.Layer, error) {
	if !gdn.PermutationOK(dt, format, be) {
		return nil, fmt.Errorf("unsupported: %s/%s/%s", dt, format, be)
	}
	cd := c.NumKeyHeads*c.KeyHeadDim*2 + c.NumValueHeads*c.ValueHeadDim
	vd := c.NumValueHeads * c.ValueHeadDim
	seed := func(n int, scale float32) []float32 {
		w := make([]float32, n)
		for i := range w {
			w[i] = float32((i%7)-3) * scale
		}
		return w
	}
	l, err := gdn.NewConfigured(c,
		seed(cd*c.HiddenSize, 0.03), seed(vd*c.HiddenSize, 0.02),
		seed(c.NumValueHeads*c.HiddenSize, 0.01), seed(c.NumValueHeads*c.HiddenSize, 0.01),
		seed(c.HiddenSize*vd, 0.03), seed(cd*c.ConvKernel, 0.05),
		seed(c.NumValueHeads, 0.01), seed(c.NumValueHeads, 0.01),
		func() []float32 {
			g := make([]float32, c.ValueHeadDim)
			for i := range g {
				g[i] = 1
			}
			return g
		}())
	if err != nil {
		return nil, err
	}
	if err := l.Pack(format); err != nil {
		return nil, err
	}
	l.Exec.Backend = be
	l.Exec.MultiCore = false
	return l, nil
}

func makeInput(c gdn.Config, batch int) *core.Tensor[float32] {
	x := core.NewTensor[float32](batch, 4, c.HiddenSize)
	for i := range x.Data {
		x.Data[i] = float32((i%7)-3) * 0.1
	}
	return x
}

func smoke(dt core.DType, format quant.Format, be core.Backend) error {
	c := matrixCfg()
	l, err := newLayer(c, dt, format, be)
	if err != nil {
		return err
	}
	x := makeInput(c, 2)
	pre, post, err := gdn.Forward(l, x)
	if err != nil {
		return fmt.Errorf("fwd: %w", err)
	}
	gy := core.NewTensor[float32](post.Shape...)
	for i := range gy.Data {
		gy.Data[i] = 1
	}
	_, _, err = gdn.Backward(l, gy, x, pre)
	if err != nil {
		return fmt.Errorf("bwd: %w", err)
	}
	return nil
}

func smokeForwardOnly(dt core.DType, format quant.Format, be core.Backend) error {
	l, err := newLayer(matrixCfg(), dt, format, be)
	if err != nil {
		return err
	}
	_, _, err = gdn.Forward(l, makeInput(matrixCfg(), 2))
	return err
}

func decodeSmoke() error {
	l, err := gdn.New(cfg())
	if err != nil {
		return err
	}
	x := make([]float32, 16)
	y := make([]float32, 16)
	for i := range x {
		x[i] = 0.01 * float32(i)
	}
	return l.ForwardDecode(x, y)
}

func seqSmoke() error {
	l, err := gdn.New(cfg())
	if err != nil {
		return err
	}
	x := core.NewTensor[float32](1, 3, 16)
	for i := range x.Data {
		x.Data[i] = 0.01
	}
	_, post, err := gdn.Forward(l, x)
	if err != nil {
		return err
	}
	if post.Shape[1] != 3 || post.Shape[2] != 16 {
		return fmt.Errorf("shape %v", post.Shape)
	}
	return nil
}

func kindSmoke() error {
	l, err := gdn.New(cfg())
	if err != nil {
		return err
	}
	if l.Kind() != seqmix.KindLinearAttn {
		return fmt.Errorf("kind %v", l.Kind())
	}
	return nil
}

func bwdSmoke() error {
	l, err := gdn.New(cfg())
	if err != nil {
		return err
	}
	x := core.NewTensor[float32](2, 3, 16)
	for i := range x.Data {
		x.Data[i] = 0.01 * float32(i%7)
	}
	pre, post, err := gdn.Forward(l, x)
	if err != nil {
		return err
	}
	gy := core.NewTensor[float32](post.Shape...)
	for i := range gy.Data {
		gy.Data[i] = 0.01
	}
	gradIn, gradW, err := gdn.Backward(l, gy, x, pre)
	if err != nil {
		return err
	}
	if gradIn.Len() != x.Len() {
		return fmt.Errorf("gradIn len %d != %d", gradIn.Len(), x.Len())
	}
	if gradW.Len() != l.GradWSize() {
		return fmt.Errorf("gradW len %d != GradWSize %d", gradW.Len(), l.GradWSize())
	}
	return nil
}

// bwdGradCheckOut verifies Backward's Out-projection gradient against a central
// finite difference — Out is one of the analytically exact paths (no state truncation).
func bwdGradCheckOut() error {
	c := cfg()
	cd := c.NumKeyHeads*c.KeyHeadDim*2 + c.NumValueHeads*c.ValueHeadDim
	vd := c.NumValueHeads * c.ValueHeadDim
	nh := c.NumValueHeads
	h := c.HiddenSize
	rnd := func(n int, seed *float32) []float32 {
		s := make([]float32, n)
		for i := range s {
			*seed = *seed*1.0000001 + 0.137
			s[i] = float32(math.Sin(float64(*seed)*12.9898)) * 0.3
		}
		return s
	}
	seed := float32(3.14)
	inQKV := rnd(cd*h, &seed)
	inZ := rnd(vd*h, &seed)
	inB := rnd(nh*h, &seed)
	inA := rnd(nh*h, &seed)
	outW := rnd(h*vd, &seed)
	convW := rnd(cd*c.ConvKernel, &seed)
	aLog := rnd(nh, &seed)
	dtBias := rnd(nh, &seed)
	gamma := rnd(c.ValueHeadDim, &seed)
	for i := range gamma {
		gamma[i] = 1 + gamma[i]*0.1
	}

	x := core.NewTensor[float32](1, 3, h)
	for i := range x.Data {
		seed = seed*1.0000001 + 0.271
		x.Data[i] = float32(math.Cos(float64(seed)*7.77)) * 0.2
	}

	loss := func(outWv []float32) (float64, error) {
		l, err := gdn.NewConfigured(c, inQKV, inZ, inB, inA, outWv, convW, aLog, dtBias, gamma)
		if err != nil {
			return 0, err
		}
		_, post, err := gdn.Forward(l, x)
		if err != nil {
			return 0, err
		}
		var s float64
		for _, v := range post.Data {
			s += float64(v) * float64(v)
		}
		return 0.5 * s, nil
	}

	l, err := gdn.NewConfigured(c, inQKV, inZ, inB, inA, outW, convW, aLog, dtBias, gamma)
	if err != nil {
		return err
	}
	_, post, err := gdn.Forward(l, x)
	if err != nil {
		return err
	}
	gy := post.Clone()
	_, gradW, err := gdn.Backward(l, gy, x, post)
	if err != nil {
		return err
	}

	eps := float32(1e-3)
	idx := 0
	outP := append([]float32(nil), outW...)
	outM := append([]float32(nil), outW...)
	outP[idx] += eps
	outM[idx] -= eps
	lp, err := loss(outP)
	if err != nil {
		return err
	}
	lm, err := loss(outM)
	if err != nil {
		return err
	}
	numGrad := (lp - lm) / float64(2*eps)
	off := cd*h + vd*h + nh*h + nh*h // dInQKV,dInZ,dInB,dInA precede dOut
	anaGrad := float64(gradW.Data[off+idx])
	diff := math.Abs(numGrad - anaGrad)
	if diff > 1e-3*math.Max(1, math.Abs(numGrad)) {
		return fmt.Errorf("Out[%d] grad mismatch num=%.6f ana=%.6f", idx, numGrad, anaGrad)
	}
	return nil
}

func sgdReducesLoss() error {
	c := cfg()
	cd := c.NumKeyHeads*c.KeyHeadDim*2 + c.NumValueHeads*c.ValueHeadDim
	vd := c.NumValueHeads * c.ValueHeadDim
	nh := c.NumValueHeads
	h := c.HiddenSize
	rnd := func(n int, seed *float32) []float32 {
		s := make([]float32, n)
		for i := range s {
			*seed = *seed*1.0000001 + 0.137
			s[i] = float32(math.Sin(float64(*seed)*12.9898)) * 0.3
		}
		return s
	}
	seed := float32(9.81)
	inQKV := rnd(cd*h, &seed)
	inZ := rnd(vd*h, &seed)
	inB := rnd(nh*h, &seed)
	inA := rnd(nh*h, &seed)
	outW := rnd(h*vd, &seed)
	convW := rnd(cd*c.ConvKernel, &seed)
	for i := range convW {
		convW[i] += 0.5
	} // avoid a degenerate all-zero conv tap
	aLog := rnd(nh, &seed)
	dtBias := rnd(nh, &seed)
	gamma := rnd(c.ValueHeadDim, &seed)
	for i := range gamma {
		gamma[i] = 1 + gamma[i]*0.1
	}
	l, err := gdn.NewConfigured(c, inQKV, inZ, inB, inA, outW, convW, aLog, dtBias, gamma)
	if err != nil {
		return err
	}
	x := core.NewTensor[float32](1, 4, 16)
	for i := range x.Data {
		x.Data[i] = 0.02 * float32(i%5-2)
	}
	target := core.NewTensor[float32](x.Shape...)
	for i := range target.Data {
		target.Data[i] = 0.1
	}

	stepLoss := func() (float64, error) {
		_, post, err := gdn.Forward(l, x)
		if err != nil {
			return 0, err
		}
		return training.MSE(post, target)
	}
	loss0, err := stepLoss()
	if err != nil {
		return err
	}

	for i := 0; i < 20; i++ {
		pre, post, err := gdn.Forward(l, x)
		if err != nil {
			return err
		}
		gy, err := training.MSEGrad(post, target)
		if err != nil {
			return err
		}
		_, gradW, err := gdn.Backward(l, gy, x, pre)
		if err != nil {
			return err
		}
		if err := gdn.ApplyGradSGD(l, gradW, 0.05); err != nil {
			return err
		}
	}
	lossN, err := stepLoss()
	if err != nil {
		return err
	}
	if !(lossN < loss0) {
		return fmt.Errorf("loss did not decrease: %.6f -> %.6f", loss0, lossN)
	}
	return nil
}

func gradVerifyBackends() error {
	if !simd.Enabled() {
		fmt.Printf("(SIMD off on %s — CPU-only) ", runtime.GOARCH)
		return nil
	}
	c := matrixCfg()
	x := makeInput(c, 2)
	run := func(be core.Backend) (*core.Tensor[float32], error) {
		l, err := newLayer(c, core.DTypeFloat32, quant.FormatNone, be)
		if err != nil {
			return nil, err
		}
		pre, post, err := gdn.Forward(l, x)
		if err != nil {
			return nil, err
		}
		gy := core.NewTensor[float32](post.Shape...)
		for i := range gy.Data {
			gy.Data[i] = 1
		}
		_, dW, err := gdn.Backward(l, gy, x, pre)
		return dW, err
	}
	cpu, err := run(core.BackendCPUTiled)
	if err != nil {
		return err
	}
	simdGrad, err := run(core.BackendSIMD)
	if err != nil {
		return err
	}
	var max float64
	for i := range cpu.Data {
		if d := math.Abs(float64(cpu.Data[i] - simdGrad.Data[i])); d > max {
			max = d
		}
	}
	if max > 1e-2 {
		return fmt.Errorf("CPU↔SIMD dW max err %v", max)
	}
	fmt.Printf("(maxΔ=%.3g) ", max)
	return nil
}

func cpuFormatNoneAll() error {
	for _, dt := range core.AllDTypes {
		if !gdn.PermutationOK(dt, quant.FormatNone, core.BackendCPUTiled) {
			rec("fwd_bwd", dt.String(), "None", core.BackendCPUTiled.String(), "-", "GAP", "GDN weights are Float32-only")
			continue
		}
		if err := smoke(dt, quant.FormatNone, core.BackendCPUTiled); err != nil {
			return err
		}
		rec("fwd_bwd", dt.String(), "None", core.BackendCPUTiled.String(), "-", "OK", "")
	}
	fmt.Printf("(Float32 supported; %d dtype gaps) ", len(core.AllDTypes)-1)
	return nil
}

func simdFormatNoneAll() error {
	if !simd.Enabled() {
		fmt.Printf("(SIMD off on %s) ", runtime.GOARCH)
		return nil
	}
	for _, dt := range core.AllDTypes {
		if !gdn.PermutationOK(dt, quant.FormatNone, core.BackendSIMD) {
			rec("fwd_bwd", dt.String(), "None", core.BackendSIMD.String(), "-", "GAP", "GDN weights are Float32-only")
			continue
		}
		if err := smoke(dt, quant.FormatNone, core.BackendSIMD); err != nil {
			return err
		}
		rec("fwd_bwd", dt.String(), "None", core.BackendSIMD.String(), "-", "OK", "")
	}
	return nil
}

func simdWebGPUQuants() error {
	for _, format := range []quant.Format{quant.FormatNone, quant.FormatBinaryPacked} {
		for _, be := range []core.Backend{core.BackendSIMD, core.BackendWebGPU} {
			if be == core.BackendSIMD && !simd.Enabled() {
				rec("fwd_bwd", "float32", format.String(), be.String(), "-", "GAP", "simd off")
				continue
			}
			if be == core.BackendWebGPU && !webgpu.Available() {
				rec("fwd_bwd", "float32", format.String(), be.String(), "-", "GAP", "no gpu")
				continue
			}
			if err := smoke(core.DTypeFloat32, format, be); err != nil {
				return err
			}
			rec("fwd_bwd", "float32", format.String(), be.String(), "-", "OK", "")
		}
	}
	return nil
}

func fullMatrixGaps() error {
	perms := gdn.AllPermutations()
	var gaps []string
	for _, p := range perms {
		if p.Backend == core.BackendSIMD && !simd.Enabled() {
			rec("census", p.DType.String(), p.Format.String(), p.Backend.String(), "-", "GAP", "simd off")
			gaps = append(gaps, p.Backend.String())
			continue
		}
		if p.Backend == core.BackendWebGPU && !webgpu.Available() {
			rec("census", p.DType.String(), p.Format.String(), p.Backend.String(), "-", "GAP", "no gpu")
			gaps = append(gaps, p.Backend.String())
			continue
		}
		run := smoke
		if p.Backend == core.BackendWebGPU {
			run = func(dt core.DType, f quant.Format, be core.Backend) error { return smokeForwardOnly(dt, f, be) }
		}
		if err := run(p.DType, p.Format, p.Backend); err != nil {
			rec("census", p.DType.String(), p.Format.String(), p.Backend.String(), "-", failOrGap(p.Backend), err.Error())
			if p.Backend == core.BackendCPUTiled {
				return err
			}
			gaps = append(gaps, p.Backend.String())
			continue
		}
		rec("census", p.DType.String(), p.Format.String(), p.Backend.String(), "-", "OK", "")
	}
	fmt.Printf("(%d supported cells, %d runtime gaps) ", len(perms), len(gaps))
	return nil
}

func failOrGap(be core.Backend) string {
	if be == core.BackendCPUTiled {
		return "FAIL"
	}
	return "GAP"
}

func webGPUNote() error {
	if webgpu.Available() {
		return nil
	}
	l, err := gdn.New(cfg())
	if err != nil {
		return err
	}
	x := core.NewTensor[float32](1, 2, 16)
	if _, _, err := gdn.ForwardWebGPU(l, x); err == nil {
		return fmt.Errorf("expected ForwardWebGPU hard error without device")
	}
	pre, post, err := gdn.Forward(l, x)
	if err != nil {
		return err
	}
	if _, _, err := gdn.BackwardWebGPU(l, post, x, pre); err == nil {
		return fmt.Errorf("expected BackwardWebGPU hard error without device")
	}
	return nil
}
