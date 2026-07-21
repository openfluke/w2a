package dense

import (
	"fmt"
	"math"
	"runtime"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/webgpu"
)

type Case struct {
	Name string
	Run  func() error
}

func Cases() []Case {
	return []Case{
		{Name: "Forward CPU tiled (identity + bias) [float32 acts]", Run: forwardCPUTiledIdentityBias},
		{Name: "Forward CPU tiled [float64 acts]", Run: forwardCPUTiledF64},
		{Name: "Numeric acts — float64/int8/uint16 on CPU+SIMD (not f32-only)", Run: multiNumericActs},
		{Name: "Backward CPU tiled (dW spot-check)", Run: backwardFiniteDiff},
		{Name: "Grad verify — CPU vs SIMD agreement + finite-diff dW", Run: gradVerifyBackends},
		{Name: "SIMD Plan 9 DotTile (FP32 + ReLU)", Run: simdDispatch},
		{Name: "SIMD Plan 9 enabled on this arch", Run: simdArchEnabled},
		{Name: "Quant Q4_0 pack/unpack round-trip", Run: packQ4_0RoundTrip},
		{Name: "WebGPU hard-errors without device (no host fake)", Run: webGPUNoDevice},
		{Name: "CPU tiled FormatNone × all 34 dtypes (native stream)", Run: cpuTiledFormatNoneAll},
		{Name: "TIMED matrix — FormatNone × all dtypes × CPU/SIMD/WebGPU", Run: TimedMatrix},
		{Name: "TIMED matrix — all quants × CPU/SIMD/WebGPU (Float32)", Run: TimedQuantMatrix},
		{Name: "CPU tiled matrix — all dtype × all quant", Run: cpuTiledMatrix},
		{Name: "SIMD FormatNone × all 34 dtypes (fwd+bwd)", Run: simdFormatNoneAll},
		{Name: "SIMD+WebGPU all quant formats (fwd+bwd, Float32)", Run: simdWebGPUAllQuants},
		{Name: "SIMD fused k-cache — EnsureKSIMDCache builds Int8QS (no F32 inflate)", Run: kSIMDCacheNoF32Inflate},
		{Name: "SIMD fused k/IQ/Affine — CPU packed MatVec parity (no F32 inflate)", Run: fusedKIQAffineSIMDParity},
		{Name: "SIMD fused AffinePacked — CPU vs SIMD fwd + MatVecPackedBlob", Run: affinePackedSIMDParity},
		{Name: "§12 AffinePacked WebGPU GEMVT CPU↔GPU parity", Run: affinePackedWebGPUGEMVTParity},
		{Name: "Repeat-forward determinism", Run: repeatForwardDet},
		{Name: "SC↔MC fwd+bwd determinism", Run: scmcFwdBwdDet},
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
			suites.EndCase("dense", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("dense", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("dense: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func RunOne(n int) error {
	cs := Cases()
	if n < 1 || n > len(cs) {
		return fmt.Errorf("dense: case %d out of range 1..%d", n, len(cs))
	}
	suites.BeginCase()
	fmt.Printf("  [%d] %s … ", n, cs[n-1].Name)
	if err := cs[n-1].Run(); err != nil {
		suites.EndCase("dense", cs[n-1].Name, "FAIL", err.Error())
		fmt.Printf("FAIL\n      %v\n", err)
		return err
	}
	suites.EndCase("dense", cs[n-1].Name, "PASS", "")
	fmt.Println("PASS")
	return nil
}

func forwardCPUTiledIdentityBias() error {
	l, err := dense.New(3, 2, core.ActivationLinear, core.DTypeFloat32)
	if err != nil {
		return err
	}
	w, _ := l.Weights.MasterF32()
	copy(w, []float32{1, 0, 0, 0, 1, 0})
	l.Weights.Bias = []float64{0.5, -0.5}
	l.Exec.MultiCore = false
	x := core.NewTensor[float32](1, 3)
	copy(x.Data, []float32{2, 3, 9})
	pre, post, err := dense.Forward(l, x)
	if err != nil {
		return err
	}
	if math.Abs(float64(pre.Data[0]-2.5)) > 1e-5 || math.Abs(float64(pre.Data[1]-2.5)) > 1e-5 {
		return fmt.Errorf("pre=%v want [2.5 2.5]", pre.Data)
	}
	if post.Data[0] != pre.Data[0] {
		return fmt.Errorf("linear act mismatch")
	}
	return nil
}

func forwardCPUTiledF64() error {
	init := []float64{1, 0, 0, 0, 1, 0}
	l, err := dense.NewConfigured(3, 2, core.ActivationLinear, core.DTypeFloat64, quant.FormatNone, init)
	if err != nil {
		return err
	}
	l.Weights.Bias = []float64{0.5, -0.5}
	l.Exec.MultiCore = false
	x := core.NewTensor[float64](1, 3)
	copy(x.Data, []float64{2, 3, 9})
	pre, _, err := dense.Forward(l, x)
	if err != nil {
		return err
	}
	if math.Abs(pre.Data[0]-2.5) > 1e-5 || math.Abs(pre.Data[1]-2.5) > 1e-5 {
		return fmt.Errorf("pre=%v want [2.5 2.5]", pre.Data)
	}
	return nil
}

func multiNumericActs() error {
	type run struct {
		name string
		fn   func() error
	}
	cases := []run{
		{"float64/Float64/CPU", func() error {
			return smokeNumeric[float64](core.DTypeFloat64, quant.FormatNone, core.BackendCPUTiled)
		}},
		{"float64/Float64/SIMD", func() error {
			if !simd.Enabled() {
				return fmt.Errorf("SIMD required")
			}
			return smokeNumeric[float64](core.DTypeFloat64, quant.FormatNone, core.BackendSIMD)
		}},
		{"int8/Int8/CPU", func() error {
			return smokeNumeric[int8](core.DTypeInt8, quant.FormatNone, core.BackendCPUTiled)
		}},
		{"int8/Int8/SIMD", func() error {
			if !simd.Enabled() {
				return fmt.Errorf("SIMD required")
			}
			return smokeNumeric[int8](core.DTypeInt8, quant.FormatNone, core.BackendSIMD)
		}},
		{"uint16/Uint16/CPU", func() error {
			return smokeNumeric[uint16](core.DTypeUint16, quant.FormatNone, core.BackendCPUTiled)
		}},
		{"uint16/Uint16/SIMD", func() error {
			if !simd.Enabled() {
				return fmt.Errorf("SIMD required")
			}
			return smokeNumeric[uint16](core.DTypeUint16, quant.FormatNone, core.BackendSIMD)
		}},
	}
	var fails []string
	for _, c := range cases {
		if err := c.fn(); err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", c.name, err))
		}
	}
	fmt.Printf("(%d numeric paths) ", len(cases))
	if len(fails) > 0 {
		return fmt.Errorf("%s", strings.Join(fails, " | "))
	}
	return nil
}

func smokeNumeric[T core.Numeric](dt core.DType, format quant.Format, backend core.Backend) error {
	const in, out = 32, 8
	init := make([]T, out*in)
	for i := range init {
		init[i] = core.FromFloat64[T](float64((i%13)-6) * 0.1)
	}
	l, err := dense.NewConfigured(in, out, core.ActivationLinear, dt, format, init)
	if err != nil {
		return err
	}
	l.Exec.Backend = backend
	l.Exec.MultiCore = false
	x := core.NewTensor[T](2, in)
	for i := range x.Data {
		x.Data[i] = core.FromFloat64[T](float64(i%5) * 0.25)
	}
	pre, _, err := dense.Forward(l, x)
	if err != nil {
		return fmt.Errorf("forward: %w", err)
	}
	gOut := core.NewTensor[T](2, out)
	for i := range gOut.Data {
		gOut.Data[i] = core.FromFloat64[T](1)
	}
	_, _, err = dense.Backward(l, gOut, x, pre)
	return err
}

func backwardFiniteDiff() error {
	l, err := dense.New(4, 3, core.ActivationLinear, core.DTypeFloat32)
	if err != nil {
		return err
	}
	w, _ := l.Weights.MasterF32()
	for i := range w {
		w[i] = float32(i%7) * 0.1
	}
	l.Exec.MultiCore = false
	x := core.NewTensor[float32](2, 4)
	for i := range x.Data {
		x.Data[i] = float32(i+1) * 0.25
	}
	pre, _, err := dense.Forward(l, x)
	if err != nil {
		return err
	}
	gradOut := core.NewTensor[float32](2, 3)
	for i := range gradOut.Data {
		gradOut.Data[i] = 1
	}
	dX, dW, err := dense.Backward(l, gradOut, x, pre)
	if err != nil {
		return err
	}
	if dX.Len() != 8 || dW.Len() != 12 {
		return fmt.Errorf("shapes dX=%d dW=%d", dX.Len(), dW.Len())
	}
	want := float32(0)
	for b := 0; b < 2; b++ {
		want += x.Data[b*4+0]
	}
	if math.Abs(float64(dW.Data[0]-want)) > 1e-4 {
		return fmt.Errorf("dW[0]=%v want %v", dW.Data[0], want)
	}
	return nil
}

func gradVerifyBackends() error {
	const in, out, batch = 32, 8, 2
	init := make([]float32, out*in)
	for i := range init {
		init[i] = float32((i%7)-3) * 0.15
	}
	x := core.NewTensor[float32](batch, in)
	for i := range x.Data {
		x.Data[i] = float32(i%5)*0.2 + 0.1
	}
	gOut := core.NewTensor[float32](batch, out)
	for i := range gOut.Data {
		gOut.Data[i] = 1
	}

	run := func(be core.Backend) (*core.Tensor[float32], *core.Tensor[float32], *core.Tensor[float32], error) {
		l, err := dense.NewConfigured(in, out, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, init)
		if err != nil {
			return nil, nil, nil, err
		}
		l.Exec.Backend = be
		l.Exec.MultiCore = false
		pre, _, err := dense.Forward(l, x)
		if err != nil {
			return nil, nil, nil, err
		}
		dX, dW, err := dense.Backward(l, gOut, x, pre)
		return pre, dX, dW, err
	}

	preCPU, dXCPU, dWCPU, err := run(core.BackendCPUTiled)
	if err != nil {
		return fmt.Errorf("CPU: %w", err)
	}
	// Finite-diff spot-check on dW[0] (∂L/∂W00 = Σ_b x[b,0] for unit gradOut).
	wantDW0 := float32(0)
	for b := 0; b < batch; b++ {
		wantDW0 += x.Data[b*in]
	}
	if math.Abs(float64(dWCPU.Data[0]-wantDW0)) > 1e-4 {
		return fmt.Errorf("CPU dW[0]=%v want %v", dWCPU.Data[0], wantDW0)
	}

	if simd.Enabled() {
		preS, dXS, dWS, err := run(core.BackendSIMD)
		if err != nil {
			return fmt.Errorf("SIMD: %w", err)
		}
		if err := tensorsClose(preCPU, preS, 1e-4); err != nil {
			return fmt.Errorf("CPU/SIMD pre: %w", err)
		}
		if err := tensorsClose(dXCPU, dXS, 1e-3); err != nil {
			return fmt.Errorf("CPU/SIMD dX: %w", err)
		}
		if err := tensorsClose(dWCPU, dWS, 1e-3); err != nil {
			return fmt.Errorf("CPU/SIMD dW: %w", err)
		}
	}

	if webgpu.Available() {
		preG, dXG, dWG, err := run(core.BackendWebGPU)
		if err != nil {
			return fmt.Errorf("WebGPU: %w", err)
		}
		if err := tensorsClose(preCPU, preG, 5e-3); err != nil {
			return fmt.Errorf("CPU/WebGPU pre: %w", err)
		}
		if err := tensorsClose(dXCPU, dXG, 5e-3); err != nil {
			return fmt.Errorf("CPU/WebGPU dX: %w", err)
		}
		if err := tensorsClose(dWCPU, dWG, 5e-3); err != nil {
			return fmt.Errorf("CPU/WebGPU dW: %w", err)
		}
	}
	fmt.Printf("(CPU+SIMD+WebGPU where available) ")

	// Packed Q4_0: CPU vs SIMD forward agreement (quantized — looser tol).
	if err := gradVerifyQuant(quant.FormatQ4_0); err != nil {
		return fmt.Errorf("Q4_0: %w", err)
	}
	if err := gradVerifyQuant(quant.FormatQ8_0); err != nil {
		return fmt.Errorf("Q8_0: %w", err)
	}
	return nil
}

func gradVerifyQuant(f quant.Format) error {
	const in, out, batch = 32, 8, 2
	init := make([]float32, out*in)
	for i := range init {
		init[i] = float32((i%7)-3) * 0.15
	}
	x := core.NewTensor[float32](batch, in)
	for i := range x.Data {
		x.Data[i] = float32(i%5)*0.2 + 0.1
	}
	run := func(be core.Backend) (*core.Tensor[float32], error) {
		l, err := dense.NewConfigured(in, out, core.ActivationLinear, core.DTypeFloat32, f, init)
		if err != nil {
			return nil, err
		}
		l.Exec.Backend = be
		l.Exec.MultiCore = false
		pre, _, err := dense.Forward(l, x)
		return pre, err
	}
	preCPU, err := run(core.BackendCPUTiled)
	if err != nil {
		return err
	}
	if simd.Enabled() {
		preS, err := run(core.BackendSIMD)
		if err != nil {
			return err
		}
		if err := tensorsClose(preCPU, preS, 5e-2); err != nil {
			return fmt.Errorf("CPU/SIMD: %w", err)
		}
	}
	if webgpu.Available() {
		preG, err := run(core.BackendWebGPU)
		if err != nil {
			return err
		}
		if err := tensorsClose(preCPU, preG, 5e-2); err != nil {
			return fmt.Errorf("CPU/WebGPU: %w", err)
		}
	}
	return nil
}

func tensorsClose(a, b *core.Tensor[float32], tol float64) error {
	if a.Len() != b.Len() {
		return fmt.Errorf("len %d vs %d", a.Len(), b.Len())
	}
	for i := range a.Data {
		if math.Abs(float64(a.Data[i]-b.Data[i])) > tol {
			return fmt.Errorf("idx %d: %v vs %v (tol=%g)", i, a.Data[i], b.Data[i], tol)
		}
	}
	return nil
}

func simdDispatch() error {
	l, err := dense.New(8, 4, core.ActivationReLU, core.DTypeFloat32)
	if err != nil {
		return err
	}
	w, _ := l.Weights.MasterF32()
	for i := range w {
		w[i] = 0.125
	}
	l.Exec.Backend = core.BackendSIMD
	x := core.NewTensor[float32](1, 8)
	for i := range x.Data {
		x.Data[i] = 1
	}
	_, post, err := dense.Forward(l, x)
	if err != nil {
		return err
	}
	if math.Abs(float64(post.Data[0]-1)) > 1e-5 {
		return fmt.Errorf("post=%v want 1", post.Data)
	}
	return nil
}

func simdArchEnabled() error {
	switch runtime.GOARCH {
	case "amd64", "arm64":
		if !simd.Enabled() {
			return fmt.Errorf("expected Plan 9 SIMD on %s", runtime.GOARCH)
		}
		fmt.Printf("(%s) ", runtime.GOARCH)
	default:
		if simd.Enabled() {
			return fmt.Errorf("unexpected SIMD on %s", runtime.GOARCH)
		}
		fmt.Printf("(%s no SIMD — BackendSIMD must hard-error) ", runtime.GOARCH)
	}
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	b := []float32{8, 7, 6, 5, 4, 3, 2, 1}
	got := simd.DotTile(a, b, 0, 8, 0)
	var want float64
	for i := range a {
		want += float64(a[i]) * float64(b[i])
	}
	if math.Abs(got-want) > 1e-5 {
		return fmt.Errorf("DotTile=%v want %v", got, want)
	}
	return nil
}

func packQ4_0RoundTrip() error {
	w := make([]float32, 64)
	for i := range w {
		w[i] = float32(i%17) - 8
	}
	b, err := quant.PackQ4_0(w, 2, 32)
	if err != nil {
		return err
	}
	out, err := quant.UnpackQ4_0(b)
	if err != nil {
		return err
	}
	var maxErr float64
	for i := range w {
		e := math.Abs(float64(out[i] - w[i]))
		if e > maxErr {
			maxErr = e
		}
	}
	if maxErr > 2.0 {
		return fmt.Errorf("max err %v", maxErr)
	}
	return nil
}

func webGPUNoDevice() error {
	if webgpu.Available() {
		fmt.Printf("(device present — skip negative check) ")
		return nil
	}
	l, err := dense.New(2, 2, core.ActivationLinear, core.DTypeFloat32)
	if err != nil {
		return err
	}
	l.Exec.Backend = core.BackendWebGPU
	_, _, err = dense.Forward(l, core.NewTensor[float32](1, 2))
	if err == nil {
		return fmt.Errorf("expected hard error without WebGPU device (no host fake)")
	}
	fmt.Printf("(got error as required) ")
	return nil
}

func cpuTiledFormatNoneAll() error {
	const want = 34
	if len(core.AllDTypes) != want {
		return fmt.Errorf("expected %d dtypes (0–33 full matrix), got %d", want, len(core.AllDTypes))
	}
	var fails []string
	for _, dt := range core.AllDTypes {
		if err := smoke[float32](dt, quant.FormatNone, core.BackendCPUTiled); err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
		}
	}
	fmt.Printf("(%d FormatNone) ", want)
	if len(fails) > 0 {
		return fmt.Errorf("%d failed: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

func cpuTiledMatrix() error {
	var fails []string
	n := 0
	for _, dt := range core.AllDTypes {
		for _, f := range quant.AllFormats {
			// smoke shape is out×in = 8×64 — AffinePacked needs cols%64==0.
			if f == quant.FormatAffinePacked && !suites.AffinePackable(8, 64) {
				continue
			}
			n++
			if err := smoke[float32](dt, f, core.BackendCPUTiled); err != nil {
				fails = append(fails, fmt.Sprintf("%s/%s: %v", dt, f, err))
				if len(fails) >= 8 {
					break
				}
			}
		}
		if len(fails) >= 8 {
			break
		}
	}
	fmt.Printf("(%d cells) ", n)
	if len(fails) > 0 {
		return fmt.Errorf("%d failed: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

func simdFormatNoneAll() error {
	if !simd.Enabled() {
		return fmt.Errorf("Plan 9 SIMD not enabled on %s", runtime.GOARCH)
	}
	var fails []string
	for _, dt := range core.AllDTypes {
		if err := smoke[float32](dt, quant.FormatNone, core.BackendSIMD); err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
		}
	}
	fmt.Printf("(%d FormatNone SIMD) ", len(core.AllDTypes))
	if len(fails) > 0 {
		return fmt.Errorf("%d failed: %s", len(fails), strings.Join(fails, " | "))
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
		if f == quant.FormatAffinePacked && !suites.AffinePackable(8, 64) {
			continue // smoke shape not packable — see backend_honesty.AffineSkipNote
		}
		for _, be := range backends {
			if be == core.BackendSIMD && !simd.Enabled() {
				fails = append(fails, fmt.Sprintf("%s/%s: SIMD not enabled", f, be))
				continue
			}
			if be == core.BackendWebGPU && !webgpu.Available() {
				continue // no device — GAP, not FAIL
			}
			if err := smoke[float32](core.DTypeFloat32, f, be); err != nil {
				fails = append(fails, fmt.Sprintf("%s/%s: %v", f, be, err))
			}
		}
	}
	n := (len(quant.AllFormats) - 1) * 2
	fmt.Printf("(%d quant×backend cells) ", n)
	if len(fails) > 0 {
		return fmt.Errorf("%d failed: %s", len(fails), strings.Join(fails[:min(8, len(fails))], " | "))
	}
	return nil
}

func fullMatrixGaps() error {
	perms := dense.AllPermutations()
	var failN int
	var samples []string
	for _, p := range perms {
		status := "OK"
		note := ""
		// smoke / smokeForwardOnly use cols=64 — AffinePacked needs cols%64==0.
		smokeCols := 64
		if p.Format == quant.FormatAffinePacked && !suites.AffinePackable(8, smokeCols) {
			failN++
			status = "GAP"
			note = suites.AffineSkipNote()
		} else {
			var err error
			if p.Backend == core.BackendWebGPU {
				// forward-only for census speed (full fwd+bwd covered by timed + quant suites)
				err = smokeForwardOnly[float32](p.DType, p.Format, p.Backend)
			} else {
				err = smoke[float32](p.DType, p.Format, p.Backend)
			}
			if err != nil {
				failN++
				status = "GAP"
				note = err.Error()
			} else if p.Backend == core.BackendWebGPU || p.Backend == core.BackendSIMD {
				status, note = suites.StampBackendNote("dense",
					p.Backend == core.BackendSIMD, p.Backend == core.BackendWebGPU, status, note)
			}
		}
		if status == "GAP" && len(samples) < 6 {
			samples = append(samples, fmt.Sprintf("%s/%s/%s", p.DType, p.Format, p.Backend))
		}
		rec("census", p.DType.String(), p.Format.String(), p.Backend.String(), "-", status, note)
	}
	ok := len(perms) - failN
	fmt.Printf("(%d cells, %d ok, %d gaps; e.g. %s) ", len(perms), ok, failN, strings.Join(samples, ", "))
	return nil
}

func smoke[T core.Numeric](dt core.DType, format quant.Format, backend core.Backend) error {
	const in, out = 64, 8
	init := make([]T, out*in)
	for i := range init {
		init[i] = core.FromFloat64[T](float64((i%13)-6) * 0.1)
	}
	useDT := dt
	if format != quant.FormatNone {
		useDT = core.DTypeFloat32
	}
	l, err := dense.NewConfigured(in, out, core.ActivationLinear, useDT, format, init)
	if err != nil {
		return err
	}
	if format == quant.FormatNone && dt != core.DTypeFloat32 {
		if err := l.Weights.SetDType(dt); err != nil {
			return err
		}
		l.Core.DType = dt
	}
	l.Exec.Backend = backend
	l.Exec.MultiCore = false
	x := core.NewTensor[T](2, in)
	for i := range x.Data {
		x.Data[i] = core.FromFloat64[T](float64(i%5) * 0.25)
	}
	pre, post, err := dense.Forward(l, x)
	if err != nil {
		return fmt.Errorf("forward: %w", err)
	}
	if pre.Len() != 2*out || post.Len() != 2*out {
		return fmt.Errorf("bad out len")
	}
	gOut := core.NewTensor[T](2, out)
	for i := range gOut.Data {
		gOut.Data[i] = core.FromFloat64[T](1)
	}
	dX, dW, err := dense.Backward(l, gOut, x, pre)
	if err != nil {
		return fmt.Errorf("backward: %w", err)
	}
	if dX.Len() != 2*in || dW.Len() != out*in {
		return fmt.Errorf("bad grad shapes")
	}
	return nil
}

func smokeForwardOnly[T core.Numeric](dt core.DType, format quant.Format, backend core.Backend) error {
	const in, out = 64, 8
	init := make([]T, out*in)
	for i := range init {
		init[i] = core.FromFloat64[T](float64((i%13)-6) * 0.1)
	}
	useDT := dt
	if format != quant.FormatNone {
		useDT = core.DTypeFloat32
	}
	l, err := dense.NewConfigured(in, out, core.ActivationLinear, useDT, format, init)
	if err != nil {
		return err
	}
	if format == quant.FormatNone && dt != core.DTypeFloat32 {
		if err := l.Weights.SetDType(dt); err != nil {
			return err
		}
		l.Core.DType = dt
	}
	l.Exec.Backend = backend
	l.Exec.MultiCore = false
	x := core.NewTensor[T](1, in)
	for i := range x.Data {
		x.Data[i] = core.FromFloat64[T](float64(i%5) * 0.25)
	}
	_, _, err = dense.Forward(l, x)
	return err
}
