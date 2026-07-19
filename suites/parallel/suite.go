package parallel

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
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
		{Name: "Forward smoke", Run: forwardSmoke},
		{Name: "Backward smoke", Run: backwardSmoke},
		{Name: "WebGPU hard-errors without device", Run: webGPUNoDevice},
		{Name: "CPU FormatNone × all dtypes (fwd)", Run: cpuFormatNoneAll},
		{Name: "SIMD FormatNone Float32 (fwd+bwd)", Run: simdSmoke},
		{Name: "GAP CENSUS", Run: fullMatrixGaps},
	}
}

func RunAll() error {
	var fails []string
	for i, c := range Cases() {
		suites.BeginCase()
		fmt.Printf("  [%d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("parallel", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("parallel", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("parallel: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func RunOne(n int) error {
	cs := Cases()
	if n < 1 || n > len(cs) {
		return fmt.Errorf("parallel: case %d out of range 1..%d", n, len(cs))
	}
	suites.BeginCase()
	fmt.Printf("  [%d] %s … ", n, cs[n-1].Name)
	if err := cs[n-1].Run(); err != nil {
		suites.EndCase("parallel", cs[n-1].Name, "FAIL", err.Error())
		fmt.Printf("FAIL\n      %v\n", err)
		return err
	}
	suites.EndCase("parallel", cs[n-1].Name, "PASS", "")
	fmt.Println("PASS")
	return nil
}

func cfg(mode parallel.CombineMode) parallel.Config {
	return parallel.Config{Dim: 8, OutFeat: 8, Branches: 2, Combine: mode}
}
func forwardSmoke() error {
	for _, mode := range []parallel.CombineMode{parallel.CombineConcat, parallel.CombineAdd, parallel.CombineAvg, parallel.CombineFilter} {
		l, err := parallel.NewConfigured[float32](cfg(mode), core.DTypeFloat32, quant.FormatNone, nil, nil)
		if err != nil { return err }
		x := core.NewTensor[float32](2, 8)
		for i := range x.Data { x.Data[i] = 0.1 }
		_, post, err := parallel.Forward(l, x)
		if err != nil { return fmt.Errorf("%s: %w", mode, err) }
		want := cfg(mode).OutDim()
		if post.Shape[1] != want { return fmt.Errorf("%s shape %v want feat %d", mode, post.Shape, want) }
	}
	return nil
}
func backwardSmoke() error {
	l, err := parallel.NewConfigured[float32](cfg(parallel.CombineConcat), core.DTypeFloat32, quant.FormatNone, nil, nil)
	if err != nil { return err }
	x := core.NewTensor[float32](2, 8)
	pre, post, err := parallel.Forward(l, x)
	if err != nil { return err }
	gy := core.NewTensor[float32](post.Shape...)
	_, _, err = parallel.Backward(l, gy, x, pre)
	return err
}
func webGPUNoDevice() error {
	if webgpu.Available() { return nil }
	l, _ := parallel.NewConfigured[float32](cfg(parallel.CombineAdd), core.DTypeFloat32, quant.FormatNone, nil, nil)
	l.Exec.Backend = core.BackendWebGPU
	x := core.NewTensor[float32](1, 8)
	_, _, err := parallel.Forward(l, x)
	if err == nil { return fmt.Errorf("expected hard error") }
	return nil
}
func cpuFormatNoneAll() error {
	for _, dt := range core.AllDTypes {
		l, err := parallel.NewConfigured[float32](cfg(parallel.CombineAdd), core.DTypeFloat32, quant.FormatNone, nil, nil)
		if err != nil { return err }
		if err := l.SetDType(dt); err != nil { return err }
		x := core.NewTensor[float32](1, 8)
		if _, _, err := parallel.Forward(l, x); err != nil { return fmt.Errorf("%v: %w", dt, err) }
	}
	fmt.Printf("(%d FormatNone) ", len(core.AllDTypes))
	return nil
}
func simdSmoke() error {
	if !simd.Enabled() { return nil }
	l, err := parallel.NewConfigured[float32](cfg(parallel.CombineFilter), core.DTypeFloat32, quant.FormatNone, nil, nil)
	if err != nil { return err }
	l.Exec.Backend = core.BackendSIMD
	x := core.NewTensor[float32](1, 8)
	pre, post, err := parallel.Forward(l, x)
	if err != nil { return err }
	gy := core.NewTensor[float32](post.Shape...)
	_, _, err = parallel.Backward(l, gy, x, pre)
	return err
}
func fullMatrixGaps() error {
	perms := parallel.AllPermutations()
	failN := 0
	for _, p := range perms {
		if p.Backend == core.BackendWebGPU && !webgpu.Available() { failN++; continue }
		if p.Backend == core.BackendSIMD && !simd.Enabled() { failN++; continue }
		l, err := parallel.NewConfigured[float32](cfg(parallel.CombineConcat), core.DTypeFloat32, quant.FormatNone, nil, nil)
		if err != nil { failN++; continue }
		_ = l.SetDType(p.DType)
		if err := l.Pack(p.Format); err != nil { failN++; continue }
		l.Exec.Backend = p.Backend
		x := core.NewTensor[float32](1, 8)
		if _, _, err := parallel.Forward(l, x); err != nil { failN++ }
	}
	fmt.Printf("(%d cells, %d ok, %d gaps) ", len(perms), len(perms)-failN, failN)
	return nil
}
