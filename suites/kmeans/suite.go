package kmeans

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/kmeans"
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
			suites.EndCase("kmeans", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("kmeans", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("kmeans: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func RunOne(n int) error {
	cs := Cases()
	if n < 1 || n > len(cs) {
		return fmt.Errorf("kmeans: case %d out of range 1..%d", n, len(cs))
	}
	suites.BeginCase()
	fmt.Printf("  [%d] %s … ", n, cs[n-1].Name)
	if err := cs[n-1].Run(); err != nil {
		suites.EndCase("kmeans", cs[n-1].Name, "FAIL", err.Error())
		fmt.Printf("FAIL\n      %v\n", err)
		return err
	}
	suites.EndCase("kmeans", cs[n-1].Name, "PASS", "")
	fmt.Println("PASS")
	return nil
}

func cfg() kmeans.Config {
	return kmeans.Config{NumClusters: 4, FeatureDim: 8, Temperature: 1, OutputMode: kmeans.OutputProbabilities, Activation: core.ActivationLinear}
}
func forwardSmoke() error {
	l, err := kmeans.NewConfigured[float32](cfg(), core.DTypeFloat32, quant.FormatNone, nil)
	if err != nil { return err }
	x := core.NewTensor[float32](2, 8)
	_, post, err := kmeans.Forward(l, x)
	if err != nil { return err }
	if post.Shape[1] != 4 { return fmt.Errorf("shape %v", post.Shape) }
	return nil
}
func backwardSmoke() error {
	l, err := kmeans.NewConfigured[float32](cfg(), core.DTypeFloat32, quant.FormatNone, nil)
	if err != nil { return err }
	x := core.NewTensor[float32](2, 8)
	for i := range x.Data { x.Data[i] = float32(i%3) * 0.1 }
	pre, post, err := kmeans.Forward(l, x)
	if err != nil { return err }
	gy := core.NewTensor[float32](post.Shape...)
	for i := range gy.Data { gy.Data[i] = 0.01 }
	_, _, err = kmeans.Backward(l, gy, x, pre)
	return err
}
func webGPUNoDevice() error {
	if webgpu.Available() { return nil }
	l, _ := kmeans.NewConfigured[float32](cfg(), core.DTypeFloat32, quant.FormatNone, nil)
	l.Exec.Backend = core.BackendWebGPU
	x := core.NewTensor[float32](1, 8)
	_, _, err := kmeans.Forward(l, x)
	if err == nil { return fmt.Errorf("expected hard error") }
	return nil
}
func cpuFormatNoneAll() error {
	for _, dt := range core.AllDTypes {
		l, err := kmeans.NewConfigured[float32](cfg(), core.DTypeFloat32, quant.FormatNone, nil)
		if err != nil { return err }
		if err := l.SetDType(dt); err != nil { return err }
		x := core.NewTensor[float32](1, 8)
		if _, _, err := kmeans.Forward(l, x); err != nil { return fmt.Errorf("%v: %w", dt, err) }
	}
	fmt.Printf("(%d FormatNone) ", len(core.AllDTypes))
	return nil
}
func simdSmoke() error {
	if !simd.Enabled() { return nil }
	l, err := kmeans.NewConfigured[float32](cfg(), core.DTypeFloat32, quant.FormatNone, nil)
	if err != nil { return err }
	l.Exec.Backend = core.BackendSIMD
	x := core.NewTensor[float32](1, 8)
	pre, post, err := kmeans.Forward(l, x)
	if err != nil { return err }
	gy := core.NewTensor[float32](post.Shape...)
	_, _, err = kmeans.Backward(l, gy, x, pre)
	return err
}
func fullMatrixGaps() error {
	perms := kmeans.AllPermutations()
	failN := 0
	for _, p := range perms {
		if p.Backend == core.BackendWebGPU && !webgpu.Available() { failN++; continue }
		if p.Backend == core.BackendSIMD && !simd.Enabled() { failN++; continue }
		l, err := kmeans.NewConfigured[float32](cfg(), core.DTypeFloat32, quant.FormatNone, nil)
		if err != nil { failN++; continue }
		_ = l.SetDType(p.DType)
		if err := l.Pack(p.Format); err != nil { failN++; continue }
		l.Exec.Backend = p.Backend
		x := core.NewTensor[float32](1, 8)
		if _, _, err := kmeans.Forward(l, x); err != nil { failN++ }
	}
	fmt.Printf("(%d cells, %d ok, %d gaps) ", len(perms), len(perms)-failN, failN)
	return nil
}
