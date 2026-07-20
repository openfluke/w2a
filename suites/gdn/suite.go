package gdn

import (
	"fmt"
	"math"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/layers/seqmix"
	"github.com/openfluke/welvet/runtime/training"
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
		{Name: "WebGPU ForwardWebGPU/BackwardWebGPU gate on device", Run: webGPUNote},
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

func decodeSmoke() error {
	l, err := gdn.New(cfg())
	if err != nil { return err }
	x := make([]float32, 16)
	y := make([]float32, 16)
	for i := range x { x[i] = 0.01 * float32(i) }
	return l.ForwardDecode(x, y)
}

func seqSmoke() error {
	l, err := gdn.New(cfg())
	if err != nil { return err }
	x := core.NewTensor[float32](1, 3, 16)
	for i := range x.Data { x.Data[i] = 0.01 }
	_, post, err := gdn.Forward(l, x)
	if err != nil { return err }
	if post.Shape[1] != 3 || post.Shape[2] != 16 { return fmt.Errorf("shape %v", post.Shape) }
	return nil
}

func kindSmoke() error {
	l, err := gdn.New(cfg())
	if err != nil { return err }
	if l.Kind() != seqmix.KindLinearAttn {
		return fmt.Errorf("kind %v", l.Kind())
	}
	return nil
}

func bwdSmoke() error {
	l, err := gdn.New(cfg())
	if err != nil { return err }
	x := core.NewTensor[float32](2, 3, 16)
	for i := range x.Data { x.Data[i] = 0.01 * float32(i%7) }
	pre, post, err := gdn.Forward(l, x)
	if err != nil { return err }
	gy := core.NewTensor[float32](post.Shape...)
	for i := range gy.Data { gy.Data[i] = 0.01 }
	gradIn, gradW, err := gdn.Backward(l, gy, x, pre)
	if err != nil { return err }
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
	for i := range gamma { gamma[i] = 1 + gamma[i]*0.1 }

	x := core.NewTensor[float32](1, 3, h)
	for i := range x.Data {
		seed = seed*1.0000001 + 0.271
		x.Data[i] = float32(math.Cos(float64(seed)*7.77)) * 0.2
	}

	loss := func(outWv []float32) (float64, error) {
		l, err := gdn.NewConfigured(c, inQKV, inZ, inB, inA, outWv, convW, aLog, dtBias, gamma)
		if err != nil { return 0, err }
		_, post, err := gdn.Forward(l, x)
		if err != nil { return 0, err }
		var s float64
		for _, v := range post.Data { s += float64(v) * float64(v) }
		return 0.5 * s, nil
	}

	l, err := gdn.NewConfigured(c, inQKV, inZ, inB, inA, outW, convW, aLog, dtBias, gamma)
	if err != nil { return err }
	_, post, err := gdn.Forward(l, x)
	if err != nil { return err }
	gy := post.Clone()
	_, gradW, err := gdn.Backward(l, gy, x, post)
	if err != nil { return err }

	eps := float32(1e-3)
	idx := 0
	outP := append([]float32(nil), outW...)
	outM := append([]float32(nil), outW...)
	outP[idx] += eps
	outM[idx] -= eps
	lp, err := loss(outP)
	if err != nil { return err }
	lm, err := loss(outM)
	if err != nil { return err }
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
	for i := range convW { convW[i] += 0.5 } // avoid a degenerate all-zero conv tap
	aLog := rnd(nh, &seed)
	dtBias := rnd(nh, &seed)
	gamma := rnd(c.ValueHeadDim, &seed)
	for i := range gamma { gamma[i] = 1 + gamma[i]*0.1 }
	l, err := gdn.NewConfigured(c, inQKV, inZ, inB, inA, outW, convW, aLog, dtBias, gamma)
	if err != nil { return err }
	x := core.NewTensor[float32](1, 4, 16)
	for i := range x.Data { x.Data[i] = 0.02 * float32(i%5-2) }
	target := core.NewTensor[float32](x.Shape...)
	for i := range target.Data { target.Data[i] = 0.1 }

	stepLoss := func() (float64, error) {
		_, post, err := gdn.Forward(l, x)
		if err != nil { return 0, err }
		return training.MSE(post, target)
	}
	loss0, err := stepLoss()
	if err != nil { return err }

	for i := 0; i < 20; i++ {
		pre, post, err := gdn.Forward(l, x)
		if err != nil { return err }
		gy, err := training.MSEGrad(post, target)
		if err != nil { return err }
		_, gradW, err := gdn.Backward(l, gy, x, pre)
		if err != nil { return err }
		if err := gdn.ApplyGradSGD(l, gradW, 0.05); err != nil { return err }
	}
	lossN, err := stepLoss()
	if err != nil { return err }
	if !(lossN < loss0) {
		return fmt.Errorf("loss did not decrease: %.6f -> %.6f", loss0, lossN)
	}
	return nil
}

func webGPUNote() error {
	if webgpu.Available() { return nil }
	l, err := gdn.New(cfg())
	if err != nil { return err }
	x := core.NewTensor[float32](1, 2, 16)
	if _, _, err := gdn.ForwardWebGPU(l, x); err == nil {
		return fmt.Errorf("expected ForwardWebGPU hard error without device")
	}
	pre, post, err := gdn.Forward(l, x)
	if err != nil { return err }
	if _, _, err := gdn.BackwardWebGPU(l, post, x, pre); err == nil {
		return fmt.Errorf("expected BackwardWebGPU hard error without device")
	}
	return nil
}
