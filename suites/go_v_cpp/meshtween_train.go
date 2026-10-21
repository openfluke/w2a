package go_v_cpp

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os/exec"
	"time"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/step"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/weights"
)

// CasesMeshTweenTrain — Dense mesh-clock MeshTween (FamilyTween @ lr*0.5) + ModeTween volume credit.
func CasesMeshTweenTrain() []Case {
	return []Case{
		{Name: "Dense MeshTween mesh-clock 1×1×1×1 Float32 — exact bits", Run: caseDenseMeshTween111F32},
		{Name: "Dense MeshTween mesh-clock 2×1×1×1 Float32 — exact bits", Run: caseDenseMeshTween211F32},
		{Name: "Dense MeshTween mesh-clock 2×2×2×1 Float32 — exact bits", Run: caseDenseMeshTween222F32},
		// 3D volume credit (= ModeTween volume wires)
		{Name: "Dense MeshTween volume 1×1×1×1 Float32 — exact bits", Run: caseDenseTweenVol111F32},
		{Name: "Dense MeshTween volume 2×1×1×1 chain Float32 — exact bits", Run: caseDenseTweenVol211F32},
		{Name: "Dense MeshTween volume 2×2×2×1 Float32 — exact bits", Run: caseDenseTweenVol222F32},
		{Name: "RMSNorm MeshTween volume 2×2×2×1 Float32 — exact bits", Run: caseRMSNormTweenVol222F32},
		{Name: "LayerNorm MeshTween volume 2×2×2×1 Float32 — exact bits", Run: caseLayerNormTweenVol222F32},
		{Name: "Softmax MeshTween volume 2×2×2×1 Float32 — exact bits", Run: caseSoftmaxTweenVol222F32},
		{Name: "SwiGLU MeshTween volume 2×2×2×1 Float32 — exact bits", Run: caseSwiGLUTweenVol222F32},
		{Name: "Sequential MeshTween volume 2×2×2×1 Float32 — exact bits", Run: caseSeqTweenVol222F32},
		{Name: "Residual MeshTween volume 2×2×2×1 Float32 — exact bits", Run: caseReTweenVol222F32},
		{Name: "Mamba MeshTween volume 2×2×2×1 Float32 — exact bits", Run: caseMbTweenVol222F32},
		{Name: "Metacognition MeshTween volume 2×2×2×1 Float32 — exact bits", Run: caseMcTweenVol222F32},
		{Name: "GDN MeshTween volume 2×2×2×1 Float32 — exact bits", Run: caseGdTweenVol222F32},
		{Name: "NestedStack MeshTween volume 2×2×2×1 Float32 — exact bits", Run: caseSkTweenVol222F32},
		{Name: "Embedding MeshTween volume 2×2×2×1 Float32 — exact bits", Run: caseEmbeddingTweenVol222F32},
		{Name: "MHA MeshTween volume 2×2×2×1 Float32 — exact bits", Run: caseMHATweenVol222F32},
		{Name: "CNN1 MeshTween volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNTweenVol222F32(1) }},
		{Name: "CNN2 MeshTween volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNTweenVol222F32(2) }},
		{Name: "CNN3 MeshTween volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNTweenVol222F32(3) }},
		{Name: "ConvT1 MeshTween volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTTweenVol222F32(1) }},
		{Name: "ConvT2 MeshTween volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTTweenVol222F32(2) }},
		{Name: "ConvT3 MeshTween volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTTweenVol222F32(3) }},
		{Name: "RNN MeshTween volume 2×2×2×1 Float32 — exact bits", Run: caseRNNTweenVol222F32},
		{Name: "LSTM MeshTween volume 2×2×2×1 Float32 — exact bits", Run: caseLSTMTweenVol222F32},
	}
}

func RunMeshTweenTrain() error {
	var fails []string
	cs := CasesMeshTweenTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [meshtween %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_meshtween_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_meshtween_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("meshtween_train: %d failed: %v", len(fails), fails)
	}
	return nil
}

type meshtweenOut struct {
	loss float64
	Ws   [][]float32
}

func encodeDenseMTW1(depth, rows, cols, lpc, batch, ticks int, cells []volCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("MTW1"))
	for _, v := range []int32{int32(depth), int32(rows), int32(cols), int32(lpc), int32(batch), 0 /*cpu*/, 0 /*n_links*/, int32(ticks)} {
		_ = binary.Write(&in, binary.LittleEndian, v)
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
	for _, v := range target {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	_ = binary.Write(&in, binary.LittleEndian, lr)
	return in.Bytes(), nil
}

func runCppMTW1(payload []byte, benchIters int) (meshtweenOut, time.Duration, error) {
	var z meshtweenOut
	var bench time.Duration
	tool, err := findDenseFwd()
	if err != nil {
		return z, 0, err
	}
	args := []string{}
	if benchIters > 0 {
		args = append(args, fmt.Sprintf("--bench=%d", benchIters))
	}
	cmd := exec.Command(tool, args...)
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return z, 0, fmt.Errorf("cpp mtw1: %v: %s", err, stderr.String())
	}
	raw := stdout.Bytes()
	if len(raw) >= 4+8+4 && string(raw[:4]) == "WVB1" {
		r := bytes.NewReader(raw[4:])
		var ns int64
		var iters int32
		if err := binary.Read(r, binary.LittleEndian, &ns); err != nil {
			return z, 0, err
		}
		if err := binary.Read(r, binary.LittleEndian, &iters); err != nil {
			return z, 0, err
		}
		rest, err := io.ReadAll(r)
		if err != nil {
			return z, 0, err
		}
		raw = rest
		bench = time.Duration(ns)
	}
	if len(raw) < 8 || string(raw[:4]) != "MTWO" {
		return z, bench, fmt.Errorf("cpp meshtween bad magic")
	}
	r := bytes.NewReader(raw[4:])
	if err := binary.Read(r, binary.LittleEndian, &z.loss); err != nil {
		return z, bench, err
	}
	var nCells int32
	if err := binary.Read(r, binary.LittleEndian, &nCells); err != nil {
		return z, bench, err
	}
	for i := int32(0); i < nCells; i++ {
		var o, inn int32
		if err := binary.Read(r, binary.LittleEndian, &o); err != nil {
			return z, bench, err
		}
		if err := binary.Read(r, binary.LittleEndian, &inn); err != nil {
			return z, bench, err
		}
		w := make([]float32, o*inn)
		for j := range w {
			if err := binary.Read(r, binary.LittleEndian, &w[j]); err != nil {
				return z, bench, err
			}
		}
		z.Ws = append(z.Ws, w)
	}
	return z, bench, nil
}

func runGoDenseMeshTween(depth, rows, cols, lpc, batch, ticks int, cells []volCell, x, target []float32, lr float64) (meshtweenOut, error) {
	var z meshtweenOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
	g.Exec.MultiCore = false
	hops := g.HopOrder()
	if len(cells) != len(hops) {
		return z, fmt.Errorf("cells %d != hops %d", len(cells), len(hops))
	}
	for i, c := range hops {
		vc := cells[i]
		if !vc.enabled {
			cell := g.At(c.Z, c.Y, c.X, c.L)
			if cell != nil {
				cell.Layer.IsDisabled = true
			}
			continue
		}
		dl, err := dense.NewConfigured(vc.in, vc.out, vc.act, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		dl.Weights.Bias = append([]float64(nil), vc.bias...)
		disableDense(dl)
		if err := g.BindOp(c.Z, c.Y, c.X, c.L, core.Layer{
			Type: core.LayerDense, DType: vc.dtype, Activation: vc.act,
			InputHeight: vc.in, OutputHeight: vc.out,
		}, dl); err != nil {
			return z, err
		}
	}
	xin := core.NewTensor[float32](batch, len(x)/batch)
	copy(xin.Data, x)
	tgt := core.NewTensor[float32](batch, len(target)/batch)
	copy(tgt.Data, target)

	st := step.New[float32](g)
	for t := 0; t < ticks; t++ {
		st.SetInput(xin)
		if _, err := step.Forward(g, st, true); err != nil {
			return z, fmt.Errorf("mesh fwd tick %d: %w", t, err)
		}
	}
	if len(st.LayerData) == 0 || st.LayerData[len(st.LayerData)-1] == nil {
		return z, fmt.Errorf("empty mesh output")
	}
	out := st.LayerData[len(st.LayerData)-1]
	if out.Len() != tgt.Len() {
		tgt = core.NewTensor[float32](out.Shape...)
		copy(tgt.Data, target)
	}
	loss, err := training.MSE(out, tgt)
	if err != nil {
		return z, err
	}
	gy, err := training.MSEGrad(out, tgt)
	if err != nil {
		return z, err
	}
	if len(st.HistoryIn) == 0 || len(st.HistoryPre) == 0 {
		return z, fmt.Errorf("empty tween history")
	}
	last := len(st.HistoryIn) - 1
	stepIn := st.HistoryIn[last]
	stepPre := st.HistoryPre[last]
	softLR := lr * 0.5
	gap := gy
	for i := len(g.Cells) - 1; i >= 0; i-- {
		cell := &g.Cells[i]
		if cell.Layer.IsDisabled {
			continue
		}
		cin := stepIn[i]
		pre := stepPre[i]
		post := st.LayerData[i]
		if cin == nil || pre == nil || post == nil {
			continue
		}
		dl, ok := cell.Op.(*dense.Layer)
		if !ok {
			return z, fmt.Errorf("expected dense at idx %d", i)
		}
		proj := gap
		if gap.Len() != post.Len() {
			// Same-shape grids in this suite; mismatch → skip update (zeros next).
			proj = core.NewTensor[float32](post.Shape...)
		}
		_, dW, err := dense.Backward(dl, proj, cin, pre)
		if err != nil {
			return z, err
		}
		if dW != nil && dW.Len() > 0 {
			if err := dense.ApplyGradSGD(dl, dW, softLR); err != nil {
				return z, err
			}
		}
		next := core.NewTensor[float32](cin.Shape...)
		if gap.Len() == next.Len() {
			copy(next.Data, gap.Data)
		}
		gap = next
	}
	z.loss = loss
	for i, c := range hops {
		if !cells[i].enabled {
			continue
		}
		cell := g.At(c.Z, c.Y, c.X, c.L)
		dl, ok := cell.Op.(*dense.Layer)
		if !ok {
			return z, fmt.Errorf("expected dense at %v", c)
		}
		Wa, err := weights.MatrixF32(dl.Weights)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, append([]float32(nil), Wa...))
	}
	return z, nil
}

func compareMeshTween(goO, cppO meshtweenOut, requireExact bool) error {
	if len(goO.Ws) != len(cppO.Ws) {
		return fmt.Errorf("n_cells go=%d cpp=%d", len(goO.Ws), len(cppO.Ws))
	}
	for i := range goO.Ws {
		m := suites.CompareF32Spectrum(goO.Ws[i], cppO.Ws[i])
		if requireExact && m.Band != suites.DetExactBit {
			return fmt.Errorf("W[%d]: %s", i, suites.FormatSpectrum(m))
		}
	}
	if goO.loss != cppO.loss {
		rel := math.Abs(goO.loss-cppO.loss) / math.Max(1e-30, math.Abs(goO.loss))
		if requireExact && rel > 0 {
			return fmt.Errorf("loss go=%g cpp=%g", goO.loss, cppO.loss)
		}
	}
	return nil
}

func runDenseMeshTweenCase(depth, rows, cols, lpc, batch, ticks, benchIters int) error {
	stack := depth * rows * cols * lpc
	dim := 4
	W, x, bias := makeSharedInit(dim, dim, batch)
	for i := range W {
		W[i] *= 0.05
	}
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, dim*batch)
	for i := range target {
		target[i] = float32(0.05 * float64(i%dim+1))
	}
	cells := make([]volCell, stack)
	for i := range cells {
		cells[i] = volCell{
			enabled: true, in: dim, out: dim, act: core.ActivationLinear,
			dtype: core.DTypeFloat32, W: append([]float32(nil), W...), bias: append([]float64(nil), bias...),
		}
	}
	goO, err := runGoDenseMeshTween(depth, rows, cols, lpc, batch, ticks, cells, x, target, 1e-2)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeDenseMTW1(depth, rows, cols, lpc, batch, ticks, cells, x, target, 1e-2)
	if err != nil {
		return err
	}
	_, cppDur, err := runCppMTW1(payload, benchIters)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	cppO, _, err := runCppMTW1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp2: %w", err)
	}
	if err := compareMeshTween(goO, cppO, true); err != nil {
		return err
	}
	t0 := time.Now()
	_, _ = runGoDenseMeshTween(depth, rows, cols, lpc, batch, ticks, cells, x, target, 1e-2)
	goUS := float64(time.Since(t0).Nanoseconds()) / 1000
	bi := benchIters
	if bi < 1 {
		bi = 1
	}
	cppUS := float64(cppDur.Nanoseconds()) / float64(bi) / 1000
	spd := suites.SpeedPctDiff(goUS, cppUS)
	fmt.Printf("\n      MeshTween %dx%dx%dx%d ticks=%d loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
		depth, rows, cols, lpc, ticks, goO.loss, goUS, cppUS, spd)
	return nil
}

func caseDenseMeshTween111F32() error {
	return runDenseMeshTweenCase(1, 1, 1, 1, 1, 1, 40)
}
func caseDenseMeshTween211F32() error {
	return runDenseMeshTweenCase(2, 1, 1, 1, 1, 2, 40)
}
func caseDenseMeshTween222F32() error {
	return runDenseMeshTweenCase(2, 2, 2, 1, 1, 8, 20)
}
