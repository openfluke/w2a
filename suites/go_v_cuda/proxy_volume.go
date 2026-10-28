package go_v_cuda

import (
	"bytes"
	"fmt"
	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"math"
	"reflect"
)

type proxyGoLeaf struct {
	cell             *architecture.Cell
	input, pre, post *core.Tensor[float32]
}
type proxyGoState struct {
	cache        [][]float64
	norm         float64
	tick, sparse int
	proxy        []float32
}

func proxyVolumeLeaves(g *architecture.Grid, fwd *forward.Result[float32]) ([]proxyGoLeaf, error) {
	var leaves []proxyGoLeaf
	for _, st := range fwd.Steps {
		cell := g.At(st.Coord.Z, st.Coord.Y, st.Coord.X, st.Coord.L)
		if cell == nil || cell.Layer.IsDisabled {
			continue
		}
		var children []*dense.Layer
		switch op := cell.Op.(type) {
		case *sequential.Layer:
			children = op.Children
		case *residual.Layer:
			children = op.Children
		case *softmax.Layer:
			continue
		}
		if children != nil {
			cur := flattenSeqLike(st.Input)
			for _, ch := range children {
				pre, post, err := dense.Forward(ch, cur)
				if err != nil {
					return nil, err
				}
				meta, err := meshMeta(ch)
				if err != nil {
					return nil, err
				}
				c := &architecture.Cell{Layer: meta, Op: ch}
				leaves = append(leaves, proxyGoLeaf{c, cur, pre, post})
				cur = post
			}
		} else {
			leaves = append(leaves, proxyGoLeaf{cell, st.Input, st.Pre, st.Post})
		}
	}
	return leaves, nil
}
func proxyLeafSGD(l proxyGoLeaf, dw *core.Tensor[float32], lr float64) error {
	if dw == nil || dw.Len() == 0 || lr == 0 {
		return nil
	}
	s, err := parallel.NewStack(l.cell.Op)
	if err != nil {
		return err
	}
	return parallel.ApplyGradSGDStack(s, dw, lr)
}
func proxyLeafDW(l proxyGoLeaf, gy *core.Tensor[float32], lr float64) error {
	var dw *core.Tensor[float32]
	var err error
	if d, ok := l.cell.Op.(*dense.Layer); ok {
		dw, err = dense.GradWOnly(d, gy, l.input, l.pre)
	} else {
		_, dw, err = backward.Cell(l.cell, gy, l.input, l.pre)
	}
	if err != nil {
		return err
	}
	return proxyLeafSGD(l, dw, lr)
}
func proxyLeafLinear(l proxyGoLeaf, gy *core.Tensor[float32], nilPre bool) (*core.Tensor[float32], error) {
	if d, ok := l.cell.Op.(*dense.Layer); ok {
		return dense.LinearGradIn(d, gy, l.input)
	}
	pre := l.pre
	if nilPre {
		pre = nil
	}
	gx, _, err := backward.Cell(l.cell, gy, l.input, pre)
	if err != nil || gx == nil || gx.Len() == 0 {
		return projectGapF32(gy, l.input.Shape...), nil
	}
	return gx, nil
}
func proxyScale(g *core.Tensor[float32], n int) {
	for i := range g.Data {
		g.Data[i] = float32(float64(g.Data[i]) * (1.0 / float64(n)))
	}
}
func proxyHidden(leaves []proxyGoLeaf, src *core.Tensor[float32], lr float64, n, selected int) error {
	for i, l := range leaves {
		if selected >= 0 && i != selected {
			continue
		}
		gi := projectGapF32(src, l.post.Shape...)
		proxyScale(gi, n)
		if err := proxyLeafDW(l, gi, lr); err != nil {
			return err
		}
	}
	return nil
}
func proxyNorm(gy *core.Tensor[float32]) float64 {
	var sum float64
	for _, v := range gy.Data {
		sum += float64(v) * float64(v)
	}
	return math.Sqrt(sum / float64(gy.Len()))
}
func proxyApplyLeaves(leaves []proxyGoLeaf, gy *core.Tensor[float32], lr float64, variant int, state *proxyGoState) error {
	n := len(leaves)
	if n == 0 {
		return nil
	}
	head := leaves[n-1]
	ghead := projectGapF32(gy, head.post.Shape...)
	if variant == 0 {
		for _, l := range leaves {
			gi := projectGapF32(gy, l.post.Shape...)
			proxyScale(gi, n)
			if err := applyTweenLocalOp(l.cell.Op, gi, l.input, l.pre, lr); err != nil {
				return err
			}
		}
		return nil
	}
	if variant == 2 || variant == 4 {
		if variant == 4 && state != nil && len(state.cache) == n && state.norm > 1e-12 && state.tick%20 != 0 {
			scale := proxyNorm(gy) / state.norm
			state.tick++
			for i, l := range leaves {
				if len(state.cache[i]) != l.post.Len() {
					continue
				}
				gi := core.NewTensor[float32](l.post.Shape...)
				for j, v := range state.cache[i] {
					gi.Data[j] = float32(v * scale)
				}
				proxyScale(gi, n)
				if err := proxyLeafDW(l, gi, lr); err != nil {
					return err
				}
			}
			return nil
		}
		var fill [][]float64
		if variant == 4 && state != nil {
			fill = make([][]float64, n)
		}
		current := gy
		for i := n - 1; i >= 0; i-- {
			l := leaves[i]
			gi := projectGapF32(current, l.post.Shape...)
			if fill != nil {
				fill[i] = make([]float64, gi.Len())
				for j, v := range gi.Data {
					fill[i][j] = float64(v)
				}
			}
			proxyScale(gi, n)
			if err := proxyLeafDW(l, gi, lr); err != nil {
				return err
			}
			gx, err := proxyLeafLinear(l, current, false)
			if err != nil {
				return err
			}
			if gx != nil {
				current = gx
			}
		}
		if fill != nil {
			state.cache = fill
			state.norm = proxyNorm(gy)
			state.tick++
		}
		return nil
	}
	if variant == 1 {
		gx, dw, err := backward.Cell(head.cell, ghead, head.input, head.pre)
		if err != nil {
			return err
		}
		if err = proxyLeafSGD(head, dw, lr); err != nil {
			return err
		}
		if n == 1 {
			return nil
		}
		if gx == nil || gx.Len() == 0 {
			gx = gy
		}
		return proxyHidden(leaves[:n-1], gx, lr, n-1, -1)
	}
	stale := variant == 5 && state != nil && len(state.proxy) > 0 && n > 1
	if stale {
		src := core.NewTensor[float32](len(state.proxy))
		copy(src.Data, state.proxy)
		if err := proxyHidden(leaves[:n-1], src, lr, n-1, -1); err != nil {
			return err
		}
	}
	var gx *core.Tensor[float32]
	var err error
	if variant != 5 {
		gx, err = proxyLeafLinear(head, gy, true)
		if err != nil {
			return err
		}
	}
	if err = proxyLeafDW(head, ghead, lr); err != nil {
		return err
	}
	if variant == 5 {
		gx, err = proxyLeafLinear(head, gy, true)
		if err != nil {
			return err
		}
		if state != nil && gx != nil {
			state.proxy = append([]float32(nil), gx.Data...)
		}
	}
	if n <= 1 || stale {
		return nil
	}
	if gx == nil || gx.Len() == 0 {
		gx = gy
	}
	if variant == 6 {
		idx := 0
		if state != nil {
			idx = state.sparse % (n - 1)
			state.sparse++
		}
		return proxyHidden(leaves[:n-1], gx, lr, 1, idx)
	}
	return proxyHidden(leaves[:n-1], gx, lr, n-1, -1)
}
func proxyGrid(f meshClockFixture) (*architecture.Grid, error) {
	g := architecture.NewGrid(f.z, f.y, f.x, f.l)
	g.Exec.MultiCore = false
	for i, c := range g.HopOrder() {
		if f.cells[i].op == nil {
			g.At(c.Z, c.Y, c.X, c.L).Layer.IsDisabled = true
			continue
		}
		meta, err := meshMeta(f.cells[i].op)
		if err != nil {
			return nil, err
		}
		if err = g.BindOp(c.Z, c.Y, c.X, c.L, meta, f.cells[i].op); err != nil {
			return nil, err
		}
	}
	return g, nil
}
func runGoProxyVolume(f meshClockFixture, variant int) (meshClockOut, error) {
	var out meshClockOut
	g, err := proxyGrid(f)
	if err != nil {
		return out, err
	}
	input := core.NewTensor[float32](ints(f.shape)...)
	copy(input.Data, f.input)
	target := core.NewTensor[float32](ints(f.targetShape)...)
	copy(target.Data, f.target)
	var accel proxyGoState
	for t := 0; t < f.ticks; t++ {
		fwd, err := forward.Forward(g, input)
		if err != nil {
			return out, err
		}
		snap := make([][]meshTensor, len(f.cells))
		for i := range snap {
			snap[i] = make([]meshTensor, 3)
		}
		for _, st := range fwd.Steps {
			c := st.Coord
			i := ((c.Z*f.y+c.Y)*f.x+c.X)*f.l + c.L
			snap[i] = []meshTensor{meshView(st.Input), meshView(st.Pre), meshView(st.Post)}
		}
		out.history = append(out.history, snap)
		out.post = meshView(fwd.Output)
		out.loss, err = training.MSE(fwd.Output, target)
		if err != nil {
			return out, err
		}
		gy, err := training.MSEGrad(fwd.Output, target)
		if err != nil {
			return out, err
		}
		leaves, err := proxyVolumeLeaves(g, fwd)
		if err != nil {
			return out, err
		}
		if err = proxyApplyLeaves(leaves, gy, meshLR(f), variant, &accel); err != nil {
			return out, err
		}
	}
	out.grads = make([]meshTensor, len(f.cells))
	out.weights = make([][]float32, len(f.cells))
	for i, c := range f.cells {
		if c.op != nil {
			out.weights[i], err = kpPackOp(c.op)
			if err != nil {
				return out, err
			}
		}
	}
	return out, nil
}
func runProxyVolume(pc polyCase, dt core.DType, z, y, x int, mode parallel.TrainMode, epochs ...int) error {
	f, err := meshFactory(pc, dt, z, y, x)
	if err != nil {
		return err
	}
	if z*y*x > 1 && len(f.cells) > 1 && f.cells[0].op == nil {
		brs, _, _, shape, targetShape, e := pc.make(dt)
		if e != nil {
			return e
		}
		if len(shape) == len(targetShape) {
			same := true
			for i := range shape {
				if shape[i] != targetShape[i] {
					same = false
				}
			}
			if same {
				f.cells[0] = brs[0]
			}
		}
	}
	f.ticks = proxyEpochs(mode)
	if len(epochs) > 0 {
		f.ticks = epochs[0]
	}
	f.warmup = 0
	base := encodeMeshClock(f)
	var payload bytes.Buffer
	payload.WriteString("PGB1")
	writeI32(&payload, int32(proxyVariant(mode)))
	payload.Write(base[4:])
	ref, err := runGoProxyVolume(f, proxyVariant(mode))
	if err != nil {
		return fmt.Errorf("Go volume: %w", err)
	}
	got, err := runCUDAMeshClock(payload.Bytes(), 0)
	if err != nil {
		return err
	}
	if err = compareProxyVolume(ref, got); err != nil {
		return err
	}
	measured, err := runCUDAMeshClock(payload.Bytes(), meshBenchIters())
	if err != nil {
		return err
	}
	return meshCompareReplay(got, measured)
}

// Proxy training may accumulate CPU/GPU rounding differences over epochs.
// Use the project's industry-to-exact spectrum while retaining shape/count
// checks and the separate exact same-backend replay gate.
func compareProxyVolume(a, b meshClockOut) error {
	if math.IsNaN(a.loss) || math.IsNaN(b.loss) || math.IsInf(a.loss, 0) || math.IsInf(b.loss, 0) {
		return fmt.Errorf("proxy volume non-finite loss: Go=%g CUDA=%g", a.loss, b.loss)
	}
	if !cudaLossWithinBand(a.loss, b.loss) {
		delta := math.Abs(a.loss - b.loss)
		rel := delta
		if den := math.Max(math.Abs(a.loss), math.Abs(b.loss)); den > 0 {
			rel = delta / den
		}
		band := suites.ClassifySpectrum(suites.DetMetrics{N: 1, MaxAbs: delta, MaxRel: rel})
		if band < suites.DetIndustry {
			return fmt.Errorf("proxy loss below industry: Go=%g CUDA=%g abs=%g rel=%g", a.loss, b.loss, delta, rel)
		}
		fmt.Printf("\n      proxy loss band=%s (%.0f%%) abs=%g rel=%g", band, band.Percent(), delta, rel)
	}
	check := func(label string, x, y meshTensor) error {
		if len(x.data) != len(y.data) || (len(x.data) > 0 && !reflect.DeepEqual(x.shape, y.shape)) {
			return fmt.Errorf("%s shape/length mismatch", label)
		}
		if cudaExactOrSubnormal(x.data, y.data) {
			return nil
		}
		metrics := suites.CompareF32Spectrum(x.data, y.data)
		if metrics.Band < suites.DetIndustry {
			return fmt.Errorf("%s below industry: %s", label, suites.FormatSpectrum(metrics))
		}
		fmt.Printf("\n      %s %s", label, suites.FormatSpectrum(metrics))
		return nil
	}
	if err := check("post", a.post, b.post); err != nil {
		return err
	}
	if err := check("gin", a.gin, b.gin); err != nil {
		return err
	}
	if len(a.weights) != len(b.weights) || len(a.grads) != len(b.grads) || len(a.history) != len(b.history) {
		return fmt.Errorf("proxy volume counts mismatch")
	}
	for i := range a.weights {
		if err := check(fmt.Sprintf("dW[%d]", i), a.grads[i], b.grads[i]); err != nil {
			return err
		}
		x, y := a.weights[i], b.weights[i]
		if err := check(fmt.Sprintf("W[%d]", i), meshTensor{shape: []int32{int32(len(x))}, data: x}, meshTensor{shape: []int32{int32(len(y))}, data: y}); err != nil {
			return err
		}
	}
	for t := range a.history {
		if len(a.history[t]) != len(b.history[t]) {
			return fmt.Errorf("proxy history %d cell count mismatch", t)
		}
		for i := range a.history[t] {
			for k := 0; k < 3; k++ {
				if err := check(fmt.Sprintf("epoch %d cell %d tensor %d", t, i, k), a.history[t][i][k], b.history[t][i][k]); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func CasesProxyVolumes(mode parallel.TrainMode) []Case {
	var cases []Case
	for _, pc := range polyTrainKinds() {
		pc := pc
		for _, top := range []struct {
			name    string
			z, y, x int
			all     bool
		}{{"111", 1, 1, 1, true}, {"211", 2, 1, 1, true}, {"222", 2, 2, 2, false}} {
			top := top
			cases = append(cases, Case{Name: "volume " + pc.name + " " + top.name, Run: func() error {
				types := []core.DType{core.DTypeFloat32}
				if top.all && pc.name != "gdn" {
					types = core.AllDTypes
				}
				for _, dt := range types {
					if err := runProxyVolume(pc, dt, top.z, top.y, top.x, mode); err != nil {
						return fmt.Errorf("%s: %w", dt, err)
					}
				}
				return nil
			}})
		}
	}
	if mode == parallel.ModeTweenSplitLinearCache {
		for _, pc := range polyTrainKinds() {
			pc := pc
			cases = append(cases, Case{Name: "volume cache refresh epoch 20 " + pc.name + " Float32", Run: func() error { return runProxyVolume(pc, core.DTypeFloat32, 2, 1, 1, mode, 22) }})
		}
	}
	return cases
}
