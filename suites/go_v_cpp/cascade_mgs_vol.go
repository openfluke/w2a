package go_v_cpp

import (
	"fmt"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/forward"
)

// volApplyFwd applies a mode after Forward (Tween / Split / Freeze / Chain / …).
type volApplyFwd func(g *architecture.Grid, fwd *forward.Result[float32], target *core.Tensor[float32], lr float64) (float64, error)

// volApplyAlt is TweenAlt (re-forwards internally).
type volApplyAlt func(g *architecture.Grid, input, target *core.Tensor[float32], lr float64) (float64, error)

func runGoMcVolApply(depth, rows, cols, lpc, batch int, cells []mcVolCell, x, target []float32, lr float64, apply volApplyFwd, clearGin bool) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
	hops := g.HopOrder()
	if len(cells) != len(hops) {
		return z, fmt.Errorf("cells %d != hops %d", len(cells), len(hops))
	}
	firstDim, firstSeq := -1, -1
	for i, c := range hops {
		vc := cells[i]
		if !vc.enabled {
			cell := g.At(c.Z, c.Y, c.X, c.L)
			if cell != nil {
				cell.Layer.IsDisabled = true
			}
			continue
		}
		if firstDim < 0 {
			firstDim, firstSeq = vc.dim, vc.seq
		}
		cfg := metacognition.Config{Dim: vc.dim, SeqLen: vc.seq}
		sl, err := metacognition.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableMetacognitionTrain(sl)
		if err := metacognition.Place(g, c.Z, c.Y, c.X, c.L, sl); err != nil {
			return z, err
		}
	}
	var xin *core.Tensor[float32]
	if firstSeq <= 1 {
		xin = core.NewTensor[float32](batch, firstDim)
	} else {
		xin = core.NewTensor[float32](batch, firstSeq, firstDim)
	}
	copy(xin.Data, x)
	fwd, err := forward.Forward(g, xin)
	if err != nil {
		return z, err
	}
	tgt := core.NewTensor[float32](fwd.Output.Shape...)
	copy(tgt.Data, target)
	loss, err := apply(g, fwd, tgt, lr)
	if err != nil {
		return z, err
	}
	z.loss = loss
	z.post = append([]float32(nil), fwd.Output.Data...)
	if clearGin {
		z.gradIn = []float32{}
	}
	for i, c := range hops {
		if !cells[i].enabled {
			continue
		}
		cell := g.At(c.Z, c.Y, c.X, c.L)
		sl, ok := cell.Op.(*metacognition.Layer)
		if !ok {
			return z, fmt.Errorf("expected metacognition at %v", c)
		}
		wa, err := packMetacognitionWeightsF32(sl)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, wa)
	}
	return z, nil
}

func runGoMcVolApplyAlt(depth, rows, cols, lpc, batch int, cells []mcVolCell, x, target []float32, lr float64, apply volApplyAlt) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
	hops := g.HopOrder()
	if len(cells) != len(hops) {
		return z, fmt.Errorf("cells %d != hops %d", len(cells), len(hops))
	}
	firstDim, firstSeq := -1, -1
	for i, c := range hops {
		vc := cells[i]
		if !vc.enabled {
			cell := g.At(c.Z, c.Y, c.X, c.L)
			if cell != nil {
				cell.Layer.IsDisabled = true
			}
			continue
		}
		if firstDim < 0 {
			firstDim, firstSeq = vc.dim, vc.seq
		}
		cfg := metacognition.Config{Dim: vc.dim, SeqLen: vc.seq}
		sl, err := metacognition.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableMetacognitionTrain(sl)
		if err := metacognition.Place(g, c.Z, c.Y, c.X, c.L, sl); err != nil {
			return z, err
		}
	}
	var xin *core.Tensor[float32]
	if firstSeq <= 1 {
		xin = core.NewTensor[float32](batch, firstDim)
	} else {
		xin = core.NewTensor[float32](batch, firstSeq, firstDim)
	}
	copy(xin.Data, x)
	fwd, err := forward.Forward(g, xin)
	if err != nil {
		return z, err
	}
	tgt := core.NewTensor[float32](fwd.Output.Shape...)
	copy(tgt.Data, target)
	loss, err := apply(g, xin, tgt, lr)
	if err != nil {
		return z, err
	}
	z.loss = loss
	z.post = append([]float32(nil), fwd.Output.Data...)
	z.gradIn = []float32{}
	for i, c := range hops {
		if !cells[i].enabled {
			continue
		}
		cell := g.At(c.Z, c.Y, c.X, c.L)
		sl, ok := cell.Op.(*metacognition.Layer)
		if !ok {
			return z, fmt.Errorf("expected metacognition at %v", c)
		}
		wa, err := packMetacognitionWeightsF32(sl)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, wa)
	}
	return z, nil
}

func runGoGdVolApply(depth, rows, cols, lpc, batch int, cells []gdVolCell, x, target []float32, lr float64, apply volApplyFwd, clearGin bool) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
	hops := g.HopOrder()
	if len(cells) != len(hops) {
		return z, fmt.Errorf("cells %d != hops %d", len(cells), len(hops))
	}
	firstH, firstSeq := -1, -1
	for i, c := range hops {
		vc := cells[i]
		if !vc.enabled {
			cell := g.At(c.Z, c.Y, c.X, c.L)
			if cell != nil {
				cell.Layer.IsDisabled = true
			}
			continue
		}
		if firstH < 0 {
			firstH, firstSeq = vc.cfg.HiddenSize, vc.seq
		}
		inQKV, inZ, inB, inA, outW, convW, aLog, dtBias, gamma, err := splitGdnPacked(vc.cfg, vc.W)
		if err != nil {
			return z, err
		}
		gl, err := gdn.NewConfigured(vc.cfg, inQKV, inZ, inB, inA, outW, convW, aLog, dtBias, gamma)
		if err != nil {
			return z, err
		}
		disableGDNTrain(gl)
		if err := gdn.Place(g, c.Z, c.Y, c.X, c.L, gl); err != nil {
			return z, err
		}
	}
	xin := core.NewTensor[float32](batch, firstSeq, firstH)
	copy(xin.Data, x)
	fwd, err := forward.Forward(g, xin)
	if err != nil {
		return z, err
	}
	tgt := core.NewTensor[float32](fwd.Output.Shape...)
	copy(tgt.Data, target)
	loss, err := apply(g, fwd, tgt, lr)
	if err != nil {
		return z, err
	}
	z.loss = loss
	z.post = append([]float32(nil), fwd.Output.Data...)
	if clearGin {
		z.gradIn = []float32{}
	}
	for i, c := range hops {
		if !cells[i].enabled {
			continue
		}
		cell := g.At(c.Z, c.Y, c.X, c.L)
		gl, ok := cell.Op.(*gdn.Layer)
		if !ok {
			return z, fmt.Errorf("expected gdn at %v", c)
		}
		wa, err := packGDNWeightsF32(gl)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, wa)
	}
	return z, nil
}

func runGoGdVolApplyAlt(depth, rows, cols, lpc, batch int, cells []gdVolCell, x, target []float32, lr float64, apply volApplyAlt) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
	hops := g.HopOrder()
	if len(cells) != len(hops) {
		return z, fmt.Errorf("cells %d != hops %d", len(cells), len(hops))
	}
	firstH, firstSeq := -1, -1
	for i, c := range hops {
		vc := cells[i]
		if !vc.enabled {
			cell := g.At(c.Z, c.Y, c.X, c.L)
			if cell != nil {
				cell.Layer.IsDisabled = true
			}
			continue
		}
		if firstH < 0 {
			firstH, firstSeq = vc.cfg.HiddenSize, vc.seq
		}
		inQKV, inZ, inB, inA, outW, convW, aLog, dtBias, gamma, err := splitGdnPacked(vc.cfg, vc.W)
		if err != nil {
			return z, err
		}
		gl, err := gdn.NewConfigured(vc.cfg, inQKV, inZ, inB, inA, outW, convW, aLog, dtBias, gamma)
		if err != nil {
			return z, err
		}
		disableGDNTrain(gl)
		if err := gdn.Place(g, c.Z, c.Y, c.X, c.L, gl); err != nil {
			return z, err
		}
	}
	xin := core.NewTensor[float32](batch, firstSeq, firstH)
	copy(xin.Data, x)
	fwd, err := forward.Forward(g, xin)
	if err != nil {
		return z, err
	}
	tgt := core.NewTensor[float32](fwd.Output.Shape...)
	copy(tgt.Data, target)
	loss, err := apply(g, xin, tgt, lr)
	if err != nil {
		return z, err
	}
	z.loss = loss
	z.post = append([]float32(nil), fwd.Output.Data...)
	z.gradIn = []float32{}
	for i, c := range hops {
		if !cells[i].enabled {
			continue
		}
		cell := g.At(c.Z, c.Y, c.X, c.L)
		gl, ok := cell.Op.(*gdn.Layer)
		if !ok {
			return z, fmt.Errorf("expected gdn at %v", c)
		}
		wa, err := packGDNWeightsF32(gl)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, wa)
	}
	return z, nil
}

func runGoStVolApply(depth, rows, cols, lpc, batch int, cells []stVolCell, x, target []float32, lr float64, apply volApplyFwd, clearGin bool) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
	hops := g.HopOrder()
	if len(cells) != len(hops) {
		return z, fmt.Errorf("cells %d != hops %d", len(cells), len(hops))
	}
	firstDim, firstSeq := -1, -1
	for i, c := range hops {
		vc := cells[i]
		if !vc.enabled {
			cell := g.At(c.Z, c.Y, c.X, c.L)
			if cell != nil {
				cell.Layer.IsDisabled = true
			}
			continue
		}
		if firstDim < 0 {
			firstDim, firstSeq = vc.dim, vc.seq
		}
		sl, err := buildNestedStackDense(vc.dim, vc.depth, vc.dtype, vc.W)
		if err != nil {
			return z, err
		}
		if err := parallel.PlaceStack(g, c.Z, c.Y, c.X, c.L, sl); err != nil {
			return z, err
		}
	}
	if firstSeq <= 0 {
		firstSeq = 1
	}
	xin := core.NewTensor[float32](batch*firstSeq, firstDim)
	copy(xin.Data, x)
	fwd, err := forward.Forward(g, xin)
	if err != nil {
		return z, err
	}
	tgt := core.NewTensor[float32](fwd.Output.Shape...)
	copy(tgt.Data, target)
	loss, err := apply(g, fwd, tgt, lr)
	if err != nil {
		return z, err
	}
	z.loss = loss
	z.post = append([]float32(nil), fwd.Output.Data...)
	if clearGin {
		z.gradIn = []float32{}
	}
	for i, c := range hops {
		if !cells[i].enabled {
			continue
		}
		cell := g.At(c.Z, c.Y, c.X, c.L)
		sl, ok := cell.Op.(*parallel.Stack)
		if !ok {
			return z, fmt.Errorf("expected stack at %v", c)
		}
		wa, err := packStackWeightsF32(sl)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, wa)
	}
	return z, nil
}

func runGoStVolApplyAlt(depth, rows, cols, lpc, batch int, cells []stVolCell, x, target []float32, lr float64, apply volApplyAlt) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
	hops := g.HopOrder()
	if len(cells) != len(hops) {
		return z, fmt.Errorf("cells %d != hops %d", len(cells), len(hops))
	}
	firstDim, firstSeq := -1, -1
	for i, c := range hops {
		vc := cells[i]
		if !vc.enabled {
			cell := g.At(c.Z, c.Y, c.X, c.L)
			if cell != nil {
				cell.Layer.IsDisabled = true
			}
			continue
		}
		if firstDim < 0 {
			firstDim, firstSeq = vc.dim, vc.seq
		}
		sl, err := buildNestedStackDense(vc.dim, vc.depth, vc.dtype, vc.W)
		if err != nil {
			return z, err
		}
		if err := parallel.PlaceStack(g, c.Z, c.Y, c.X, c.L, sl); err != nil {
			return z, err
		}
	}
	if firstSeq <= 0 {
		firstSeq = 1
	}
	xin := core.NewTensor[float32](batch*firstSeq, firstDim)
	copy(xin.Data, x)
	fwd, err := forward.Forward(g, xin)
	if err != nil {
		return z, err
	}
	tgt := core.NewTensor[float32](fwd.Output.Shape...)
	copy(tgt.Data, target)
	loss, err := apply(g, xin, tgt, lr)
	if err != nil {
		return z, err
	}
	z.loss = loss
	z.post = append([]float32(nil), fwd.Output.Data...)
	z.gradIn = []float32{}
	for i, c := range hops {
		if !cells[i].enabled {
			continue
		}
		cell := g.At(c.Z, c.Y, c.X, c.L)
		sl, ok := cell.Op.(*parallel.Stack)
		if !ok {
			return z, fmt.Errorf("expected stack at %v", c)
		}
		wa, err := packStackWeightsF32(sl)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, wa)
	}
	return z, nil
}
