package go_v_cpp

import (
	"fmt"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/forward"
)

// cascadeVolKmCfg — OutputProbabilities with K==D so identical cells chain in volume.
func cascadeVolKmCfg() kmeans.Config {
	return kmeans.Config{
		NumClusters: 4, FeatureDim: 4, Temperature: 1,
		OutputMode: kmeans.OutputProbabilities, Activation: core.ActivationLinear,
	}
}

func kmCascadeVolPostLen(batch, seq int, cfg kmeans.Config) int {
	if seq <= 0 {
		seq = 1
	}
	return batch * seq * cfg.OutDim()
}

func runGoKmVolApply(depth, rows, cols, lpc, batch int, cells []kmVolCell, x, target []float32, lr float64, apply volApplyFwd, clearGin bool) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
	hops := g.HopOrder()
	if len(cells) != len(hops) {
		return z, fmt.Errorf("cells %d != hops %d", len(cells), len(hops))
	}
	firstD, firstSeq := -1, -1
	for i, c := range hops {
		vc := cells[i]
		if !vc.enabled {
			cell := g.At(c.Z, c.Y, c.X, c.L)
			if cell != nil {
				cell.Layer.IsDisabled = true
			}
			continue
		}
		if firstD < 0 {
			firstD, firstSeq = vc.cfg.FeatureDim, vc.seq
		}
		sl, err := kmeans.NewConfigured(vc.cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableKMeansTrain(sl)
		if err := kmeans.Place(g, c.Z, c.Y, c.X, c.L, sl); err != nil {
			return z, err
		}
	}
	var xin *core.Tensor[float32]
	if firstSeq <= 1 {
		xin = core.NewTensor[float32](batch, firstD)
	} else {
		xin = core.NewTensor[float32](batch, firstSeq, firstD)
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
		sl, ok := cell.Op.(*kmeans.Layer)
		if !ok {
			return z, fmt.Errorf("expected kmeans at %v", c)
		}
		wa, err := packKMeansWeightsF32(sl)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, wa)
	}
	return z, nil
}

func runGoKmVolApplyAlt(depth, rows, cols, lpc, batch int, cells []kmVolCell, x, target []float32, lr float64, apply volApplyAlt) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
	hops := g.HopOrder()
	if len(cells) != len(hops) {
		return z, fmt.Errorf("cells %d != hops %d", len(cells), len(hops))
	}
	firstD, firstSeq := -1, -1
	for i, c := range hops {
		vc := cells[i]
		if !vc.enabled {
			cell := g.At(c.Z, c.Y, c.X, c.L)
			if cell != nil {
				cell.Layer.IsDisabled = true
			}
			continue
		}
		if firstD < 0 {
			firstD, firstSeq = vc.cfg.FeatureDim, vc.seq
		}
		sl, err := kmeans.NewConfigured(vc.cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableKMeansTrain(sl)
		if err := kmeans.Place(g, c.Z, c.Y, c.X, c.L, sl); err != nil {
			return z, err
		}
	}
	var xin *core.Tensor[float32]
	if firstSeq <= 1 {
		xin = core.NewTensor[float32](batch, firstD)
	} else {
		xin = core.NewTensor[float32](batch, firstSeq, firstD)
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
		sl, ok := cell.Op.(*kmeans.Layer)
		if !ok {
			return z, fmt.Errorf("expected kmeans at %v", c)
		}
		wa, err := packKMeansWeightsF32(sl)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, wa)
	}
	return z, nil
}
