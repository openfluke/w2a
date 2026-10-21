package go_v_cpp

import (
	"bytes"
	"fmt"
	"os/exec"
	"time"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/layers/convt1"
	"github.com/openfluke/welvet/layers/convt2"
	"github.com/openfluke/welvet/layers/convt3"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/layers/mamba"
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/weights"
)

// ModeTweenSplitSparse volume credit: proxy=Wᵀgy (nil pre); head GradWOnly;
// one hidden at sparse%(N-1) GradWOnly on P(proxy) (no 1/(N-1)).
// Matches systems::tween_split::train_volume_mse_sparse.

func applySparseLeafDW(op any, gy, input, pre *core.Tensor[float32], lr float64) error {
	if op == nil || gy == nil || input == nil {
		return nil
	}
	if d, ok := op.(*dense.Layer); ok {
		dW, err := dense.GradWOnly(d, gy, input, pre)
		if err != nil || dW == nil || dW.Len() == 0 {
			return err
		}
		return dense.ApplyGradSGD(d, dW, lr)
	}
	return applyTweenLocalOp(op, gy, input, pre, lr)
}

func applySparseGradIn(op any, gy, input, pre *core.Tensor[float32]) *core.Tensor[float32] {
	if op == nil || gy == nil {
		return nil
	}
	if d, ok := op.(*dense.Layer); ok && input != nil {
		gx, err := dense.LinearGradIn(d, gy, input)
		if err == nil && gx != nil && gx.Len() > 0 {
			return gx
		}
	}
	if input == nil {
		return nil
	}
	switch l := op.(type) {
	case *rmsnorm.Layer:
		gx, _, err := rmsnorm.Backward(l, gy, input, pre)
		if err == nil {
			return gx
		}
	case *layernorm.Layer:
		gx, _, err := layernorm.Backward(l, gy, input, pre)
		if err == nil {
			return gx
		}
	case *swiglu.Layer:
		gx, _, err := swiglu.Backward(l, gy, input, pre)
		if err == nil {
			return gx
		}
	case *softmax.Layer:
		gx, _, err := softmax.Backward(l, gy, input, pre)
		if err == nil {
			return gx
		}
	case *embedding.Layer:
		gx, _, err := embedding.Backward(l, gy, input, pre)
		if err == nil {
			return gx
		}
	case *mha.Layer:
		gx, _, err := mha.Backward(l, gy, input, pre)
		if err == nil {
			return gx
		}
	case *cnn1.Layer:
		gx, _, err := cnn1.Backward(l, gy, input, pre)
		if err == nil {
			return gx
		}
	case *cnn2.Layer:
		gx, _, err := cnn2.Backward(l, gy, input, pre)
		if err == nil {
			return gx
		}
	case *cnn3.Layer:
		gx, _, err := cnn3.Backward(l, gy, input, pre)
		if err == nil {
			return gx
		}
	case *convt1.Layer:
		gx, _, err := convt1.Backward(l, gy, input, pre)
		if err == nil {
			return gx
		}
	case *convt2.Layer:
		gx, _, err := convt2.Backward(l, gy, input, pre)
		if err == nil {
			return gx
		}
	case *convt3.Layer:
		gx, _, err := convt3.Backward(l, gy, input, pre)
		if err == nil {
			return gx
		}
	case *rnn.Layer:
		gx, _, err := rnn.Backward(l, gy, input, pre)
		if err == nil {
			return gx
		}
	case *lstm.Layer:
		gx, _, err := lstm.Backward(l, gy, input, pre)
		if err == nil {
			return gx
		}
	case *mamba.Layer:
		gx, _, err := mamba.Backward(l, gy, input, pre)
		if err == nil {
			return gx
		}
	case *metacognition.Layer:
		gx, _, err := metacognition.Backward(l, gy, input, pre)
		if err == nil {
			return gx
		}
	case *kmeans.Layer:
		gx, _, err := kmeans.Backward(l, gy, input, pre)
		if err == nil {
			return gx
		}
	case *gdn.Layer:
		gx, _, err := gdn.Backward(l, gy, input, pre)
		if err == nil {
			return gx
		}
	case *parallel.Layer:
		gx, _, err := parallel.Backward(l, gy, input, pre)
		if err == nil {
			return gx
		}
	case *parallel.Stack:
		gx, _, err := parallel.BackwardStack(l, gy, input, pre)
		if err == nil {
			return gx
		}
	case *sequential.Layer:
		gx, _, err := sequential.Backward(l, gy, input, pre)
		if err == nil {
			return gx
		}
	case *residual.Layer:
		gx, _, err := residual.Backward(l, gy, input, pre)
		if err == nil {
			return gx
		}
	}
	return projectGapF32(gy, input.Shape...)
}

func applyModeTweenSplitSparseVolume(g *architecture.Grid, fwd *forward.Result[float32], target *core.Tensor[float32], lr float64) (float64, error) {
	// Matches systems::tween_split::train_volume_mse_sparse / Go trainTweenSplitSparseLeaves.
	// Fresh Grid ⇒ sparse counter 0 → hidden leaf index 0.
	loss, err := training.MSE(fwd.Output, target)
	if err != nil {
		return 0, err
	}
	gy, err := training.MSEGrad(fwd.Output, target)
	if err != nil {
		return loss, err
	}
	type leaf struct {
		op            any
		in, pre, post *core.Tensor[float32]
	}
	var leaves []leaf
	for _, st := range fwd.Steps {
		cell := g.At(st.Coord.Z, st.Coord.Y, st.Coord.X, st.Coord.L)
		if cell == nil || cell.Layer.IsDisabled {
			continue
		}
		switch op := cell.Op.(type) {
		case *sequential.Layer:
			flatIn := flattenSeqLike(st.Input)
			cur := flatIn
			for _, ch := range op.Children {
				p, o, err := dense.Forward(ch, cur)
				if err != nil {
					return loss, err
				}
				leaves = append(leaves, leaf{op: ch, in: cur, pre: p, post: o})
				cur = o
			}
		case *residual.Layer:
			flatIn := flattenSeqLike(st.Input)
			cur := flatIn
			for _, ch := range op.Children {
				p, o, err := dense.Forward(ch, cur)
				if err != nil {
					return loss, err
				}
				leaves = append(leaves, leaf{op: ch, in: cur, pre: p, post: o})
				cur = o
			}
		case *softmax.Layer:
			// weightless — not a leaf
		default:
			leaves = append(leaves, leaf{op: cell.Op, in: st.Input, pre: st.Pre, post: st.Post})
		}
	}
	n := len(leaves)
	if n == 0 {
		return loss, nil
	}
	head := leaves[n-1]
	proxy := applySparseGradIn(head.op, gy, head.in, nil) // before head SGD
	if head.post != nil && head.post.Len() > 0 {
		if err := applySparseLeafDW(head.op, projectGapF32(gy, head.post.Shape...), head.in, head.pre, lr); err != nil {
			return loss, err
		}
	}
	if n == 1 {
		return loss, nil
	}
	idx := 0 // fresh accel.sparse
	src := proxy
	if src == nil || src.Len() == 0 {
		src = gy
	}
	lf := leaves[idx]
	if lf.post == nil || lf.post.Len() == 0 {
		return loss, nil
	}
	gi := projectGapF32(src, lf.post.Shape...)
	if err := applySparseLeafDW(lf.op, gi, lf.in, lf.pre, lr); err != nil {
		return loss, err
	}
	return loss, nil
}


func runCppTweenSplitSparseVolRetarget(payload []byte, inFrom, inTo, outFrom, outTo string, benchIters int) ([]byte, time.Duration, error) {
	payload = swapMagic4(payload, inFrom, inTo)
	tool, err := findDenseFwd()
	if err != nil {
		return nil, 0, err
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
		return nil, 0, fmt.Errorf("cpp tween vol: %v: %s", err, stderr.String())
	}
	raw := rewriteOutMagic(stdout.Bytes(), outFrom, outTo)
	var bench time.Duration
	if len(raw) >= 16 && string(raw[:4]) == "WVB1" {
		// leave WVB1 for decoders that understand it
		_ = bench
	}
	return raw, bench, nil
}

// --- RMSNorm ModeTween volume (from runGoRMSNormVolNBP) ---

func runGoRMSNormVolTweenSplitSparse(depth, rows, cols, lpc, batch int, cells []rnVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
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
		cfg := rmsnorm.Config{Dim: vc.dim, Eps: vc.eps}
		rl, err := rmsnorm.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.gamma)
		if err != nil {
			return z, err
		}
		rl.Exec.Backend = core.BackendCPUTiled
		rl.Exec.MultiCore = false
		rl.Core.MultiCore = false
		if err := g.BindOp(c.Z, c.Y, c.X, c.L, core.Layer{
			Type: core.LayerRMSNorm, DType: vc.dtype, Activation: core.ActivationLinear,
			InputHeight: vc.dim, OutputHeight: vc.dim,
		}, rl); err != nil {
			return z, err
		}
	}
	xin := core.NewTensor[float32](batch, len(x)/batch)
	copy(xin.Data, x)
	fwd, err := forward.Forward(g, xin)
	if err != nil {
		return z, err
	}
	tgt := core.NewTensor[float32](fwd.Output.Shape...)
	copy(tgt.Data, target)
	loss, err := applyModeTweenSplitSparseVolume(g, fwd, tgt, lr)
	if err != nil {
		return z, err
	}
	z.loss = loss
	z.post = append([]float32(nil), fwd.Output.Data...)
	z.gradIn = []float32{} // ModeTweenSplitSparse: no chain-rule gin
	for i, c := range hops {
		if !cells[i].enabled {
			continue
		}
		cell := g.At(c.Z, c.Y, c.X, c.L)
		rl, ok := cell.Op.(*rmsnorm.Layer)
		if !ok {
			return z, fmt.Errorf("expected rmsnorm at %v", c)
		}
		ga, err := weights.MatrixF32(rl.Gamma)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, append([]float32(nil), ga...))
	}
	return z, nil
}

// --- LayerNorm ModeTween volume (from runGoLayerNormVolNBP) ---

func runGoLayerNormVolTweenSplitSparse(depth, rows, cols, lpc, batch int, cells []lnVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
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
		cfg := layernorm.Config{Dim: vc.dim, Eps: vc.eps}
		ll, err := layernorm.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.gamma, vc.beta)
		if err != nil {
			return z, err
		}
		ll.Exec.Backend = core.BackendCPUTiled
		ll.Exec.MultiCore = false
		ll.Core.MultiCore = false
		if err := g.BindOp(c.Z, c.Y, c.X, c.L, core.Layer{
			Type: core.LayerLayerNorm, DType: vc.dtype, Activation: core.ActivationLinear,
			InputHeight: vc.dim, OutputHeight: vc.dim,
		}, ll); err != nil {
			return z, err
		}
	}
	xin := core.NewTensor[float32](batch, len(x)/batch)
	copy(xin.Data, x)
	fwd, err := forward.Forward(g, xin)
	if err != nil {
		return z, err
	}
	tgt := core.NewTensor[float32](fwd.Output.Shape...)
	copy(tgt.Data, target)
	loss, err := applyModeTweenSplitSparseVolume(g, fwd, tgt, lr)
	if err != nil {
		return z, err
	}
	z.loss = loss
	z.post = append([]float32(nil), fwd.Output.Data...)
	z.gradIn = []float32{} // ModeTweenSplitSparse: no chain-rule gin
	for i, c := range hops {
		if !cells[i].enabled {
			continue
		}
		cell := g.At(c.Z, c.Y, c.X, c.L)
		ll, ok := cell.Op.(*layernorm.Layer)
		if !ok {
			return z, fmt.Errorf("expected layernorm at %v", c)
		}
		ga, err := weights.MatrixF32(ll.Gamma)
		if err != nil {
			return z, err
		}
		ba, err := weights.MatrixF32(ll.Beta)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, concatGB(ga, ba))
	}
	return z, nil
}

// --- SwiGLU ModeTween volume (from runGoSwiGLUVolNBP) ---

func runGoSwiGLUVolTweenSplitSparse(depth, rows, cols, lpc, batch int, cells []sgVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
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
		cfg := swiglu.Config{InputDim: vc.in, IntermediateDim: vc.inter}
		sl, err := swiglu.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.gate, vc.up, vc.down)
		if err != nil {
			return z, err
		}
		sl.Exec.Backend = core.BackendCPUTiled
		sl.Exec.MultiCore = false
		sl.Core.MultiCore = false
		sl.Gate.Exec.Backend = core.BackendCPUTiled
		sl.Up.Exec.Backend = core.BackendCPUTiled
		sl.Down.Exec.Backend = core.BackendCPUTiled
		sl.Gate.Exec.MultiCore = false
		sl.Up.Exec.MultiCore = false
		sl.Down.Exec.MultiCore = false
		if err := g.BindOp(c.Z, c.Y, c.X, c.L, core.Layer{
			Type: core.LayerSwiGLU, DType: vc.dtype, Activation: core.ActivationSilu,
			InputHeight: vc.in, OutputHeight: vc.in,
		}, sl); err != nil {
			return z, err
		}
	}
	xin := core.NewTensor[float32](batch, len(x)/batch)
	copy(xin.Data, x)
	fwd, err := forward.Forward(g, xin)
	if err != nil {
		return z, err
	}
	tgt := core.NewTensor[float32](fwd.Output.Shape...)
	copy(tgt.Data, target)
	loss, err := applyModeTweenSplitSparseVolume(g, fwd, tgt, lr)
	if err != nil {
		return z, err
	}
	z.loss = loss
	z.post = append([]float32(nil), fwd.Output.Data...)
	z.gradIn = []float32{} // ModeTweenSplitSparse: no chain-rule gin
	for i, c := range hops {
		if !cells[i].enabled {
			continue
		}
		cell := g.At(c.Z, c.Y, c.X, c.L)
		sl, ok := cell.Op.(*swiglu.Layer)
		if !ok {
			return z, fmt.Errorf("expected swiglu at %v", c)
		}
		Wa, err := exportSwiGLUW(sl)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, append([]float32(nil), Wa...))
	}
	return z, nil
}

// --- Softmax ModeTween volume (from runGoSoftmaxVolNBP) ---

func runGoSoftmaxVolTweenSplitSparse(depth, rows, cols, lpc, batch int, cells []smVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
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
		cfg := vc.cfg
		if err := cfg.Validate(); err != nil {
			return z, err
		}
		sl, err := softmax.New(cfg)
		if err != nil {
			return z, err
		}
		sl.Exec.Backend = core.BackendCPUTiled
		sl.Exec.MultiCore = false
		sl.Core.MultiCore = false
		_ = sl.SetDType(vc.dtype)
		if err := g.BindOp(c.Z, c.Y, c.X, c.L, core.Layer{
			Type: core.LayerSoftmax, DType: vc.dtype, Activation: core.ActivationLinear,
			InputHeight: cfg.Dim, OutputHeight: cfg.Dim,
		}, sl); err != nil {
			return z, err
		}
	}
	xin := core.NewTensor[float32](batch, len(x)/batch)
	copy(xin.Data, x)
	fwd, err := forward.Forward(g, xin)
	if err != nil {
		return z, err
	}
	tgt := core.NewTensor[float32](fwd.Output.Shape...)
	copy(tgt.Data, target)
	loss, err := applyModeTweenSplitSparseVolume(g, fwd, tgt, lr)
	if err != nil {
		return z, err
	}
	z.loss = loss
	z.post = append([]float32(nil), fwd.Output.Data...)
	z.gradIn = []float32{} // ModeTweenSplitSparse: no chain-rule gin
	for i := range hops {
		if !cells[i].enabled {
			continue
		}
		z.Ws = append(z.Ws, []float32{}) // weightless
	}
	return z, nil
}

// --- Embedding ModeTween volume (from runGoEmbeddingVolNBP) ---

func runGoEmbeddingVolTweenSplitSparse(depth, rows, cols, lpc, batch int, cells []embVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
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
		cfg := embedding.Config{VocabSize: vc.vocab, EmbeddingDim: vc.emb, SeqLen: vc.seq}
		el, err := embedding.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		el.Exec.Backend = core.BackendCPUTiled
		el.Exec.MultiCore = false
		el.Core.MultiCore = false
		el.Core.Activation = vc.act
		if err := g.BindOp(c.Z, c.Y, c.X, c.L, core.Layer{
			Type: core.LayerEmbedding, DType: vc.dtype, Activation: vc.act,
			InputHeight: vc.vocab, OutputHeight: vc.emb,
		}, el); err != nil {
			return z, err
		}
	}
	xin := core.NewTensor[float32](batch, len(x)/batch)
	copy(xin.Data, x)
	fwd, err := forward.Forward(g, xin)
	if err != nil {
		return z, err
	}
	tgt := core.NewTensor[float32](fwd.Output.Shape...)
	copy(tgt.Data, target)
	loss, err := applyModeTweenSplitSparseVolume(g, fwd, tgt, lr)
	if err != nil {
		return z, err
	}
	z.loss = loss
	z.post = append([]float32(nil), fwd.Output.Data...)
	z.gradIn = []float32{} // ModeTweenSplitSparse: no chain-rule gin
	for i, c := range hops {
		if !cells[i].enabled {
			continue
		}
		cell := g.At(c.Z, c.Y, c.X, c.L)
		el, ok := cell.Op.(*embedding.Layer)
		if !ok {
			return z, fmt.Errorf("expected embedding at %v", c)
		}
		wa, err := weights.MatrixF32(el.Weights)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, append([]float32(nil), wa...))
	}
	return z, nil
}

// --- MHA ModeTween volume (from runGoMHAVolNBP) ---

func runGoMHAVolTweenSplitSparse(depth, rows, cols, lpc, batch int, cells []mhVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
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
		cfg := vc.cfg
		ml, err := mha.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.q, vc.k, vc.v, vc.o)
		if err != nil {
			return z, err
		}
		ml.Exec.Backend = core.BackendCPUTiled
		ml.Exec.MultiCore = false
		ml.Core.MultiCore = false
		if err := g.BindOp(c.Z, c.Y, c.X, c.L, core.Layer{
			Type: core.LayerMultiHeadAttention, DType: vc.dtype, Activation: core.ActivationLinear,
			InputHeight: cfg.DModel, OutputHeight: cfg.DModel,
		}, ml); err != nil {
			return z, err
		}
	}
	// x layout [batch, seq, d]
	d := cells[0].cfg.DModel
	seq := len(x) / (batch * d)
	xin := core.NewTensor[float32](batch, seq, d)
	copy(xin.Data, x)
	fwd, err := forward.Forward(g, xin)
	if err != nil {
		return z, err
	}
	tgt := core.NewTensor[float32](fwd.Output.Shape...)
	copy(tgt.Data, target)
	loss, err := applyModeTweenSplitSparseVolume(g, fwd, tgt, lr)
	if err != nil {
		return z, err
	}
	z.loss = loss
	z.post = append([]float32(nil), fwd.Output.Data...)
	z.gradIn = []float32{} // ModeTweenSplitSparse: no chain-rule gin
	for i, c := range hops {
		if !cells[i].enabled {
			continue
		}
		cell := g.At(c.Z, c.Y, c.X, c.L)
		ml, ok := cell.Op.(*mha.Layer)
		if !ok {
			return z, fmt.Errorf("expected mha at %v", c)
		}
		wa, err := exportMHAWeights(ml)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, append([]float32(nil), wa...))
	}
	return z, nil
}

// --- RNN ModeTween volume (from runGoRNNVolNBP) ---

func runGoRNNVolTweenSplitSparse(depth, rows, cols, lpc, batch int, cells []rnnVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
	hops := g.HopOrder()
	if len(cells) != len(hops) {
		return z, fmt.Errorf("cells %d != hops %d", len(cells), len(hops))
	}
	firstIn, firstSeq := -1, -1
	for i, c := range hops {
		vc := cells[i]
		if !vc.enabled {
			cell := g.At(c.Z, c.Y, c.X, c.L)
			if cell != nil {
				cell.Layer.IsDisabled = true
			}
			continue
		}
		if firstIn < 0 {
			firstIn, firstSeq = vc.in, vc.seq
		}
		cfg := rnn.Config{InputSize: vc.in, HiddenSize: vc.hid, SeqLen: vc.seq}
		rl, err := rnn.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableRNN(rl)
		if err := rnn.Place(g, c.Z, c.Y, c.X, c.L, rl); err != nil {
			return z, err
		}
	}
	xin := core.NewTensor[float32](batch, firstSeq, firstIn)
	copy(xin.Data, x)
	fwd, err := forward.Forward(g, xin)
	if err != nil {
		return z, err
	}
	tgt := core.NewTensor[float32](fwd.Output.Shape...)
	copy(tgt.Data, target)
	loss, err := applyModeTweenSplitSparseVolume(g, fwd, tgt, lr)
	if err != nil {
		return z, err
	}
	z.loss = loss
	z.post = append([]float32(nil), fwd.Output.Data...)
	z.gradIn = []float32{} // ModeTweenSplitSparse: no chain-rule gin
	for i, c := range hops {
		if !cells[i].enabled {
			continue
		}
		cell := g.At(c.Z, c.Y, c.X, c.L)
		rl, ok := cell.Op.(*rnn.Layer)
		if !ok {
			return z, fmt.Errorf("expected rnn at %v", c)
		}
		wa, err := packRNNWeightsF32(rl)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, wa)
	}
	return z, nil
}

// --- LSTM ModeTween volume (from runGoLSTMVolNBP) ---

func runGoLSTMVolTweenSplitSparse(depth, rows, cols, lpc, batch int, cells []lstmVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
	hops := g.HopOrder()
	if len(cells) != len(hops) {
		return z, fmt.Errorf("cells %d != hops %d", len(cells), len(hops))
	}
	firstIn, firstSeq := -1, -1
	for i, c := range hops {
		vc := cells[i]
		if !vc.enabled {
			cell := g.At(c.Z, c.Y, c.X, c.L)
			if cell != nil {
				cell.Layer.IsDisabled = true
			}
			continue
		}
		if firstIn < 0 {
			firstIn, firstSeq = vc.in, vc.seq
		}
		cfg := lstm.Config{InputSize: vc.in, HiddenSize: vc.hid, SeqLen: vc.seq}
		ll, err := lstm.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableLSTMTrainGates(ll)
		if err := lstm.Place(g, c.Z, c.Y, c.X, c.L, ll); err != nil {
			return z, err
		}
	}
	xin := core.NewTensor[float32](batch, firstSeq, firstIn)
	copy(xin.Data, x)
	fwd, err := forward.Forward(g, xin)
	if err != nil {
		return z, err
	}
	tgt := core.NewTensor[float32](fwd.Output.Shape...)
	copy(tgt.Data, target)
	loss, err := applyModeTweenSplitSparseVolume(g, fwd, tgt, lr)
	if err != nil {
		return z, err
	}
	z.loss = loss
	z.post = append([]float32(nil), fwd.Output.Data...)
	z.gradIn = []float32{} // ModeTweenSplitSparse: no chain-rule gin
	for i, c := range hops {
		if !cells[i].enabled {
			continue
		}
		cell := g.At(c.Z, c.Y, c.X, c.L)
		ll, ok := cell.Op.(*lstm.Layer)
		if !ok {
			return z, fmt.Errorf("expected lstm at %v", c)
		}
		wa, err := packLSTMWeightsF32(ll)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, wa)
	}
	return z, nil
}

// --- Sequential ModeTween volume (from runGoSeqVolNBP) ---

func runGoSeqVolTweenSplitSparse(depth, rows, cols, lpc, batch int, cells []seqVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
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
		cfg := sequential.Config{Dim: vc.dim, SeqLen: vc.seq, Depth: vc.depth}
		sl, err := sequential.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableSequentialTrain(sl)
		if err := sequential.Place(g, c.Z, c.Y, c.X, c.L, sl); err != nil {
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
	loss, err := applyModeTweenSplitSparseVolume(g, fwd, tgt, lr)
	if err != nil {
		return z, err
	}
	z.loss = loss
	z.post = append([]float32(nil), fwd.Output.Data...)
	z.gradIn = []float32{} // ModeTweenSplitSparse: no chain-rule gin
	for i, c := range hops {
		if !cells[i].enabled {
			continue
		}
		cell := g.At(c.Z, c.Y, c.X, c.L)
		sl, ok := cell.Op.(*sequential.Layer)
		if !ok {
			return z, fmt.Errorf("expected sequential at %v", c)
		}
		wa, err := packSequentialWeightsF32(sl)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, wa)
	}
	return z, nil
}

// --- Residual ModeTween volume (from runGoReVolNBP) ---

func runGoReVolTweenSplitSparse(depth, rows, cols, lpc, batch int, cells []reVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
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
		cfg := residual.Config{Dim: vc.dim, SeqLen: vc.seq, Depth: vc.depth}
		sl, err := residual.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableResidualTrain(sl)
		if err := residual.Place(g, c.Z, c.Y, c.X, c.L, sl); err != nil {
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
	loss, err := applyModeTweenSplitSparseVolume(g, fwd, tgt, lr)
	if err != nil {
		return z, err
	}
	z.loss = loss
	z.post = append([]float32(nil), fwd.Output.Data...)
	z.gradIn = []float32{} // ModeTweenSplitSparse: no chain-rule gin
	for i, c := range hops {
		if !cells[i].enabled {
			continue
		}
		cell := g.At(c.Z, c.Y, c.X, c.L)
		sl, ok := cell.Op.(*residual.Layer)
		if !ok {
			return z, fmt.Errorf("expected residual at %v", c)
		}
		wa, err := packResidualWeightsF32(sl)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, wa)
	}
	return z, nil
}

// --- Mamba ModeTween volume (from runGoMbVolNBP) ---

func runGoMbVolTweenSplitSparse(depth, rows, cols, lpc, batch int, cells []mbVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
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
			firstD, firstSeq = vc.dModel, vc.seqLen
		}
		cfg := mamba.Config{DModel: vc.dModel, DState: vc.dState, Expand: vc.expand, SeqLen: vc.seqLen}
		inInit, outInit, aLog, dSkip, err := splitMambaPacked(cfg, vc.W)
		if err != nil {
			return z, err
		}
		ml, err := mamba.NewConfigured(cfg, vc.dtype, quant.FormatNone, inInit, outInit, aLog, dSkip)
		if err != nil {
			return z, err
		}
		disableMambaTrain(ml)
		if err := mamba.Place(g, c.Z, c.Y, c.X, c.L, ml); err != nil {
			return z, err
		}
	}
	xin := core.NewTensor[float32](batch, firstSeq, firstD)
	copy(xin.Data, x)
	fwd, err := forward.Forward(g, xin)
	if err != nil {
		return z, err
	}
	tgt := core.NewTensor[float32](fwd.Output.Shape...)
	copy(tgt.Data, target)
	loss, err := applyModeTweenSplitSparseVolume(g, fwd, tgt, lr)
	if err != nil {
		return z, err
	}
	z.loss = loss
	z.post = append([]float32(nil), fwd.Output.Data...)
	z.gradIn = []float32{} // ModeTweenSplitSparse: no chain-rule gin
	for i, c := range hops {
		if !cells[i].enabled {
			continue
		}
		cell := g.At(c.Z, c.Y, c.X, c.L)
		ml, ok := cell.Op.(*mamba.Layer)
		if !ok {
			return z, fmt.Errorf("expected mamba at %v", c)
		}
		wa, err := packMambaWeightsF32(ml)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, wa)
	}
	return z, nil
}

func finishCNNVolTweenSplitSparse[T any](
	g *architecture.Grid,
	hops []architecture.Coord,
	cells []T,
	xin *core.Tensor[float32],
	target []float32,
	lr float64,
	exportW func(int, *architecture.Cell) ([]float32, error),
) (nbpVolOut, error) {
	var z nbpVolOut
	fwd, err := forward.Forward(g, xin)
	if err != nil {
		return z, err
	}
	tgt := core.NewTensor[float32](fwd.Output.Shape...)
	copy(tgt.Data, target)
	loss, err := applyModeTweenSplitSparseVolume(g, fwd, tgt, lr)
	if err != nil {
		return z, err
	}
	z.loss = loss
	z.post = append([]float32(nil), fwd.Output.Data...)
	z.gradIn = []float32{}
	for i, c := range hops {
		cell := g.At(c.Z, c.Y, c.X, c.L)
		if cell == nil {
			continue
		}
		wa, err := exportW(i, cell)
		if err != nil {
			return z, err
		}
		if wa != nil {
			z.Ws = append(z.Ws, wa)
		}
	}
	return z, nil
}

func runGoCNN1VolTweenSplitSparse(depth, rows, cols, lpc, batch int, cells []cnn1VolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
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
		cfg := vc.cfg
		cl, err := cnn1.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableCNN1(cl)
		if len(vc.bias) > 0 {
			cl.Proj.Weights.Bias = append([]float64(nil), vc.bias...)
		}
		if err := cnn1.Place(g, c.Z, c.Y, c.X, c.L, cl); err != nil {
			return z, err
		}
	}
	cfg := cells[0].cfg
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.SeqLen)
	copy(xin.Data, x)
	return finishCNNVolTweenSplitSparse(g, hops, cells, xin, target, lr, func(i int, cell *architecture.Cell) ([]float32, error) {
		if !cells[i].enabled {
			return nil, nil
		}
		cl, ok := cell.Op.(*cnn1.Layer)
		if !ok {
			return nil, fmt.Errorf("expected cnn1 at hop %d", i)
		}
		return weights.MatrixF32(cl.Proj.Weights)
	})
}

func runGoCNN2VolTweenSplitSparse(depth, rows, cols, lpc, batch int, cells []cnn2VolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
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
		cfg := vc.cfg
		cl, err := cnn2.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableCNN2(cl)
		if len(vc.bias) > 0 {
			cl.Proj.Weights.Bias = append([]float64(nil), vc.bias...)
		}
		if err := cnn2.Place(g, c.Z, c.Y, c.X, c.L, cl); err != nil {
			return z, err
		}
	}
	cfg := cells[0].cfg
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.Height, cfg.Width)
	copy(xin.Data, x)
	return finishCNNVolTweenSplitSparse(g, hops, cells, xin, target, lr, func(i int, cell *architecture.Cell) ([]float32, error) {
		if !cells[i].enabled {
			return nil, nil
		}
		cl, ok := cell.Op.(*cnn2.Layer)
		if !ok {
			return nil, fmt.Errorf("expected cnn2 at hop %d", i)
		}
		return weights.MatrixF32(cl.Proj.Weights)
	})
}

func runGoCNN3VolTweenSplitSparse(depth, rows, cols, lpc, batch int, cells []cnn3VolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
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
		cfg := vc.cfg
		cl, err := cnn3.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableCNN3(cl)
		if len(vc.bias) > 0 {
			cl.Proj.Weights.Bias = append([]float64(nil), vc.bias...)
		}
		if err := cnn3.Place(g, c.Z, c.Y, c.X, c.L, cl); err != nil {
			return z, err
		}
	}
	cfg := cells[0].cfg
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.Depth, cfg.Height, cfg.Width)
	copy(xin.Data, x)
	return finishCNNVolTweenSplitSparse(g, hops, cells, xin, target, lr, func(i int, cell *architecture.Cell) ([]float32, error) {
		if !cells[i].enabled {
			return nil, nil
		}
		cl, ok := cell.Op.(*cnn3.Layer)
		if !ok {
			return nil, fmt.Errorf("expected cnn3 at hop %d", i)
		}
		return weights.MatrixF32(cl.Proj.Weights)
	})
}

func runGoConvT1VolTweenSplitSparse(depth, rows, cols, lpc, batch int, cells []convt1VolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
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
		cfg := vc.cfg
		cl, err := convt1.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableConvT1(cl)
		if err := convt1.Place(g, c.Z, c.Y, c.X, c.L, cl); err != nil {
			return z, err
		}
	}
	cfg := cells[0].cfg
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.SeqLen)
	copy(xin.Data, x)
	return finishCNNVolTweenSplitSparse(g, hops, cells, xin, target, lr, func(i int, cell *architecture.Cell) ([]float32, error) {
		if !cells[i].enabled {
			return nil, nil
		}
		cl, ok := cell.Op.(*convt1.Layer)
		if !ok {
			return nil, fmt.Errorf("expected convt1 at hop %d", i)
		}
		return weights.MatrixF32(cl.Proj.Weights)
	})
}

func runGoConvT2VolTweenSplitSparse(depth, rows, cols, lpc, batch int, cells []convt2VolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
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
		cfg := vc.cfg
		cl, err := convt2.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableConvT2(cl)
		if err := convt2.Place(g, c.Z, c.Y, c.X, c.L, cl); err != nil {
			return z, err
		}
	}
	cfg := cells[0].cfg
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.Height, cfg.Width)
	copy(xin.Data, x)
	return finishCNNVolTweenSplitSparse(g, hops, cells, xin, target, lr, func(i int, cell *architecture.Cell) ([]float32, error) {
		if !cells[i].enabled {
			return nil, nil
		}
		cl, ok := cell.Op.(*convt2.Layer)
		if !ok {
			return nil, fmt.Errorf("expected convt2 at hop %d", i)
		}
		return weights.MatrixF32(cl.Proj.Weights)
	})
}

func runGoConvT3VolTweenSplitSparse(depth, rows, cols, lpc, batch int, cells []convt3VolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
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
		cfg := vc.cfg
		cl, err := convt3.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableConvT3(cl)
		if err := convt3.Place(g, c.Z, c.Y, c.X, c.L, cl); err != nil {
			return z, err
		}
	}
	cfg := cells[0].cfg
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.Depth, cfg.Height, cfg.Width)
	copy(xin.Data, x)
	return finishCNNVolTweenSplitSparse(g, hops, cells, xin, target, lr, func(i int, cell *architecture.Cell) ([]float32, error) {
		if !cells[i].enabled {
			return nil, nil
		}
		cl, ok := cell.Op.(*convt3.Layer)
		if !ok {
			return nil, fmt.Errorf("expected convt3 at hop %d", i)
		}
		return weights.MatrixF32(cl.Proj.Weights)
	})
}

func compareTweenVolSparse(goO, cppO nbpVolOut) error {
	goO.gradIn = []float32{}
	cppO.gradIn = []float32{}
	return compareVolNBP(goO, cppO, true)
}

func caseRMSNormTweenSplitSparseVol222F32() error {
	const depth, rows, cols, lpc, batch, dim, benchIters = 2, 2, 2, 1, 1, 4, 40
	stack := depth * rows * cols * lpc
	gamma, x := makeRMSGammaInit(dim, batch)
	target := make([]float32, dim)
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := make([]rnVolCell, stack)
	for i := range cells {
		cells[i] = rnVolCell{
			enabled: true, dim: dim, dtype: core.DTypeFloat32, eps: 0,
			gamma: append([]float32(nil), gamma...),
		}
	}
	goO, err := runGoRMSNormVolTweenSplitSparse(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeRMSNormVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
	if err != nil {
		return err
	}
	raw, _, err := runCppTweenSplitSparseVolRetarget(payload, "WRV1", "SRP1", "SRPO", "WRVO", 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	cppO, _, err := decodeRMSNormVolNBP(raw)
	if err != nil {
		return err
	}
	if err := compareTweenVolSparse(goO, cppO); err != nil {
		return err
	}
	fmt.Printf("\n      Tween vol 2x2x2x1 %s loss=%.6g exact_bit", "RMSNorm", goO.loss)
	return nil
}

func caseLayerNormTweenSplitSparseVol222F32() error {
	const depth, rows, cols, lpc, batch, dim, benchIters = 2, 2, 2, 1, 1, 4, 40
	stack := depth * rows * cols * lpc
	gamma, beta, x := makeLNAffineInit(dim, batch)
	target := make([]float32, dim)
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := make([]lnVolCell, stack)
	for i := range cells {
		cells[i] = lnVolCell{
			enabled: true, dim: dim, dtype: core.DTypeFloat32, eps: 0,
			gamma: append([]float32(nil), gamma...), beta: append([]float32(nil), beta...),
		}
	}
	goO, err := runGoLayerNormVolTweenSplitSparse(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeLayerNormVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
	if err != nil {
		return err
	}
	raw, _, err := runCppTweenSplitSparseVolRetarget(payload, "WLV1", "SLP1", "SLPO", "WLVO", 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	cppO, _, err := decodeLayerNormVolNBP(raw)
	if err != nil {
		return err
	}
	if err := compareTweenVolSparse(goO, cppO); err != nil {
		return err
	}
	fmt.Printf("\n      Tween vol 2x2x2x1 %s loss=%.6g exact_bit", "LayerNorm", goO.loss)
	return nil
}

func caseSwiGLUTweenSplitSparseVol222F32() error {
	const depth, rows, cols, lpc, batch, in, inter, benchIters = 2, 2, 2, 1, 1, 4, 8, 40
	stack := depth * rows * cols * lpc
	gate, up, down, x := makeSwiGLUTrainInit(in, inter, batch, 0x53574924)
	scaleSwiGLU(gate, up, down, x, 0.05)
	target := make([]float32, in)
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := make([]sgVolCell, stack)
	for i := range cells {
		cells[i] = sgVolCell{
			enabled: true, in: in, inter: inter, dtype: core.DTypeFloat32,
			gate: append([]float32(nil), gate...), up: append([]float32(nil), up...),
			down: append([]float32(nil), down...),
		}
	}
	goO, err := runGoSwiGLUVolTweenSplitSparse(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeSwiGLUVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
	if err != nil {
		return err
	}
	raw, _, err := runCppTweenSplitSparseVolRetarget(payload, "WSV1", "SGP1", "SGPO", "WSVO", 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	cppO, _, err := decodeSwiGLUVolNBP(raw)
	if err != nil {
		return err
	}
	if err := compareTweenVolSparse(goO, cppO); err != nil {
		return err
	}
	fmt.Printf("\n      Tween vol 2x2x2x1 %s loss=%.6g exact_bit", "SwiGLU", goO.loss)
	return nil
}

func caseSoftmaxTweenSplitSparseVol222F32() error {
	const batch, dim, benchIters = 1, 4, 20
	// scale logits so chained Softmax stays numerically calm
	x := smbpRandInit(batch, dim, 0x534d5637)
	for i := range x {
		x[i] *= 0.25
	}
	target := makeSoftmaxNBPTarget(batch*dim, 0x534d5638)
	cfg := softmax.Config{Dim: dim, SeqLen: 1, Kind: softmax.KindStandard, Temperature: 1}
	cells := make([]smVolCell, 8)
	for i := range cells {
		cells[i] = smVolCell{enabled: true, cfg: cfg, dtype: core.DTypeFloat32}
	}
	const lr = 1e-2
	goO, err := runGoSoftmaxVolTweenSplitSparse(2, 2, 2, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeSoftmaxVolNBP(2, 2, 2, 1, batch, cells, x, target, lr)
	if err != nil {
		return err
	}
	raw, _, err := runCppTweenSplitSparseVolRetarget(payload, "WXV1", "SXP1", "SXPO", "WXVO", 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	cppO, _, err := decodeSoftmaxVolNBP(raw)
	if err != nil {
		return err
	}
	if err := compareTweenVolSparse(goO, cppO); err != nil {
		return err
	}
	fmt.Printf("\n      Tween vol 2x2x2x1 %s loss=%.6g exact_bit", "Softmax", goO.loss)
	return nil
}

func caseEmbeddingTweenSplitSparseVol222F32() error {
	const batch, vocab, emb, seq, benchIters = 1, 8, 4, 2, 20
	x := tokenIDs(batch*seq, vocab, 0x454d5637)
	target := make([]float32, batch*seq*emb)
	for i := range target {
		target[i] = float32(0.01 * float64(i%3+1))
	}
	cells := make([]embVolCell, 8)
	for i := range cells {
		W, _ := makeEmbeddingTrainInit(vocab, emb, batch, seq, 0x454d5700+uint32(i))
		cells[i] = embVolCell{
			enabled: true, vocab: vocab, emb: emb, seq: seq,
			dtype: core.DTypeFloat32, act: core.ActivationLinear, W: W,
		}
	}
	const lr = 1e-2
	goO, err := runGoEmbeddingVolTweenSplitSparse(2, 2, 2, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeEmbeddingVolNBP(2, 2, 2, 1, batch, cells, x, target, lr)
	if err != nil {
		return err
	}
	raw, _, err := runCppTweenSplitSparseVolRetarget(payload, "WEV1", "SEP1", "SEPO", "WEVO", 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	cppO, _, err := decodeEmbeddingVolNBP(raw)
	if err != nil {
		return err
	}
	if err := compareTweenVolSparse(goO, cppO); err != nil {
		return err
	}
	fmt.Printf("\n      Tween vol 2x2x2x1 %s loss=%.6g exact_bit", "Embedding", goO.loss)
	return nil
}

func caseMHATweenSplitSparseVol222F32() error {
	const batch, seq, benchIters = 1, 2, 20
	cfg := mha.DecoderCausal(16, 2, 2)
	cfg.MaxSeqLen = 32
	x := randInit(batch*seq, cfg.DModel, 0x4d485638)
	target := make([]float32, batch*seq*cfg.DModel)
	for i := range target {
		target[i] = float32(0.01 * float64(i%3+1))
	}
	cells := make([]mhVolCell, 8)
	for i := range cells {
		c, err := makeMHVolCell(cfg, core.DTypeFloat32, 0x4d485700+uint32(i))
		if err != nil {
			return err
		}
		// Scale down for 8-hop stability (attention product grows fast).
		scaleMHA(c.q, c.k, c.v, c.o, nil, 0.05)
		cells[i] = c
	}
	scaleMHA(nil, nil, nil, nil, x, 0.05)
	const lr = 1e-2
	goO, err := runGoMHAVolTweenSplitSparse(2, 2, 2, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeMHAVolNBP(2, 2, 2, 1, batch, cells, x, target, lr)
	if err != nil {
		return err
	}
	raw, _, err := runCppTweenSplitSparseVolRetarget(payload, "WHV1", "SMP1", "SMPO", "WHVO", 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	cppO, _, err := decodeMHAVolNBP(raw)
	if err != nil {
		return err
	}
	if err := compareTweenVolSparse(goO, cppO); err != nil {
		return err
	}
	fmt.Printf("\n      Tween vol 2x2x2x1 %s loss=%.6g exact_bit", "MHA", goO.loss)
	return nil
}

func caseRNNTweenSplitSparseVol222F32() error {
	const depth, rows, cols, lpc, batch, benchIters = 2, 2, 2, 1, 1, 20
	stack := depth * rows * cols * lpc
	const lr = 1e-2
	cfg := volRNNCfg()
	W := randInit(1, cfg.WeightCount(), 0x524e5638)
	for i := range W {
		W[i] *= 0.05
	}
	x := randInit(batch*cfg.SeqLen, cfg.InputSize, 0x524e5639)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, rnnVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := make([]rnnVolCell, stack)
	for i := range cells {
		cells[i] = rnnVolCell{
			enabled: true, in: cfg.InputSize, hid: cfg.HiddenSize, seq: cfg.SeqLen,
			dtype: core.DTypeFloat32, W: append([]float32(nil), W...),
		}
	}
	goO, err := runGoRNNVolTweenSplitSparse(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeRNNVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return err
	}
	raw, _, err := runCppTweenSplitSparseVolRetarget(payload, "RNV1", "RNSP", "RNSK", "RNVO", 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	cppO, _, err := decodeRNNVolNBP(raw)
	if err != nil {
		return err
	}
	if err := compareTweenVolSparse(goO, cppO); err != nil {
		return err
	}
	fmt.Printf("\n      Tween vol 2x2x2x1 %s loss=%.6g exact_bit", "RNN", goO.loss)
	return nil
}

func caseLSTMTweenSplitSparseVol222F32() error {
	const depth, rows, cols, lpc, batch, benchIters = 2, 2, 2, 1, 1, 20
	stack := depth * rows * cols * lpc
	const lr = 1e-2
	cfg := volLSTMCfg()
	W := randInit(1, cfg.WeightCount(), 0x4c535638)
	for i := range W {
		W[i] *= 0.05
	}
	x := randInit(batch*cfg.SeqLen, cfg.InputSize, 0x4c535639)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, lstmVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := make([]lstmVolCell, stack)
	for i := range cells {
		cells[i] = lstmVolCell{
			enabled: true, in: cfg.InputSize, hid: cfg.HiddenSize, seq: cfg.SeqLen,
			dtype: core.DTypeFloat32, W: append([]float32(nil), W...),
		}
	}
	goO, err := runGoLSTMVolTweenSplitSparse(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeLSTMVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return err
	}
	raw, _, err := runCppTweenSplitSparseVolRetarget(payload, "LSV1", "LSSP", "LSSK", "LSVO", 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	cppO, _, err := decodeLSTMVolNBP(raw)
	if err != nil {
		return err
	}
	if err := compareTweenVolSparse(goO, cppO); err != nil {
		return err
	}
	fmt.Printf("\n      Tween vol 2x2x2x1 %s loss=%.6g exact_bit", "LSTM", goO.loss)
	return nil
}

func caseSeqTweenSplitSparseVol222F32() error {
	const depth, rows, cols, lpc, batch, benchIters = 2, 2, 2, 1, 1, 20
	stack := depth * rows * cols * lpc
	const lr = 1e-2
	cfg := volSeqCfg()
	W := randInit(1, cfg.WeightCount(), 0x53515638)
	for i := range W {
		W[i] *= 0.05
	}
	x := randInit(batch*cfg.SeqLen, cfg.Dim, 0x53515639)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, seqVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := make([]seqVolCell, stack)
	for i := range cells {
		cells[i] = seqVolCell{
			enabled: true, dim: cfg.Dim, seq: cfg.SeqLen, depth: cfg.Depth,
			dtype: core.DTypeFloat32, W: append([]float32(nil), W...),
		}
	}
	goO, err := runGoSeqVolTweenSplitSparse(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeSeqVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return err
	}
	raw, _, err := runCppTweenSplitSparseVolRetarget(payload, "SQV1", "SQSP", "SQSK", "SQVO", 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	cppO, _, err := decodeSeqVolNBP(raw)
	if err != nil {
		return err
	}
	if err := compareTweenVolSparse(goO, cppO); err != nil {
		return err
	}
	fmt.Printf("\n      Tween vol 2x2x2x1 %s loss=%.6g exact_bit", "Sequential", goO.loss)
	return nil
}

func caseReTweenSplitSparseVol222F32() error {
	const depth, rows, cols, lpc, batch, benchIters = 2, 2, 2, 1, 1, 20
	stack := depth * rows * cols * lpc
	const lr = 1e-2
	cfg := volReCfg()
	W := randInit(1, cfg.WeightCount(), 0x52455638)
	for i := range W {
		W[i] *= 0.05
	}
	x := randInit(batch*cfg.SeqLen, cfg.Dim, 0x52455639)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, reVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := make([]reVolCell, stack)
	for i := range cells {
		cells[i] = reVolCell{
			enabled: true, dim: cfg.Dim, seq: cfg.SeqLen, depth: cfg.Depth,
			dtype: core.DTypeFloat32, W: append([]float32(nil), W...),
		}
	}
	goO, err := runGoReVolTweenSplitSparse(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeReVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return err
	}
	raw, _, err := runCppTweenSplitSparseVolRetarget(payload, "REV1", "RESP", "RESK", "REVO", 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	cppO, _, err := decodeReVolNBP(raw)
	if err != nil {
		return err
	}
	if err := compareTweenVolSparse(goO, cppO); err != nil {
		return err
	}
	fmt.Printf("\n      Tween vol 2x2x2x1 %s loss=%.6g exact_bit", "Residual", goO.loss)
	return nil
}

func caseMbTweenSplitSparseVol222F32() error {
	const depth, rows, cols, lpc, batch, benchIters = 2, 2, 2, 1, 1, 20
	stack := depth * rows * cols * lpc
	const lr = 1e-2
	cfg := volMbCfg()
	W := randInit(1, mambaWeightCount(cfg), 0x4d425638)
	for i := range W {
		W[i] *= 0.05
	}
	x := randInit(batch*cfg.SeqLen, cfg.DModel, 0x4d425639)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, mbVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := make([]mbVolCell, stack)
	for i := range cells {
		cells[i] = mbVolCell{
			enabled: true, dModel: cfg.DModel, dState: cfg.DState, expand: cfg.Expand, seqLen: cfg.SeqLen,
			dtype: core.DTypeFloat32, W: append([]float32(nil), W...),
		}
	}
	goO, err := runGoMbVolTweenSplitSparse(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeMbVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return err
	}
	raw, _, err := runCppTweenSplitSparseVolRetarget(payload, "MBV1", "MBSP", "MBSK", "MBVO", 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	cppO, _, err := decodeMbVolNBP(raw)
	if err != nil {
		return err
	}
	if err := compareTweenVolSparse(goO, cppO); err != nil {
		return err
	}
	fmt.Printf("\n      Tween vol 2x2x2x1 %s loss=%.6g exact_bit", "Mamba", goO.loss)
	return nil
}

func cnnVolMagicsSparse(tier int) (inFrom, outTo string) {
	switch tier {
	case 1:
		return "C1V1", "C1VO"
	case 2:
		return "C2V1", "C2VO"
	case 3:
		return "C3V1", "C3VO"
	}
	return "", ""
}

func cnnTweenVolMagicsSparse(tier int) (inTo, outFrom string) {
	switch tier {
	case 1:
		return "C1SP", "C1SK"
	case 2:
		return "C2SP", "C2SK"
	case 3:
		return "C3SP", "C3SK"
	}
	return "", ""
}

func convtVolMagicsSparse(tier int) (inFrom, outTo string) {
	switch tier {
	case 1:
		return "T1V1", "T1VO"
	case 2:
		return "T2V1", "T2VO"
	case 3:
		return "T3V1", "T3VO"
	}
	return "", ""
}

func convtTweenVolMagicsSparse(tier int) (inTo, outFrom string) {
	switch tier {
	case 1:
		return "T1SP", "T1SK"
	case 2:
		return "T2SP", "T2SK"
	case 3:
		return "T3SP", "T3SK"
	}
	return "", ""
}

func caseCNNTweenSplitSparseVol222F32(tier int) error {
	const depth, rows, cols, lpc, batch = 2, 2, 2, 1, 1
	stack := depth * rows * cols * lpc
	const lr = 1e-2
	inFrom, outTo := cnnVolMagicsSparse(tier)
	inTo, outFrom := cnnTweenVolMagicsSparse(tier)
	switch tier {
	case 1:
		cfg := preservingCNN1()
		W := randInit(cfg.Filters, cfg.PatchDim(), 0x54572200)
		for i := range W {
			W[i] *= 0.05
		}
		x := randInit(batch*cfg.InChannels, cfg.SeqLen, 0x54572201)
		for i := range x {
			x[i] *= 0.05
		}
		target := make([]float32, cnn1VolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.05 * float64(i+1))
		}
		cells := make([]cnn1VolCell, stack)
		for i := range cells {
			cells[i] = cnn1VolCell{enabled: true, cfg: cfg, dtype: core.DTypeFloat32, W: append([]float32(nil), W...)}
		}
		goO, err := runGoCNN1VolTweenSplitSparse(depth, rows, cols, lpc, batch, cells, x, target, lr)
		if err != nil {
			return fmt.Errorf("go: %w", err)
		}
		payload, err := encodeCNN1VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		if err != nil {
			return err
		}
		raw, _, err := runCppTweenSplitSparseVolRetarget(payload, inFrom, inTo, outFrom, outTo, 0)
		if err != nil {
			return fmt.Errorf("cpp: %w", err)
		}
		cppO, _, err := decodeCNNVolNBP(raw, tier)
		if err != nil {
			return err
		}
		if err := compareTweenVolSparse(goO, cppO); err != nil {
			return err
		}
		fmt.Printf("\n      Tween vol 2x2x2x1 CNN1 loss=%.6g exact_bit", goO.loss)
		return nil
	case 2:
		cfg := preservingCNN2()
		W := randInit(cfg.Filters, cfg.PatchDim(), 0x54572210)
		for i := range W {
			W[i] *= 0.05
		}
		x := randInit(batch*cfg.InChannels*cfg.Height, cfg.Width, 0x54572211)
		for i := range x {
			x[i] *= 0.05
		}
		target := make([]float32, cnn2VolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.05 * float64(i+1))
		}
		cells := make([]cnn2VolCell, stack)
		for i := range cells {
			cells[i] = cnn2VolCell{enabled: true, cfg: cfg, dtype: core.DTypeFloat32, W: append([]float32(nil), W...)}
		}
		goO, err := runGoCNN2VolTweenSplitSparse(depth, rows, cols, lpc, batch, cells, x, target, lr)
		if err != nil {
			return fmt.Errorf("go: %w", err)
		}
		payload, err := encodeCNN2VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		if err != nil {
			return err
		}
		raw, _, err := runCppTweenSplitSparseVolRetarget(payload, inFrom, inTo, outFrom, outTo, 0)
		if err != nil {
			return fmt.Errorf("cpp: %w", err)
		}
		cppO, _, err := decodeCNNVolNBP(raw, tier)
		if err != nil {
			return err
		}
		if err := compareTweenVolSparse(goO, cppO); err != nil {
			return err
		}
		fmt.Printf("\n      Tween vol 2x2x2x1 CNN2 loss=%.6g exact_bit", goO.loss)
		return nil
	case 3:
		cfg := preservingCNN3()
		W := randInit(cfg.Filters, cfg.PatchDim(), 0x54572220)
		for i := range W {
			W[i] *= 0.05
		}
		x := randInit(cfg.InChannels*cfg.Depth*cfg.Height, cfg.Width, 0x54572221)
		for i := range x {
			x[i] *= 0.05
		}
		target := make([]float32, cnn3VolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.05 * float64(i+1))
		}
		cells := make([]cnn3VolCell, stack)
		for i := range cells {
			cells[i] = cnn3VolCell{enabled: true, cfg: cfg, dtype: core.DTypeFloat32, W: append([]float32(nil), W...)}
		}
		goO, err := runGoCNN3VolTweenSplitSparse(depth, rows, cols, lpc, batch, cells, x, target, lr)
		if err != nil {
			return fmt.Errorf("go: %w", err)
		}
		payload, err := encodeCNN3VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		if err != nil {
			return err
		}
		raw, _, err := runCppTweenSplitSparseVolRetarget(payload, inFrom, inTo, outFrom, outTo, 0)
		if err != nil {
			return fmt.Errorf("cpp: %w", err)
		}
		cppO, _, err := decodeCNNVolNBP(raw, tier)
		if err != nil {
			return err
		}
		if err := compareTweenVolSparse(goO, cppO); err != nil {
			return err
		}
		fmt.Printf("\n      Tween vol 2x2x2x1 CNN3 loss=%.6g exact_bit", goO.loss)
		return nil
	default:
		return fmt.Errorf("bad cnn tier %d", tier)
	}
}

func caseConvTTweenSplitSparseVol222F32(tier int) error {
	const depth, rows, cols, lpc, batch = 2, 2, 2, 1, 1
	stack := depth * rows * cols * lpc
	const lr = 1e-2
	inFrom, outTo := convtVolMagicsSparse(tier)
	inTo, outFrom := convtTweenVolMagicsSparse(tier)
	switch tier {
	case 1:
		cfg := preservingConvT1()
		W := randInit(cfg.Filters, cfg.PatchDim(), 0x54572200)
		for i := range W {
			W[i] *= 0.05
		}
		x := randInit(batch*cfg.InChannels, cfg.SeqLen, 0x54572201)
		for i := range x {
			x[i] *= 0.05
		}
		target := make([]float32, convt1VolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.05 * float64(i+1))
		}
		cells := make([]convt1VolCell, stack)
		for i := range cells {
			cells[i] = convt1VolCell{enabled: true, cfg: cfg, dtype: core.DTypeFloat32, W: append([]float32(nil), W...)}
		}
		goO, err := runGoConvT1VolTweenSplitSparse(depth, rows, cols, lpc, batch, cells, x, target, lr)
		if err != nil {
			return fmt.Errorf("go: %w", err)
		}
		payload, err := encodeConvT1VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		if err != nil {
			return err
		}
		raw, _, err := runCppTweenSplitSparseVolRetarget(payload, inFrom, inTo, outFrom, outTo, 0)
		if err != nil {
			return fmt.Errorf("cpp: %w", err)
		}
		cppO, _, err := decodeConvTVolNBP(raw, tier)
		if err != nil {
			return err
		}
		if err := compareTweenVolSparse(goO, cppO); err != nil {
			return err
		}
		fmt.Printf("\n      Tween vol 2x2x2x1 ConvT1 loss=%.6g exact_bit", goO.loss)
		return nil
	case 2:
		cfg := preservingConvT2()
		W := randInit(cfg.Filters, cfg.PatchDim(), 0x54572210)
		for i := range W {
			W[i] *= 0.05
		}
		x := randInit(cfg.InChannels*cfg.Height, cfg.Width, 0x54572211)
		for i := range x {
			x[i] *= 0.05
		}
		target := make([]float32, convt2VolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.05 * float64(i+1))
		}
		cells := make([]convt2VolCell, stack)
		for i := range cells {
			cells[i] = convt2VolCell{enabled: true, cfg: cfg, dtype: core.DTypeFloat32, W: append([]float32(nil), W...)}
		}
		goO, err := runGoConvT2VolTweenSplitSparse(depth, rows, cols, lpc, batch, cells, x, target, lr)
		if err != nil {
			return fmt.Errorf("go: %w", err)
		}
		payload, err := encodeConvT2VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		if err != nil {
			return err
		}
		raw, _, err := runCppTweenSplitSparseVolRetarget(payload, inFrom, inTo, outFrom, outTo, 0)
		if err != nil {
			return fmt.Errorf("cpp: %w", err)
		}
		cppO, _, err := decodeConvTVolNBP(raw, tier)
		if err != nil {
			return err
		}
		if err := compareTweenVolSparse(goO, cppO); err != nil {
			return err
		}
		fmt.Printf("\n      Tween vol 2x2x2x1 ConvT2 loss=%.6g exact_bit", goO.loss)
		return nil
	case 3:
		cfg := preservingConvT3()
		W := randInit(cfg.Filters, cfg.PatchDim(), 0x54572220)
		for i := range W {
			W[i] *= 0.05
		}
		x := randInit(cfg.InChannels*cfg.Depth*cfg.Height, cfg.Width, 0x54572221)
		for i := range x {
			x[i] *= 0.05
		}
		target := make([]float32, convt3VolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.05 * float64(i+1))
		}
		cells := make([]convt3VolCell, stack)
		for i := range cells {
			cells[i] = convt3VolCell{enabled: true, cfg: cfg, dtype: core.DTypeFloat32, W: append([]float32(nil), W...)}
		}
		goO, err := runGoConvT3VolTweenSplitSparse(depth, rows, cols, lpc, batch, cells, x, target, lr)
		if err != nil {
			return fmt.Errorf("go: %w", err)
		}
		payload, err := encodeConvT3VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		if err != nil {
			return err
		}
		raw, _, err := runCppTweenSplitSparseVolRetarget(payload, inFrom, inTo, outFrom, outTo, 0)
		if err != nil {
			return fmt.Errorf("cpp: %w", err)
		}
		cppO, _, err := decodeConvTVolNBP(raw, tier)
		if err != nil {
			return err
		}
		if err := compareTweenVolSparse(goO, cppO); err != nil {
			return err
		}
		fmt.Printf("\n      Tween vol 2x2x2x1 ConvT3 loss=%.6g exact_bit", goO.loss)
		return nil
	default:
		return fmt.Errorf("bad convt tier %d", tier)
	}
}
// --- CASCADE Meta/GDN/Stack (auto) ---
func caseMcTweenSplitSparseVol222F32() error {
	const depth, rows, cols, lpc, batch = 2, 2, 2, 1, 1
	stack := depth * rows * cols * lpc
	const lr = 1e-2
	cfg := volMcCfg()
	W := randInit(1, mcWeightCount(cfg), 0x4d435638)
	for i := range W {
		W[i] *= 0.05
	}
	x := randInit(batch*cfg.SeqLen, cfg.Dim, 0x4d435639)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, mcVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := make([]mcVolCell, stack)
	for i := range cells {
		cells[i] = mcVolCell{
			enabled: true, dim: cfg.Dim, seq: cfg.SeqLen,
			dtype: core.DTypeFloat32, W: append([]float32(nil), W...),
		}
	}
	goO, err := runGoMcVolApply(depth, rows, cols, lpc, batch, cells, x, target, lr, applyModeTweenSplitSparseVolume, true)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeMcVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return err
	}
	raw, _, err := runCppTweenSplitSparseVolRetarget(payload, "MCV1", "MCSP", "MCPK", "MCVO", 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	cppO, _, err := decodeMcVolNBP(raw)
	if err != nil {
		return err
	}
	if err := compareTweenVolSparse(goO, cppO); err != nil {
		return err
	}
	fmt.Printf("\n      TweenSplitSparse vol 2x2x2x1 %s loss=%.6g exact_bit", "Metacognition", goO.loss)
	return nil
}

func caseGdTweenSplitSparseVol222F32() error {
	const depth, rows, cols, lpc, batch = 2, 2, 2, 1, 1
	stack := depth * rows * cols * lpc
	const lr = 1e-2
	cfg := volGdCfg()
	seq := 2
	W := randInit(1, gdnWeightCount(cfg), 0x47445638)
	for i := range W {
		W[i] *= 0.05
	}
	x := randInit(batch*seq, cfg.HiddenSize, 0x47445639)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, batch*seq*cfg.HiddenSize)
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := make([]gdVolCell, stack)
	for i := range cells {
		cells[i] = gdVolCell{
			enabled: true, cfg: cfg, seq: seq, W: append([]float32(nil), W...),
		}
	}
	goO, err := runGoGdVolApply(depth, rows, cols, lpc, batch, cells, x, target, lr, applyModeTweenSplitSparseVolume, true)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeGdVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return err
	}
	raw, _, err := runCppTweenSplitSparseVolRetarget(payload, "GDV1", "GDSP", "GDPK", "GDVO", 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	cppO, _, err := decodeGdVolNBP(raw)
	if err != nil {
		return err
	}
	if err := compareTweenVolSparse(goO, cppO); err != nil {
		return err
	}
	fmt.Printf("\n      TweenSplitSparse vol 2x2x2x1 %s loss=%.6g exact_bit", "GDN", goO.loss)
	return nil
}

func caseSkTweenSplitSparseVol222F32() error {
	const depth, rows, cols, lpc, batch = 2, 2, 2, 1, 1
	stack := depth * rows * cols * lpc
	const lr = 1e-2
	dim, nChildren, seq := volStCfg()
	W := randInit(1, stWeightCount(dim, nChildren), 0x53544b38)
	for i := range W {
		W[i] *= 0.05
	}
	x := randInit(batch*seq, dim, 0x53544b39)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, batch*seq*dim)
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := make([]stVolCell, stack)
	for i := range cells {
		cells[i] = stVolCell{
			enabled: true, dim: dim, seq: seq, depth: nChildren,
			dtype: core.DTypeFloat32, W: append([]float32(nil), W...),
		}
	}
	goO, err := runGoStVolApply(depth, rows, cols, lpc, batch, cells, x, target, lr, applyModeTweenSplitSparseVolume, true)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeStVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return err
	}
	raw, _, err := runCppTweenSplitSparseVolRetarget(payload, "STKV", "SKSP", "SKPK", "STVO", 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	cppO, _, err := decodeStVolNBP(raw)
	if err != nil {
		return err
	}
	if err := compareTweenVolSparse(goO, cppO); err != nil {
		return err
	}
	fmt.Printf("\n      TweenSplitSparse vol 2x2x2x1 %s loss=%.6g exact_bit", "NestedStack", goO.loss)
	return nil
}

// --- CASCADE KMeans vol (auto) ---
func caseKmTweenSplitSparseVol222F32() error {
	const depth, rows, cols, lpc, batch = 2, 2, 2, 1, 1
	stack := depth * rows * cols * lpc
	const lr = 1e-2
	cfg := cascadeVolKmCfg()
	seq := 2
	W := randInit(cfg.NumClusters, cfg.FeatureDim, 0x4b4d5638)
	for i := range W {
		W[i] *= 0.05
	}
	x := randInit(batch*seq, cfg.FeatureDim, 0x4b4d5639)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, kmCascadeVolPostLen(batch, seq, cfg))
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := make([]kmVolCell, stack)
	for i := range cells {
		cells[i] = kmVolCell{
			enabled: true, cfg: cfg, seq: seq,
			dtype: core.DTypeFloat32, W: append([]float32(nil), W...),
		}
	}
	goO, err := runGoKmVolApply(depth, rows, cols, lpc, batch, cells, x, target, lr, applyModeTweenSplitSparseVolume, true)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeKmVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return err
	}
	raw, _, err := runCppTweenSplitSparseVolRetarget(payload, "KMV1", "KMSP", "KMPK", "KMVO", 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	cppO, _, err := decodeKmVolNBP(raw)
	if err != nil {
		return err
	}
	if err := compareTweenVol(goO, cppO); err != nil {
		return err
	}
	fmt.Printf("\n      TweenSplitSparse vol 2x2x2x1 %s loss=%.6g exact_bit", "KMeans", goO.loss)
	return nil
}

