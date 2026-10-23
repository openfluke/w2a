package go_v_qnn

import (
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
)

// hostTiming is Go / native f32 exec-only µs on the same matrix shapes as NPU.
type hostTiming struct {
	goUS, natUS float64
	goOK, natOK bool
	skip        bool
	note        string
}

func measureHosts(layer string) hostTiming {
	switch layer {
	case "dense":
		return hostDense(mx.DenseB, mx.DenseIn, mx.DenseOut)
	case "softmax":
		return hostSoftmax(mx.SoftRows, mx.SoftCols)
	case "rmsnorm":
		return hostRMSNorm(mx.NormB, mx.NormSeq, mx.NormDim)
	case "layernorm":
		return hostLayerNorm(mx.NormB, mx.NormSeq, mx.NormDim)
	case "cnn2":
		return hostCnn2Ref(mx.Cnn2N, mx.Cnn2H, mx.Cnn2W, mx.Cnn2Cin, mx.Cnn2Cout)
	case "cnn3":
		return hostCnn3Ref(mx.Cnn3N, mx.Cnn3D, mx.Cnn3H, mx.Cnn3W, mx.Cnn3Cin, mx.Cnn3Cout)
	case "convt2":
		// QNN convt 1×1 ≈ same spatial GEMV as cnn2 ref.
		return hostCnn2Ref(mx.ConvT2N, mx.ConvT2H, mx.ConvT2W, mx.ConvT2Cin, mx.ConvT2Cout)
	case "convt3":
		return hostCnn3Ref(mx.ConvT3N, mx.ConvT3D, mx.ConvT3H, mx.ConvT3W, mx.ConvT3Cin, mx.ConvT3Cout)
	case "lstm":
		return hostLstmRef(mx.LstmT, mx.LstmB, mx.LstmIn, mx.LstmHid)
	case "embedding":
		return hostEmbedding(mx.EmbVocab, mx.EmbDim, mx.EmbN)
	case "residual":
		return hostResidualAdd(mx.ResN)
	case "swiglu":
		return hostSwiGlu(mx.SwiB, mx.SwiIn, mx.SwiHid)
	case "mha":
		return hostMHA(mx.MhaB, mx.MhaSeq, mx.MhaDim, mx.MhaHeads)
	default:
		return hostTiming{skip: true}
	}
}

func benchBackend(fn func() error) (us float64, ok bool) {
	us, err := benchUS(fn, 30)
	if err != nil {
		return 0, false
	}
	return us, true
}

func hostDense(batch, in, out int) hostTiming {
	W := make([]float32, out*in)
	fillRand(W, 1, 0.02)
	bias := make([]float64, out)
	x := make([]float32, batch*in)
	fillRand(x, 2, 0.1)
	h := hostTiming{}
	mk := func(b core.Backend) (*dense.Layer, *core.Tensor[float32], error) {
		l, err := dense.NewConfigured(in, out, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, W)
		if err != nil {
			return nil, nil, err
		}
		l.Weights.Bias = append([]float64(nil), bias...)
		l.Exec.Backend = b
		l.Exec.MultiCore = false
		t := core.NewTensor[float32](batch, in)
		copy(t.Data, x)
		return l, t, nil
	}
	if goL, goX, err := mk(core.BackendCPUTiled); err == nil {
		_, _, _ = dense.Forward(goL, goX)
		h.goUS, h.goOK = benchBackend(func() error { _, _, e := dense.Forward(goL, goX); return e })
	}
	if nativecpp.NativeCPUAvailable() {
		if natL, natX, err := mk(core.BackendNativeCPU); err == nil && natL.SyncToNative() == nil {
			_, _, _ = dense.Forward(natL, natX)
			h.natUS, h.natOK = benchBackend(func() error { _, _, e := dense.Forward(natL, natX); return e })
		}
	}
	return h
}

func hostSoftmax(rows, cols int) hostTiming {
	x := make([]float32, rows*cols)
	fillRand(x, 3, 1)
	h := hostTiming{}
	mk := func(b core.Backend) (*softmax.Layer, *core.Tensor[float32], error) {
		l, err := softmax.New(softmax.Config{Dim: cols})
		if err != nil {
			return nil, nil, err
		}
		l.Exec.Backend = b
		l.Exec.MultiCore = false
		t := core.NewTensor[float32](rows, cols)
		copy(t.Data, x)
		return l, t, nil
	}
	if goL, goX, err := mk(core.BackendCPUTiled); err == nil {
		_, _, _ = softmax.Forward(goL, goX)
		h.goUS, h.goOK = benchBackend(func() error { _, _, e := softmax.Forward(goL, goX); return e })
	}
	if nativecpp.NativeCPUAvailable() {
		if natL, natX, err := mk(core.BackendNativeCPU); err == nil && natL.SyncToNative() == nil {
			_, _, _ = softmax.Forward(natL, natX)
			h.natUS, h.natOK = benchBackend(func() error { _, _, e := softmax.Forward(natL, natX); return e })
		}
	}
	return h
}

func hostRMSNorm(batch, seq, dim int) hostTiming {
	g := make([]float32, dim)
	for i := range g {
		g[i] = 1
	}
	x := make([]float32, batch*seq*dim)
	fillRand(x, 4, 0.5)
	h := hostTiming{}
	mk := func(b core.Backend) (*rmsnorm.Layer, *core.Tensor[float32], error) {
		l, err := rmsnorm.NewConfigured(rmsnorm.Config{Dim: dim, Eps: 1e-6}, core.DTypeFloat32, quant.FormatNone, g)
		if err != nil {
			return nil, nil, err
		}
		l.Exec.Backend = b
		l.Exec.MultiCore = false
		t := core.NewTensor[float32](batch, seq, dim)
		copy(t.Data, x)
		return l, t, nil
	}
	if goL, goX, err := mk(core.BackendCPUTiled); err == nil {
		_, _, _ = rmsnorm.Forward(goL, goX)
		h.goUS, h.goOK = benchBackend(func() error { _, _, e := rmsnorm.Forward(goL, goX); return e })
	}
	if nativecpp.NativeCPUAvailable() {
		if natL, natX, err := mk(core.BackendNativeCPU); err == nil && natL.SyncToNative() == nil {
			_, _, _ = rmsnorm.Forward(natL, natX)
			h.natUS, h.natOK = benchBackend(func() error { _, _, e := rmsnorm.Forward(natL, natX); return e })
		}
	}
	return h
}

func hostLayerNorm(batch, seq, dim int) hostTiming {
	g := make([]float32, dim)
	beta := make([]float32, dim)
	for i := range g {
		g[i] = 1
	}
	x := make([]float32, batch*seq*dim)
	fillRand(x, 5, 0.4)
	h := hostTiming{}
	mk := func(b core.Backend) (*layernorm.Layer, *core.Tensor[float32], error) {
		l, err := layernorm.NewConfigured(layernorm.Config{Dim: dim, Eps: 1e-5}, core.DTypeFloat32, quant.FormatNone, g, beta)
		if err != nil {
			return nil, nil, err
		}
		l.Exec.Backend = b
		l.Exec.MultiCore = false
		t := core.NewTensor[float32](batch, seq, dim)
		copy(t.Data, x)
		return l, t, nil
	}
	if goL, goX, err := mk(core.BackendCPUTiled); err == nil {
		_, _, _ = layernorm.Forward(goL, goX)
		h.goUS, h.goOK = benchBackend(func() error { _, _, e := layernorm.Forward(goL, goX); return e })
	}
	if nativecpp.NativeCPUAvailable() {
		if natL, natX, err := mk(core.BackendNativeCPU); err == nil && natL.SyncToNative() == nil {
			_, _, _ = layernorm.Forward(natL, natX)
			h.natUS, h.natOK = benchBackend(func() error { _, _, e := layernorm.Forward(natL, natX); return e })
		}
	}
	return h
}

func hostCnn2Ref(n, h, w, cin, cout int) hostTiming {
	W := make([]float32, cout*cin)
	fillRand(W, 6, 0.05)
	x := make([]float32, n*h*w*cin)
	y := make([]float32, n*h*w*cout)
	fillRand(x, 7, 0.1)
	ht := hostTiming{note: "Go = 1×1 spatial GEMV (matches QNN cnn/convt k=1)"}
	ht.goUS, ht.goOK = benchBackend(func() error {
		cnn2Ref1x1(n, h, w, cin, cout, W, x, y)
		return nil
	})
	return ht
}

func hostCnn3Ref(n, d, h, w, cin, cout int) hostTiming {
	W := make([]float32, cout*cin)
	fillRand(W, 8, 0.05)
	x := make([]float32, n*d*h*w*cin)
	y := make([]float32, n*d*h*w*cout)
	fillRand(x, 9, 0.1)
	ht := hostTiming{note: "Go = 1×1×1 spatial GEMV"}
	ht.goUS, ht.goOK = benchBackend(func() error {
		spatial := d * h * w
		for ni := 0; ni < n; ni++ {
			for s := 0; s < spatial; s++ {
				xi := (ni*spatial + s) * cin
				yi := (ni*spatial + s) * cout
				for oc := 0; oc < cout; oc++ {
					var acc float32
					base := oc * cin
					for ic := 0; ic < cin; ic++ {
						acc += W[base+ic] * x[xi+ic]
					}
					y[yi+oc] = acc
				}
			}
		}
		return nil
	})
	return ht
}

// hostLstmRef — naive Go LSTM (same math shape as QNN); native not wired.
func hostLstmRef(seq, batch, inSz, hid int) hostTiming {
	wIn := make([]float32, 4*hid*inSz)
	wRec := make([]float32, 4*hid*hid)
	bias := make([]float32, 4*hid)
	x := make([]float32, seq*batch*inSz)
	y := make([]float32, seq*batch*hid)
	fillRand(wIn, 14, 0.02)
	fillRand(wRec, 15, 0.02)
	fillRand(x, 16, 0.1)
	ht := hostTiming{note: "Go = naive LSTM ref"}
	ht.goUS, ht.goOK = benchBackend(func() error {
		lstmRef(seq, batch, inSz, hid, wIn, wRec, bias, x, y)
		return nil
	})
	return ht
}

func lstmRef(seq, batch, inSz, hid int, wIn, wRec, bias, x, y []float32) {
	hPrev := make([]float32, batch*hid)
	cPrev := make([]float32, batch*hid)
	gates := make([]float32, batch*4*hid)
	for t := 0; t < seq; t++ {
		xOff := t * batch * inSz
		for b := 0; b < batch; b++ {
			for g := 0; g < 4*hid; g++ {
				acc := bias[g]
				for i := 0; i < inSz; i++ {
					acc += wIn[g*inSz+i] * x[xOff+b*inSz+i]
				}
				for i := 0; i < hid; i++ {
					acc += wRec[g*hid+i] * hPrev[b*hid+i]
				}
				gates[b*4*hid+g] = acc
			}
			for i := 0; i < hid; i++ {
				base := b*4*hid + i
				iG := sigmoid32(gates[base])
				fG := sigmoid32(gates[base+hid])
				gG := tanh32(gates[base+2*hid])
				oG := sigmoid32(gates[base+3*hid])
				c := fG*cPrev[b*hid+i] + iG*gG
				h := oG * tanh32(c)
				cPrev[b*hid+i] = c
				hPrev[b*hid+i] = h
				y[t*batch*hid+b*hid+i] = h
			}
		}
	}
}

func sigmoid32(v float32) float32 {
	return float32(1 / (1 + math.Exp(-float64(v))))
}

func tanh32(v float32) float32 {
	return float32(math.Tanh(float64(v)))
}

func hostEmbedding(vocab, dim, n int) hostTiming {
	table := make([]float32, vocab*dim)
	fillRand(table, 17, 0.05)
	ids := make([]float32, n) // embedding Forward reads numeric IDs
	for i := range ids {
		ids[i] = float32(i % vocab)
	}
	h := hostTiming{}
	mk := func(b core.Backend) (*embedding.Layer, *core.Tensor[float32], error) {
		l, err := embedding.NewConfigured(embedding.Config{
			VocabSize: vocab, EmbeddingDim: dim, SeqLen: n,
		}, core.DTypeFloat32, quant.FormatNone, table)
		if err != nil {
			return nil, nil, err
		}
		l.Exec.Backend = b
		l.Exec.MultiCore = false
		t := core.NewTensor[float32](n)
		copy(t.Data, ids)
		return l, t, nil
	}
	if goL, goX, err := mk(core.BackendCPUTiled); err == nil {
		_, _, _ = embedding.Forward(goL, goX)
		h.goUS, h.goOK = benchBackend(func() error { _, _, e := embedding.Forward(goL, goX); return e })
	}
	if nativecpp.NativeCPUAvailable() {
		if natL, natX, err := mk(core.BackendNativeCPU); err == nil {
			_, _, _ = embedding.Forward(natL, natX)
			h.natUS, h.natOK = benchBackend(func() error { _, _, e := embedding.Forward(natL, natX); return e })
		}
	}
	return h
}

func hostResidualAdd(n int) hostTiming {
	a := make([]float32, n)
	b := make([]float32, n)
	y := make([]float32, n)
	fillRand(a, 18, 0.5)
	fillRand(b, 19, 0.5)
	ht := hostTiming{note: "Go = a+b (QNN residual is elementwise add)"}
	ht.goUS, ht.goOK = benchBackend(func() error {
		for i := 0; i < n; i++ {
			y[i] = a[i] + b[i]
		}
		return nil
	})
	return ht
}

func hostSwiGlu(batch, in, hid int) hostTiming {
	wG := make([]float32, hid*in)
	wU := make([]float32, hid*in)
	wD := make([]float32, in*hid)
	x := make([]float32, batch*in)
	fillRand(wG, 20, 0.02)
	fillRand(wU, 21, 0.02)
	fillRand(wD, 22, 0.02)
	fillRand(x, 23, 0.1)
	h := hostTiming{}
	mk := func(b core.Backend) (*swiglu.Layer, *core.Tensor[float32], error) {
		l, err := swiglu.NewConfigured(swiglu.Config{InputDim: in, IntermediateDim: hid},
			core.DTypeFloat32, quant.FormatNone, wG, wU, wD)
		if err != nil {
			return nil, nil, err
		}
		l.Exec.Backend = b
		l.Exec.MultiCore = false
		l.Gate.Exec.Backend = b
		l.Up.Exec.Backend = b
		l.Down.Exec.Backend = b
		t := core.NewTensor[float32](batch, in)
		copy(t.Data, x)
		return l, t, nil
	}
	if goL, goX, err := mk(core.BackendCPUTiled); err == nil {
		_, _, _ = swiglu.Forward(goL, goX)
		h.goUS, h.goOK = benchBackend(func() error { _, _, e := swiglu.Forward(goL, goX); return e })
	}
	if nativecpp.NativeCPUAvailable() {
		if natL, natX, err := mk(core.BackendNativeCPU); err == nil {
			_, _, _ = swiglu.Forward(natL, natX)
			h.natUS, h.natOK = benchBackend(func() error { _, _, e := swiglu.Forward(natL, natX); return e })
		}
	}
	return h
}

func hostMHA(batch, seq, dim, heads int) hostTiming {
	wQ := make([]float32, dim*dim)
	wK := make([]float32, dim*dim)
	wV := make([]float32, dim*dim)
	wO := make([]float32, dim*dim)
	x := make([]float32, batch*seq*dim)
	for i := 0; i < dim; i++ {
		wQ[i*dim+i], wK[i*dim+i], wV[i*dim+i], wO[i*dim+i] = 1, 1, 1, 1
	}
	fillRand(x, 24, 0.05)
	h := hostTiming{}
	mk := func(b core.Backend) (*mha.Layer, *core.Tensor[float32], error) {
		l, err := mha.NewConfigured(mha.Config{
			DModel: dim, NumHeads: heads, MaxSeqLen: seq,
			Mask: mha.MaskBidirectional,
		}, core.DTypeFloat32, quant.FormatNone, wQ, wK, wV, wO)
		if err != nil {
			return nil, nil, err
		}
		l.Exec.Backend = b
		l.Exec.MultiCore = false
		for _, p := range []*dense.Layer{l.Q, l.K, l.V, l.O} {
			p.Exec.Backend = b
			p.Exec.MultiCore = false
		}
		t := core.NewTensor[float32](batch, seq, dim)
		copy(t.Data, x)
		return l, t, nil
	}
	if goL, goX, err := mk(core.BackendCPUTiled); err == nil {
		_, _, _ = mha.Forward(goL, goX)
		h.goUS, h.goOK = benchBackend(func() error { _, _, e := mha.Forward(goL, goX); return e })
	}
	if nativecpp.NativeCPUAvailable() {
		if natL, natX, err := mk(core.BackendNativeCPU); err == nil {
			_, _, _ = mha.Forward(natL, natX)
			h.natUS, h.natOK = benchBackend(func() error { _, _, e := mha.Forward(natL, natX); return e })
		}
	}
	return h
}
