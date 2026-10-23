package go_v_qnn

import (
	"fmt"
	"math/rand"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/nativeqnn"
	"github.com/openfluke/welvet/quant"
)

func fillRand(dst []float32, seed int64, scale float32) {
	r := rand.New(rand.NewSource(seed))
	for i := range dst {
		dst[i] = (r.Float32()*2 - 1) * scale
	}
}

func cloneF32(src []float32) []float32 {
	return append([]float32(nil), src...)
}

// measureOnline runs a one-shot QNN forward that rebuilds the graph each call.
// Transfer is timed separately; create/finalize/exec come from LastPhase (exec is headline).
func measureOnline(r *LayerReport, sess *nativeqnn.Session, transfer func(), run func() error) {
	if transfer != nil {
		r.TransferUS = timeUS(transfer)
	}
	if err := run(); err != nil {
		r.Err = err
		return
	}
	if err := run(); err != nil {
		r.Err = err
		return
	}
	takePhase(r, sess)
	finishE2E(r)
}

// measureCompose times multi-graph layers (LSTM/MHA/SwiGLU). LastPhase only covers the
// last subgraph, so we report wall as e2e prep+exec lumped into create (exec=0 until prepared).
func measureCompose(r *LayerReport, sess *nativeqnn.Session, transfer func(), run func() error) {
	if transfer != nil {
		r.TransferUS = timeUS(transfer)
	}
	_ = sess
	if err := run(); err != nil {
		r.Err = err
		return
	}
	wall := timeUS(func() {
		if err := run(); err != nil {
			r.Err = err
		}
	})
	if r.Err != nil {
		return
	}
	r.CreateUS = wall
	r.FinalizeUS = 0
	r.ExecUS = 0
	finishE2E(r)
}

func measureDenseBig(sess *nativeqnn.Session) LayerReport {
	const batch, in, out = 64, 1024, 1024
	r := LayerReport{Name: "dense", Geom: fmt.Sprintf("B=%d in=%d out=%d", batch, in, out)}
	W := make([]float32, out*in)
	bias := make([]float32, out)
	x := make([]float32, batch*in)
	fillRand(W, 1, 0.02)
	fillRand(bias, 2, 0.01)
	fillRand(x, 3, 0.1)
	yQ := make([]float32, batch*out)
	yG := make([]float32, batch*out)

	r.TransferUS = timeUS(func() {
		_ = cloneF32(W)
		_ = cloneF32(bias)
	})

	var prepErr error
	_ = timeUS(func() {
		prepErr = sess.DensePrepareF32(batch, in, out, W, bias)
	})
	if prepErr != nil {
		r.Err = prepErr
		return r
	}
	takePhase(&r, sess) // create+finalize from prepare; execute=0

	_ = sess.DenseExecuteF32(x, yQ) // warm
	execUS, err := benchUS(func() error {
		return sess.DenseExecuteF32(x, yQ)
	}, 20)
	if err != nil {
		r.Err = err
		return r
	}
	r.ExecUS = execUS
	finishE2E(&r)

	goL, err := dense.NewConfigured(in, out, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, W)
	if err != nil {
		r.Err = err
		return r
	}
	bf64 := make([]float64, out)
	for i := range bias {
		bf64[i] = float64(bias[i])
	}
	goL.Weights.Bias = bf64
	goL.Exec.Backend = core.BackendCPUTiled
	goL.Exec.MultiCore = false
	tx := core.NewTensor[float32](batch, in)
	copy(tx.Data, x)
	_, post, err := dense.Forward(goL, tx)
	if err != nil {
		r.Err = err
		return r
	}
	copy(yG, post.Data)
	r.GoExecUS, _ = benchUS(func() error {
		_, _, e := dense.Forward(goL, tx)
		return e
	}, 10)
	fillAccuracy(&r, yG, yQ)
	return r
}

func measureSoftmaxBig(sess *nativeqnn.Session) LayerReport {
	const rows, cols = 128, 8192
	r := LayerReport{Name: "softmax", Geom: fmt.Sprintf("rows=%d cols=%d", rows, cols)}
	x := make([]float32, rows*cols)
	fillRand(x, 4, 2)
	yQ := make([]float32, len(x))
	yG := make([]float32, len(x))
	measureOnline(&r, sess, nil, func() error {
		return sess.SoftmaxForwardF32(rows, cols, x, yQ)
	})
	if r.Err != nil {
		return r
	}

	l, _ := softmax.New(softmax.Config{Dim: cols})
	l.Exec.Backend = core.BackendCPUTiled
	t := core.NewTensor[float32](rows, cols)
	copy(t.Data, x)
	_, post, err := softmax.Forward(l, t)
	if err != nil {
		r.Err = err
		return r
	}
	copy(yG, post.Data)
	r.GoExecUS, _ = benchUS(func() error {
		_, _, e := softmax.Forward(l, t)
		return e
	}, 5)
	fillAccuracy(&r, yG, yQ)
	return r
}

func measureRmsNormBig(sess *nativeqnn.Session) LayerReport {
	const batch, seq, dim = 16, 256, 1024
	r := LayerReport{Name: "rmsnorm", Geom: fmt.Sprintf("B=%d S=%d D=%d", batch, seq, dim)}
	gamma := make([]float32, dim)
	x := make([]float32, batch*seq*dim)
	for i := range gamma {
		gamma[i] = 1
	}
	fillRand(x, 5, 0.5)
	yQ := make([]float32, len(x))
	yG := make([]float32, len(x))
	measureOnline(&r, sess, func() { _ = cloneF32(gamma) }, func() error {
		return sess.RmsNormForwardF32(batch, seq, dim, 1e-6, gamma, x, yQ)
	})
	if r.Err != nil {
		return r
	}

	l, _ := rmsnorm.NewConfigured(rmsnorm.Config{Dim: dim, Eps: 1e-6}, core.DTypeFloat32, quant.FormatNone, gamma)
	l.Exec.Backend = core.BackendCPUTiled
	t := core.NewTensor[float32](batch, seq, dim)
	copy(t.Data, x)
	_, post, err := rmsnorm.Forward(l, t)
	if err != nil {
		r.Err = err
		return r
	}
	copy(yG, post.Data)
	r.GoExecUS, _ = benchUS(func() error {
		_, _, e := rmsnorm.Forward(l, t)
		return e
	}, 5)
	fillAccuracy(&r, yG, yQ)
	return r
}

func measureLayerNormBig(sess *nativeqnn.Session) LayerReport {
	const batch, seq, dim = 16, 256, 1024
	r := LayerReport{Name: "layernorm", Geom: fmt.Sprintf("B=%d S=%d D=%d", batch, seq, dim)}
	gamma := make([]float32, dim)
	beta := make([]float32, dim)
	x := make([]float32, batch*seq*dim)
	for i := range gamma {
		gamma[i] = 1
	}
	fillRand(x, 6, 0.4)
	yQ := make([]float32, len(x))
	yG := make([]float32, len(x))
	measureOnline(&r, sess, func() {
		_ = cloneF32(gamma)
		_ = cloneF32(beta)
	}, func() error {
		return sess.LayerNormForwardF32(batch, seq, dim, 1e-5, gamma, beta, x, yQ)
	})
	if r.Err != nil {
		return r
	}

	l, _ := layernorm.NewConfigured(layernorm.Config{Dim: dim, Eps: 1e-5}, core.DTypeFloat32, quant.FormatNone, gamma, beta)
	l.Exec.Backend = core.BackendCPUTiled
	t := core.NewTensor[float32](batch, seq, dim)
	copy(t.Data, x)
	_, post, err := layernorm.Forward(l, t)
	if err != nil {
		r.Err = err
		return r
	}
	copy(yG, post.Data)
	r.GoExecUS, _ = benchUS(func() error {
		_, _, e := layernorm.Forward(l, t)
		return e
	}, 5)
	fillAccuracy(&r, yG, yQ)
	return r
}

func measureCnn2Big(sess *nativeqnn.Session) LayerReport {
	const n, h, w, cin, cout = 8, 128, 128, 64, 64
	r := LayerReport{Name: "cnn2", Geom: fmt.Sprintf("N=%d %dx%d cin=%d cout=%d k=1x1", n, h, w, cin, cout)}
	W := make([]float32, cout*cin)
	bias := make([]float32, cout)
	x := make([]float32, n*h*w*cin)
	fillRand(W, 7, 0.05)
	fillRand(x, 8, 0.1)
	yQ := make([]float32, n*h*w*cout)
	measureOnline(&r, sess, func() { _ = cloneF32(W) }, func() error {
		return sess.Cnn2ForwardF32(n, h, w, cin, cout, W, bias, x, yQ)
	})
	if r.Err != nil {
		return r
	}
	// Go 1×1 conv = per-pixel GEMV
	yG := make([]float32, len(yQ))
	r.GoExecUS = timeUS(func() {
		cnn2Ref1x1(n, h, w, cin, cout, W, x, yG) // HTP path ignores bias today
	})
	fillAccuracy(&r, yG, yQ)
	return r
}

func cnn2Ref1x1(n, h, w, cin, cout int, W, x, y []float32) {
	spatial := h * w
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
}

func measureCnn3Big(sess *nativeqnn.Session) LayerReport {
	const n, d, h, w, cin, cout = 2, 16, 16, 16, 32, 32
	r := LayerReport{Name: "cnn3", Geom: fmt.Sprintf("N=%d %dx%dx%d cin=%d cout=%d", n, d, h, w, cin, cout)}
	W := make([]float32, cout*cin)
	bias := make([]float32, cout)
	x := make([]float32, n*d*h*w*cin)
	fillRand(W, 9, 0.05)
	fillRand(x, 10, 0.1)
	yQ := make([]float32, n*d*h*w*cout)
	measureOnline(&r, sess, func() { _ = cloneF32(W) }, func() error {
		return sess.Cnn3ForwardF32(n, d, h, w, cin, cout, W, bias, x, yQ)
	})
	return r
}

func measureConvT2Big(sess *nativeqnn.Session) LayerReport {
	const n, h, w, cin, cout = 8, 64, 64, 64, 64
	r := LayerReport{Name: "convt2", Geom: fmt.Sprintf("N=%d %dx%d cin=%d cout=%d", n, h, w, cin, cout)}
	W := make([]float32, cout*cin)
	bias := make([]float32, cout)
	x := make([]float32, n*h*w*cin)
	fillRand(W, 11, 0.05)
	fillRand(x, 12, 0.1)
	yQ := make([]float32, n*h*w*cout)
	measureOnline(&r, sess, func() { _ = cloneF32(W) }, func() error {
		return sess.ConvT2ForwardF32(n, h, w, cin, cout, W, bias, x, yQ)
	})
	return r
}

func measureConvT3Big(sess *nativeqnn.Session) LayerReport {
	const n, d, h, w, cin, cout = 2, 16, 16, 16, 32, 32
	r := LayerReport{Name: "convt3", Geom: fmt.Sprintf("N=%d %dx%dx%d c=%d→%d", n, d, h, w, cin, cout)}
	W := make([]float32, cout*cin)
	bias := make([]float32, cout)
	x := make([]float32, n*d*h*w*cin)
	fillRand(W, 13, 0.05)
	fillRand(x, 14, 0.1)
	yQ := make([]float32, n*d*h*w*cout)
	measureOnline(&r, sess, func() { _ = cloneF32(W) }, func() error {
		return sess.ConvT3ForwardF32(n, d, h, w, cin, cout, W, bias, x, yQ)
	})
	return r
}

func measureLstmBig(sess *nativeqnn.Session) LayerReport {
	// Compose = many online graphs/step; large enough for NPU, small enough HTP can finish.
	const seq, batch, inSz, hid = 8, 8, 128, 128
	r := LayerReport{Name: "lstm", Geom: fmt.Sprintf("T=%d B=%d in=%d hid=%d", seq, batch, inSz, hid)}
	wIn := make([]float32, 4*hid*inSz)
	wRec := make([]float32, 4*hid*hid)
	b := make([]float32, 4*hid)
	x := make([]float32, seq*batch*inSz)
	fillRand(wIn, 15, 0.02)
	fillRand(wRec, 16, 0.02)
	fillRand(x, 17, 0.1)
	yQ := make([]float32, seq*batch*hid)
	measureCompose(&r, sess, func() {
		_ = cloneF32(wIn)
		_ = cloneF32(wRec)
	}, func() error {
		return sess.LstmForwardF32(seq, batch, inSz, hid, wIn, wRec, b, x, yQ)
	})
	return r
}

func measureEmbeddingBig(sess *nativeqnn.Session) LayerReport {
	const vocab, dim, n = 8192, 512, 512
	r := LayerReport{Name: "embedding", Geom: fmt.Sprintf("vocab=%d dim=%d n=%d", vocab, dim, n)}
	table := make([]float32, vocab*dim)
	ids := make([]int32, n)
	fillRand(table, 18, 0.05)
	for i := range ids {
		ids[i] = int32(i % vocab)
	}
	yQ := make([]float32, n*dim)
	yG := make([]float32, n*dim)
	measureOnline(&r, sess, func() { _ = cloneF32(table) }, func() error {
		return sess.EmbeddingForwardF32(vocab, dim, n, table, ids, yQ)
	})
	if r.Err != nil {
		return r
	}
	r.GoExecUS = timeUS(func() {
		for i, id := range ids {
			off := int(id) * dim
			copy(yG[i*dim:(i+1)*dim], table[off:off+dim])
		}
	})
	fillAccuracy(&r, yG, yQ)
	return r
}

func measureResidualBig(sess *nativeqnn.Session) LayerReport {
	const n = 256 * 1024 // 2M hung HTP Graph Optimizations; 256k still NPU-relevant
	r := LayerReport{Name: "residual", Geom: fmt.Sprintf("n=%d", n)}
	a := make([]float32, n)
	b := make([]float32, n)
	fillRand(a, 19, 0.5)
	fillRand(b, 20, 0.5)
	yQ := make([]float32, n)
	yG := make([]float32, n)
	measureOnline(&r, sess, nil, func() error {
		return sess.ResidualForwardF32(int32(n), a, b, yQ)
	})
	if r.Err != nil {
		return r
	}
	r.GoExecUS = timeUS(func() {
		for i := 0; i < n; i++ {
			yG[i] = a[i] + b[i]
		}
	})
	fillAccuracy(&r, yG, yQ)
	return r
}

func measureSwiGluBig(sess *nativeqnn.Session) LayerReport {
	const batch, in, hidden = 16, 512, 1408
	r := LayerReport{Name: "swiglu", Geom: fmt.Sprintf("B=%d in=%d hidden=%d", batch, in, hidden)}
	wG := make([]float32, hidden*in)
	wU := make([]float32, hidden*in)
	wD := make([]float32, in*hidden)
	x := make([]float32, batch*in)
	fillRand(wG, 21, 0.02)
	fillRand(wU, 22, 0.02)
	fillRand(wD, 23, 0.02)
	fillRand(x, 24, 0.1)
	yQ := make([]float32, batch*in)
	measureCompose(&r, sess, func() {
		_ = cloneF32(wG)
		_ = cloneF32(wU)
		_ = cloneF32(wD)
	}, func() error {
		return sess.SwiGluForwardF32(batch, in, hidden, wG, wU, wD, x, yQ)
	})
	return r
}

func measureMhaBig(sess *nativeqnn.Session) LayerReport {
	const batch, seq, dim, heads = 2, 64, 256, 4
	r := LayerReport{Name: "mha", Geom: fmt.Sprintf("B=%d S=%d D=%d H=%d", batch, seq, dim, heads)}
	wQ := make([]float32, dim*dim)
	wK := make([]float32, dim*dim)
	wV := make([]float32, dim*dim)
	wO := make([]float32, dim*dim)
	x := make([]float32, batch*seq*dim)
	for i := 0; i < dim; i++ {
		wQ[i*dim+i], wK[i*dim+i], wV[i*dim+i], wO[i*dim+i] = 1, 1, 1, 1
	}
	fillRand(x, 25, 0.05)
	yQ := make([]float32, len(x))
	measureCompose(&r, sess, func() {
		_ = cloneF32(wQ)
		_ = cloneF32(wK)
		_ = cloneF32(wV)
		_ = cloneF32(wO)
	}, func() error {
		return sess.MhaForwardF32(batch, seq, dim, heads, wQ, wK, wV, wO, x, yQ)
	})
	return r
}

// AllREADMELayers runs every HTP row from welvet.qnn README (f32).
func AllREADMELayers(sess *nativeqnn.Session) []LayerReport {
	return []LayerReport{
		measureDenseBig(sess),
		measureSoftmaxBig(sess),
		measureRmsNormBig(sess),
		measureLayerNormBig(sess),
		measureCnn2Big(sess),
		measureCnn3Big(sess),
		measureConvT2Big(sess),
		measureConvT3Big(sess),
		measureLstmBig(sess),
		measureEmbeddingBig(sess),
		measureResidualBig(sess),
		measureSwiGluBig(sess),
		measureMhaBig(sess),
	}
}
