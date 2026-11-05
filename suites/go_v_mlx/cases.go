package go_v_mlx

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/nativemlx"
	"github.com/openfluke/welvet/quant"
)

func fillRand(dst []float32, seed int64, scale float32) {
	r := rand.New(rand.NewSource(seed))
	for i := range dst {
		dst[i] = (r.Float32()*2 - 1) * scale
	}
}

func caseCapsInventory() error {
	layers := nativemlx.SupportedLayers()
	if len(layers) == 0 {
		return fmt.Errorf("no supported layers")
	}
	for _, l := range layers {
		if !nativemlx.LayerSupported(l, "float32") {
			return fmt.Errorf("%s/float32 should be supported", l)
		}
		fmt.Printf("      cap ok  %s/float32\n", l)
	}
	return nil
}

func caseLayerDense() error {
	sess, err := openSess()
	if err != nil {
		return err
	}
	const batch, in, out = 32, 512, 512
	r := LayerReport{Name: "dense", Geom: fmt.Sprintf("B=%d in=%d out=%d", batch, in, out)}
	W := make([]float32, out*in)
	bias := make([]float32, out)
	x := make([]float32, batch*in)
	fillRand(W, 1, 0.02)
	fillRand(bias, 2, 0.01)
	fillRand(x, 3, 0.1)
	yM := make([]float32, batch*out)

	r.TransferUS = timeUS(func() { _ = append([]float32(nil), W...) })

	var prepErr error
	_ = timeUS(func() { prepErr = sess.DensePrepareF32(batch, in, out, W, bias) })
	if prepErr != nil {
		return prepErr
	}
	takePhase(&r, sess)

	_ = sess.DenseExecuteF32(x, yM)
	execUS, err := benchUS(func() error { return sess.DenseExecuteF32(x, yM) }, 10)
	if err != nil {
		return err
	}
	r.ExecUS = execUS
	finishE2E(&r)

	goL, err := dense.NewConfigured(in, out, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, W)
	if err != nil {
		return err
	}
	bf64 := make([]float64, out)
	for i := range bias {
		bf64[i] = float64(bias[i])
	}
	goL.Weights.Bias = bf64
	goL.Exec.Backend = core.BackendCPUTiled
	tx := core.NewTensor[float32](batch, in)
	copy(tx.Data, x)
	_, goPost, err := dense.Forward(goL, tx)
	if err != nil {
		return err
	}
	fillAccuracy(&r, goPost.Data, yM)
	r.GoExecUS, _ = benchUS(func() error {
		_, _, e := dense.Forward(goL, tx)
		return e
	}, 10)
	printReport(r)
	if r.MaxAbs > 1e-3 {
		return fmt.Errorf("dense max|Δ|=%g too high", r.MaxAbs)
	}
	return nil
}

func caseLayerSoftmax() error {
	sess, err := openSess()
	if err != nil {
		return err
	}
	const rows, cols = 64, 4096
	r := LayerReport{Name: "softmax", Geom: fmt.Sprintf("%d×%d", rows, cols)}
	x := make([]float32, rows*cols)
	fillRand(x, 4, 1)
	yM := make([]float32, rows*cols)
	r.TransferUS = timeUS(func() { _ = append([]float32(nil), x...) })
	if err := sess.SoftmaxForwardF32(rows, cols, x, yM); err != nil {
		return err
	}
	takePhase(&r, sess)
	execUS, err := benchUS(func() error { return sess.SoftmaxForwardF32(rows, cols, x, yM) }, 5)
	if err != nil {
		return err
	}
	r.ExecUS = execUS
	finishE2E(&r)

	s, err := softmax.New(softmax.Config{Dim: cols})
	if err != nil {
		return err
	}
	s.Exec.Backend = core.BackendCPUTiled
	tx := core.NewTensor[float32](rows, cols)
	copy(tx.Data, x)
	_, goPost, err := softmax.Forward(s, tx)
	if err != nil {
		return err
	}
	fillAccuracy(&r, goPost.Data, yM)
	printReport(r)
	if r.MaxAbs > 1e-4 {
		return fmt.Errorf("softmax max|Δ|=%g", r.MaxAbs)
	}
	return nil
}

func caseLayerRMSNorm() error {
	sess, err := openSess()
	if err != nil {
		return err
	}
	const batch, seq, dim = 8, 128, 512
	r := LayerReport{Name: "rmsnorm", Geom: fmt.Sprintf("[%d,%d,%d]", batch, seq, dim)}
	gamma := make([]float32, dim)
	x := make([]float32, batch*seq*dim)
	for i := range gamma {
		gamma[i] = 1
	}
	fillRand(x, 5, 0.5)
	yM := make([]float32, len(x))
	const eps = float32(1e-5)
	r.TransferUS = timeUS(func() { _ = append([]float32(nil), gamma...) })
	if err := sess.RmsNormForwardF32(batch, seq, dim, eps, gamma, x, yM); err != nil {
		return err
	}
	takePhase(&r, sess)
	execUS, err := benchUS(func() error {
		return sess.RmsNormForwardF32(batch, seq, dim, eps, gamma, x, yM)
	}, 5)
	if err != nil {
		return err
	}
	r.ExecUS = execUS
	finishE2E(&r)

	l, err := rmsnorm.NewConfigured(rmsnorm.Config{Dim: dim, Eps: float64(eps)}, core.DTypeFloat32, quant.FormatNone, gamma)
	if err != nil {
		return err
	}
	l.Exec.Backend = core.BackendCPUTiled
	tx := core.NewTensor[float32](batch*seq, dim)
	copy(tx.Data, x)
	_, goPost, err := rmsnorm.Forward(l, tx)
	if err != nil {
		return err
	}
	fillAccuracy(&r, goPost.Data, yM)
	printReport(r)
	if r.MaxAbs > 1e-3 {
		return fmt.Errorf("rmsnorm max|Δ|=%g", r.MaxAbs)
	}
	return nil
}

func caseLayerLayerNorm() error {
	sess, err := openSess()
	if err != nil {
		return err
	}
	const batch, seq, dim = 8, 128, 512
	r := LayerReport{Name: "layernorm", Geom: fmt.Sprintf("[%d,%d,%d]", batch, seq, dim)}
	gamma := make([]float32, dim)
	beta := make([]float32, dim)
	x := make([]float32, batch*seq*dim)
	for i := range gamma {
		gamma[i] = 1
	}
	fillRand(x, 6, 0.5)
	yM := make([]float32, len(x))
	const eps = float32(1e-5)
	r.TransferUS = timeUS(func() { _ = append([]float32(nil), gamma...) })
	if err := sess.LayerNormForwardF32(batch, seq, dim, eps, gamma, beta, x, yM); err != nil {
		return err
	}
	takePhase(&r, sess)
	execUS, err := benchUS(func() error {
		return sess.LayerNormForwardF32(batch, seq, dim, eps, gamma, beta, x, yM)
	}, 5)
	if err != nil {
		return err
	}
	r.ExecUS = execUS
	finishE2E(&r)

	l, err := layernorm.NewConfigured(layernorm.Config{Dim: dim, Eps: float64(eps)}, core.DTypeFloat32, quant.FormatNone, gamma, beta)
	if err != nil {
		return err
	}
	l.Exec.Backend = core.BackendCPUTiled
	tx := core.NewTensor[float32](batch*seq, dim)
	copy(tx.Data, x)
	_, goPost, err := layernorm.Forward(l, tx)
	if err != nil {
		return err
	}
	fillAccuracy(&r, goPost.Data, yM)
	printReport(r)
	if r.MaxAbs > 1e-3 {
		return fmt.Errorf("layernorm max|Δ|=%g", r.MaxAbs)
	}
	return nil
}

func caseLayerResidual() error {
	sess, err := openSess()
	if err != nil {
		return err
	}
	const n = 65536
	r := LayerReport{Name: "residual", Geom: fmt.Sprintf("n=%d", n)}
	a := make([]float32, n)
	b := make([]float32, n)
	fillRand(a, 7, 1)
	fillRand(b, 8, 1)
	yM := make([]float32, n)
	r.TransferUS = timeUS(func() { _ = append([]float32(nil), a...) })
	if err := sess.ResidualForwardF32(n, a, b, yM); err != nil {
		return err
	}
	takePhase(&r, sess)
	execUS, err := benchUS(func() error { return sess.ResidualForwardF32(n, a, b, yM) }, 10)
	if err != nil {
		return err
	}
	r.ExecUS = execUS
	finishE2E(&r)
	yG := make([]float32, n)
	for i := range yG {
		yG[i] = a[i] + b[i]
	}
	fillAccuracy(&r, yG, yM)
	printReport(r)
	if r.ExactPct < 99.9 {
		return fmt.Errorf("residual exact=%g%%", r.ExactPct)
	}
	return nil
}

func caseSequentialTower() error {
	sess, err := openSess()
	if err != nil {
		return err
	}
	const dim, depth, batch = 64, 3, 8
	seq, err := nativemlx.NewSequential()
	if err != nil {
		return err
	}
	defer seq.Free()

	weights := make([][]float32, depth)
	for i := 0; i < depth; i++ {
		W := make([]float32, dim*dim)
		bias := make([]float32, dim)
		fillRand(W, int64(20+i), 0.02)
		weights[i] = W
		if err := seq.PushDense(dim, dim, W, bias); err != nil {
			return err
		}
	}
	x := make([]float32, batch*dim)
	fillRand(x, 30, 0.1)
	yM := make([]float32, batch*dim)
	if err := seq.ForwardF32(sess, batch, 1, x, yM); err != nil {
		return err
	}

	// Go reference: chain Dense
	cur := append([]float32(nil), x...)
	for i := 0; i < depth; i++ {
		goL, err := dense.NewConfigured(dim, dim, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, weights[i])
		if err != nil {
			return err
		}
		goL.Exec.Backend = core.BackendCPUTiled
		tx := core.NewTensor[float32](batch, dim)
		copy(tx.Data, cur)
		_, post, err := dense.Forward(goL, tx)
		if err != nil {
			return err
		}
		cur = append([]float32(nil), post.Data...)
	}
	r := LayerReport{Name: "sequential", Geom: fmt.Sprintf("dim=%d depth=%d B=%d", dim, depth, batch)}
	fillAccuracy(&r, cur, yM)
	printReport(r)
	maxOK := 1e-3 * float64(depth)
	if r.MaxAbs > maxOK && r.MaxAbs > 1e-2 {
		return fmt.Errorf("sequential max|Δ|=%g (tol %g)", r.MaxAbs, math.Max(maxOK, 1e-2))
	}
	return nil
}
