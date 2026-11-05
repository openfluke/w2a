package go_v_mlx

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/nativemlx"
)

// Caps dtype bands from welvet.mlx mlx_supports().
var (
	floatOnly = []string{"float32", "float16", "bfloat16"}
	floatInts = []string{
		"float32", "float16", "bfloat16",
		"int8", "int16", "int32", "int64",
		"uint8", "uint16", "uint32", "uint64",
	}
	floatIntsC64 = append(append([]string{}, floatInts...), "complex64")
)

func isFloatDT(dt string) bool {
	return dt == "float32" || dt == "float16" || dt == "bfloat16"
}

func checkFinite(y []float32) error {
	for _, v := range y {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("non-finite output")
		}
	}
	return nil
}

func floatTol(dt string) float64 {
	switch dt {
	case "float32":
		return 1e-4
	case "float16", "bfloat16":
		return 5e-2
	default:
		return 1e-3
	}
}

// caseSoftmaxAllDtypes — Softmax Session × f32/f16/bf16 (+ Go accuracy on floats).
func caseSoftmaxAllDtypes() error {
	const rows, cols = 16, 256
	var fails []string
	for _, dt := range floatOnly {
		if !nativemlx.LayerSupported("softmax", dt) {
			fails = append(fails, dt+": missing from caps")
			continue
		}
		sess, err := openSess()
		if err != nil {
			return err
		}
		x := make([]float32, rows*cols)
		fillRand(x, 40, 1)
		yM := make([]float32, len(x))
		if err := sess.SoftmaxForward(rows, cols, x, yM, dt); err != nil {
			fails = append(fails, dt+": "+err.Error())
			continue
		}
		if err := checkFinite(yM); err != nil {
			fails = append(fails, dt+": "+err.Error())
			continue
		}
		if isFloatDT(dt) {
			s, _ := softmax.New(softmax.Config{Dim: cols})
			s.Exec.Backend = core.BackendCPUTiled
			tx := core.NewTensor[float32](rows, cols)
			copy(tx.Data, x)
			_, goPost, err := softmax.Forward(s, tx)
			if err != nil {
				fails = append(fails, dt+": go "+err.Error())
				continue
			}
			r := LayerReport{Name: "softmax/" + dt, Geom: fmt.Sprintf("%d×%d", rows, cols)}
			fillAccuracy(&r, goPost.Data, yM)
			printReport(r)
			if r.MaxAbs > floatTol(dt) {
				fails = append(fails, fmt.Sprintf("%s: max|Δ|=%g", dt, r.MaxAbs))
			}
		} else {
			fmt.Printf("      softmax/%s exec-ok\n", dt)
		}
	}
	if len(fails) > 0 {
		return fmt.Errorf("softmax: %v", fails)
	}
	return nil
}

func caseRMSNormAllDtypes() error {
	const batch, seq, dim = 4, 16, 64
	const eps = float32(1e-5)
	var fails []string
	for _, dt := range floatOnly {
		if !nativemlx.LayerSupported("rmsnorm", dt) {
			fails = append(fails, dt+": missing from caps")
			continue
		}
		sess, err := openSess()
		if err != nil {
			return err
		}
		gamma := make([]float32, dim)
		for i := range gamma {
			gamma[i] = 1
		}
		x := make([]float32, batch*seq*dim)
		fillRand(x, 41, 0.5)
		yM := make([]float32, len(x))
		if err := sess.RmsNormForward(batch, seq, dim, eps, gamma, x, yM, dt); err != nil {
			fails = append(fails, dt+": "+err.Error())
			continue
		}
		if err := checkFinite(yM); err != nil {
			fails = append(fails, dt+": "+err.Error())
			continue
		}
		fmt.Printf("      rmsnorm/%s exec-ok\n", dt)
	}
	if len(fails) > 0 {
		return fmt.Errorf("rmsnorm: %v", fails)
	}
	return nil
}

func caseLayerNormAllDtypes() error {
	const batch, seq, dim = 4, 16, 64
	const eps = float32(1e-5)
	var fails []string
	for _, dt := range floatOnly {
		if !nativemlx.LayerSupported("layernorm", dt) {
			fails = append(fails, dt+": missing from caps")
			continue
		}
		sess, err := openSess()
		if err != nil {
			return err
		}
		gamma := make([]float32, dim)
		beta := make([]float32, dim)
		for i := range gamma {
			gamma[i] = 1
		}
		x := make([]float32, batch*seq*dim)
		fillRand(x, 42, 0.5)
		yM := make([]float32, len(x))
		if err := sess.LayerNormForward(batch, seq, dim, eps, gamma, beta, x, yM, dt); err != nil {
			fails = append(fails, dt+": "+err.Error())
			continue
		}
		if err := checkFinite(yM); err != nil {
			fails = append(fails, dt+": "+err.Error())
			continue
		}
		fmt.Printf("      layernorm/%s exec-ok\n", dt)
	}
	if len(fails) > 0 {
		return fmt.Errorf("layernorm: %v", fails)
	}
	return nil
}

func caseCNN2AllDtypes() error {
	const n, h, w, cin, cout = 2, 8, 8, 4, 4
	var fails []string
	for _, dt := range floatInts {
		if !nativemlx.LayerSupported("cnn2", dt) {
			fails = append(fails, dt+": missing from caps")
			continue
		}
		sess, err := openSess()
		if err != nil {
			return err
		}
		W := make([]float32, cout*cin)
		bias := make([]float32, cout)
		x := make([]float32, n*h*w*cin)
		scale := float32(0.05)
		if dt == "int8" || dt == "uint8" {
			scale = 0.5
		}
		fillRand(W, 50, scale)
		fillRand(x, 51, scale)
		yM := make([]float32, n*h*w*cout)
		if err := sess.Cnn2Forward(n, h, w, cin, cout, W, bias, x, yM, dt); err != nil {
			fails = append(fails, dt+": "+err.Error())
			continue
		}
		if err := checkFinite(yM); err != nil {
			fails = append(fails, dt+": "+err.Error())
			continue
		}
		if isFloatDT(dt) {
			yG := make([]float32, len(yM))
			cnn2Ref1x1(n, h, w, cin, cout, W, x, yG)
			r := LayerReport{Name: "cnn2/" + dt, Geom: fmt.Sprintf("%dx%dx%dx%d", n, h, w, cin)}
			fillAccuracy(&r, yG, yM)
			printReport(r)
			if r.MaxAbs > floatTol(dt) {
				fails = append(fails, fmt.Sprintf("%s: max|Δ|=%g", dt, r.MaxAbs))
			}
		} else {
			fmt.Printf("      cnn2/%s exec-ok (int — no Go f32 bit parity)\n", dt)
		}
	}
	if len(fails) > 0 {
		return fmt.Errorf("cnn2: %v", fails)
	}
	return nil
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

func caseCNN3AllDtypes() error {
	const n, d, h, w, cin, cout = 1, 4, 4, 4, 2, 2
	var fails []string
	for _, dt := range floatInts {
		if !nativemlx.LayerSupported("cnn3", dt) {
			fails = append(fails, dt+": missing from caps")
			continue
		}
		sess, err := openSess()
		if err != nil {
			return err
		}
		W := make([]float32, cout*cin)
		bias := make([]float32, cout)
		x := make([]float32, n*d*h*w*cin)
		fillRand(W, 52, 0.05)
		fillRand(x, 53, 0.1)
		yM := make([]float32, n*d*h*w*cout)
		if err := sess.Cnn3Forward(n, d, h, w, cin, cout, W, bias, x, yM, dt); err != nil {
			fails = append(fails, dt+": "+err.Error())
			continue
		}
		if err := checkFinite(yM); err != nil {
			fails = append(fails, dt+": "+err.Error())
			continue
		}
		fmt.Printf("      cnn3/%s exec-ok\n", dt)
	}
	if len(fails) > 0 {
		return fmt.Errorf("cnn3: %v", fails)
	}
	return nil
}

func caseConvT2AllDtypes() error {
	const n, h, w, cin, cout = 2, 4, 4, 2, 2
	var fails []string
	for _, dt := range floatInts {
		if !nativemlx.LayerSupported("convt2", dt) {
			fails = append(fails, dt+": missing from caps")
			continue
		}
		sess, err := openSess()
		if err != nil {
			return err
		}
		W := make([]float32, cout*cin)
		bias := make([]float32, cout)
		x := make([]float32, n*h*w*cin)
		fillRand(W, 54, 0.05)
		fillRand(x, 55, 0.1)
		yM := make([]float32, n*h*w*cout)
		if err := sess.ConvT2Forward(n, h, w, cin, cout, W, bias, x, yM, dt); err != nil {
			fails = append(fails, dt+": "+err.Error())
			continue
		}
		if err := checkFinite(yM); err != nil {
			fails = append(fails, dt+": "+err.Error())
			continue
		}
		fmt.Printf("      convt2/%s exec-ok\n", dt)
	}
	if len(fails) > 0 {
		return fmt.Errorf("convt2: %v", fails)
	}
	return nil
}

func caseConvT3AllDtypes() error {
	const n, d, h, w, cin, cout = 1, 2, 2, 2, 2, 2
	var fails []string
	for _, dt := range floatInts {
		if !nativemlx.LayerSupported("convt3", dt) {
			fails = append(fails, dt+": missing from caps")
			continue
		}
		sess, err := openSess()
		if err != nil {
			return err
		}
		W := make([]float32, cout*cin)
		bias := make([]float32, cout)
		x := make([]float32, n*d*h*w*cin)
		fillRand(W, 56, 0.05)
		fillRand(x, 57, 0.1)
		yM := make([]float32, n*d*h*w*cout)
		if err := sess.ConvT3Forward(n, d, h, w, cin, cout, W, bias, x, yM, dt); err != nil {
			fails = append(fails, dt+": "+err.Error())
			continue
		}
		if err := checkFinite(yM); err != nil {
			fails = append(fails, dt+": "+err.Error())
			continue
		}
		fmt.Printf("      convt3/%s exec-ok\n", dt)
	}
	if len(fails) > 0 {
		return fmt.Errorf("convt3: %v", fails)
	}
	return nil
}

func caseLSTMAllDtypes() error {
	const seq, batch, inSz, hid = 4, 2, 8, 8
	var fails []string
	for _, dt := range floatOnly {
		if !nativemlx.LayerSupported("lstm", dt) {
			fails = append(fails, dt+": missing from caps")
			continue
		}
		sess, err := openSess()
		if err != nil {
			return err
		}
		wIn := make([]float32, 4*hid*inSz)
		wRec := make([]float32, 4*hid*hid)
		b := make([]float32, 4*hid)
		x := make([]float32, seq*batch*inSz)
		fillRand(wIn, 60, 0.02)
		fillRand(wRec, 61, 0.02)
		fillRand(x, 62, 0.1)
		yM := make([]float32, seq*batch*hid)
		if err := sess.LstmForward(seq, batch, inSz, hid, wIn, wRec, b, x, yM, dt); err != nil {
			fails = append(fails, dt+": "+err.Error())
			continue
		}
		if err := checkFinite(yM); err != nil {
			fails = append(fails, dt+": "+err.Error())
			continue
		}
		fmt.Printf("      lstm/%s exec-ok\n", dt)
	}
	if len(fails) > 0 {
		return fmt.Errorf("lstm: %v", fails)
	}
	return nil
}

func caseEmbeddingAllDtypes() error {
	const vocab, dim, n = 64, 16, 32
	var fails []string
	for _, dt := range floatInts {
		if !nativemlx.LayerSupported("embedding", dt) {
			fails = append(fails, dt+": missing from caps")
			continue
		}
		sess, err := openSess()
		if err != nil {
			return err
		}
		table := make([]float32, vocab*dim)
		ids := make([]int32, n)
		fillRand(table, 63, 0.05)
		for i := range ids {
			ids[i] = int32(i % vocab)
		}
		yM := make([]float32, n*dim)
		if err := sess.EmbeddingForward(vocab, dim, n, table, ids, yM, dt); err != nil {
			fails = append(fails, dt+": "+err.Error())
			continue
		}
		if err := checkFinite(yM); err != nil {
			fails = append(fails, dt+": "+err.Error())
			continue
		}
		if isFloatDT(dt) {
			yG := make([]float32, n*dim)
			for i, id := range ids {
				off := int(id) * dim
				copy(yG[i*dim:(i+1)*dim], table[off:off+dim])
			}
			r := LayerReport{Name: "embedding/" + dt, Geom: fmt.Sprintf("v=%d d=%d n=%d", vocab, dim, n)}
			fillAccuracy(&r, yG, yM)
			printReport(r)
			if r.MaxAbs > floatTol(dt) {
				fails = append(fails, fmt.Sprintf("%s: max|Δ|=%g", dt, r.MaxAbs))
			}
		} else {
			fmt.Printf("      embedding/%s exec-ok (int — no Go f32 bit parity)\n", dt)
		}
	}
	if len(fails) > 0 {
		return fmt.Errorf("embedding: %v", fails)
	}
	return nil
}

func caseResidualAllDtypes() error {
	const n = 4096
	var fails []string
	for _, dt := range floatIntsC64 {
		if !nativemlx.LayerSupported("residual", dt) {
			fails = append(fails, dt+": missing from caps")
			continue
		}
		sess, err := openSess()
		if err != nil {
			return err
		}
		a := make([]float32, n)
		b := make([]float32, n)
		fillRand(a, 64, 0.5)
		fillRand(b, 65, 0.5)
		yM := make([]float32, n)
		if err := sess.ResidualForward(int32(n), a, b, yM, dt); err != nil {
			fails = append(fails, dt+": "+err.Error())
			continue
		}
		if err := checkFinite(yM); err != nil {
			fails = append(fails, dt+": "+err.Error())
			continue
		}
		if isFloatDT(dt) {
			yG := make([]float32, n)
			for i := range yG {
				yG[i] = a[i] + b[i]
			}
			r := LayerReport{Name: "residual/" + dt, Geom: fmt.Sprintf("n=%d", n)}
			fillAccuracy(&r, yG, yM)
			printReport(r)
			if r.ExactPct < 99.0 && r.MaxAbs > floatTol(dt) {
				fails = append(fails, fmt.Sprintf("%s: exact=%g max|Δ|=%g", dt, r.ExactPct, r.MaxAbs))
			}
		} else {
			fmt.Printf("      residual/%s exec-ok (int/c64 — no Go f32 bit parity)\n", dt)
		}
	}
	if len(fails) > 0 {
		return fmt.Errorf("residual: %v", fails)
	}
	return nil
}

func caseSwiGLUAllDtypes() error {
	const batch, in, hidden = 4, 32, 64
	var fails []string
	for _, dt := range floatOnly {
		if !nativemlx.LayerSupported("swiglu", dt) {
			fails = append(fails, dt+": missing from caps")
			continue
		}
		sess, err := openSess()
		if err != nil {
			return err
		}
		wG := make([]float32, hidden*in)
		wU := make([]float32, hidden*in)
		wD := make([]float32, in*hidden)
		x := make([]float32, batch*in)
		fillRand(wG, 70, 0.02)
		fillRand(wU, 71, 0.02)
		fillRand(wD, 72, 0.02)
		fillRand(x, 73, 0.1)
		yM := make([]float32, batch*in)
		if err := sess.SwiGluForward(batch, in, hidden, wG, wU, wD, x, yM, dt); err != nil {
			fails = append(fails, dt+": "+err.Error())
			continue
		}
		if err := checkFinite(yM); err != nil {
			fails = append(fails, dt+": "+err.Error())
			continue
		}
		fmt.Printf("      swiglu/%s exec-ok\n", dt)
	}
	if len(fails) > 0 {
		return fmt.Errorf("swiglu: %v", fails)
	}
	return nil
}

func caseMHAAllDtypes() error {
	const batch, seq, dim, heads = 2, 8, 32, 4
	var fails []string
	for _, dt := range floatOnly {
		if !nativemlx.LayerSupported("mha", dt) {
			fails = append(fails, dt+": missing from caps")
			continue
		}
		sess, err := openSess()
		if err != nil {
			return err
		}
		wQ := make([]float32, dim*dim)
		wK := make([]float32, dim*dim)
		wV := make([]float32, dim*dim)
		wO := make([]float32, dim*dim)
		x := make([]float32, batch*seq*dim)
		for i := 0; i < dim; i++ {
			wQ[i*dim+i], wK[i*dim+i], wV[i*dim+i], wO[i*dim+i] = 1, 1, 1, 1
		}
		fillRand(x, 74, 0.05)
		yM := make([]float32, len(x))
		if err := sess.MhaForward(batch, seq, dim, heads, wQ, wK, wV, wO, x, yM, dt); err != nil {
			fails = append(fails, dt+": "+err.Error())
			continue
		}
		if err := checkFinite(yM); err != nil {
			fails = append(fails, dt+": "+err.Error())
			continue
		}
		fmt.Printf("      mha/%s exec-ok\n", dt)
	}
	if len(fails) > 0 {
		return fmt.Errorf("mha: %v", fails)
	}
	return nil
}

// caseSequentialF32 — f32 Dense tower via Session Sequential (baseline).
func caseSequentialF32() error {
	return caseSequentialTower()
}
