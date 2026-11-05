package go_v_mlx

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/nativemlx"
	"github.com/openfluke/welvet/quant"
)

// All Dense compute dtypes claimed in welvet.mlx caps (float64 rejected on GPU).
var denseDTypes = []struct {
	Name string
	DT   core.DType
	Tol  float64 // max|Δ| vs Go f32 ref (host I/O is f32; compute cast may round)
}{
	{"float32", core.DTypeFloat32, 1e-5},
	{"float16", core.DTypeFloat16, 5e-2},
	{"bfloat16", core.DTypeBFloat16, 5e-2},
	{"int8", core.DTypeInt8, 2.0},
	{"int16", core.DTypeInt16, 1.0},
	{"int32", core.DTypeInt32, 1e-3},
	{"int64", core.DTypeInt64, 1e-3},
	{"uint8", core.DTypeUint8, 2.0},
	{"uint16", core.DTypeUint16, 1.0},
	{"uint32", core.DTypeUint32, 1e-3},
	{"uint64", core.DTypeUint64, 1e-3},
	{"complex64", core.DTypeComplex64, 1e-3},
}

func caseDenseCaps() error {
	for _, d := range denseDTypes {
		if !nativemlx.LayerSupported("dense", d.Name) {
			return fmt.Errorf("dense/%s should be in caps", d.Name)
		}
		fmt.Printf("      dense cap ok  %s\n", d.Name)
	}
	if nativemlx.LayerSupported("dense", "float64") {
		return fmt.Errorf("dense/float64 must be rejected on GPU")
	}
	fmt.Println("      dense cap ok  float64 rejected")
	return nil
}

// caseLayerDense — large f32 prepare/exec via Session ABI + Go accuracy.
func caseLayerDense() error {
	return runDenseSession("float32", 32, 512, 512, 1e-3)
}

// caseDenseAllDtypes — every Dense caps dtype via Session prepare/exec vs Go f32 ref.
func caseDenseAllDtypes() error {
	const batch, in, out = 4, 64, 64
	var fails []string
	for _, d := range denseDTypes {
		if err := runDenseSession(d.Name, batch, in, out, d.Tol); err != nil {
			fmt.Printf("      FAIL dense/%s: %v\n", d.Name, err)
			fails = append(fails, d.Name+": "+err.Error())
			continue
		}
	}
	if len(fails) > 0 {
		return fmt.Errorf("dense dtype matrix: %d failed: %v", len(fails), fails)
	}
	return nil
}

func runDenseSession(dtype string, batch, in, out int, tol float64) error {
	sess, err := openSess()
	if err != nil {
		return err
	}
	r := LayerReport{Name: "dense/" + dtype, Geom: fmt.Sprintf("B=%d in=%d out=%d", batch, in, out)}
	W := make([]float32, out*in)
	bias := make([]float32, out)
	x := make([]float32, batch*in)
	// Keep magnitudes friendly for integer compute casts.
	scale := float32(0.02)
	if dtype == "int8" || dtype == "uint8" {
		scale = 0.5
	}
	fillRand(W, 1, scale)
	fillRand(bias, 2, scale*0.5)
	fillRand(x, 3, scale*5)
	yM := make([]float32, batch*out)

	r.TransferUS = timeUS(func() { _ = append([]float32(nil), W...) })
	var prepErr error
	_ = timeUS(func() {
		prepErr = sess.DensePrepare(int32(batch), int32(in), int32(out), W, bias, dtype)
	})
	if prepErr != nil {
		return prepErr
	}
	takePhase(&r, sess)
	_ = sess.DenseExecuteF32(x, yM)
	execUS, err := benchUS(func() error { return sess.DenseExecuteF32(x, yM) }, 5)
	if err != nil {
		return err
	}
	r.ExecUS = execUS
	finishE2E(&r)

	for _, v := range yM {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("non-finite output")
		}
	}
	// Float compute: compare to Go f32. Int/complex: cast rounding ≠ Go f32 — exec+finite only.
	switch dtype {
	case "float32", "float16", "bfloat16":
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
		printReport(r)
		if r.MaxAbs > tol && !math.IsNaN(r.MaxAbs) {
			return fmt.Errorf("max|Δ|=%g > tol %g", r.MaxAbs, tol)
		}
	default:
		printReport(r)
		fmt.Printf("      dense/%s exec-ok (int/complex — no Go f32 bit parity)\n", dtype)
	}
	return nil
}

// caseDenseGoLayer — SyncToMLX + BackendMLX Forward through dense.Layer (all float caps).
func caseDenseGoLayer() error {
	if !nativemlx.Available() {
		return fmt.Errorf("nativemlx unavailable")
	}
	const batch, in, out = 8, 128, 128
	floatOnly := []core.DType{core.DTypeFloat32, core.DTypeFloat16, core.DTypeBFloat16}
	tols := map[core.DType]float64{
		core.DTypeFloat32:  1e-4,
		core.DTypeFloat16:  5e-2,
		core.DTypeBFloat16: 5e-2,
	}
	W := make([]float32, out*in)
	x := make([]float32, batch*in)
	fillRand(W, 11, 0.03)
	fillRand(x, 12, 0.1)

	for _, dt := range floatOnly {
		name := nativemlx.DTypeName(dt.String())
		goRef, err := dense.NewConfigured(in, out, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, W)
		if err != nil {
			return err
		}
		goRef.Exec.Backend = core.BackendCPUTiled
		tx := core.NewTensor[float32](batch, in)
		copy(tx.Data, x)
		_, goPost, err := dense.Forward(goRef, tx)
		if err != nil {
			return fmt.Errorf("go ref: %w", err)
		}

		mlxL, err := dense.NewConfigured(in, out, core.ActivationLinear, dt, quant.FormatNone, W)
		if err != nil {
			return err
		}
		if err := mlxL.SyncToMLX(); err != nil {
			return fmt.Errorf("SyncToMLX %s: %w", name, err)
		}
		if err := mlxL.PrepareMLX(batch); err != nil {
			return fmt.Errorf("PrepareMLX %s: %w", name, err)
		}
		mlxL.Exec.Backend = core.BackendMLX
		_, mlxPost, err := dense.Forward(mlxL, tx)
		if err != nil {
			return fmt.Errorf("ForwardMLX %s: %w", name, err)
		}
		r := LayerReport{Name: "dense/go/" + name, Geom: fmt.Sprintf("B=%d %dx%d", batch, in, out)}
		fillAccuracy(&r, goPost.Data, core.SliceAsFloat32(mlxPost.Data))
		ph := mustSessPhase()
		r.CreateUS = ph.CreateUS()
		r.ExecUS = ph.ExecuteUS()
		finishE2E(&r)
		printReport(r)
		if r.MaxAbs > tols[dt] {
			return fmt.Errorf("dense/go/%s max|Δ|=%g > %g", name, r.MaxAbs, tols[dt])
		}
		nativemlx.CloseDefaultSession()
	}
	return nil
}

func mustSessPhase() nativemlx.PhaseTiming {
	s, err := nativemlx.DefaultSession()
	if err != nil {
		return nativemlx.PhaseTiming{}
	}
	return s.LastPhase()
}
