package go_v_cuda

import (
	"fmt"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/nativecuda"
	"github.com/openfluke/welvet/runtime/dispatch"
)

// CasesNativeCudaRouting exercises the normal public layer dispatchers, rather
// than calling Session training directly: global gradients in, dInput out, SGD.
func CasesNativeCudaRouting() []Case {
	var out []Case
	for _, pc := range polyTrainKinds() {
		pc := pc
		out = append(out, Case{Name: "Routed backward SGD " + pc.name + " native dtypes", Run: func() error {
			router, e := nativecuda.EnableRouting()
			if e != nil {
				return e
			}
			defer router.Close()
			dtypes := core.AllDTypes
			if pc.name == "gdn" {
				dtypes = []core.DType{core.DTypeFloat32}
			}
			for _, dt := range dtypes {
				if e := runRoutedCuda(pc, dt); e != nil {
					return fmt.Errorf("%s: %w", dt, e)
				}
			}
			stats := router.Stats()
			if stats.Forwards == 0 || stats.Backwards == 0 || (pc.name != "softmax" && stats.Updates == 0) {
				return fmt.Errorf("no native dispatch: %+v", stats)
			}
			return nil
		}})
	}
	return out
}
func runRoutedCuda(pc polyCase, dt core.DType) error {
	branches, x, _, shape, _, e := pc.make(dt)
	if e != nil {
		return e
	}
	refs, _, _, _, _, e := pc.make(dt)
	if e != nil {
		return e
	}
	op, ref := branches[0].op, refs[0].op
	dispatch.SyncExec(ref, core.ExecConfig{Backend: core.BackendCPUTiled})
	setter, ok := op.(interface {
		ApplyExecConfig(nativecpp.ExecConfig, string, string, string)
	})
	if !ok {
		return fmt.Errorf("missing placement setter %T", op)
	}
	setter.ApplyExecConfig(nativecpp.ExecConfig{Default: "cuda"}, "main", "0,0,0,0", "0")
	input := camInput(shape, x)
	pre, post, e := dispatch.Forward(op, input)
	if e != nil {
		return e
	}
	rp, ro, e := dispatch.Forward(ref, input)
	if e != nil {
		return e
	}
	if e = camCompare(camResult{post: meshView(ro)}, camResult{post: meshView(post)}); e != nil {
		return fmt.Errorf("initial forward: %w", e)
	}
	gy := core.NewTensor[float32](post.Shape...)
	copy(gy.Data, seedPat(gy.Len(), 0xcafe))
	gx, dw, e := dispatch.Backward(op, gy, input, pre)
	if e != nil {
		return e
	}
	rx, rw, e := dispatch.Backward(ref, gy, input, rp)
	if e != nil {
		return e
	}
	// A native segment returns flat concatenated dW; Go leaf shapes can be matrices.
	if e = camCompare(camResult{model: parallelNBPOut{WAfter: [][]float32{routingData(rx), routingData(rw)}}}, camResult{model: parallelNBPOut{WAfter: [][]float32{routingData(gx), routingData(dw)}}}); e != nil {
		return fmt.Errorf("gradient: %w", e)
	}
	if e = dispatch.ApplyGradSGD(op, dw, .017); e != nil {
		return e
	}
	if e = dispatch.ApplyGradSGD(ref, rw, .017); e != nil {
		return e
	}
	wa, e := kpPackOp(op)
	if e != nil {
		return e
	}
	wr, e := kpPackOp(ref)
	if e != nil {
		return e
	}
	_, post, e = dispatch.Forward(op, input)
	if e != nil {
		return e
	}
	_, ro, e = dispatch.Forward(ref, input)
	if e != nil {
		return e
	}
	return camCompare(camResult{model: parallelNBPOut{WAfter: [][]float32{wr}}, post: meshView(ro)}, camResult{model: parallelNBPOut{WAfter: [][]float32{wa}}, post: meshView(post)})
}

// All native mode IDs, including Line state, go through public TrainStackMSE
// and TrainMSE. An explicit native session supplies the independent route oracle.
func CasesNativeCudaRoutedModes() []Case {
	var out []Case
	for mode := parallel.TrainMode(0); mode <= parallel.ModeMemory; mode++ {
		mode := mode
		for _, root := range []int{0, 1} {
			root := root
			out = append(out, Case{Name: fmt.Sprintf("Routed %s root%d", mode, root), Run: func() error {
				r, e := nativecuda.EnableRouting()
				if e != nil {
					return e
				}
				defer r.Close()
				f, e := camDenseRoot(core.DTypeFloat32, parallel.CombineAdd)
				if e != nil {
					return e
				}
				ref, e := camDenseRoot(core.DTypeFloat32, parallel.CombineAdd)
				if e != nil {
					return e
				}
				f.kind = root
				ref.kind = root
				if e = camPrepare(&f); e != nil {
					return e
				}
				if e = camPrepare(&ref); e != nil {
					return e
				}
				var model any = f.root
				if root == 0 {
					f.root.SetBackend(core.BackendCuda)
				} else {
					model = f.ops[0]
					f.ops[0].(*parallel.Layer).SetBackend(core.BackendCuda)
				}
				oracle, e := nativecuda.Mount(nativeCudaRoot(&ref))
				if e != nil {
					return e
				}
				defer oracle.Close()
				post, e := oracle.Forward(ref.input)
				if e != nil {
					return e
				}
				target := core.NewTensor[float32](post.Shape...)
				copy(target.Data, seedPat(target.Len(), 123))
				for i := 0; i < 5; i++ {
					var a float64
					if root == 0 {
						a, e = parallel.TrainStackMSE(f.root, f.input, target, mode, .017)
					} else {
						a, e = parallel.TrainMSE(model.(*parallel.Layer), f.input, target, mode, .017)
					}
					if e != nil {
						return e
					}
					b, err := oracle.TrainMSE(ref.input, target, mode, .017)
					if err != nil {
						return err
					}
					if a != b {
						return fmt.Errorf("loss %g != native %g", a, b)
					}
				}
				if e = oracle.SyncToGo(); e != nil {
					return e
				}
				a, e := nativeGoParameters(&f)
				if e != nil {
					return e
				}
				b, e := nativeGoParameters(&ref)
				if e != nil {
					return e
				}
				if r.Stats().Mounts != 1 || r.Stats().TrainingSteps != 5 {
					return fmt.Errorf("state remounted: %+v", r.Stats())
				}
				return camCompare(camResult{model: parallelNBPOut{WAfter: a}}, camResult{model: parallelNBPOut{WAfter: b}})
			}})
		}
	}
	return out
}

func routingData(t *core.Tensor[float32]) []float32 {
	if t == nil {
		return nil
	}
	return t.Data
}
