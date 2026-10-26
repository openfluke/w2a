package go_v_metal_test

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/nativemetal"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/runtime/dispatch"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/placement"
)

func routingDense(t *testing.T, i int) *dense.Layer {
	t.Helper()
	l, err := dense.NewConfigured(2, 2, core.ActivationTanh, core.DTypeFloat32, quant.FormatNone, []float32{.8, .12, -.15, .9})
	if err != nil {
		t.Fatal(err)
	}
	l.Weights.Bias = []float64{.01 * float64(i+1), -.02}
	return l
}
func routingNear(t *testing.T, a, b []float32) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("length %d != %d", len(a), len(b))
	}
	for i := range a {
		if math.IsNaN(float64(a[i])) || math.IsNaN(float64(b[i])) || math.Abs(float64(a[i]-b[i])) > 2e-5*(1+math.Abs(float64(b[i]))) {
			t.Fatalf("value %d: %g != %g", i, a[i], b[i])
		}
	}
}
func routingInput() *core.Tensor[float32] {
	return &core.Tensor[float32]{Shape: []int{1, 2}, Data: []float32{.5, -.3}}
}
func TestGoNativeMetalRoutingCPPRequired(t *testing.T) {
	if os.Getenv("WELVET_REQUIRE_CPP_ROUTING") == "1" && !nativecpp.NativeCPUAvailable() {
		t.Fatal("mixed C++/Metal verification requires the normal build and C++ artifact; do not use -tags welvet_metal")
	}
}
func TestGoNativeMetalRoutingStack(t *testing.T) {
	for _, useCPP := range []bool{false, true} {
		t.Run(fmt.Sprint("cpp=", useCPP), func(t *testing.T) {
			if useCPP && !nativecpp.NativeCPUAvailable() {
				t.Skip("requires normal build with shipped C++ library")
			}
			router, err := nativemetal.EnableRouting()
			if err != nil {
				t.Fatal(err)
			}
			defer router.Close()
			var ops, refs []any
			var layers []*dense.Layer
			cfg := nativecpp.ExecConfig{Default: "go_cpu", Volumes: map[string]nativecpp.VolumeBackend{"main": {Sectors: map[string]nativecpp.SectorBackend{"0,0,0,0": {Layers: map[string]nativecpp.LayerBackend{}}}}}}
			for i := 0; i < 13; i++ {
				l := routingDense(t, i)
				layers = append(layers, l)
				ops = append(ops, l)
				refs = append(refs, routingDense(t, i))
				b := "go_cpu"
				if i >= 4 && i <= 8 && useCPP {
					b = "native_cpu"
				}
				if i >= 9 {
					b = "metal"
				}
				cfg.Volumes["main"].Sectors["0,0,0,0"].Layers[fmt.Sprint(i)] = nativecpp.LayerBackend{Backend: b}
			}
			raw, _ := json.Marshal(cfg)
			cfg, err = nativecpp.ParseExecConfigJSON(raw)
			if err != nil {
				t.Fatal(err)
			}
			s, err := parallel.NewStack(ops...)
			if err != nil {
				t.Fatal(err)
			}
			ref, err := parallel.NewStack(refs...)
			if err != nil {
				t.Fatal(err)
			}
			if err = placement.Stack(s, cfg, "main", "0,0,0,0"); err != nil {
				t.Fatal(err)
			}
			x := routingInput()
			gy := &core.Tensor[float32]{Shape: []int{1, 2}, Data: []float32{.3, -.2}}
			for epoch := 0; epoch < 3; epoch++ {
				pre, post, e := parallel.ForwardStack(s, x)
				if e != nil {
					t.Fatal(e)
				}
				rp, ro, e := parallel.ForwardStack(ref, x)
				if e != nil {
					t.Fatal(e)
				}
				routingNear(t, post.Data, ro.Data)
				gx, dw, e := parallel.BackwardStack(s, gy, x, pre)
				if e != nil {
					t.Fatal(e)
				}
				rx, rw, e := parallel.BackwardStack(ref, gy, x, rp)
				if e != nil {
					t.Fatal(e)
				}
				routingNear(t, gx.Data, rx.Data)
				routingNear(t, dw.Data, rw.Data)
				if e = parallel.ApplyGradSGDStack(s, dw, .03); e != nil {
					t.Fatal(e)
				}
				if e = parallel.ApplyGradSGDStack(ref, rw, .03); e != nil {
					t.Fatal(e)
				}
				for i, l := range layers {
					a, _ := l.Weights.FlattenF32()
					b, _ := refs[i].(*dense.Layer).Weights.FlattenF32()
					routingNear(t, a, b)
				}
			}
			stats := router.Stats()
			if stats.Mounts != 4 || stats.Backwards != 12 || stats.Updates != 12 {
				t.Fatalf("native routing not retained: %+v", stats)
			}
			// Switching to Go sees the latest native updates without an explicit pull.
			layers[9].SetBackend(core.BackendCPUTiled)
			_, _, err = parallel.ForwardStack(s, x)
			if err != nil {
				t.Fatal(err)
			}
			if e := router.Close(); e != nil {
				t.Fatal(e)
			}
			if _, _, e := dense.Forward(layers[10], x); e == nil {
				t.Fatal("Metal fell back after router close")
			}
		})
	}
}
func TestGoNativeMetalRoutingGrid(t *testing.T) {
	if !nativecpp.NativeCPUAvailable() {
		t.Skip("requires C++ library")
	}
	for _, remote := range []bool{false, true} {
		t.Run(fmt.Sprint("remote=", remote), func(t *testing.T) {
			r, e := nativemetal.EnableRouting()
			if e != nil {
				t.Fatal(e)
			}
			defer r.Close()
			makeGrid := func(mixed bool) *architecture.Grid {
				g := architecture.NewGrid(3, 3, 3, 1)
				for _, c := range g.HopOrder() {
					g.At(c.Z, c.Y, c.X, c.L).Layer.IsDisabled = true
				}
				for i := 0; i < 3; i++ {
					l := routingDense(t, i)

					c := g.At(i, i, i, 0)
					c.Layer = l.Core
					c.Op = l
				}
				if mixed {
					cfg := nativecpp.ExecConfig{Default: "go_cpu", Volumes: map[string]nativecpp.VolumeBackend{"main": {Sectors: map[string]nativecpp.SectorBackend{"1,1,1,0": {Backend: "native_cpu"}, "2,2,2,0": {Backend: "metal"}}}}}
					if e := placement.Grid(g, cfg, "main"); e != nil {
						t.Fatal(e)
					}
				}
				if remote {
					if e := g.SetRemoteLink(2, 2, 2, 0, 0, 0, 0, 0); e != nil {
						t.Fatal(e)
					}
				}
				return g
			}
			g, ref := makeGrid(true), makeGrid(false)
			x := routingInput()
			f, e := forward.Forward(g, x)
			if e != nil {
				t.Fatal(e)
			}
			rf, e := forward.Forward(ref, x)
			if e != nil {
				t.Fatal(e)
			}
			routingNear(t, f.Output.Data, rf.Output.Data)
			gy := &core.Tensor[float32]{Shape: []int{1, 2}, Data: []float32{.1, -.2}}
			b, e := backward.Backward(f, gy)
			if e != nil {
				t.Fatal(e)
			}
			rb, e := backward.Backward(rf, gy)
			if e != nil {
				t.Fatal(e)
			}
			routingNear(t, b.GradIn.Data, rb.GradIn.Data)
			for i, dw := range b.GradWs {
				routingNear(t, dw.DW.Data, rb.GradWs[i].DW.Data)
				c := g.At(dw.Coord.Z, dw.Coord.Y, dw.Coord.X, dw.Coord.L)
				if e := dispatch.ApplyGradSGD(c.Op, dw.DW, .02); e != nil {
					t.Fatal(e)
				}
			}
			if r.Stats().Backwards != 1 || r.Stats().Updates != 1 {
				t.Fatal(r.Stats())
			}
		})
	}
}
func TestGoNativeMetalRoutingGuards(t *testing.T) {
	l, _, x, _ := metalLifecycleModel(t)
	l.SetBackend(core.BackendMetal)
	if _, _, e := dense.Forward(l, x); e == nil {
		t.Fatal("missing router silently fell back")
	}
	if core.ParseBackend("metal") != core.BackendMetal || nativecpp.FromCoreBackend(core.BackendMetal) != "metal" {
		t.Fatal("Metal backend name lost")
	}
	r, e := nativemetal.EnableRouting()
	if e != nil {
		t.Fatal(e)
	}
	defer r.Close()
	if _, e := nativemetal.EnableRouting(); e == nil {
		t.Fatal("duplicate router accepted")
	}
	if _, _, e := dense.Forward(l, core.NewTensor[float64](1, 2)); e == nil {
		t.Fatal("unsupported activation dtype fell back")
	}
	r.Close()
	r.Close()
	r2, e := nativemetal.EnableRouting()
	if e != nil {
		t.Fatal(e)
	}
	defer r2.Close()
	if _, _, e := dense.Forward(l, x); e != nil {
		t.Fatal(e)
	}
	// Native updates remain invisible for explicitly mounted sessions until SyncToGo.
	before, _ := l.Weights.FlattenF32()
	s, e := nativemetal.Mount(l)
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	_, dw, e := s.Backward(x, x)
	if e != nil {
		t.Fatal(e)
	}
	if e = s.ApplySGD(dw, .1); e != nil {
		t.Fatal(e)
	}
	after, _ := l.Weights.FlattenF32()
	if !reflect.DeepEqual(before, after) {
		t.Fatal("explicit segment update leaked into Go")
	}
}

func TestGoNativeMetalRoutingMixedModes(t *testing.T) {
	if !nativecpp.NativeCPUAvailable() {
		t.Skip("requires normal build with C++ library")
	}
	for _, mode := range parallel.AllStackLocalTrainModes() {
		t.Run(mode.String(), func(t *testing.T) {
			r, e := nativemetal.EnableRouting()
			if e != nil {
				t.Fatal(e)
			}
			defer r.Close()
			var ops, refs []any
			for i := 0; i < 3; i++ {
				l := routingDense(t, i)
				l.SetBackend([]core.Backend{core.BackendCPUTiled, core.BackendNativeCPU, core.BackendMetal}[i])
				ops = append(ops, l)
				ref := routingDense(t, i)
				ref.SetBackend(core.BackendCPUTiled)
				refs = append(refs, ref)
			}
			s, e := parallel.NewStack(ops...)
			if e != nil {
				t.Fatal(e)
			}
			ref, e := parallel.NewStack(refs...)
			if e != nil {
				t.Fatal(e)
			}
			target := core.NewTensor[float32](1, 2)
			for i := 0; i < 8; i++ {
				a, err := parallel.TrainStackMSE(s, routingInput(), target, mode, .01)
				if err != nil {
					t.Fatal(err)
				}
				b, err := parallel.TrainStackMSE(ref, routingInput(), target, mode, .01)
				if err != nil {
					t.Fatal(err)
				}
				if math.Abs(a-b) > 1e-3*(1+math.Abs(b)) {
					t.Fatalf("loss %g != Go %g", a, b)
				}
			}
			for i, l := range ops {
				a, _ := l.(*dense.Layer).Weights.FlattenF32()
				b, _ := refs[i].(*dense.Layer).Weights.FlattenF32()
				routingNear(t, a, b)
			}
			if r.Stats().Forwards == 0 || r.Stats().Updates == 0 {
				t.Fatal("no native work", r.Stats())
			}
		})
	}
}
func TestGoNativeMetalRoutingOverrides(t *testing.T) {
	r, e := nativemetal.EnableRouting()
	if e != nil {
		t.Fatal(e)
	}
	defer r.Close()
	for _, parent := range []core.Backend{core.BackendMetal, core.BackendNativeCPU} {
		a, b := routingDense(t, 0), routingDense(t, 1)
		a.SetBackend(core.BackendCPUTiled)
		b.SetBackend(core.BackendMetal)
		s, e := parallel.NewStack(a, b)
		if e != nil {
			t.Fatal(e)
		}
		s.SetBackend(parent)
		if _, e = parallel.TrainStackMSE(s, routingInput(), core.NewTensor[float32](1, 2), parallel.ModeNormalBP, .02); e != nil {
			t.Fatal(e)
		}
		if a.Exec.Backend != core.BackendCPUTiled || b.Exec.Backend != core.BackendMetal {
			t.Fatal("parent erased explicit placement")
		}
	}
	if r.Stats().Updates != 2 {
		t.Fatal("container swallowed child placement", r.Stats())
	}
}
func TestGoNativeMetalRoutingPendingCPP(t *testing.T) {
	if !nativecpp.NativeCPUAvailable() {
		t.Skip("requires C++ library")
	}
	l, _, x, target := metalLifecycleModel(t)
	if _, e := dense.TrainNativeMSE(l, x, target, .1); e != nil {
		t.Fatal(e)
	}
	if !l.NativeWeightsPending() {
		t.Fatal("expected pending C++ updates")
	}
	r, e := nativemetal.EnableRouting()
	if e != nil {
		t.Fatal(e)
	}
	defer r.Close()
	l.SetBackend(core.BackendMetal)
	_, post, e := dense.Forward(l, x)
	if e != nil {
		t.Fatal(e)
	}
	if l.NativeWeightsPending() {
		t.Fatal("C++ updates were not pulled before Metal mount")
	}
	l.SetBackend(core.BackendCPUTiled)
	_, ref, e := dense.Forward(l, x)
	if e != nil {
		t.Fatal(e)
	}
	routingNear(t, post.Data, ref.Data)
}
func TestGoNativeMetalRoutingBadLibrary(t *testing.T) {
	r, e := nativemetal.EnableRouting(nativemetal.Options{LibraryPath: "/missing/metal-library.dylib"})
	if e != nil {
		t.Fatal(e)
	}
	defer r.Close()
	l := routingDense(t, 0)
	l.SetBackend(core.BackendMetal)
	if _, _, e = dense.Forward(l, routingInput()); e == nil {
		t.Fatal("missing library fell back to CPU")
	}
}

func TestGoNativeMetalRoutingNestedOverrides(t *testing.T) {
	for _, family := range []string{"sequential", "residual"} {
		t.Run(family, func(t *testing.T) {
			r, e := nativemetal.EnableRouting()
			if e != nil {
				t.Fatal(e)
			}
			defer r.Close()
			var op any
			var children []*dense.Layer
			w := []float32{.8, .1, -.1, .9, .7, -.1, .2, .8}
			if family == "sequential" {
				l, err := sequential.NewConfigured(sequential.Config{Dim: 2, Depth: 2}, core.DTypeFloat32, quant.FormatNone, w)
				if err != nil {
					t.Fatal(err)
				}
				l.ApplyExecConfig(nativecpp.ExecConfig{Default: "metal"}, "main", "0,0,0,0", "0")
				op = l
				children = l.Children
			} else {
				l, err := residual.NewConfigured(residual.Config{Dim: 2, Depth: 2}, core.DTypeFloat32, quant.FormatNone, w)
				if err != nil {
					t.Fatal(err)
				}
				l.ApplyExecConfig(nativecpp.ExecConfig{Default: "metal"}, "main", "0,0,0,0", "0")
				op = l
				children = l.Children
			}
			children[0].SetBackend(core.BackendCPUTiled)
			x := routingInput()
			pre, post, e := dispatch.Forward(op, x)
			if e != nil {
				t.Fatal(e)
			}
			_, dw, e := dispatch.Backward(op, post, x, pre)
			if e != nil {
				t.Fatal(e)
			}
			if e = dispatch.ApplyGradSGD(op, dw, .01); e != nil {
				t.Fatal(e)
			}
			if children[0].Exec.Backend != core.BackendCPUTiled || children[1].Exec.Backend != core.BackendMetal {
				t.Fatal("container erased override")
			}
			if r.Stats().Mounts != 1 || r.Stats().Updates != 1 {
				t.Fatal("container swallowed override", r.Stats())
			}
		})
	}
}
func TestGoNativeMetalRoutingCredit(t *testing.T) {
	for _, mode := range parallel.AllStackLocalTrainModes() {
		t.Run(mode.String(), func(t *testing.T) {
			r, e := nativemetal.EnableRouting()
			if e != nil {
				t.Fatal(e)
			}
			defer r.Close()
			l, s, x, _ := metalLifecycleModel(t)
			ref, rs, _, _ := metalLifecycleModel(t)
			ref.SetBackend(core.BackendCPUTiled)
			s.SetBackend(core.BackendMetal)
			pre, post, e := parallel.ForwardStack(s, x)
			if e != nil {
				t.Fatal(e)
			}
			rp, _, e := parallel.ForwardStack(rs, x)
			if e != nil {
				t.Fatal(e)
			}
			gy := core.NewTensor[float32](post.Shape...)
			copy(gy.Data, []float32{.1, -.2})
			if e = parallel.TrainStack(s, gy, x, pre, mode, .01); e != nil {
				t.Fatal(e)
			}
			if e = parallel.TrainStack(rs, gy, x, rp, mode, .01); e != nil {
				t.Fatal(e)
			}
			a, _ := l.Weights.FlattenF32()
			b, _ := ref.Weights.FlattenF32()
			routingNear(t, a, b)
			if r.Stats().TrainingSteps != 1 {
				t.Fatal(r.Stats())
			}
		})
	}
}

func TestGoNativeMetalRoutingParallelGate(t *testing.T) {
	r, e := nativemetal.EnableRouting()
	if e != nil {
		t.Fatal(e)
	}
	defer r.Close()
	a, b, gate := routingDense(t, 0), routingDense(t, 1), routingDense(t, 2)
	l, e := parallel.NewFromBranches(parallel.Config{Dim: 2, OutFeat: 2, Branches: 2, Combine: parallel.CombineFilter}, []any{a, b}, gate)
	if e != nil {
		t.Fatal(e)
	}
	l.SetBackend(core.BackendMetal)
	gate.SetBackend(core.BackendCPUTiled)
	x := routingInput()
	pre, post, e := parallel.Forward(l, x)
	if e != nil {
		t.Fatal(e)
	}
	_, dw, e := parallel.Backward(l, post, x, pre)
	if e != nil {
		t.Fatal(e)
	}
	if e = parallel.ApplyGradSGD(l, dw, .01); e != nil {
		t.Fatal(e)
	}
	if gate.Exec.Backend != core.BackendCPUTiled || r.Stats().Mounts != 2 || r.Stats().Updates != 2 {
		t.Fatal("gate override lost", gate.Exec, r.Stats())
	}
}
