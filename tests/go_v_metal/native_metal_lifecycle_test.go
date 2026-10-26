package go_v_metal_test

import (
	"errors"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/model/entity"
	"github.com/openfluke/welvet/nativemetal"
	"github.com/openfluke/welvet/quant"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
)

func metalLifecycleModel(t *testing.T) (*dense.Layer, *parallel.Stack, *core.Tensor[float32], *core.Tensor[float32]) {
	return metalLifecycleModelType(t, core.DTypeFloat32)
}
func metalLifecycleModelType(t *testing.T, dt core.DType) (*dense.Layer, *parallel.Stack, *core.Tensor[float32], *core.Tensor[float32]) {
	t.Helper()
	l, e := dense.NewConfigured(2, 2, core.ActivationLinear, dt, quant.FormatNone, []float32{.3, -.2, .1, .4})
	if e != nil {
		t.Fatal(e)
	}
	l.Weights.Bias = []float64{.1, -.2}
	s, e := parallel.NewStack(l)
	if e != nil {
		t.Fatal(e)
	}
	x := core.NewTensor[float32](1, 2)
	copy(x.Data, []float32{.5, .25})
	target := core.NewTensor[float32](1, 2)
	return l, s, x, target
}
func TestGoNativeMetalSaveReload(t *testing.T) {
	for _, dt := range core.AllDTypes {
		t.Run(dt.String(), func(t *testing.T) { metalSaveReload(t, dt) })
	}
}
func metalSaveReload(t *testing.T, dt core.DType) {
	l, model, x, target := metalLifecycleModelType(t, dt)
	_, _, e := dense.Forward(l, x)
	if e != nil {
		t.Fatal(e)
	} // prime Go compute caches
	initial, _ := l.Weights.FlattenF32()
	s, e := nativemetal.Mount(model)
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	for i := 0; i < 4; i++ {
		if _, e = s.TrainMSE(x, target, parallel.ModeNormalBP, .1); e != nil {
			t.Fatal(e)
		}
	}
	before, _ := l.Weights.FlattenF32()
	if !reflect.DeepEqual(before, initial) {
		t.Fatal("Go weights changed before explicit sync")
	}
	native, e := s.Forward(x)
	if e != nil {
		t.Fatal(e)
	}
	if e = s.SyncToGo(); e != nil {
		t.Fatal(e)
	}
	_, post, e := dense.Forward(l, x)
	if e != nil {
		t.Fatal(e)
	}
	for i := range post.Data {
		if d := post.Data[i] - native.Data[i]; d > 1e-6 || d < -1e-6 {
			t.Fatal("stale Go cache after weight return")
		}
	}
	path := filepath.Join(t.TempDir(), "trained.entity")
	if e = entity.WriteCameralFile(path, model, parallel.ModeNormalBP); e != nil {
		t.Fatal(e)
	}
	loaded, mode, e := entity.LoadCameral(path)
	if e != nil {
		t.Fatal(e)
	}
	if mode != parallel.ModeNormalBP {
		t.Fatal("saved TrainMode changed")
	}
	s2, e := nativemetal.Mount(loaded)
	if e != nil {
		t.Fatal(e)
	}
	defer s2.Close()
	originalWeights, e := s.Weights()
	if e != nil {
		t.Fatal(e)
	}
	loadedWeights, e := s2.Weights()
	if e != nil {
		t.Fatal(e)
	}
	if !reflect.DeepEqual(originalWeights, loadedWeights) {
		t.Fatal("ENTITY roundtrip changed native bytes, scales or biases")
	}
	restored, e := s2.Forward(x)
	if e != nil {
		t.Fatal(e)
	}
	if !reflect.DeepEqual(native, restored) {
		t.Fatalf("saved/reloaded native result changed: %v vs %v", native, restored)
	}
	if e = s.Close(); e != nil {
		t.Fatal(e)
	}
	if _, e = s.Weights(); e == nil {
		t.Fatal("closed session exported")
	}
	if e = s.SyncToGo(); e == nil {
		t.Fatal("closed session synchronized")
	}
}
func TestGoNativeMetalErrorsAndConcurrency(t *testing.T) {
	attention, err := mha.New(mha.DecoderCausal(2, 1, 1))
	if err != nil {
		t.Fatal(err)
	}
	attention.KVOffset = 1
	if _, err = nativemetal.Mount(attention); err == nil {
		t.Fatal("live KV cache silently discarded at mount")
	}
	l, model, x, target := metalLifecycleModel(t)
	if _, e := nativemetal.Mount(model, nativemetal.Options{LibraryPath: filepath.Join(t.TempDir(), "missing.dylib")}); e == nil {
		t.Fatal("missing library silently accepted")
	}
	if _, e := nativemetal.Mount((*dense.Layer)(nil)); e == nil {
		t.Fatal("nil model accepted")
	}
	if _, e := nativemetal.Mount(&dense.Layer{}); e == nil {
		t.Fatal("invalid empty model accepted")
	}
	cycle, _ := parallel.NewStack(l)
	cycle.Children = []any{cycle}
	if _, e := nativemetal.Mount(cycle); e == nil {
		t.Fatal("cyclic model accepted")
	}
	shared, _ := parallel.NewStack(l, l)
	if _, e := nativemetal.Mount(shared); e == nil {
		t.Fatal("unsupported shared stores accepted")
	}
	s, e := nativemetal.Mount(model)
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	bad := &core.Tensor[float32]{Shape: []int{1, 2}, Data: []float32{1}}
	if _, e = s.Forward(bad); e == nil {
		t.Fatal("short tensor accepted")
	}
	if _, e = s.TrainMSE(x, target, parallel.ModeNormalBP, .1, 0); e == nil {
		t.Fatal("zero ticks accepted")
	}
	if e = s.SyncCams(parallel.SyncWhen(99)); e == nil {
		t.Fatal("invalid sync event accepted")
	}
	want, e := s.Forward(x)
	if e != nil {
		t.Fatal(e)
	}
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 4; j++ {
				got, e := s.Forward(x)
				if e != nil {
					t.Error(e)
					return
				}
				if !reflect.DeepEqual(got, want) {
					t.Error("concurrent session result changed")
				}
			}
		}()
	}
	wg.Wait()
	old := l.Weights.Rows
	l.Weights.Rows++
	if e = s.SyncToGo(); e == nil {
		t.Fatal("mutated Go store metadata accepted")
	}
	l.Weights.Rows = old
	if e = s.SyncToGo(); e != nil {
		t.Fatal(e)
	}
}

func TestGoNativeMetalCamSync(t *testing.T) {
	for _, dt := range core.AllDTypes {
		t.Run(dt.String(), func(t *testing.T) {
			a, e := dense.NewConfigured(1, 1, core.ActivationLinear, dt, quant.FormatNone, []float32{1})
			if e != nil {
				t.Fatal(e)
			}
			b, e := dense.NewConfigured(1, 1, core.ActivationLinear, dt, quant.FormatNone, []float32{3})
			if e != nil {
				t.Fatal(e)
			}
			a.Weights.Bias = []float64{1}
			b.Weights.Bias = []float64{3}
			l, e := parallel.NewFromBranches(parallel.Config{Dim: 1, OutFeat: 1, Branches: 2, Combine: parallel.CombineAdd}, []any{a, b}, nil)
			if e != nil {
				t.Fatal(e)
			}
			cfg := parallel.DefaultCamSync()
			cfg.When = parallel.SyncAfterPulse
			cfg.BranchAlpha = []float64{0, 1}
			l.SetCamSync(cfg)
			s, e := nativemetal.Mount(l)
			if e != nil {
				t.Fatal(e)
			}
			defer s.Close()
			x := core.NewTensor[float32](1, 1)
			x.Data[0] = 1
			if _, e = s.Forward(x); e != nil {
				t.Fatal(e)
			}
			before, e := s.Weights()
			if e != nil {
				t.Fatal(e)
			}
			if e = s.SyncCams(parallel.SyncAfterSample); e != nil {
				t.Fatal(e)
			}
			same, e := s.Weights()
			if e != nil {
				t.Fatal(e)
			}
			if !reflect.DeepEqual(before, same) {
				t.Fatal("wrong sync boundary wrote weights")
			}
			if e = s.SyncCams(parallel.SyncAfterPulse); e != nil {
				t.Fatal(e)
			}
			after, e := s.Weights()
			if e != nil {
				t.Fatal(e)
			}
			if !reflect.DeepEqual(before.Stores[0], after.Stores[0]) {
				t.Fatal("alpha-zero teacher changed")
			}
			post, e := s.Forward(x)
			if e != nil {
				t.Fatal(e)
			}
			if d := post.Data[0] - 6; d > 1e-5 || d < -1e-5 {
				t.Fatalf("wrong blend or stale native cache: %v", post.Data)
			}
			old, _ := b.Weights.FlattenF32()
			if old[0] != 3 {
				t.Fatal("CamSync modified Go implicitly")
			}
			if e = s.SyncToGo(); e != nil {
				t.Fatal(e)
			}
			updated, _ := b.Weights.FlattenF32()
			if updated[0] != 2 || b.Weights.Bias[0] != 2 {
				t.Fatalf("weight/bias pull failed: %v %v", updated, b.Weights.Bias)
			}
		})
	}
}

func TestGoNativeMetalUnsupportedGridMode(t *testing.T) {
	l, _, x, target := metalLifecycleModel(t)
	g := architecture.NewGrid(1, 1, 1, 1)
	g.Cells[0].Op = l
	g.Cells[0].Layer = l.Core
	s, e := nativemetal.Mount(g)
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	before, e := s.Weights()
	if e != nil {
		t.Fatal(e)
	}
	_, e = s.TrainMSE(x, target, parallel.ModeStepBP, .1)
	var native *nativemetal.NativeError
	if !errors.As(e, &native) || native.Code != 2 {
		t.Fatalf("unsupported mode did not retain native code: %v", e)
	}
	after, e := s.Weights()
	if e != nil {
		t.Fatal(e)
	}
	if !reflect.DeepEqual(before, after) {
		t.Fatal("rejected mode changed weights")
	}
}
