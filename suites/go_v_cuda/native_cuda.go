package go_v_cuda

import (
	"fmt"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/nativecuda"
	"github.com/openfluke/welvet/systems/dna"
	"github.com/openfluke/welvet/weights"
	"math"
	"reflect"
)

func nativeCudaRoot(f *camFixture) any {
	if f.kind == 1 {
		return f.ops[0]
	}
	if f.kind == 2 {
		return f.grid
	}
	return f.root
}
func nativeGoStores(f *camFixture) []*weights.Store {
	var out []*weights.Store
	for _, op := range f.ops {
		out = append(out, dna.CollectStores(op)...)
	}
	return out
}
func nativeGoParameters(f *camFixture) ([][]float32, error) {
	var all [][]float32
	for _, op := range f.ops {
		p, e := kpPackOp(op)
		if e != nil {
			return nil, e
		}
		all = append(all, p)
	}
	return all, nil
}
func nativeRoundtrip(snap nativecuda.Snapshot, stores []*weights.Store) error {
	if len(snap.Stores) != len(stores) {
		return fmt.Errorf("store count native=%d Go=%d", len(snap.Stores), len(stores))
	}
	for i, s := range stores {
		v := snap.Stores[i]
		if v.DType != s.DType || v.Rows != s.Rows || v.Cols != s.Cols || v.Scale != s.Scale || !reflect.DeepEqual(v.Native, append([]byte{}, s.Native...)) && !(len(v.Native) == 0 && len(s.Native) == 0) || !reflect.DeepEqual(v.Bias, append([]float64{}, s.Bias...)) && !(len(v.Bias) == 0 && len(s.Bias) == 0) {
			return fmt.Errorf("native store %d bytes/scale/bias changed during roundtrip", i)
		}
		if s.DType == core.DTypeFloat32 {
			p, e := weights.MatrixF32(s)
			if e != nil {
				return e
			}
			if !reflect.DeepEqual(p, v.F32) {
				return fmt.Errorf("Float32 payload changed during roundtrip")
			}
		}
	}
	return nil
}
func runNativeCudaFixture(dt core.DType, build func(core.DType) (camFixture, error)) error {
	f, e := build(dt)
	if e != nil {
		return e
	}
	if e = camPrepare(&f); e != nil {
		return e
	}
	ref, e := build(dt)
	if e != nil {
		return e
	}
	if e = camPrepare(&ref); e != nil {
		return e
	}
	initial, e := nativeGoParameters(&f)
	if e != nil {
		return e
	}
	s, e := nativecuda.Mount(nativeCudaRoot(&f))
	if e != nil {
		return e
	}
	defer s.Close()
	if s.Version() == "" || s.Identity() == "" {
		return fmt.Errorf("missing CUDA engine identity")
	}
	before, e := s.Weights()
	if e != nil {
		return e
	}
	if e = nativeRoundtrip(before, nativeGoStores(&f)); e != nil {
		return e
	}
	post, e := s.Forward(f.input)
	if e != nil {
		return e
	}
	refPost, e := camForward(&ref)
	if e != nil {
		return e
	}
	if e = camCompare(camResult{post: meshView(refPost)}, camResult{post: meshView(post)}); e != nil {
		return e
	}
	target := core.NewTensor[float32](post.Shape...)
	copy(target.Data, seedPat(target.Len(), 0xcab1))
	var loss, refLoss float64
	for epoch := 0; epoch < f.epochs; epoch++ {
		loss, e = s.TrainMSE(f.input, target, f.mode, .017)
		if e != nil {
			return e
		}
		if f.kind == 0 {
			refLoss, e = parallel.TrainStackMSE(ref.root, ref.input, target, ref.mode, .017)
		} else if f.kind == 1 {
			refLoss, e = parallel.TrainMSE(ref.ops[0].(*parallel.Layer), ref.input, target, ref.mode, .017)
		} else {
			refLoss, e = camVolume(&ref, target)
		}
		if e != nil {
			return e
		}
	}
	unchanged, e := nativeGoParameters(&f)
	if e != nil {
		return e
	}
	if !reflect.DeepEqual(initial, unchanged) {
		return fmt.Errorf("native training modified Go before SyncToGo")
	}
	after, e := s.Weights()
	if e != nil {
		return e
	}
	if e = s.SyncToGo(); e != nil {
		return e
	}
	if e = nativeRoundtrip(after, nativeGoStores(&f)); e != nil {
		return e
	}
	gotW, e := nativeGoParameters(&f)
	if e != nil {
		return e
	}
	wantW, e := nativeGoParameters(&ref)
	if e != nil {
		return e
	}
	post, e = s.Forward(f.input)
	if e != nil {
		return e
	}
	refPost, e = camForward(&ref)
	if e != nil {
		return e
	}
	if e = camCompare(camResult{model: parallelNBPOut{loss: refLoss, WAfter: wantW}, post: meshView(refPost)}, camResult{model: parallelNBPOut{loss: loss, WAfter: gotW}, post: meshView(post)}); e != nil {
		return e
	}
	if math.IsNaN(loss) {
		return fmt.Errorf("nonfinite loss")
	}
	if e = s.Close(); e != nil {
		return e
	}
	if _, e = s.Forward(f.input); e == nil {
		return fmt.Errorf("closed session executed")
	}
	return s.Close()
}

func nativeDTypes(pc polyCase, build func(core.DType) (camFixture, error)) func() error {
	return func() error {
		types := core.AllDTypes
		if pc.name == "gdn" {
			types = []core.DType{core.DTypeFloat32}
		}
		for _, dt := range types {
			if e := runNativeCudaFixture(dt, build); e != nil {
				return fmt.Errorf("%s: %w", dt, e)
			}
		}
		return nil
	}
}
func CasesNativeCudaIntegration() []Case {
	var out []Case
	for _, pc := range polyTrainKinds() {
		pc := pc
		for _, placement := range []int{0, 1, 2} {
			placement := placement
			out = append(out, Case{Name: fmt.Sprintf("Mount train sync %s placement%d native dtypes", pc.name, placement), Run: nativeDTypes(pc, camPlain(pc, parallel.ModeNormalBP, placement))})
		}
	}
	for mode := parallel.TrainMode(0); mode <= parallel.ModeMemory; mode++ {
		mode := mode
		for _, root := range []int{0, 1} {
			root := root
			out = append(out, Case{Name: fmt.Sprintf("Persistent mode %s root%d native dtypes", mode, root), Run: nativeDTypes(polyCase{}, func(dt core.DType) (camFixture, error) {
				f, e := camDenseRoot(dt, parallel.CombineAdd)
				if e != nil {
					return f, e
				}
				f.mode = mode
				f.epochs = 5
				if root == 0 {
					f.kind = 0
				}
				return f, nil
			})})
		}
	}
	for _, combine := range []parallel.CombineMode{parallel.CombineConcat, parallel.CombineAdd, parallel.CombineAvg, parallel.CombineFilter, parallel.CombineMax, parallel.CombineSparseK, parallel.CombineDisagree} {
		combine := combine
		out = append(out, Case{Name: "Mounted mixed " + string(combine) + " native dtypes", Run: nativeDTypes(polyCase{}, func(dt core.DType) (camFixture, error) {
			f, e := camDenseRoot(dt, combine)
			if e != nil {
				return f, e
			}
			l := f.ops[0].(*parallel.Layer)
			l.SetBranchModes(parallel.ModeShadow, parallel.ModeAdversarial, parallel.ModeMemory)
			l.SetBranchLRs(.5, .75, 0)
			l.SetCamKit(parallel.CamKit{ShadowCoef: .35})
			cfg := parallel.DefaultCamSync()
			cfg.Alpha = .25
			l.SetCamSync(cfg)
			return f, nil
		})})
	}
	return out
}

func nativeMeshRun(pc polyCase, dt core.DType, mode parallel.TrainMode) error {
	f, e := meshFactory(pc, dt, 2, 1, 1)
	if e != nil {
		return e
	}
	f.credit = mode
	grid := architecture.NewGrid(f.z, f.y, f.x, f.l)
	for i, coord := range grid.HopOrder() {
		cell := grid.At(coord.Z, coord.Y, coord.X, coord.L)
		if f.cells[i].op == nil {
			cell.Layer.IsDisabled = true
			continue
		}
		meta, e := meshMeta(f.cells[i].op)
		if e != nil {
			return e
		}
		cell.Layer = meta
		cell.Op = f.cells[i].op
	}
	s, e := nativecuda.Mount(grid)
	if e != nil {
		return e
	}
	defer s.Close()
	x := camInput(f.shape, f.input)
	target := camInput(f.targetShape, f.target)
	loss, e := s.TrainMSE(x, target, mode, meshLR(f), f.ticks)
	if e != nil {
		return e
	}
	if e = s.SyncToGo(); e != nil {
		return e
	}
	var got [][]float32
	for _, c := range f.cells {
		if c.op == nil {
			got = append(got, nil)
			continue
		}
		w, e := kpPackOp(c.op)
		if e != nil {
			return e
		}
		got = append(got, w)
	}
	ref, e := meshFactory(pc, dt, 2, 1, 1)
	if e != nil {
		return e
	}
	ref.credit = mode
	want, e := runGoMeshClock(ref)
	if e != nil {
		return e
	}
	return camCompare(camResult{model: parallelNBPOut{loss: want.loss, WAfter: want.weights}}, camResult{model: parallelNBPOut{loss: loss, WAfter: got}})
}
func CasesNativeCudaMesh() []Case {
	var out []Case
	for _, pc := range polyTrainKinds() {
		pc := pc
		for _, mode := range []parallel.TrainMode{parallel.ModeMeshBP, parallel.ModeMeshTween, parallel.ModeMeshTweenChain, parallel.ModeMeshTweenSplit, parallel.ModeMeshTweenAlt, parallel.ModeMeshTweenSplitFastProxy, parallel.ModeMeshTweenSplitSparse} {
			mode := mode
			out = append(out, Case{Name: fmt.Sprintf("Mounted mesh %s %s native dtypes", pc.name, mode), Run: func() error {
				types := core.AllDTypes
				if pc.name == "gdn" {
					types = []core.DType{core.DTypeFloat32}
				}
				for _, dt := range types {
					if e := nativeMeshRun(pc, dt, mode); e != nil {
						return fmt.Errorf("%s: %w", dt, e)
					}
				}
				return nil
			}})
		}
	}
	return out
}
