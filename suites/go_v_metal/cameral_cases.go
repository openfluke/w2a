package go_v_metal

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
)

func camInput(shape []int32, x []float32) *core.Tensor[float32] {
	t := core.NewTensor[float32](ints(shape)...)
	copy(t.Data, x)
	return t
}
func camPlain(pc polyCase, mode parallel.TrainMode, volume int) func(core.DType) (camFixture, error) {
	return func(dt core.DType) (camFixture, error) {
		brs, x, _, shape, _, e := pc.make(dt)
		if e != nil {
			return camFixture{}, e
		}
		f := camFixture{mode: mode, ops: []any{brs[0].op}, wires: [][]byte{brs[0].bytes}, input: camInput(shape, x)}
		if volume > 0 {
			f.kind = 2
			f.dims = [4]int{1, 1, 1, 1}
			if volume > 1 {
				pair, e := proxyFamilyFixture(pc, dt)
				if e != nil {
					return f, e
				}
				f.ops = pair.ops
				f.wires = pair.wires
				f.dims = [4]int{2, 1, 1, 1}
			}
		}
		return f, nil
	}
}
func camPoly(pc polyCase, role string, volume bool) func(core.DType) (camFixture, error) {
	return func(dt core.DType) (camFixture, error) {
		brs, x, _, shape, _, e := pc.make(dt)
		if e != nil {
			return camFixture{}, e
		}
		cfg := parallel.Config{Dim: pc.cfgDim, OutFeat: pc.cfgOutFeat, Branches: len(brs), Combine: parallel.CombineAdd, SeqLen: pc.cfgSeq}
		var ops []any
		var wires [][]byte
		for _, br := range brs {
			ops = append(ops, br.op)
			wires = append(wires, br.bytes)
		}
		l, e := parallel.NewFromBranches(cfg, ops, nil)
		if e != nil {
			return camFixture{}, e
		}
		f := camFixture{kind: 1, mode: parallel.ModeNormalBP, ops: []any{l}, wires: [][]byte{encodeParallelChild(cfg, wires, nil)}, input: camInput(shape, x)}
		switch role {
		case "adversarial":
			l.SetBranchModes(parallel.ModeAdversarial, parallel.ModeFreeze)
		case "memory_sleep":
			l.SetBranchModes(parallel.ModeMemory, parallel.ModeFreeze)
			l.SetCamKit(parallel.CamKit{SurpriseThresh: 1e10})
		case "memory_awake":
			l.SetBranchModes(parallel.ModeMemory, parallel.ModeFreeze)
			l.SetCamKit(parallel.CamKit{SurpriseThresh: 0})
		case "shadow":
			l.SetBranchModes(parallel.ModeShadow, parallel.ModeNormalBP)
			l.SetCamKit(parallel.CamKit{ShadowCoef: .35})
		case "sync":
			cfgSync := parallel.DefaultCamSync()
			cfgSync.Alpha = .375
			cfgSync.When = parallel.SyncManual
			l.SetCamSync(cfgSync)
			f.action = 1
		}
		if volume {
			f.kind = 2
			f.dims = [4]int{1, 1, 1, 1}
		}
		return f, nil
	}
}
func camDenseRoot(dt core.DType, combine parallel.CombineMode) (camFixture, error) {
	f, e := kpParallelFixture(dt, combine)
	if e != nil {
		return camFixture{}, e
	}
	// The first Op is Parallel; keep the optional stem/tail out of root tests.
	return camFixture{kind: 1, mode: parallel.ModeNormalBP, ops: f.ops[:1], wires: f.wires[:1], input: camInput(f.xShape, f.x)}, nil
}
func camEachDType(pc polyCase, build func(core.DType) (camFixture, error)) func() error {
	return func() error {
		types := core.AllDTypes
		if pc.name == "gdn" {
			types = []core.DType{core.DTypeFloat32}
		}
		for _, dt := range types {
			if e := camRun(dt, build); e != nil {
				return fmt.Errorf("%s: %w", dt, e)
			}
		}
		return nil
	}
}

// Build a real 2×2×2 volume: the requested cameral Op occupies the first
// cell, with shape-preserving independent RMSNorm cells downstream.
func camExpand(build func(core.DType) (camFixture, error)) func(core.DType) (camFixture, error) {
	return func(dt core.DType) (camFixture, error) {
		probe, e := build(dt)
		if e != nil {
			return probe, e
		}
		if _, e = meshMeta(probe.ops[0]); e != nil {
			return probe, e
		}
		_, post, e := parallel.Forward(probe.ops[0].(*parallel.Layer), probe.input)
		if e != nil {
			return probe, e
		}
		f, e := build(dt)
		if e != nil {
			return f, e
		}
		f.kind = 2
		f.dims = [4]int{2, 2, 2, 1}
		for i := 1; i < 8; i++ {
			pairs, _, _, _, _, e := makeRMSPair(dt, 2, post.Shape[len(post.Shape)-1])
			if e != nil {
				return f, e
			}
			f.ops = append(f.ops, pairs[i%2].op)
			f.wires = append(f.wires, pairs[i%2].bytes)
		}
		return f, nil
	}
}
func CasesCameralTrain() []Case {
	var cases []Case
	for _, pc := range polyTrainKinds() {
		pc := pc
		for _, mode := range []parallel.TrainMode{parallel.ModeFreeze, parallel.ModeShadow} {
			mode := mode
			for _, volume := range []int{0, 1, 2} {
				volume := volume
				cases = append(cases, Case{Name: fmt.Sprintf("%s %s placement%d native dtypes", mode, pc.name, volume), Run: camEachDType(pc, camPlain(pc, mode, volume))})
			}
		}
		for _, role := range []string{"adversarial", "memory_sleep", "memory_awake", "shadow"} {
			role := role
			cases = append(cases, Case{Name: role + " " + pc.name + " branch native dtypes", Run: camEachDType(pc, camPoly(pc, role, false))})
			cases = append(cases, Case{Name: role + " " + pc.name + " volume native dtypes", Run: camEachDType(pc, camPoly(pc, role, true))})
			cases = append(cases, Case{Name: role + " " + pc.name + " volume 222 Float32", Run: func() error { return camRun(core.DTypeFloat32, camExpand(camPoly(pc, role, true))) }})
		}
	}
	for _, combine := range []parallel.CombineMode{parallel.CombineConcat, parallel.CombineAdd, parallel.CombineAvg, parallel.CombineFilter, parallel.CombineMax, parallel.CombineSparseK, parallel.CombineDisagree} {
		combine := combine
		cases = append(cases, Case{Name: "mixed seven combines " + string(combine) + " native dtypes", Run: func() error {
			for _, dt := range core.AllDTypes {
				e := camRun(dt, func(dt core.DType) (camFixture, error) {
					f, e := camDenseRoot(dt, combine)
					if e != nil {
						return f, e
					}
					l := f.ops[0].(*parallel.Layer)
					l.SetBranchModes(parallel.ModeShadow, parallel.ModeAdversarial, parallel.ModeMemory)
					l.SetBranchLRs(0, .375, 1)
					l.SetCamKit(parallel.CamKit{ShadowCoef: .75, SurpriseThresh: 1e10})
					return f, nil
				})
				if e != nil {
					return fmt.Errorf("%s: %w", dt, e)
				}
			}
			return nil
		}})
	}
	for mode := parallel.ModeInherit; mode <= parallel.ModeMemory; mode++ {
		mode := mode
		cases = append(cases, Case{Name: "branch dispatcher " + mode.String() + " native dtypes", Run: func() error {
			for _, dt := range core.AllDTypes {
				e := camRun(dt, func(dt core.DType) (camFixture, error) {
					f, e := camDenseRoot(dt, parallel.CombineAdd)
					if e != nil {
						return f, e
					}
					f.ops[0].(*parallel.Layer).SetBranchModes(mode, parallel.ModeShadow, parallel.ModeNormalBP)
					return f, nil
				})
				if e != nil {
					return fmt.Errorf("%s: %w", dt, e)
				}
			}
			return nil
		}})
	}
	// Child overrides must preserve frozen input gradients through the stem, and
	// nested Parallel modes must force the public Stack mixed walker.
	for _, nested := range []bool{false, true} {
		nested := nested
		for _, mode := range []parallel.TrainMode{parallel.ModeInherit, parallel.ModeFreeze, parallel.ModeShadow, parallel.ModeTween, parallel.ModeTweenSplit, parallel.ModeNormalBP, parallel.ModeStepBP} {
			mode := mode
			cases = append(cases, Case{Name: fmt.Sprintf("Stack overrides nested=%v %s native dtypes", nested, mode), Run: func() error {
				for _, dt := range core.AllDTypes {
					e := camRun(dt, func(dt core.DType) (camFixture, error) {
						base, e := kpParallelFixture(dt, parallel.CombineAdd)
						if e != nil {
							return camFixture{}, e
						}
						l := base.ops[0].(*parallel.Layer)
						l.SetBranchModes(parallel.ModeShadow, parallel.ModeAdversarial, parallel.ModeTween)
						ops, wires := base.ops, base.wires
						policy := func(s *parallel.Stack) {
							if mode != parallel.ModeInherit {
								s.SetChildModes(mode, parallel.ModeNormalBP)
							}
						}
						if nested {
							inner, e := parallel.NewStack(ops...)
							if e != nil {
								return camFixture{}, e
							}
							if mode != parallel.ModeInherit {
								inner.SetChildModes(mode, parallel.ModeNormalBP)
							}
							ops = []any{inner}
							var b bytes.Buffer
							writeI32(&b, skNestedStack, int32(len(wires)))
							for _, w := range wires {
								b.Write(w)
							}
							wires = [][]byte{b.Bytes()}
							policy = nil
						}
						return camFixture{mode: parallel.ModeNormalBP, ops: ops, wires: wires, input: camInput(base.xShape, base.x), rootPolicy: policy}, nil
					})
					if e != nil {
						return fmt.Errorf("%s: %w", dt, e)
					}
				}
				return nil
			}})
		}
	}
	for _, mode := range []parallel.TrainMode{parallel.ModeFreeze, parallel.ModeShadow, parallel.ModeNormalBP} {
		mode := mode
		cases = append(cases, Case{Name: "frozen gradient stem " + mode.String() + " native dtypes", Run: func() error {
			for _, dt := range core.AllDTypes {
				e := camRun(dt, func(dt core.DType) (camFixture, error) {
					base, e := kpParallelFixture(dt, parallel.CombineAdd)
					if e != nil {
						return camFixture{}, e
					}
					w := seedPat(16, 0xcaf0)
					stem, e := dense.NewConfigured(4, 4, core.ActivationSilu, dt, 0, w)
					if e != nil {
						return camFixture{}, e
					}
					l := base.ops[0].(*parallel.Layer)
					if mode == parallel.ModeNormalBP {
						l.SetBranchModes(parallel.ModeShadow, parallel.ModeAdversarial, parallel.ModeTween)
					}
					stemWire := polyEncDense(4, 4, dt, w)
					binary.LittleEndian.PutUint32(stemWire[12:16], uint32(core.ActivationSilu))
					return camFixture{mode: parallel.ModeNormalBP, ops: append([]any{stem}, base.ops...), wires: append([][]byte{stemWire}, base.wires...), input: camInput(base.xShape, base.x), rootPolicy: func(s *parallel.Stack) { s.SetChildModes(parallel.ModeNormalBP, mode, parallel.ModeNormalBP) }}, nil
				})
				if e != nil {
					return fmt.Errorf("%s: %w", dt, e)
				}
			}
			return nil
		}})
	}

	for _, name := range []string{"rmsnorm", "layernorm"} {
		name := name
		for _, pc := range polyTrainKinds() {
			pc := pc
			if pc.name != name {
				continue
			}
			cases = append(cases, Case{Name: "singleton sequence " + name + " native dtypes", Run: camEachDType(pc, func(dt core.DType) (camFixture, error) {
				brs, x, _, shape, _, e := pc.make(dt)
				if e != nil {
					return camFixture{}, e
				}
				dim := shape[len(shape)-1]
				batch := int32(len(x)) / dim
				return camFixture{mode: parallel.ModeNormalBP, ops: []any{brs[0].op}, wires: [][]byte{brs[0].bytes}, input: camInput([]int32{batch, 1, dim}, x)}, nil
			})})
		}
	}

	return cases
}
func CasesCamSync() []Case {
	var cases []Case
	for _, pc := range polyTrainKinds() {
		pc := pc
		cases = append(cases, Case{Name: "CamSync " + pc.name + " native dtypes", Run: camEachDType(pc, camPoly(pc, "sync", false))})
		build := func(dt core.DType) (camFixture, error) {
			f, e := camPoly(pc, "sync", true)(dt)
			if e != nil {
				return f, e
			}
			f.action = 0
			l := f.ops[0].(*parallel.Layer)
			l.CamSync.When = parallel.SyncAfterSample
			l.SetBranchModes(parallel.ModeMemory, parallel.ModeMemory)
			l.SetCamKit(parallel.CamKit{SurpriseThresh: 1e10})
			return f, nil
		}
		cases = append(cases, Case{Name: "CamSync " + pc.name + " volume native dtypes", Run: camEachDType(pc, build)})
		cases = append(cases, Case{Name: "CamSync " + pc.name + " volume 222 Float32", Run: func() error { return camRun(core.DTypeFloat32, camExpand(build)) }})

	}
	for _, when := range []parallel.SyncWhen{parallel.SyncManual, parallel.SyncAfterSample, parallel.SyncAfterStep, parallel.SyncAfterPulse} {
		when := when
		for _, action := range []int{0, 1, 2, 3, 4, 5, 6} {
			action := action
			cases = append(cases, Case{Name: fmt.Sprintf("sync boundary %d action %d native dtypes", when, action), Run: func() error {
				for _, dt := range core.AllDTypes {
					e := camRun(dt, func(dt core.DType) (camFixture, error) {
						f, e := camDenseRoot(dt, parallel.CombineFilter)
						if e != nil {
							return f, e
						}
						f.action = action
						l := f.ops[0].(*parallel.Layer)
						l.SetBranchModes(parallel.ModeFreeze, parallel.ModeNormalBP, parallel.ModeNormalBP)
						l.SetCamSync(parallel.CamSyncConfig{Enabled: true, When: when, Alpha: .25, BranchAlpha: []float64{1, 0, .75}, Groups: [][]int{{-1, 0, 1, 2, 99}}})
						return f, nil
					})
					if e != nil {
						return fmt.Errorf("%s: %w", dt, e)
					}
				}
				return nil
			}})
		}
	}
	for _, alpha := range []float64{0, -.2, .01, .5, 1, 2} {
		alpha := alpha
		cases = append(cases, Case{Name: fmt.Sprintf("Stack cross links alpha %g native dtypes", alpha), Run: func() error {
			for _, dt := range core.AllDTypes {
				e := camRun(dt, func(dt core.DType) (camFixture, error) {
					base, e := kpParallelFixture(dt, parallel.CombineAdd)
					if e != nil {
						return camFixture{}, e
					}
					l := base.ops[0].(*parallel.Layer)
					l.Branches[1].(*dense.Layer).Weights.Bias = []float64{-.1, .25, -.2, .4}
					var branchWires [][]byte
					raw := seedPat(48, 0x6811)
					for i := 0; i < 3; i++ {
						child := polyEncDense(4, 4, dt, raw[i*16:(i+1)*16])
						if i == 1 {
							var bw bytes.Buffer
							bw.Write(child[:len(child)-4])
							writeI32(&bw, 4)
							for _, v := range []float64{-.1, .25, -.2, .4} {
								writeF64(&bw, v)
							}
							child = bw.Bytes()
						}
						branchWires = append(branchWires, child)
					}
					base.wires[0] = encodeParallelChild(l.Cfg, branchWires, nil)
					d := base.ops[1].(*dense.Layer)
					// Explicit nonzero biases exercise bias averaging and resident reload.
					d.Weights.Bias = []float64{.2, -.3, .4, .1}
					wire := append([]byte(nil), base.wires[1][:len(base.wires[1])-4]...)
					var b bytes.Buffer
					b.Write(wire)
					writeI32(&b, 4)
					for _, v := range []float64{.2, -.3, .4, .1} {
						writeF64(&b, v)
					}
					cfg := parallel.CamSyncConfig{Enabled: true, Alpha: alpha, When: parallel.SyncManual, Cross: []parallel.SyncPair{{A: parallel.SyncEndpoint{StackIdx: 0, Branch: 1, Store: 0}, B: parallel.SyncEndpoint{StackIdx: 1, Branch: 0, Store: 0}}}}
					return camFixture{action: 1, mode: parallel.ModeNormalBP, ops: base.ops, wires: [][]byte{base.wires[0], b.Bytes()}, input: camInput(base.xShape, base.x), rootPolicy: func(s *parallel.Stack) { s.SetCamSync(cfg) }}, nil
				})
				if e != nil {
					return fmt.Errorf("%s: %w", dt, e)
				}
			}
			return nil
		}})
	}
	for _, which := range []string{"disabled", "groups", "mismatched_slots", "nested_cross", "line_after_step", "line_after_sample"} {
		which := which
		cases = append(cases, Case{Name: "CamSync policy " + which + " native dtypes", Run: func() error {
			for _, dt := range core.AllDTypes {
				e := camRun(dt, func(dt core.DType) (camFixture, error) {
					base, e := kpParallelFixture(dt, parallel.CombineAdd)
					if e != nil {
						return camFixture{}, e
					}
					l := base.ops[0].(*parallel.Layer)
					cfg := parallel.DefaultCamSync()
					cfg.When = parallel.SyncManual
					cfg.Alpha = .4
					f := camFixture{mode: parallel.ModeNormalBP, action: 1, ops: base.ops, wires: base.wires, input: camInput(base.xShape, base.x)}
					switch which {
					case "disabled":
						f.kind = 1
						f.ops = f.ops[:1]
						f.wires = f.wires[:1]
						cfg.Enabled = false
						l.SetCamSync(cfg)
					case "groups":
						f.kind = 1
						f.ops = f.ops[:1]
						f.wires = f.wires[:1]
						cfg.Groups = [][]int{{0, 1}, {1, 2}}
						cfg.BranchAlpha = []float64{2, -1, .2}
						l.SetBranchModes(parallel.ModeShadow, parallel.ModeNormalBP, parallel.ModeNormalBP)
						l.SetCamSync(cfg)
					case "mismatched_slots":
						pairs, _, _, _, _, e := makeRMSPair(dt, 2, 4)
						if e != nil {
							return f, e
						}
						p, e := parallel.NewFromBranches(parallel.Config{Dim: 4, OutFeat: 4, Branches: 2, Combine: parallel.CombineAdd}, []any{l.Branches[0], pairs[0].op}, nil)
						if e != nil {
							return f, e
						}
						p.SetCamSync(cfg)
						f.kind = 1
						f.ops = []any{p}
						raw := seedPat(48, 0x6811)
						f.wires = [][]byte{encodeParallelChild(p.Cfg, [][]byte{polyEncDense(4, 4, dt, raw[:16]), pairs[0].bytes}, nil)}
					case "nested_cross":
						inner, e := parallel.NewStack(base.ops...)
						if e != nil {
							return f, e
						}
						var bw bytes.Buffer
						writeI32(&bw, skNestedStack, int32(len(base.wires)))
						for _, w := range base.wires {
							bw.Write(w)
						}
						f.ops = []any{inner}
						f.wires = [][]byte{bw.Bytes()}
						cfg.Cross = []parallel.SyncPair{{A: parallel.SyncEndpoint{StackIdx: 0, Branch: 0, Store: 1}, B: parallel.SyncEndpoint{StackIdx: 0, Branch: 1, Store: 0}}}
						f.rootPolicy = func(s *parallel.Stack) { s.SetCamSync(cfg) }
					case "line_after_step", "line_after_sample":
						f.action = 0
						f.mode = parallel.ModeStepBP
						f.epochs = 5
						cfg.When = parallel.SyncAfterStep
						if which == "line_after_sample" {
							cfg.When = parallel.SyncAfterSample
						}
						l.SetBranchModes(parallel.ModeFreeze, parallel.ModeNormalBP, parallel.ModeTween)
						f.rootPolicy = func(s *parallel.Stack) { s.SetCamSync(cfg) }
					}
					return f, nil
				})
				if e != nil {
					return fmt.Errorf("%s: %w", dt, e)
				}
			}
			return nil
		}})
	}

	return cases
}
