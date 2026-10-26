package go_v_metal

import (
	"fmt"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/runtime/dispatch"
	"reflect"
)

// CasesNativeCPPRouting verifies real Go -> C ABI execution rather than merely
// setting BackendNativeCPU (which previously left many host-only dispatchers).
func CasesNativeCPPRouting() []Case {
	var out []Case
	for _, pc := range polyTrainKinds() {
		pc := pc
		out = append(out, Case{Name: "C ABI forward backward " + pc.name + " native dtypes", Run: func() error {
			if !nativecpp.NativeCPUAvailable() {
				return fmt.Errorf("C++ native library unavailable")
			}
			dtypes := core.AllDTypes
			if pc.name == "gdn" {
				dtypes = []core.DType{core.DTypeFloat32}
			}
			for _, dt := range dtypes {
				if e := runNativeCPPRoute(pc, dt); e != nil {
					return fmt.Errorf("%s: %w", dt, e)
				}
			}
			return nil
		}})
	}
	return out
}
func runNativeCPPRoute(pc polyCase, dt core.DType) error {
	branches, x, _, shape, _, e := pc.make(dt)
	if e != nil {
		return e
	}
	refs, _, _, _, _, e := pc.make(dt)
	if e != nil {
		return e
	}
	op, ref := branches[0].op, refs[0].op
	dispatch.SyncExec(op, core.ExecConfig{Backend: core.BackendNativeCPU})
	dispatch.SyncExec(ref, core.ExecConfig{Backend: core.BackendCPUTiled})
	input := camInput(shape, x)
	operation := pc.name
	if operation == "stack" || operation == "metacognition" {
		operation = "dense"
	}
	readCount := func(back bool) uint64 {
		if operation == "nested_parallel" {
			return nativecpp.CompletedCalls()
		}
		suffix := "_forward"
		if back {
			suffix = "_normalbp"
		}
		return nativecpp.CompletedCallsFor(operation + suffix)
	}
	count := readCount(false)
	pre, post, e := dispatch.Forward(op, input)
	if e != nil {
		return e
	}
	if readCount(false) == count {
		return fmt.Errorf("forward ran no successful C ABI calls")
	}
	rp, ro, e := dispatch.Forward(ref, input)
	if e != nil {
		return e
	}
	if e = camCompare(camResult{post: meshView(ro)}, camResult{post: meshView(post)}); e != nil {
		return fmt.Errorf("forward: %w", e)
	}
	gy := core.NewTensor[float32](post.Shape...)
	copy(gy.Data, seedPat(gy.Len(), 0xcafe))
	count = readCount(true)
	gx, dw, e := dispatch.Backward(op, gy, input, pre)
	if e != nil {
		return e
	}
	if readCount(true) == count {
		return fmt.Errorf("backward ran no successful C ABI calls")
	}
	rx, rw, e := dispatch.Backward(ref, gy, input, rp)
	if e != nil {
		return e
	}
	if gx != nil && rx != nil && !reflect.DeepEqual(gx.Shape, rx.Shape) {
		return fmt.Errorf("input gradient shape native=%v Go=%v", gx.Shape, rx.Shape)
	}
	switch pc.name {
	case "cnn1", "cnn2", "cnn3", "convt1", "convt2", "convt3", "embedding", "rnn", "lstm", "mamba", "mha", "gdn", "kmeans", "sequential", "residual":
		if dw != nil && rw != nil && !reflect.DeepEqual(dw.Shape, rw.Shape) {
			return fmt.Errorf("weight gradient shape native=%v Go=%v", dw.Shape, rw.Shape)
		}
	}
	if e = camCompare(camResult{model: parallelNBPOut{WAfter: [][]float32{routingData(rx), routingData(rw)}}}, camResult{model: parallelNBPOut{WAfter: [][]float32{routingData(gx), routingData(dw)}}}); e != nil {
		return fmt.Errorf("gradient: %w", e)
	}
	if e = dispatch.ApplyGradSGD(op, dw, .001); e != nil {
		return e
	}
	if e = dispatch.ApplyGradSGD(ref, rw, .001); e != nil {
		return e
	}
	a, e := kpPackOp(op)
	if e != nil {
		return e
	}
	b, e := kpPackOp(ref)
	if e != nil {
		return e
	}
	return camCompare(camResult{model: parallelNBPOut{WAfter: [][]float32{b}}}, camResult{model: parallelNBPOut{WAfter: [][]float32{a}}})
}
