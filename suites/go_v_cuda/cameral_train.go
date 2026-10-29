package go_v_cuda

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"math"
	"os/exec"
	"reflect"
)

type camFixture struct {
	kind, action, epochs int
	mode                 parallel.TrainMode
	ops                  []any
	wires                [][]byte
	input                *core.Tensor[float32]
	root                 *parallel.Stack
	grid                 *architecture.Grid
	dims                 [4]int
	rootPolicy           func(*parallel.Stack)
}
func wLens(ws [][]float32) []int {
	out := make([]int, len(ws))
	for i, w := range ws {
		out[i] = len(w)
	}
	return out
}

type camResult struct {
	model parallelNBPOut
	post  meshTensor
}

func camChildren(op any) []any {
	switch v := op.(type) {
	case *parallel.Layer:
		return v.Branches
	case *parallel.Stack:
		return v.Children
	}
	return nil
}
func camWriteControls(b *bytes.Buffer, op any) {
	var modes []parallel.TrainMode
	var lrs []float64
	var coef, threshold, last float64
	var sync *parallel.CamSyncConfig
	switch v := op.(type) {
	case *parallel.Layer:
		modes = v.BranchModes
		lrs = v.BranchLRs
		sync = v.CamSync
		if v.CamKit != nil {
			coef = v.CamKit.ShadowCoef
			threshold = v.CamKit.SurpriseThresh
			last = v.CamKit.LastLoss
		}
	case *parallel.Stack:
		modes = v.ChildModes
		sync = v.CamSync
	}
	writeI32(b, int32(len(modes)))
	for _, m := range modes {
		writeI32(b, int32(m))
	}
	writeI32(b, int32(len(lrs)))
	for _, x := range lrs {
		writeF64(b, x)
	}
	for _, v := range []float64{coef, threshold, last} {
		writeF64(b, v)
	}
	if sync == nil {
		writeI32(b, 0)
		return
	}
	writeI32(b, 1)
	enabled := int32(0)
	if sync.Enabled {
		enabled = 1
	}
	writeI32(b, enabled)
	writeF64(b, sync.Alpha)
	writeI32(b, int32(sync.When))
	writeI32(b, int32(len(sync.BranchAlpha)))
	for _, x := range sync.BranchAlpha {
		writeF64(b, x)
	}
	writeI32(b, int32(len(sync.Groups)))
	for _, group := range sync.Groups {
		writeI32(b, int32(len(group)))
		for _, i := range group {
			writeI32(b, int32(i))
		}
	}
	writeI32(b, int32(len(sync.Cross)))
	for _, pair := range sync.Cross {
		for _, e := range []parallel.SyncEndpoint{pair.A, pair.B} {
			writeI32(b, int32(e.StackIdx), int32(e.Branch), int32(e.Store))
		}
	}
}
func camWriteTree(b *bytes.Buffer, op any) {
	camWriteControls(b, op)
	for _, c := range camChildren(op) {
		camWriteTree(b, c)
	}
}
func camPrepare(f *camFixture) error {
	for _, op := range f.ops {
		if _, e := meshMeta(op); e != nil {
			return e
		}
	}
	var err error
	f.root, err = parallel.NewStack(f.ops...)
	if err != nil {
		return err
	}
	f.root.Exec.Backend = core.BackendCPUTiled
	f.root.Exec.MultiCore = false
	if f.rootPolicy != nil {
		f.rootPolicy(f.root)
	}
	if f.epochs == 0 {
		f.epochs = 3
	}
	if f.dims == [4]int{} {
		f.dims = [4]int{1, 1, len(f.ops), 1}
	}
	if f.kind == 2 {
		f.grid = architecture.NewGrid(f.dims[0], f.dims[1], f.dims[2], f.dims[3])
		coords := f.grid.HopOrder()
		if len(coords) != len(f.ops) {
			return fmt.Errorf("cameral grid count")
		}
		for i, op := range f.ops {
			meta, e := meshMeta(op)
			if e != nil {
				return e
			}
			c := f.grid.At(coords[i].Z, coords[i].Y, coords[i].X, coords[i].L)
			c.Layer = meta
			c.Op = op
		}
	}
	return nil
}
func camForward(f *camFixture) (*core.Tensor[float32], error) {
	if f.kind == 1 {
		_, post, e := parallel.Forward(f.ops[0].(*parallel.Layer), f.input)
		return post, e
	}
	if f.kind == 2 {
		fw, e := forward.Forward(f.grid, f.input)
		if e != nil {
			return nil, e
		}
		return fw.Output, nil
	}
	_, post, e := parallel.ForwardStack(f.root, f.input)
	return post, e
}

// Grid reference composes Go's actual per-cell backward, native SGD and public
// Parallel/Stack credit APIs. The global loss supplies Memory's volume gate.
func camVolume(f *camFixture, target *core.Tensor[float32]) (float64, error) {
	fw, e := forward.Forward(f.grid, f.input)
	if e != nil {
		return 0, e
	}
	loss, e := training.MSE(fw.Output, target)
	if e != nil {
		return loss, e
	}
	if f.mode.IsFrozen() {
		return loss, nil
	}
	gy, e := training.MSEGrad(fw.Output, target)
	if e != nil {
		return loss, e
	}
	type update struct {
		op                 any
		input, pre, gy, dw *core.Tensor[float32]
	}
	var updates []update
	for i := len(fw.Steps) - 1; i >= 0; i-- {
		st := fw.Steps[i]
		c := f.grid.At(st.Coord.Z, st.Coord.Y, st.Coord.X, st.Coord.L)
		gx, dw, e := backward.Cell(c, gy, st.Input, st.Pre)
		if e != nil {
			return loss, e
		}
		updates = append(updates, update{c.Op, st.Input, st.Pre, gy, dw})
		gy = gx
	}
	// Ordinary cells apply accumulated gradients first, as the CUDA volume API.
	for _, u := range updates {
		switch u.op.(type) {
		case *parallel.Layer, *parallel.Stack:
			continue
		}
		if u.dw != nil && u.dw.Len() > 0 {
			s, e := parallel.NewStack(u.op)
			if e != nil {
				return loss, e
			}
			if e = parallel.ApplyGradSGDStack(s, u.dw, .017); e != nil {
				return loss, e
			}
		}
	}
	for i := len(updates) - 1; i >= 0; i-- {
		u := updates[i]
		switch v := u.op.(type) {
		case *parallel.Layer:
			v.NoteLoss(loss)
			if e = parallel.Train(v, u.gy, u.input, u.pre, f.mode, .017); e != nil {
				return loss, e
			}
			if e = v.MaybeSync(parallel.SyncAfterStep); e != nil {
				return loss, e
			}
			if e = v.MaybeSync(parallel.SyncAfterSample); e != nil {
				return loss, e
			}
		case *parallel.Stack:
			if e = parallel.TrainStack(v, u.gy, u.input, u.pre, f.mode, .017); e != nil {
				return loss, e
			}
			if e = v.MaybeSync(parallel.SyncAfterStep); e != nil {
				return loss, e
			}
			if e = v.MaybeSync(parallel.SyncAfterSample); e != nil {
				return loss, e
			}
		}
	}
	return loss, nil
}
func camSync(f *camFixture, action int) error {
	if f.kind == 2 {
		return fmt.Errorf("manual grid sync uses per-cell policy")
	}
	if f.kind == 1 {
		l := f.ops[0].(*parallel.Layer)
		switch action {
		case 1:
			return l.SyncNow()
		case 2:
			return l.Pulse()
		case 3:
			return l.MaybeSync(parallel.SyncAfterStep)
		case 4:
			return l.MaybeSync(parallel.SyncAfterSample)
		}
	}
	switch action {
	case 1:
		return f.root.SyncNow()
	case 2:
		return f.root.Pulse()
	case 3:
		return f.root.MaybeSync(parallel.SyncAfterStep)
	case 4:
		return f.root.MaybeSync(parallel.SyncAfterSample)
	}
	return fmt.Errorf("unknown sync action")
}
func runCUDACam(payload []byte, iters int) (camResult, error) {
	var out camResult
	tool, e := findCUDAFwd()
	if e != nil {
		return out, e
	}
	args := []string{}
	if iters > 0 {
		args = append(args, fmt.Sprintf("--bench=%d", iters))
	}
	cmd := exec.Command(tool, args...)
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if e = cmd.Run(); e != nil {
		return out, fmt.Errorf("CUDA cameral: %w: %s", e, stderr.String())
	}
	data := stdout.Bytes()
	if len(data) >= 16 && string(data[:4]) == "WVB1" {
		data = data[16:]
	}
	if len(data) < 4 || string(data[:4]) != "CMBO" {
		return out, fmt.Errorf("cameral output magic")
	}
	r := bytes.NewReader(data[4:])
	if e = binary.Read(r, binary.LittleEndian, &out.model.loss); e != nil {
		return out, e
	}
	var count int32
	if e = binary.Read(r, binary.LittleEndian, &count); e != nil || count < 0 || count > 4096 {
		return out, fmt.Errorf("cameral output count")
	}
	out.model.WAfter = make([][]float32, count)
	for i := range out.model.WAfter {
		var n int32
		if e = binary.Read(r, binary.LittleEndian, &n); e != nil || n < 0 || n > 1<<26 {
			return out, fmt.Errorf("cameral output size")
		}
		out.model.WAfter[i] = make([]float32, n)
		if e = binary.Read(r, binary.LittleEndian, out.model.WAfter[i]); e != nil {
			return out, e
		}
	}
	out.post, e = readMeshTensor(r)
	if e != nil {
		return out, e
	}
	if r.Len() != 0 {
		return out, fmt.Errorf("cameral trailing output")
	}
	return out, nil
}
func camRun(dt core.DType, makeFixture func(core.DType) (camFixture, error)) error {
	f, e := makeFixture(dt)
	if e != nil {
		return e
	}
	if e = camPrepare(&f); e != nil {
		return e
	}
	post, e := camForward(&f)
	if e != nil {
		return fmt.Errorf("shape discovery: %w", e)
	}
	target := core.NewTensor[float32](post.Shape...)
	copy(target.Data, seedPat(target.Len(), 0xcab1))
	f, e = makeFixture(dt)
	if e != nil {
		return e
	}
	if e = camPrepare(&f); e != nil {
		return e
	}
	var wire bytes.Buffer
	wire.WriteString("CMB1")
	writeI32(&wire, int32(f.kind), int32(f.mode), int32(f.epochs), int32(f.action), int32(len(f.ops)))
	for _, d := range f.dims {
		writeI32(&wire, int32(d))
	}
	writeF64(&wire, .017)
	for _, w := range f.wires {
		wire.Write(w)
	}
	camWriteControls(&wire, f.root)
	for _, op := range f.ops {
		camWriteTree(&wire, op)
	}
	writeMeshTensor(&wire, meshView(f.input))
	writeMeshTensor(&wire, meshView(target))
	protected := map[int][]float32{}
	if f.mode.IsFrozen() && f.action == 0 {
		for i, op := range f.ops {
			w, err := kpPackOp(op)
			if err != nil {
				return err
			}
			protected[i] = w
		}
	}
	var protectedBranches = map[int][]float32{}
	if f.kind == 1 {
		l := f.ops[0].(*parallel.Layer)
		for i, op := range l.Branches {
			bm := l.EffectiveBranchMode(i, f.mode)
			sleep := bm == parallel.ModeMemory && l.CamKit != nil && l.CamKit.SurpriseThresh >= 1e10 && (l.CamSync == nil || !l.CamSync.Enabled)
			if bm.IsFrozen() || sleep {
				w, err := kpPackOp(op)
				if err != nil {
					return err
				}
				protectedBranches[i] = w
			}
		}
	}
	ref := camResult{}
	if f.action >= 1 && f.action <= 4 {
		if _, e = camForward(&f); e != nil {
			return e
		}
	}
	for epoch := 0; epoch < f.epochs; epoch++ {
		if f.action >= 1 && f.action <= 4 {
			if e = camSync(&f, f.action); e != nil {
				return e
			}
			continue
		}
		if f.kind == 0 {
			ref.model.loss, e = parallel.TrainStackMSE(f.root, f.input, target, f.mode, .017)
		} else if f.kind == 1 {
			ref.model.loss, e = parallel.TrainMSE(f.ops[0].(*parallel.Layer), f.input, target, f.mode, .017)
		} else {
			ref.model.loss, e = camVolume(&f, target)
		}
		if e != nil {
			return fmt.Errorf("Go epoch %d: %w", epoch, e)
		}
		if f.action >= 5 {
			a := 1
			if f.action == 6 {
				a = 2
			}
			if e = camSync(&f, a); e != nil {
				return e
			}
		}
	}
	post, e = camForward(&f)
	if e != nil {
		return e
	}
	ref.post = meshView(post)
	for _, op := range f.ops {
		w, e := kpPackOp(op)
		if e != nil {
			return e
		}
		ref.model.WAfter = append(ref.model.WAfter, w)
	}
	got, e := runCUDACam(wire.Bytes(), 0)
	if e != nil {
		return e
	}
	for i, w := range protected {
		if len(w) > 0 && !reflect.DeepEqual(w, got.model.WAfter[i]) {
			return fmt.Errorf("frozen Op %d changed native weights", i)
		}
	}
	if f.kind == 1 {
		l := f.ops[0].(*parallel.Layer)
		offset := 0
		for i, op := range l.Branches {
			w, err := kpPackOp(op)
			if err != nil {
				return err
			}
			// Metacognition can reset its observed store during Forward, even
			// while optimizer credit is frozen. Preserve the Go forward result.
			if initial, ok := protectedBranches[i]; ok && len(initial) > 0 && !reflect.DeepEqual(w, got.model.WAfter[0][offset:offset+len(w)]) {
				return fmt.Errorf("frozen/sleeping branch %d changed weights", i)
			}
			offset += len(w)
		}
	}
	if e = camCompare(ref, got); e != nil {
		return e
	}
	replay, e := runCUDACam(wire.Bytes(), 2)
	if e != nil {
		return e
	}
	if !reflect.DeepEqual(got, replay) {
		return fmt.Errorf("cameral replay changed loss, weights or post")
	}
	return nil
}

// Standard scale-aware absolute+relative tolerance for CPU/GPU training.
// A score is a tolerance band, not the fraction of exact elements. Near-zero
// cancellation must not turn a tiny absolute error into a spurious failure.
func camCompare(a, b camResult) error {
	if len(a.model.WAfter) != len(b.model.WAfter) || !reflect.DeepEqual(a.post.shape, b.post.shape) {
		return fmt.Errorf("cameral shape/count mismatch GoW=%d CudaW=%d GoPost=%v CudaPost=%v GoWLens=%v CudaWLens=%v",
			len(a.model.WAfter), len(b.model.WAfter), a.post.shape, b.post.shape,
			wLens(a.model.WAfter), wLens(b.model.WAfter))
	}
	check := func(label string, x, y []float32) error {
		if len(x) != len(y) {
			return fmt.Errorf("%s length mismatch", label)
		}
		exact := true
		worst := 0.0
		for i := range x {
			av, bv := float64(x[i]), float64(y[i])
			if math.IsNaN(av) || math.IsNaN(bv) || math.IsInf(av, 0) || math.IsInf(bv, 0) {
				return fmt.Errorf("%s non-finite value", label)
			}
			if math.Float32bits(x[i]) != math.Float32bits(y[i]) {
				exact = false
			}
			ratio := math.Abs(av-bv) / (1 + math.Max(math.Abs(av), math.Abs(bv)))
			if ratio > worst {
				worst = ratio
			}
		}
		if worst > 1e-3 {
			return fmt.Errorf("%s below industry: normalized error=%g; %s", label, worst, suites.FormatSpectrum(suites.CompareF32Spectrum(x, y)))
		}
		if !exact {
			band := suites.DetStrict
			if worst > 1e-5 {
				band = suites.DetIndustry
			}
			m := suites.CompareF32Spectrum(x, y)
			fmt.Printf("\n      %s band=%s (%.0f%%) normalized=%g exact_bits=%.2f%% max_abs=%g", label, band, band.Percent(), worst, m.ExactPct, m.MaxAbs)
		}
		return nil
	}
	av, bv := a.model.loss, b.model.loss
	if math.IsNaN(av) || math.IsNaN(bv) || math.IsInf(av, 0) || math.IsInf(bv, 0) || math.Abs(av-bv) > 1e-3*(1+math.Max(math.Abs(av), math.Abs(bv))) {
		return fmt.Errorf("cameral loss Go=%g CUDA=%g", av, bv)
	}
	for i := range a.model.WAfter {
		if e := check(fmt.Sprintf("W[%d]", i), a.model.WAfter[i], b.model.WAfter[i]); e != nil {
			return e
		}
	}
	return check("post", a.post.data, b.post.data)
}
