package go_v_cuda

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os/exec"
	"reflect"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/layers/convt1"
	"github.com/openfluke/welvet/layers/convt2"
	"github.com/openfluke/welvet/layers/convt3"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/layers/mamba"
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/runtime/step"
	"github.com/openfluke/welvet/runtime/training"
)

type meshClockFixture struct {
	credit                    parallel.TrainMode
	lr                        *float64
	z, y, x, l, ticks, warmup int
	cells                     []polyBranchEnc
	shape                     []int32
	input, target             []float32
	targetShape               []int32
	links                     [][8]int32
	family                    string
}

func meshMeta(op any) (core.Layer, error) {
	switch v := op.(type) {
	case *sequential.Layer:
		for _, c := range v.ChildOps() {
			if _, err := meshMeta(c); err != nil {
				return core.Layer{}, err
			}
		}
	case *residual.Layer:
		for _, c := range v.ChildOps() {
			if _, err := meshMeta(c); err != nil {
				return core.Layer{}, err
			}
		}
	case *parallel.Layer:
		for _, c := range v.Branches {
			if _, err := meshMeta(c); err != nil {
				return core.Layer{}, err
			}
		}
		if v.Gate != nil {
			if _, err := meshMeta(v.Gate); err != nil {
				return core.Layer{}, err
			}
		}
	case *parallel.Stack:
		for _, c := range v.Children {
			if _, err := meshMeta(c); err != nil {
				return core.Layer{}, err
			}
		}
	case *kmeans.Layer:
		disableKMeansTrain(v)
	case *metacognition.Layer:
		disableMetacognitionTrain(v)
	case *mamba.Layer:
		disableMambaTrain(v)
	case *rnn.Layer:
		disableRNN(v)
	case *lstm.Layer:
		disableLSTMTrainGates(v)
	}

	switch v := op.(type) {
	case *dense.Layer:
		v.Exec.Backend = core.BackendCPUTiled
		v.Exec.MultiCore = false
		v.Core.MultiCore = false
		return v.Core, nil
	case *rmsnorm.Layer:
		v.Exec.Backend = core.BackendCPUTiled
		v.Exec.MultiCore = false
		v.Core.MultiCore = false
		return v.Core, nil
	case *layernorm.Layer:
		v.Exec.Backend = core.BackendCPUTiled
		v.Exec.MultiCore = false
		v.Core.MultiCore = false
		return v.Core, nil
	case *swiglu.Layer:
		v.Exec.Backend = core.BackendCPUTiled
		v.Exec.MultiCore = false
		v.Core.MultiCore = false
		return v.Core, nil
	case *softmax.Layer:
		v.Exec.Backend = core.BackendCPUTiled
		v.Exec.MultiCore = false
		v.Core.MultiCore = false
		return v.Core, nil
	case *embedding.Layer:
		v.Exec.Backend = core.BackendCPUTiled
		v.Exec.MultiCore = false
		v.Core.MultiCore = false
		return v.Core, nil
	case *mha.Layer:
		v.Exec.Backend = core.BackendCPUTiled
		v.Exec.MultiCore = false
		v.Core.MultiCore = false
		return v.Core, nil
	case *cnn1.Layer:
		v.Exec.Backend = core.BackendCPUTiled
		v.Exec.MultiCore = false
		v.Core.MultiCore = false
		return v.Core, nil
	case *cnn2.Layer:
		v.Exec.Backend = core.BackendCPUTiled
		v.Exec.MultiCore = false
		v.Core.MultiCore = false
		return v.Core, nil
	case *cnn3.Layer:
		v.Exec.Backend = core.BackendCPUTiled
		v.Exec.MultiCore = false
		v.Core.MultiCore = false
		return v.Core, nil
	case *convt1.Layer:
		v.Exec.Backend = core.BackendCPUTiled
		v.Exec.MultiCore = false
		v.Core.MultiCore = false
		return v.Core, nil
	case *convt2.Layer:
		v.Exec.Backend = core.BackendCPUTiled
		v.Exec.MultiCore = false
		v.Core.MultiCore = false
		return v.Core, nil
	case *convt3.Layer:
		v.Exec.Backend = core.BackendCPUTiled
		v.Exec.MultiCore = false
		v.Core.MultiCore = false
		return v.Core, nil
	case *rnn.Layer:
		v.Exec.Backend = core.BackendCPUTiled
		v.Exec.MultiCore = false
		v.Core.MultiCore = false
		return v.Core, nil
	case *lstm.Layer:
		v.Exec.Backend = core.BackendCPUTiled
		v.Exec.MultiCore = false
		v.Core.MultiCore = false
		return v.Core, nil
	case *sequential.Layer:
		v.Exec.Backend = core.BackendCPUTiled
		v.Exec.MultiCore = false
		v.Core.MultiCore = false
		return v.Core, nil
	case *residual.Layer:
		v.Exec.Backend = core.BackendCPUTiled
		v.Exec.MultiCore = false
		v.Core.MultiCore = false
		return v.Core, nil
	case *mamba.Layer:
		v.Exec.Backend = core.BackendCPUTiled
		v.Exec.MultiCore = false
		v.Core.MultiCore = false
		return v.Core, nil
	case *metacognition.Layer:
		v.Exec.Backend = core.BackendCPUTiled
		v.Exec.MultiCore = false
		v.Core.MultiCore = false
		return v.Core, nil
	case *gdn.Layer:
		disableGDNTrain(v)
		return core.Layer{Type: core.LayerGDN, DType: core.DTypeFloat32}, nil
	case *kmeans.Layer:
		v.Exec.Backend = core.BackendCPUTiled
		v.Exec.MultiCore = false
		v.Core.MultiCore = false
		return v.Core, nil
	case *parallel.Layer:
		v.Exec.Backend = core.BackendCPUTiled
		v.Exec.MultiCore = false
		v.Core.MultiCore = false
		return v.Core, nil
	case *parallel.Stack:
		v.Exec.Backend = core.BackendCPUTiled
		v.Exec.MultiCore = false
		v.Core.MultiCore = false
		meta := v.Core
		meta.Type = core.LayerStack
		return meta, nil
	default:
		return core.Layer{}, fmt.Errorf("mesh metadata %T", op)
	}
}
func meshView(t *core.Tensor[float32]) meshTensor {
	if t == nil {
		return meshTensor{}
	}
	shape := make([]int32, len(t.Shape))
	for i, d := range t.Shape {
		shape[i] = int32(d)
	}
	return meshTensor{shape: shape, data: append([]float32(nil), t.Data...)}
}
func writeMeshTensor(b *bytes.Buffer, t meshTensor) {
	writeI32(b, int32(len(t.shape)))
	writeI32(b, t.shape...)
	polyWriteF32s(b, t.data)
}
func encodeMeshClock(f meshClockFixture) []byte {
	var b bytes.Buffer
	switch f.credit {
	case parallel.ModeMeshTweenSplit:
		b.WriteString("MGS1")
	case parallel.ModeMeshTweenAlt:
		b.WriteString("MGA1")
	case parallel.ModeMeshTweenSplitFastProxy:
		b.WriteString("MGF1")
	case parallel.ModeMeshTweenSplitSparse:
		b.WriteString("MGQ1")
	default:
		if f.credit == parallel.ModeMeshTween {
			b.WriteString("MGW1")
		} else if f.credit == parallel.ModeMeshTweenChain {
			b.WriteString("MGC1")
		} else {
			b.WriteString("MGB1")
		}
	}
	writeI32(&b, int32(f.z), int32(f.y), int32(f.x), int32(f.l), int32(f.ticks), int32(f.warmup))
	lr := 0.001
	if f.lr != nil {
		lr = *f.lr
	}
	writeF64(&b, lr)
	writeI32(&b, int32(len(f.links)))
	for _, link := range f.links {
		writeI32(&b, link[:]...)
	}
	for _, c := range f.cells {
		if c.op == nil {
			writeI32(&b, 0)
		} else {
			writeI32(&b, 1)
			b.Write(c.bytes)
		}
	}
	writeMeshTensor(&b, meshTensor{shape: f.shape, data: f.input})
	writeMeshTensor(&b, meshTensor{shape: f.targetShape, data: f.target})
	return b.Bytes()
}
func runGoMeshClock(f meshClockFixture) (meshClockOut, error) {
	var out meshClockOut
	grid := architecture.NewGrid(f.z, f.y, f.x, f.l)
	grid.Exec.MultiCore = false
	coords := grid.HopOrder()
	if len(coords) != len(f.cells) {
		return out, fmt.Errorf("mesh cell count")
	}
	for i, c := range coords {
		cell := f.cells[i]
		if cell.op == nil {
			grid.At(c.Z, c.Y, c.X, c.L).Layer.IsDisabled = true
			continue
		}
		meta, err := meshMeta(cell.op)
		if err != nil {
			return out, err
		}
		if err = grid.BindOp(c.Z, c.Y, c.X, c.L, meta, cell.op); err != nil {
			return out, err
		}
	}
	input := core.NewTensor[float32](ints(f.shape)...)
	copy(input.Data, f.input)
	state := step.New[float32](grid)
	for t := 0; t < f.warmup; t++ {
		state.SetInput(input)
		if _, err := step.Forward(grid, state, false); err != nil {
			return out, err
		}
	}
	for _, q := range f.links {
		if err := grid.SetRemoteLink(int(q[0]), int(q[1]), int(q[2]), int(q[3]), int(q[4]), int(q[5]), int(q[6]), int(q[7])); err != nil {
			return out, err
		}
	}
	for t := 0; t < f.ticks; t++ {
		state.SetInput(input)
		if _, err := step.Forward(grid, state, true); err != nil {
			return out, fmt.Errorf("tick %d: %w", t, err)
		}
		snapshot := make([][]meshTensor, len(f.cells))
		for i := range f.cells {
			snapshot[i] = []meshTensor{meshView(state.HistoryIn[t][i]), meshView(state.HistoryPre[t][i]), meshView(state.LayerData[i])}
		}
		out.history = append(out.history, snapshot)
	}
	post := state.LayerData[len(f.cells)-1]
	target := core.NewTensor[float32](ints(f.targetShape)...)
	copy(target.Data, f.target)
	var err error
	out.loss, err = training.MSE(post, target)
	if err != nil {
		return out, err
	}
	out.post = meshView(post)
	gy, err := training.MSEGrad(post, target)
	if err != nil {
		return out, err
	}
	var gin *core.Tensor[float32]
	var grads [][2]*core.Tensor[float32]
	if f.credit == parallel.ModeMeshTween || (f.credit >= parallel.ModeMeshTweenSplit && f.credit <= parallel.ModeMeshTweenSplitSparse) {
		if err = meshGoLocalCredit(grid, state, gy, meshLR(f), f.credit); err != nil {
			return out, err
		}
		grads = make([][2]*core.Tensor[float32], len(f.cells))
	} else {
		gin, grads, err = step.Backward(grid, state, gy)
		if err != nil {
			return out, err
		}
	}
	out.gin = meshView(gin)
	out.weights = make([][]float32, len(f.cells))
	out.grads = make([]meshTensor, len(f.cells))
	for i, c := range f.cells {
		out.grads[i] = meshView(grads[i][1])
		if len(out.grads[i].data) > 0 {
			out.grads[i].shape = []int32{int32(len(out.grads[i].data))}
		}
		if c.op == nil {
			continue
		}
		if meshLR(f) != 0 && grads[i][1] != nil && len(grads[i][1].Data) > 0 {
			wrapper, e := parallel.NewStack(c.op)
			if e != nil {
				return out, e
			}
			if e = parallel.ApplyGradSGDStack(wrapper, grads[i][1], meshLR(f)); e != nil {
				return out, e
			}
		}
		out.weights[i], err = kpPackOp(c.op)
		if err != nil {
			return out, err
		}
	}
	return out, nil
}
func readMeshTensor(r *bytes.Reader) (meshTensor, error) {
	var t meshTensor
	var rank, n int32
	if err := binary.Read(r, binary.LittleEndian, &rank); err != nil {
		return t, err
	}
	if rank < 0 || rank > 5 {
		return t, fmt.Errorf("mesh rank %d", rank)
	}
	t.shape = make([]int32, rank)
	if err := binary.Read(r, binary.LittleEndian, &t.shape); err != nil {
		return t, err
	}
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return t, err
	}
	if n < 0 || int64(n)*4 > int64(r.Len()) {
		return t, fmt.Errorf("mesh tensor length %d", n)
	}
	t.data = make([]float32, n)
	if err := binary.Read(r, binary.LittleEndian, &t.data); err != nil {
		return t, err
	}
	return t, nil
}
func runCUDAMeshClock(payload []byte, iters int) (meshClockOut, error) {
	var out meshClockOut
	tool, err := findCUDAFwd()
	if err != nil {
		return out, err
	}
	var args []string
	if iters > 0 {
		args = append(args, fmt.Sprintf("--bench=%d", iters))
	}
	cmd := exec.Command(tool, args...)
	cmd.Stdin = bytes.NewReader(payload)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	raw, err := cmd.Output()
	if err != nil {
		return out, fmt.Errorf("CUDA mesh: %w: %s", err, stderr.String())
	}
	if len(raw) >= 16 && string(raw[:4]) == "WVB1" {
		raw = raw[16:]
	}
	magic := "MGBO"
	localMagic := map[string]string{"MGS1": "MGSO", "MGA1": "MGAO", "MGF1": "MGFO", "MGQ1": "MGQO"}
	if value, ok := localMagic[string(payload[:4])]; ok {
		magic = value
	}
	if string(payload[:4]) == "MGW1" {
		magic = "MGWO"
	} else if string(payload[:4]) == "PGB1" {
		magic = "PGVO"
	} else if string(payload[:4]) == "MGC1" {
		magic = "MGCO"
	}
	if len(raw) < 4 || string(raw[:4]) != magic {
		return out, fmt.Errorf("mesh magic")
	}
	r := bytes.NewReader(raw[4:])
	if err = binary.Read(r, binary.LittleEndian, &out.loss); err != nil {
		return out, err
	}
	if out.post, err = readMeshTensor(r); err != nil {
		return out, err
	}
	if out.gin, err = readMeshTensor(r); err != nil {
		return out, err
	}
	var count int32
	if err = binary.Read(r, binary.LittleEndian, &count); err != nil || count < 0 || count > 4096 {
		return out, fmt.Errorf("mesh count")
	}
	out.grads = make([]meshTensor, count)
	out.weights = make([][]float32, count)
	for i := range out.grads {
		if out.grads[i], err = readMeshTensor(r); err != nil {
			return out, err
		}
		var n int32
		if err = binary.Read(r, binary.LittleEndian, &n); err != nil || n < 0 || int64(n)*4 > int64(r.Len()) {
			return out, fmt.Errorf("mesh weights")
		}
		out.weights[i] = make([]float32, n)
		if err = binary.Read(r, binary.LittleEndian, &out.weights[i]); err != nil {
			return out, err
		}
	}
	var ticks int32
	if err = binary.Read(r, binary.LittleEndian, &ticks); err != nil || ticks < 0 || ticks > 4096 {
		return out, fmt.Errorf("mesh ticks")
	}
	out.history = make([][][]meshTensor, ticks)
	for t := range out.history {
		out.history[t] = make([][]meshTensor, count)
		for i := range out.grads {
			out.history[t][i] = make([]meshTensor, 3)
			for k := 0; k < 3; k++ {
				if out.history[t][i][k], err = readMeshTensor(r); err != nil {
					return out, err
				}
			}
		}
	}
	if r.Len() != 0 {
		return out, fmt.Errorf("mesh trailing bytes %d", r.Len())
	}
	return out, nil
}
func meshCompareTensor(label string, a, b meshTensor, family string, gradient bool) error {
	if len(a.data) != len(b.data) {
		return fmt.Errorf("%s length %d != %d", label, len(a.data), len(b.data))
	}
	if len(a.data) > 0 && !reflect.DeepEqual(a.shape, b.shape) {
		return fmt.Errorf("%s shape %v != %v", label, a.shape, b.shape)
	}
	if cudaExactOrSubnormal(a.data, b.data) {
		return nil
	}
	if label == "gin" && family == "kmeans" && kmeansInputParity(a.data, b.data) {
		return nil
	}
	floor := suites.DetExactBit
	if family == "gdn" {
		floor = suites.DetLoose
	}
	if gradient && family == "mha" {
		floor = suites.DetStrict
	}
	if gradient && family == "softmax" {
		floor = suites.DetLoose
	}
	m := suites.CompareF32Spectrum(a.data, b.data)
	if m.Band.Percent() < floor.Percent() {
		return fmt.Errorf("%s: %s", label, suites.FormatSpectrum(m))
	}
	return nil
}
func compareMeshClock(a, b meshClockOut, family string) error {
	if !cudaLossWithinBand(a.loss, b.loss) {
		return fmt.Errorf("mesh loss %g != %g", a.loss, b.loss)
	}
	if err := meshCompareTensor("post", a.post, b.post, family, false); err != nil {
		return err
	}
	if err := meshCompareTensor("gin", a.gin, b.gin, family, true); err != nil {
		return err
	}
	if len(a.weights) != len(b.weights) || len(a.history) != len(b.history) {
		return fmt.Errorf("mesh counts")
	}
	for i := range a.weights {
		if err := meshCompareTensor(fmt.Sprintf("dW[%d]", i), a.grads[i], b.grads[i], family, true); err != nil {
			return err
		}
		if len(a.weights[i]) != len(b.weights[i]) {
			return fmt.Errorf("W[%d] length", i)
		}
		if !cudaExactOrSubnormal(a.weights[i], b.weights[i]) {
			m := suites.CompareF32Spectrum(a.weights[i], b.weights[i])
			if family != "gdn" || m.Band.Percent() < suites.DetStrict.Percent() {
				return fmt.Errorf("W[%d]: %s", i, suites.FormatSpectrum(m))
			}
		}
	}
	for t := range a.history {
		for i := range a.weights {
			for k := 0; k < 3; k++ {
				if err := meshCompareTensor(fmt.Sprintf("tick %d cell %d tensor %d", t, i, k), a.history[t][i][k], b.history[t][i][k], family, false); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// Reuse independent native layer factories, not the Parallel container around
// them. Every enabled mesh cell binds its actual layer kind.
func meshFactory(pc polyCase, dt core.DType, depth, rows, cols int) (meshClockFixture, error) {
	f := meshClockFixture{z: depth, y: rows, x: cols, l: 1, family: pc.name}
	count := depth * rows * cols
	f.cells = make([]polyBranchEnc, count)
	for i := 0; i < count; i++ {
		branches, input, target, shape, tshape, err := pc.make(dt)
		if err != nil {
			return f, err
		}
		if i == 0 {
			f.input = input
			f.target = target
			f.shape = shape
			f.targetShape = tshape
		}
		f.cells[i] = branches[i%len(branches)]
	}
	// A dedicated disabled ingress keeps externally injected data separate from
	// shape-changing cell outputs in a double-buffered mesh.
	if count > 1 {
		f.cells[0] = polyBranchEnc{}
	}
	f.ticks = count + 2
	if !reflect.DeepEqual(f.shape, f.targetShape) {
		if count == 1 {
			f.ticks = 1
		} else {
			for i := 2; i < count; i++ {
				dim := int(f.targetShape[len(f.targetShape)-1])
				pair, _, _, _, _, err := makeRMSPair(dt, 2, dim)
				if err != nil {
					return f, err
				}
				f.cells[i] = pair[i%2]
			}
		}
	}
	return f, nil
}
func runMeshFamily(pc polyCase, dt core.DType, z, y, x int, credit parallel.TrainMode) error {
	f, err := meshFactory(pc, dt, z, y, x)
	f.credit = credit
	if err != nil {
		return err
	}
	payload := encodeMeshClock(f)
	ref, err := runGoMeshClock(f)
	if err != nil {
		return fmt.Errorf("Go: %w", err)
	}
	got, err := runCUDAMeshClock(payload, 0)
	if err != nil {
		return err
	}
	if err = compareMeshClock(ref, got, pc.name); err != nil {
		return err
	}
	bench, err := runCUDAMeshClock(payload, meshBenchIters())
	if err != nil {
		return err
	}
	return meshCompareReplay(got, bench)
}
func CasesMeshBPTrain() []Case { return meshTrainCases(parallel.ModeMeshBP) }
func CasesMeshTweenClockTrain() []Case { return meshTrainCases(parallel.ModeMeshTween) }
func CasesMeshTweenChainClockTrain() []Case {
	cases := meshTrainCases(parallel.ModeMeshTweenChain)
	return append(cases, Case{Name: "MeshTweenChain equals MeshBP all 23 families Float32", Run: func() error {
		for _, pc := range polyTrainKinds() {
			f, err := meshFactory(pc, core.DTypeFloat32, 2, 1, 1)
			if err != nil {
				return err
			}
			f.credit = parallel.ModeMeshBP
			bp, err := runCUDAMeshClock(encodeMeshClock(f), 0)
			if err != nil {
				return err
			}
			f.credit = parallel.ModeMeshTweenChain
			chain, err := runCUDAMeshClock(encodeMeshClock(f), 0)
			if err != nil {
				return err
			}
			if err = meshCompareReplay(bp, chain); err != nil {
				return fmt.Errorf("%s: %w", pc.name, err)
			}
		}
		return nil
	}})
}

func meshTrainCases(credit parallel.TrainMode) []Case {
	mode := credit.String()
	var cases []Case
	for _, pc := range polyTrainKinds() {
		pc := pc
		for _, top := range []struct {
			name    string
			z, y, x int
			all     bool
		}{{"111", 1, 1, 1, true}, {"211", 2, 1, 1, true}, {"222", 2, 2, 2, false}} {
			top := top
			label := pc.name + " " + mode + " clock " + top.name + " Float32"
			if top.all && pc.name != "gdn" {
				label = pc.name + " " + mode + " clock " + top.name + " ×34"
			}
			cases = append(cases, Case{Name: label, Run: func() error {
				types := []core.DType{core.DTypeFloat32}
				if top.all && pc.name != "gdn" {
					types = core.AllDTypes
				}
				for _, dt := range types {
					if err := runMeshFamily(pc, dt, top.z, top.y, top.x, credit); err != nil {
						return fmt.Errorf("%s: %w", dt, err)
					}
				}
				return nil
			}})
		}
	}
	return append(cases, meshEdgeCases(credit)...)
}

func meshLR(f meshClockFixture) float64 {
	if f.lr != nil {
		return *f.lr
	}
	return 0.001
}
func meshCheckFixture(f meshClockFixture) error {
	payload := encodeMeshClock(f)
	ref, err := runGoMeshClock(f)
	if err != nil {
		return fmt.Errorf("Go: %w", err)
	}
	got, err := runCUDAMeshClock(payload, 0)
	if err != nil {
		return err
	}
	if err = compareMeshClock(ref, got, f.family); err != nil {
		return err
	}
	bench, err := runCUDAMeshClock(payload, meshBenchIters())
	if err != nil {
		return err
	}
	return meshCompareReplay(got, bench)
}
func meshBenchIters() int {
	if defaultGoBenchIters < 2 {
		return 2
	}
	return defaultGoBenchIters
}

func meshCompareReplay(fresh, measured meshClockOut) error {
	if !reflect.DeepEqual(fresh, measured) {
		return fmt.Errorf("mesh benchmark replay changed outputs, tapes, gradients or weights")
	}
	return nil
}

// An ingress node keeps the injected input separate from shape-changing nodes.
// Parallel Concat and KMeans probabilities feed a real Dense adapter node.
func meshCompositeFixture(dt core.DType, build func(core.DType) (kpModeFixture, error), family string, credit parallel.TrainMode) error {
	k, err := build(dt)
	if err != nil {
		return err
	}
	f := meshClockFixture{credit: credit, z: 3, y: 1, x: 1, l: 1, ticks: 5,
		shape: k.xShape, input: k.x, targetShape: []int32{2, 4},
		target: seedPat(8, 0x7911), family: family,
		cells: make([]polyBranchEnc, 3)}
	for i := range k.ops {
		f.cells[i+1] = polyBranchEnc{op: k.ops[i], bytes: k.wires[i]}
	}
	return meshCheckFixture(f)
}

func meshRecurrence(pc polyCase, dt core.DType, credit parallel.TrainMode) error {
	f, err := meshFactory(pc, dt, 2, 1, 1)
	f.credit = credit
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(f.shape, f.targetShape) {
		return fmt.Errorf("recurrence needs shape-preserving fixture")
	}
	f.warmup = 3
	f.ticks = 4
	if pc.name == "swiglu" {
		f.warmup = 1
		f.ticks = 2
	}
	f.links = [][8]int32{{1, 0, 0, 0, 1, 0, 0, 0}}
	return meshCheckFixture(f)
}
func meshEdgeCases(credit parallel.TrainMode) []Case {
	var cases []Case
	addComposite := func(name, family string, build func(core.DType) (kpModeFixture, error)) {
		cases = append(cases, Case{Name: name + " clock multi-hop ×34", Run: func() error {
			for _, dt := range core.AllDTypes {
				if err := meshCompositeFixture(dt, build, family, credit); err != nil {
					return fmt.Errorf("%s: %w", dt, err)
				}
			}
			return nil
		}})
	}
	for _, output := range []kmeans.OutputMode{kmeans.OutputProbabilities, kmeans.OutputFeatures} {
		output := output
		addComposite("KMeans "+string(output), "kmeans", func(dt core.DType) (kpModeFixture, error) { return kpKMeansFixture(dt, output) })
	}
	for _, combine := range []parallel.CombineMode{parallel.CombineConcat, parallel.CombineAdd, parallel.CombineAvg, parallel.CombineFilter, parallel.CombineMax, parallel.CombineSparseK, parallel.CombineDisagree} {
		combine := combine
		addComposite(fmt.Sprintf("Parallel %v", combine), "parallel", func(dt core.DType) (kpModeFixture, error) { return kpParallelFixture(dt, combine) })
	}
	for _, pc := range polyTrainKinds() {
		pc := pc
		if pc.name == "embedding" {
			continue
		}
		label := pc.name + " recurrent remote clock ×34"
		if pc.name == "gdn" {
			label = "gdn recurrent remote clock Float32"
		}
		cases = append(cases, Case{Name: label, Run: func() error {
			types := core.AllDTypes
			if pc.name == "gdn" {
				types = []core.DType{core.DTypeFloat32}
			}
			for _, dt := range types {
				if err := meshRecurrence(pc, dt, credit); err != nil {
					return fmt.Errorf("%s: %w", dt, err)
				}
			}
			return nil
		}})
	}
	for _, family := range []string{"dense", "kmeans", "nested_parallel"} {
		family := family
		cases = append(cases, Case{Name: family + " zero LR preserves native weights ×34", Run: func() error {
			for _, pc := range polyTrainKinds() {
				if pc.name != family {
					continue
				}
				for _, dt := range core.AllDTypes {
					f, err := meshFactory(pc, dt, 2, 1, 1)
					f.credit = credit
					if err != nil {
						return err
					}
					zero := 0.0
					f.lr = &zero
					if err = meshCheckFixture(f); err != nil {
						return fmt.Errorf("%s: %w", dt, err)
					}
				}
			}
			return nil
		}})
	}
	return cases
}


// MeshTween local credit lands with MGW1; MeshBP uses step. Backward + SGD.
func meshGoLocalCredit(g *architecture.Grid, st *step.State[float32], gap *core.Tensor[float32], lr float64, mode parallel.TrainMode) error {
	if mode == parallel.ModeMeshTween {
		return meshGoTween(g, st, gap, lr)
	}
	return fmt.Errorf("mesh local credit %v not wired on CUDA MeshBP suite", mode)
}

// Mirror the mesh last-tick contract with Go's concrete backward/SGD dispatch.
// Sequential and Residual use Go's public local-Tween leaf collector.
func meshGoTween(g *architecture.Grid, st *step.State[float32], gap *core.Tensor[float32], lr float64) error {
	if lr == 0 {
		return nil
	}
	last := len(st.HistoryIn) - 1
	for i := len(g.Cells) - 1; i >= 0; i-- {
		c := &g.Cells[i]
		input, pre, post := st.HistoryIn[last][i], st.HistoryPre[last][i], st.LayerData[i]
		if c.Layer.IsDisabled || input == nil || pre == nil || post == nil {
			continue
		}
		switch c.Op.(type) {
		case *sequential.Layer, *residual.Layer:
			wrapper, err := parallel.NewStack(c.Op)
			if err != nil {
				return err
			}
			wrapper.Exec.Backend = core.BackendCPUTiled
			wrapper.Exec.MultiCore = false
			if err = parallel.TrainStack(wrapper, gap, input, pre, parallel.ModeTween, lr); err != nil {
				return err
			}
		default:
			projected := projectGapF32(gap, post.Shape...)
			_, dw, err := backward.Cell(c, projected, input, pre)
			if err != nil {
				return err
			}
			if dw != nil && dw.Len() > 0 {
				wrapper, err := parallel.NewStack(c.Op)
				if err != nil {
					return err
				}
				if err = parallel.ApplyGradSGDStack(wrapper, dw, lr*0.5); err != nil {
					return err
				}
			}
		}
		next := core.NewTensor[float32](input.Shape...)
		if next.Len() == gap.Len() {
			copy(next.Data, gap.Data)
		}
		gap = next
	}
	return nil
}
