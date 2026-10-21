package go_v_cpp

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os/exec"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/weights"
)

// Parallel Dense Add volume 2×2×2×1 — Phase D OpKind::Parallel / PNV1 + Mesh retargets.

type pnVolCell struct {
	enabled  bool
	dim      int
	outFeat  int
	branches int
	dtype    core.DType
	combine  parallel.CombineMode
	W        []float32
}

func packParallelWeightsF32(l *parallel.Layer) ([]float32, error) {
	var out []float32
	for _, ch := range l.Branches {
		dl, ok := ch.(*dense.Layer)
		if !ok {
			return nil, fmt.Errorf("parallel vol: branch %T want *dense.Layer", ch)
		}
		wa, err := weights.MatrixF32(dl.Weights)
		if err != nil {
			return nil, err
		}
		out = append(out, wa...)
	}
	return out, nil
}

func placePnVolGrid(depth, rows, cols, lpc, batch int, cells []pnVolCell, x []float32) (*architecture.Grid, *core.Tensor[float32], error) {
	g := architecture.NewGrid(depth, rows, cols, lpc)
	hops := g.HopOrder()
	if len(cells) != len(hops) {
		return nil, nil, fmt.Errorf("cells %d != hops %d", len(cells), len(hops))
	}
	firstDim := -1
	for i, c := range hops {
		vc := cells[i]
		if !vc.enabled {
			cell := g.At(c.Z, c.Y, c.X, c.L)
			if cell != nil {
				cell.Layer.IsDisabled = true
			}
			continue
		}
		if firstDim < 0 {
			firstDim = vc.dim
		}
		cfg := parallel.Config{Dim: vc.dim, OutFeat: vc.outFeat, Branches: vc.branches, Combine: vc.combine}
		pl, err := parallel.NewConfigured[float32](cfg, vc.dtype, quant.FormatNone, vc.W, nil)
		if err != nil {
			return nil, nil, err
		}
		pl.Exec.Backend = core.BackendCPUTiled
		pl.Exec.MultiCore = false
		if err := parallel.Place(g, c.Z, c.Y, c.X, c.L, pl); err != nil {
			return nil, nil, err
		}
	}
	xin := core.NewTensor[float32](batch, firstDim)
	copy(xin.Data, x)
	return g, xin, nil
}

func exportPnVolWs(g *architecture.Grid, cells []pnVolCell) ([][]float32, error) {
	hops := g.HopOrder()
	var Ws [][]float32
	for i, c := range hops {
		if !cells[i].enabled {
			continue
		}
		cell := g.At(c.Z, c.Y, c.X, c.L)
		pl, ok := cell.Op.(*parallel.Layer)
		if !ok {
			return nil, fmt.Errorf("expected parallel at %v", c)
		}
		wa, err := packParallelWeightsF32(pl)
		if err != nil {
			return nil, err
		}
		Ws = append(Ws, wa)
	}
	return Ws, nil
}

func runGoPnVolApply(depth, rows, cols, lpc, batch int, cells []pnVolCell, x, target []float32, lr float64,
	apply volApplyFwd, clearGin bool) (nbpVolOut, error) {
	var z nbpVolOut
	g, xin, err := placePnVolGrid(depth, rows, cols, lpc, batch, cells, x)
	if err != nil {
		return z, err
	}
	fwd, err := forward.Forward(g, xin)
	if err != nil {
		return z, err
	}
	tgt := core.NewTensor[float32](fwd.Output.Shape...)
	copy(tgt.Data, target)
	loss, err := apply(g, fwd, tgt, lr)
	if err != nil {
		return z, err
	}
	z.loss = loss
	z.post = append([]float32(nil), fwd.Output.Data...)
	if clearGin {
		z.gradIn = []float32{}
	}
	z.Ws, err = exportPnVolWs(g, cells)
	return z, err
}

func runGoPnVolNBP(depth, rows, cols, lpc, batch int, cells []pnVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	var z nbpVolOut
	g, xin, err := placePnVolGrid(depth, rows, cols, lpc, batch, cells, x)
	if err != nil {
		return z, err
	}
	fwd, err := forward.Forward(g, xin)
	if err != nil {
		return z, err
	}
	tgt := core.NewTensor[float32](fwd.Output.Shape...)
	copy(tgt.Data, target)
	loss, err := training.MSE(fwd.Output, tgt)
	if err != nil {
		return z, err
	}
	gy, err := training.MSEGrad(fwd.Output, tgt)
	if err != nil {
		return z, err
	}
	bwd, err := backward.Backward(fwd, gy)
	if err != nil {
		return z, err
	}
	if err := training.SGD(fwd, bwd, lr); err != nil {
		return z, err
	}
	z.loss = loss
	z.post = append([]float32(nil), fwd.Output.Data...)
	z.gradIn = append([]float32(nil), bwd.GradIn.Data...)
	z.Ws, err = exportPnVolWs(g, cells)
	return z, err
}

func runGoPnVolAlt(depth, rows, cols, lpc, batch int, cells []pnVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	var z nbpVolOut
	g, xin, err := placePnVolGrid(depth, rows, cols, lpc, batch, cells, x)
	if err != nil {
		return z, err
	}
	tgt := core.NewTensor[float32](batch, cells[0].outFeat)
	for i := range cells {
		if cells[i].enabled {
			tgt = core.NewTensor[float32](batch, cells[i].outFeat)
			break
		}
	}
	copy(tgt.Data, target)
	// Match C++ wire: post from initial Forward before Alt credit (same as KMeans Alt vol).
	fwd, err := forward.Forward(g, xin)
	if err != nil {
		return z, err
	}
	z.post = append([]float32(nil), fwd.Output.Data...)
	loss, err := applyModeTweenAltVolume(g, xin, tgt, lr)
	if err != nil {
		return z, err
	}
	z.loss = loss
	z.gradIn = []float32{}
	z.Ws, err = exportPnVolWs(g, cells)
	return z, err
}

func encodePnVolNBP(depth, rows, cols, lpc, batch int, cells []pnVolCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("PNV1"))
	for _, v := range []int32{int32(depth), int32(rows), int32(cols), int32(lpc), int32(batch), 0, 0} {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	if len(cells) != depth*rows*cols*lpc {
		return nil, fmt.Errorf("cells len %d != stack", len(cells))
	}
	for _, c := range cells {
		en := int32(0)
		if c.enabled {
			en = 1
		}
		_ = binary.Write(&in, binary.LittleEndian, en)
		if !c.enabled {
			continue
		}
		_ = binary.Write(&in, binary.LittleEndian, int32(c.dim))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.outFeat))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.branches))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.dtype))
		_ = binary.Write(&in, binary.LittleEndian, parallelCombineCode(c.combine))
		for _, v := range c.W {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
	}
	for _, v := range x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range target {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	_ = binary.Write(&in, binary.LittleEndian, lr)
	return in.Bytes(), nil
}

func decodePnVolNBP(raw []byte) (nbpVolOut, error) {
	o, _, err := decodeRecVolNBP(raw, "PNVO")
	return o, err
}

func runCppPnVolRetarget(payload []byte, inFrom, inTo, outFrom, outTo string) ([]byte, error) {
	payload = swapMagic4(payload, inFrom, inTo)
	tool, err := findDenseFwd()
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(tool)
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("cpp parallel vol: %v: %s", err, stderr.String())
	}
	return rewriteOutMagic(stdout.Bytes(), outFrom, outTo), nil
}

func makePnVolCells(stack, dim, outFeat, branches int, combine parallel.CombineMode, seed uint32) []pnVolCell {
	nw := branches * outFeat * dim
	W := randInit(1, nw, seed)
	for i := range W {
		W[i] *= 0.05
	}
	cells := make([]pnVolCell, stack)
	for i := range cells {
		cells[i] = pnVolCell{
			enabled: true, dim: dim, outFeat: outFeat, branches: branches,
			dtype: core.DTypeFloat32, combine: combine, W: append([]float32(nil), W...),
		}
	}
	return cells
}

func pnVolFixture() (cells []pnVolCell, x, target []float32, depth, rows, cols, lpc, batch int, lr float64) {
	depth, rows, cols, lpc, batch = 2, 2, 2, 1, 1
	const dim, outFeat, branches = 4, 4, 2
	lr = 1e-2
	cells = makePnVolCells(depth*rows*cols*lpc, dim, outFeat, branches, parallel.CombineAdd, 0x504e5631)
	x = randInit(batch, dim, 0x504e5632)
	for i := range x {
		x[i] *= 0.05
	}
	target = make([]float32, batch*outFeat)
	for i := range target {
		target[i] = float32(0.05 * float64(i%outFeat+1))
	}
	return
}

func compareNBPVol(goO, cppO nbpVolOut) error {
	if err := compareTweenVol(goO, cppO); err != nil {
		return err
	}
	if len(goO.gradIn) != len(cppO.gradIn) {
		return fmt.Errorf("gin len go=%d cpp=%d", len(goO.gradIn), len(cppO.gradIn))
	}
	for i := range goO.gradIn {
		if goO.gradIn[i] != cppO.gradIn[i] {
			return fmt.Errorf("gin[%d] go=%g cpp=%g", i, goO.gradIn[i], cppO.gradIn[i])
		}
	}
	return nil
}

func casePnNBPVol222F32() error {
	cells, x, target, depth, rows, cols, lpc, batch, lr := pnVolFixture()
	goO, err := runGoPnVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodePnVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return err
	}
	tool, err := findDenseFwd()
	if err != nil {
		return err
	}
	cmd := exec.Command(tool)
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("cpp: %v: %s", err, stderr.String())
	}
	cppO, err := decodePnVolNBP(stdout.Bytes())
	if err != nil {
		return err
	}
	if err := compareNBPVol(goO, cppO); err != nil {
		return err
	}
	fmt.Printf("\n      Parallel vol 2x2x2x1 NormalBP/MeshBP loss=%.6g exact_bit", goO.loss)
	return nil
}

func runPnVolFamily(apply volApplyFwd, cppIn, cppOut string, label string) error {
	cells, x, target, depth, rows, cols, lpc, batch, lr := pnVolFixture()
	goO, err := runGoPnVolApply(depth, rows, cols, lpc, batch, cells, x, target, lr, apply, true)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodePnVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return err
	}
	raw, err := runCppPnVolRetarget(payload, "PNV1", cppIn, "PNVO", cppOut)
	if err != nil {
		return err
	}
	cppO, _, err := decodeRecVolNBP(raw, cppOut)
	if err != nil {
		return err
	}
	if err := compareTweenVol(goO, cppO); err != nil {
		return err
	}
	fmt.Printf("\n      Parallel vol 2x2x2x1 %s loss=%.6g exact_bit", label, goO.loss)
	return nil
}

func casePnTweenVol222F32() error {
	return runPnVolFamily(applyModeTweenVolume, "PNTV", "PNTO", "Tween/MeshTween")
}

func casePnTweenChainVol222F32() error {
	cells, x, target, depth, rows, cols, lpc, batch, lr := pnVolFixture()
	goO, err := runGoPnVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodePnVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return err
	}
	raw, err := runCppPnVolRetarget(payload, "PNV1", "PNTC", "PNVO", "PNCO")
	if err != nil {
		return err
	}
	cppO, _, err := decodeRecVolNBP(raw, "PNCO")
	if err != nil {
		return err
	}
	if err := compareNBPVol(goO, cppO); err != nil {
		return err
	}
	fmt.Printf("\n      Parallel vol 2x2x2x1 TweenChain/MeshTweenChain loss=%.6g exact_bit", goO.loss)
	return nil
}

func casePnTweenSplitVol222F32() error {
	return runPnVolFamily(applyModeTweenSplitVolume, "PNTP", "PNPO", "TweenSplit/MeshTweenSplit")
}

func casePnTweenAltVol222F32() error {
	cells, x, target, depth, rows, cols, lpc, batch, lr := pnVolFixture()
	goO, err := runGoPnVolAlt(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodePnVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return err
	}
	raw, err := runCppPnVolRetarget(payload, "PNV1", "PNTA", "PNVO", "PNAO")
	if err != nil {
		return err
	}
	cppO, _, err := decodeRecVolNBP(raw, "PNAO")
	if err != nil {
		return err
	}
	if err := compareTweenVol(goO, cppO); err != nil {
		return err
	}
	fmt.Printf("\n      Parallel vol 2x2x2x1 TweenAlt/MeshTweenAlt loss=%.6g exact_bit", goO.loss)
	return nil
}

func casePnFastProxyVol222F32() error {
	return runPnVolFamily(applyModeTweenSplitFastProxyVolume, "PNFP", "PNFO", "FastProxy/MeshTweenSplitFastProxy")
}

func casePnSparseVol222F32() error {
	return runPnVolFamily(applyModeTweenSplitSparseVolume, "PNSP", "PNSO", "Sparse/MeshTweenSplitSparse")
}

func casePnFreezeVol222F32() error {
	return runPnVolFamily(applyModeFreezeVolume, "PNFR", "PNFK", "Freeze/Shadow")
}
