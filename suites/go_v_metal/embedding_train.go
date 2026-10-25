package go_v_metal

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os/exec"
	"strings"
	"time"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/weights"
)

// CasesEmbeddingTrain — NormalBP Embedding Go↔Metal (leaf + volumetric).
func CasesEmbeddingTrain() []Case {
	return []Case{
		{Name: "Embedding NormalBP leaf Float32 — exact bits (dX/dW/W/loss)", Run: caseEmbeddingNBPLeafF32},
		{Name: "Embedding NormalBP leaf × all 34 dtypes — spectrum + speed", Run: caseEmbeddingNBPLeafAllDTypes},
		{Name: "Embedding NormalBP volume 1×1×1×1 Float32 — exact bits", Run: caseEmbeddingNBPVol1x1x1},
		{Name: "Embedding NormalBP volume 1×1×1×2 chain Float32 — exact bits", Run: caseEmbeddingNBPVolChain},
		{Name: "Embedding NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum", Run: caseEmbeddingNBPVol111AllDTypes},
		{Name: "Embedding NormalBP volume 2×2×2×1 Float32 — exact bits", Run: caseEmbeddingNBPVol222F32},
		{Name: "Embedding NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum", Run: caseEmbeddingNBPVol211AllDTypes},
	}
}

func RunEmbeddingTrain() error {
	var fails []string
	cs := CasesEmbeddingTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [embp %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_metal_embedding_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_metal_embedding_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_metal embedding_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type embpLeaf struct {
	batch, vocab, emb, seq int
	inRank                 int
	dtype                  core.DType
	act                    core.ActivationType
	W                      []float32
	x                      []float32
	target                 []float32
	lr                     float64
}

type embpLeafOut struct {
	pre, post, dX, dW, WAfter []float32
	loss                      float64
}

func makeEmbeddingTrainInit(vocab, emb, batch, seq int, seed uint32) (W, x []float32) {
	W = randInit(vocab, emb, seed)
	x = tokenIDs(batch*seq, vocab, seed^0x9e3779b9)
	return W, x
}

func runGoEmbeddingNBP(fc embpLeaf) (embpLeafOut, error) {
	var z embpLeafOut
	cfg := embedding.Config{VocabSize: fc.vocab, EmbeddingDim: fc.emb, SeqLen: fc.seq}
	l, err := embedding.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.W)
	if err != nil {
		return z, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	l.Core.Activation = fc.act

	var xin *core.Tensor[float32]
	switch fc.inRank {
	case 1:
		xin = core.NewTensor[float32](fc.seq)
	case 3:
		xin = core.NewTensor[float32](fc.batch, fc.seq, 1)
	default:
		xin = core.NewTensor[float32](fc.batch, fc.seq)
	}
	copy(xin.Data, fc.x)
	preT, postT, err := embedding.Forward(l, xin)
	if err != nil {
		return z, err
	}
	tgt := core.NewTensor[float32](postT.Shape...)
	copy(tgt.Data, fc.target)
	loss, err := training.MSE(postT, tgt)
	if err != nil {
		return z, err
	}
	gy, err := training.MSEGrad(postT, tgt)
	if err != nil {
		return z, err
	}
	dX, dW, err := embedding.Backward(l, gy, xin, preT)
	if err != nil {
		return z, err
	}
	if fc.lr != 0 {
		if err := embedding.ApplyGradSGD(l, dW, fc.lr); err != nil {
			return z, err
		}
	}
	wa, err := weights.MatrixF32(l.Weights)
	if err != nil {
		return z, err
	}
	z.pre = append([]float32(nil), preT.Data...)
	z.post = append([]float32(nil), postT.Data...)
	z.dX = append([]float32(nil), dX.Data...)
	z.dW = append([]float32(nil), dW.Data...)
	z.WAfter = append([]float32(nil), wa...)
	z.loss = loss
	return z, nil
}

func encodeEmbeddingNBP(fc embpLeaf) []byte {
	inRank := int32(fc.inRank)
	if inRank < 1 || inRank > 3 {
		inRank = 2
	}
	inLast := int32(1)
	var in bytes.Buffer
	_, _ = in.Write([]byte("WEB1"))
	for _, v := range []int32{
		int32(fc.batch), int32(fc.vocab), int32(fc.emb), int32(fc.seq),
		inRank, inLast, int32(fc.act), int32(fc.dtype), 0, // cpu
	} {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.W {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	_ = binary.Write(&in, binary.LittleEndian, int32(1)) // use_target
	for _, v := range fc.target {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	_ = binary.Write(&in, binary.LittleEndian, fc.lr)
	return in.Bytes()
}

func decodeEmbeddingNBP(raw []byte) (embpLeafOut, time.Duration, error) {
	var z embpLeafOut
	var bench time.Duration
	if len(raw) >= 4+8+4 && string(raw[:4]) == "WVB1" {
		r := bytes.NewReader(raw[4:])
		var ns int64
		var iters int32
		if err := binary.Read(r, binary.LittleEndian, &ns); err != nil {
			return z, 0, err
		}
		if err := binary.Read(r, binary.LittleEndian, &iters); err != nil {
			return z, 0, err
		}
		rest, err := io.ReadAll(r)
		if err != nil {
			return z, 0, err
		}
		raw = rest
		bench = time.Duration(ns)
	}
	if len(raw) < 8 || string(raw[:4]) != "WEBO" {
		return z, bench, fmt.Errorf("metal embedding nbp bad magic")
	}
	r := bytes.NewReader(raw[4:])
	readVec := func() ([]float32, error) {
		var n int32
		if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
			return nil, err
		}
		v := make([]float32, n)
		for i := range v {
			if err := binary.Read(r, binary.LittleEndian, &v[i]); err != nil {
				return nil, err
			}
		}
		return v, nil
	}
	var err error
	if z.pre, err = readVec(); err != nil {
		return z, bench, err
	}
	if z.post, err = readVec(); err != nil {
		return z, bench, err
	}
	if z.dX, err = readVec(); err != nil {
		return z, bench, err
	}
	if z.dW, err = readVec(); err != nil {
		return z, bench, err
	}
	if z.WAfter, err = readVec(); err != nil {
		return z, bench, err
	}
	if err := binary.Read(r, binary.LittleEndian, &z.loss); err != nil {
		return z, bench, err
	}
	return z, bench, nil
}

func runMetalEmbeddingNBP(fc embpLeaf) (embpLeafOut, error) {
	out, _, err := runMetalEmbeddingNBPBench(fc, 0)
	return out, err
}

func runMetalEmbeddingNBPBench(fc embpLeaf, benchIters int) (embpLeafOut, time.Duration, error) {
	tool, err := findMetalFwd()
	if err != nil {
		return embpLeafOut{}, 0, err
	}
	var cmd *exec.Cmd
	if benchIters > 0 {
		cmd = exec.Command(tool, fmt.Sprintf("--bench=%d", benchIters))
	} else {
		cmd = exec.Command(tool)
	}
	cmd.Stdin = bytes.NewReader(encodeEmbeddingNBP(fc))
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return embpLeafOut{}, 0, fmt.Errorf("metal embedding nbp: %v: %s", err, stderr.String())
	}
	return decodeEmbeddingNBP(stdout.Bytes())
}

func compareEmbeddingNBP(goO, mtlO embpLeafOut, requireExact bool) error {
	checks := []struct {
		name string
		a, b []float32
	}{
		{"pre", goO.pre, mtlO.pre},
		{"post", goO.post, mtlO.post},
		{"dX", goO.dX, mtlO.dX},
		{"dW", goO.dW, mtlO.dW},
		{"W_after", goO.WAfter, mtlO.WAfter},
	}
	for _, c := range checks {
		m := suites.CompareF32Spectrum(c.a, c.b)
		if requireExact && m.Band != suites.DetExactBit {
			return fmt.Errorf("%s: %s", c.name, suites.FormatSpectrum(m))
		}
		if !requireExact && (m.Band == suites.DetBroken || m.Band == suites.DetBallpark) {
			return fmt.Errorf("%s: %s", c.name, suites.FormatSpectrum(m))
		}
	}
	if !metalLossWithinBand(goO.loss, mtlO.loss) {
		rel := math.Abs(goO.loss-mtlO.loss) / math.Max(1e-30, math.Abs(goO.loss))
		if requireExact || rel > 1e-5 {
			return fmt.Errorf("loss go=%g mtl=%g", goO.loss, mtlO.loss)
		}
	}
	return nil
}

func caseEmbeddingNBPLeafF32() error {
	var benchIters = defaultGoBenchIters
	vocab, emb, batch, seq := 16, 8, 2, 4
	W, x := makeEmbeddingTrainInit(vocab, emb, batch, seq, 0x454d4250)
	target := make([]float32, batch*seq*emb)
	for i := range target {
		target[i] = float32(0.05 * float64(i%7+1))
	}
	fc := embpLeaf{
		batch: batch, vocab: vocab, emb: emb, seq: seq, inRank: 2,
		dtype: core.DTypeFloat32, act: core.ActivationLinear,
		W: W, x: x, target: target, lr: 1e-2,
	}
	goO, err := runGoEmbeddingNBP(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	mtlO, err := runMetalEmbeddingNBP(fc)
	if err != nil {
		return fmt.Errorf("metal: %w", err)
	}
	if err := compareEmbeddingNBP(goO, mtlO, true); err != nil {
		return err
	}
	goUS, err := benchGoUS(func() error {
		_, err := runGoEmbeddingNBP(fc)
		return err
	}, benchIters)
	if err != nil {
		return fmt.Errorf("go bench: %w", err)
	}
	_, mtlDur, err := runMetalEmbeddingNBPBench(fc, benchIters)
	if err != nil {
		return fmt.Errorf("metal bench: %w", err)
	}
	mtlUS := perIterUS(mtlDur, benchIters)
	spdPct := suites.SpeedPctDiff(goUS, mtlUS)
	fmt.Printf("\n      loss=%.6g exact_bit  go=%s  mtl=%s  spd%%Δ=%+.1f%%  (neg ⇒ Metal faster)",
		goO.loss, FormatUS(goUS), FormatUS(mtlUS), spdPct)
	return nil
}

func caseEmbeddingNBPLeafAllDTypes() error {
	var benchIters = defaultGoBenchIters
	vocab, emb, batch, seq := 12, 6, 2, 3
	W, x := makeEmbeddingTrainInit(vocab, emb, batch, seq, 0x454d4434)
	target := make([]float32, batch*seq*emb)
	for i := range target {
		target[i] = float32(0.04 * float64(i%5+1))
	}
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      Go CPU vs Metal — Embedding NBP ×34\n")
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "go_µs", "mtl_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 80))
	for _, dt := range core.AllDTypes {
		fc := embpLeaf{
			batch: batch, vocab: vocab, emb: emb, seq: seq, inRank: 2,
			dtype: dt, act: core.ActivationLinear,
			W: W, x: x, target: target, lr: 1e-2,
		}
		goO, err := runGoEmbeddingNBP(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		mtlO, err := runMetalEmbeddingNBP(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s metal: %v", dt, err))
			continue
		}
		mW := suites.CompareF32Spectrum(goO.WAfter, mtlO.WAfter)
		mDX := suites.CompareF32Spectrum(goO.dX, mtlO.dX)
		mDW := suites.CompareF32Spectrum(goO.dW, mtlO.dW)
		band := mW.Band
		for _, b := range []suites.DetBand{mDX.Band, mDW.Band} {
			if b.Percent() < band.Percent() {
				band = b
			}
		}
		if band.Percent() < worst.Percent() {
			worst = band
		}
		goUS, err := benchGoUS(func() error {
			_, err := runGoEmbeddingNBP(fc)
			return err
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		_, mtlDur, err := runMetalEmbeddingNBPBench(fc, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s metal bench: %v", dt, err))
			fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %9s %s\n",
				dt, band, mW.ExactPct, FormatUS(goUS), "?", "?", fmt.Sprintf("%.3g", mW.MaxAbs))
			continue
		}
		mtlUS := perIterUS(mtlDur, benchIters)
		spdPct := suites.SpeedPctDiff(goUS, mtlUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %+8.1f%% %s\n",
			dt, band, mW.ExactPct, FormatUS(goUS), FormatUS(mtlUS), spdPct, fmt.Sprintf("%.3g", mW.MaxAbs))
		if band != suites.DetExactBit {
			fails = append(fails, fmt.Sprintf("%s: %s", dt, band))
		}
	}
	fmt.Printf("      worst_band=%s — want exact_bit on all 34\n", worst)
	if len(fails) > 0 {
		return fmt.Errorf("%d dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

type embVolCell struct {
	enabled bool
	vocab   int
	emb     int
	seq     int
	dtype   core.DType
	act     core.ActivationType
	W       []float32
}

func runGoEmbeddingVolNBP(depth, rows, cols, lpc, batch int, cells []embVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
	hops := g.HopOrder()
	if len(cells) != len(hops) {
		return z, fmt.Errorf("cells %d != hops %d", len(cells), len(hops))
	}
	for i, c := range hops {
		vc := cells[i]
		if !vc.enabled {
			cell := g.At(c.Z, c.Y, c.X, c.L)
			if cell != nil {
				cell.Layer.IsDisabled = true
			}
			continue
		}
		cfg := embedding.Config{VocabSize: vc.vocab, EmbeddingDim: vc.emb, SeqLen: vc.seq}
		el, err := embedding.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		el.Exec.Backend = core.BackendCPUTiled
		el.Exec.MultiCore = false
		el.Core.MultiCore = false
		el.Core.Activation = vc.act
		if err := g.BindOp(c.Z, c.Y, c.X, c.L, core.Layer{
			Type: core.LayerEmbedding, DType: vc.dtype, Activation: vc.act,
			InputHeight: vc.vocab, OutputHeight: vc.emb,
		}, el); err != nil {
			return z, err
		}
	}
	xin := core.NewTensor[float32](batch, len(x)/batch)
	copy(xin.Data, x)
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
	for i, c := range hops {
		if !cells[i].enabled {
			continue
		}
		cell := g.At(c.Z, c.Y, c.X, c.L)
		el, ok := cell.Op.(*embedding.Layer)
		if !ok {
			return z, fmt.Errorf("expected embedding at %v", c)
		}
		wa, err := weights.MatrixF32(el.Weights)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, append([]float32(nil), wa...))
	}
	return z, nil
}

func encodeEmbeddingVolNBP(depth, rows, cols, lpc, batch int, cells []embVolCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("WEV1"))
	for _, v := range []int32{int32(depth), int32(rows), int32(cols), int32(lpc), int32(batch), 0 /*cpu*/, 0 /*n_links*/} {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	if len(cells) != depth*rows*cols*lpc {
		return nil, fmt.Errorf("cells len %d != stack %d", len(cells), depth*rows*cols*lpc)
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
		_ = binary.Write(&in, binary.LittleEndian, int32(c.vocab))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.emb))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.seq))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.act))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.dtype))
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

func decodeEmbeddingVolNBP(raw []byte) (nbpVolOut, time.Duration, error) {
	var z nbpVolOut
	var bench time.Duration
	if len(raw) >= 4+8+4 && string(raw[:4]) == "WVB1" {
		r := bytes.NewReader(raw[4:])
		var ns int64
		var iters int32
		if err := binary.Read(r, binary.LittleEndian, &ns); err != nil {
			return z, 0, err
		}
		if err := binary.Read(r, binary.LittleEndian, &iters); err != nil {
			return z, 0, err
		}
		rest, err := io.ReadAll(r)
		if err != nil {
			return z, 0, err
		}
		raw = rest
		bench = time.Duration(ns)
	}
	if len(raw) < 8 || string(raw[:4]) != "WEVO" {
		return z, bench, fmt.Errorf("metal embedding vol nbp bad magic")
	}
	r := bytes.NewReader(raw[4:])
	if err := binary.Read(r, binary.LittleEndian, &z.loss); err != nil {
		return z, bench, err
	}
	var ginN int32
	if err := binary.Read(r, binary.LittleEndian, &ginN); err != nil {
		return z, bench, err
	}
	z.gradIn = make([]float32, ginN)
	for i := range z.gradIn {
		if err := binary.Read(r, binary.LittleEndian, &z.gradIn[i]); err != nil {
			return z, bench, err
		}
	}
	var nCells int32
	if err := binary.Read(r, binary.LittleEndian, &nCells); err != nil {
		return z, bench, err
	}
	for i := int32(0); i < nCells; i++ {
		var nw int32
		if err := binary.Read(r, binary.LittleEndian, &nw); err != nil {
			return z, bench, err
		}
		w := make([]float32, nw)
		for j := range w {
			if err := binary.Read(r, binary.LittleEndian, &w[j]); err != nil {
				return z, bench, err
			}
		}
		z.Ws = append(z.Ws, w)
	}
	var outN int32
	if err := binary.Read(r, binary.LittleEndian, &outN); err != nil {
		return z, bench, err
	}
	z.post = make([]float32, outN)
	for i := range z.post {
		if err := binary.Read(r, binary.LittleEndian, &z.post[i]); err != nil {
			return z, bench, err
		}
	}
	return z, bench, nil
}

func runMetalEmbeddingVolNBP(depth, rows, cols, lpc, batch int, cells []embVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	out, _, err := runMetalEmbeddingVolNBPBench(depth, rows, cols, lpc, batch, cells, x, target, lr, 0)
	return out, err
}

func runMetalEmbeddingVolNBPBench(depth, rows, cols, lpc, batch int, cells []embVolCell, x, target []float32, lr float64, benchIters int) (nbpVolOut, time.Duration, error) {
	tool, err := findMetalFwd()
	if err != nil {
		return nbpVolOut{}, 0, err
	}
	payload, err := encodeEmbeddingVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return nbpVolOut{}, 0, err
	}
	var cmd *exec.Cmd
	if benchIters > 0 {
		cmd = exec.Command(tool, fmt.Sprintf("--bench=%d", benchIters))
	} else {
		cmd = exec.Command(tool)
	}
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nbpVolOut{}, 0, fmt.Errorf("metal embedding vol nbp: %v: %s", err, stderr.String())
	}
	return decodeEmbeddingVolNBP(stdout.Bytes())
}

func benchEmbeddingVolNBP(depth, rows, cols, lpc, batch int, cells []embVolCell, x, target []float32, lr float64, benchIters int) (goUS, mtlUS, spdPct float64, err error) {
	goUS, err = benchGoUS(func() error {
		_, err := runGoEmbeddingVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		return err
	}, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	_, mtlDur, err := runMetalEmbeddingVolNBPBench(depth, rows, cols, lpc, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	mtlUS = perIterUS(mtlDur, benchIters)
	spdPct = suites.SpeedPctDiff(goUS, mtlUS)
	return goUS, mtlUS, spdPct, nil
}

func caseEmbeddingNBPVol1x1x1() error {
	const batch, vocab, emb, seq = 2, 8, 4, 3
	var benchIters = defaultGoBenchIters
	W, x := makeEmbeddingTrainInit(vocab, emb, batch, seq, 0x454d5631)
	target := make([]float32, batch*seq*emb)
	for i := range target {
		target[i] = float32(0.03 * float64(i%4+1))
	}
	cells := []embVolCell{{
		enabled: true, vocab: vocab, emb: emb, seq: seq,
		dtype: core.DTypeFloat32, act: core.ActivationLinear, W: W,
	}}
	const lr = 1e-2
	goO, err := runGoEmbeddingVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	mtlO, err := runMetalEmbeddingVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("metal: %w", err)
	}
	if err := compareVolNBP(goO, mtlO, true); err != nil {
		return err
	}
	goUS, mtlUS, spdPct, err := benchEmbeddingVolNBP(1, 1, 1, 1, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×1 loss=%.6g exact_bit  go=%s  mtl=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(mtlUS), spdPct)
	return nil
}

func caseEmbeddingNBPVolChain() error {
	const batch, vocab, emb, seq = 1, 8, 4, 3
	var benchIters = defaultGoBenchIters
	W0, x := makeEmbeddingTrainInit(vocab, emb, batch, seq, 0x454d5633)
	W1, _ := makeEmbeddingTrainInit(vocab, emb, batch, seq, 0x454d5634)
	target := make([]float32, batch*seq*emb)
	for i := range target {
		target[i] = float32(0.02 * float64(i%5+1))
	}
	cells := []embVolCell{
		{enabled: true, vocab: vocab, emb: emb, seq: seq, dtype: core.DTypeFloat32, act: core.ActivationLinear, W: W0},
		{enabled: true, vocab: vocab, emb: emb, seq: seq, dtype: core.DTypeFloat32, act: core.ActivationLinear, W: W1},
	}
	const lr = 1e-2
	goO, err := runGoEmbeddingVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	mtlO, err := runMetalEmbeddingVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("metal: %w", err)
	}
	if err := compareVolNBP(goO, mtlO, true); err != nil {
		return err
	}
	goUS, mtlUS, spdPct, err := benchEmbeddingVolNBP(1, 1, 1, 2, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×2 loss=%.6g exact_bit  go=%s  mtl=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(mtlUS), spdPct)
	return nil
}

func caseEmbeddingNBPVol111AllDTypes() error {
	const batch, vocab, emb, seq = 2, 8, 4, 3
	W, x := makeEmbeddingTrainInit(vocab, emb, batch, seq, 0x454d5635)
	target := make([]float32, batch*seq*emb)
	for i := range target {
		target[i] = float32(0.03 * float64(i%4+1))
	}
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      Embedding NBP vol 1×1×1×1 ×34\n")
	for _, dt := range core.AllDTypes {
		cells := []embVolCell{{
			enabled: true, vocab: vocab, emb: emb, seq: seq,
			dtype: dt, act: core.ActivationLinear, W: W,
		}}
		goO, err := runGoEmbeddingVolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		mtlO, err := runMetalEmbeddingVolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s metal: %v", dt, err))
			continue
		}
		if err := compareVolNBP(goO, mtlO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			continue
		}
		m := suites.CompareF32Spectrum(goO.Ws[0], mtlO.Ws[0])
		if m.Band.Percent() < worst.Percent() {
			worst = m.Band
		}
	}
	fmt.Printf("      worst_band=%s\n", worst)
	if len(fails) > 0 {
		return fmt.Errorf("%d vol dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

func caseEmbeddingNBPVol222F32() error {
	const batch, vocab, emb, seq = 1, 8, 4, 2
	var benchIters = defaultGoBenchIters
	x := tokenIDs(batch*seq, vocab, 0x454d5637)
	target := make([]float32, batch*seq*emb)
	for i := range target {
		target[i] = float32(0.01 * float64(i%3+1))
	}
	cells := make([]embVolCell, 8)
	for i := range cells {
		W, _ := makeEmbeddingTrainInit(vocab, emb, batch, seq, 0x454d5700+uint32(i))
		cells[i] = embVolCell{
			enabled: true, vocab: vocab, emb: emb, seq: seq,
			dtype: core.DTypeFloat32, act: core.ActivationLinear, W: W,
		}
	}
	const lr = 1e-2
	goO, err := runGoEmbeddingVolNBP(2, 2, 2, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	mtlO, err := runMetalEmbeddingVolNBP(2, 2, 2, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("metal: %w", err)
	}
	if err := compareVolNBP(goO, mtlO, true); err != nil {
		return err
	}
	goUS, mtlUS, spdPct, err := benchEmbeddingVolNBP(2, 2, 2, 1, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      2×2×2×1 loss=%.6g exact_bit  go=%s  mtl=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(mtlUS), spdPct)
	return nil
}

func caseEmbeddingNBPVol211AllDTypes() error {
	const batch, vocab, emb, seq = 1, 8, 4, 2
	W0, x := makeEmbeddingTrainInit(vocab, emb, batch, seq, 0x454d5639)
	W1, _ := makeEmbeddingTrainInit(vocab, emb, batch, seq, 0x454d563a)
	target := make([]float32, batch*seq*emb)
	for i := range target {
		target[i] = float32(0.02 * float64(i%4+1))
	}
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      Embedding NBP vol 2×1×1×1 ×34\n")
	for _, dt := range core.AllDTypes {
		cells := []embVolCell{
			{enabled: true, vocab: vocab, emb: emb, seq: seq, dtype: dt, act: core.ActivationLinear, W: W0},
			{enabled: true, vocab: vocab, emb: emb, seq: seq, dtype: dt, act: core.ActivationLinear, W: W1},
		}
		goO, err := runGoEmbeddingVolNBP(2, 1, 1, 1, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		mtlO, err := runMetalEmbeddingVolNBP(2, 1, 1, 1, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s metal: %v", dt, err))
			continue
		}
		if err := compareVolNBP(goO, mtlO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			continue
		}
		m := suites.CompareF32Spectrum(goO.Ws[0], mtlO.Ws[0])
		if m.Band.Percent() < worst.Percent() {
			worst = m.Band
		}
	}
	fmt.Printf("      worst_band=%s\n", worst)
	if len(fails) > 0 {
		return fmt.Errorf("%d vol211 dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}
