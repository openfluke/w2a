package go_v_cpp

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os/exec"
	"time"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
)

// CasesMeshTweenChainTrain — Dense mesh-clock MeshTweenChain (= MeshBP FamilyBP) + TweenChain volume.
func CasesMeshTweenChainTrain() []Case {
	return []Case{
		{Name: "Dense MeshTweenChain mesh-clock 1×1×1×1 Float32 — exact bits", Run: caseDenseMeshTweenChain111F32},
		{Name: "Dense MeshTweenChain mesh-clock 2×1×1×1 Float32 — exact bits", Run: caseDenseMeshTweenChain211F32},
		{Name: "Dense MeshTweenChain mesh-clock 2×2×2×1 Float32 — exact bits", Run: caseDenseMeshTweenChain222F32},
		// 3D volume credit (= ModeTweenChain volume wires)
		{Name: "Dense MeshTweenChain volume 1×1×1×1 Float32 — exact bits", Run: caseDenseTweenChainVol111F32},
		{Name: "Dense MeshTweenChain volume 2×1×1×1 chain Float32 — exact bits", Run: caseDenseTweenChainVol211F32},
		{Name: "Dense MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseDenseTweenChainVol222F32},
		{Name: "RMSNorm MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseRMSNormTweenChainVol222F32},
		{Name: "LayerNorm MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseLayerNormTweenChainVol222F32},
		{Name: "Softmax MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseSoftmaxTweenChainVol222F32},
		{Name: "SwiGLU MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseSwiGLUTweenChainVol222F32},
		{Name: "Sequential MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseSeqTweenChainVol222F32},
		{Name: "Residual MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseReTweenChainVol222F32},
		{Name: "Mamba MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseMbTweenChainVol222F32},
		{Name: "Metacognition MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseMcTweenChainVol222F32},
		{Name: "GDN MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseGdTweenChainVol222F32},
		{Name: "NestedStack MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseSkTweenChainVol222F32},
		{Name: "Embedding MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseEmbeddingTweenChainVol222F32},
		{Name: "MHA MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseMHATweenChainVol222F32},
		{Name: "CNN1 MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNTweenChainVol222F32(1) }},
		{Name: "CNN2 MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNTweenChainVol222F32(2) }},
		{Name: "CNN3 MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNTweenChainVol222F32(3) }},
		{Name: "ConvT1 MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTTweenChainVol222F32(1) }},
		{Name: "ConvT2 MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTTweenChainVol222F32(2) }},
		{Name: "ConvT3 MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTTweenChainVol222F32(3) }},
		{Name: "RNN MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseRNNTweenChainVol222F32},
		{Name: "LSTM MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseLSTMTweenChainVol222F32},
	}
}

func RunMeshTweenChainTrain() error {
	var fails []string
	cs := CasesMeshTweenChainTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [meshtweenchain %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_meshtweenchain_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_meshtweenchain_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("meshtweenchain_train: %d failed: %v", len(fails), fails)
	}
	return nil
}

type meshtweenchainOut struct {
	loss float64
	Ws   [][]float32
}

func encodeDenseMTC1(depth, rows, cols, lpc, batch, ticks int, cells []volCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("MTC1"))
	for _, v := range []int32{int32(depth), int32(rows), int32(cols), int32(lpc), int32(batch), 0 /*cpu*/, 0 /*n_links*/, int32(ticks)} {
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
		_ = binary.Write(&in, binary.LittleEndian, int32(c.in))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.out))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.act))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.dtype))
		for _, v := range c.W {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
		_ = binary.Write(&in, binary.LittleEndian, uint32(len(c.bias)))
		for _, v := range c.bias {
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

func runCppMTC1(payload []byte, benchIters int) (meshtweenchainOut, time.Duration, error) {
	var z meshtweenchainOut
	var bench time.Duration
	tool, err := findDenseFwd()
	if err != nil {
		return z, 0, err
	}
	args := []string{}
	if benchIters > 0 {
		args = append(args, fmt.Sprintf("--bench=%d", benchIters))
	}
	cmd := exec.Command(tool, args...)
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return z, 0, fmt.Errorf("cpp mtc1: %v: %s", err, stderr.String())
	}
	raw := stdout.Bytes()
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
	if len(raw) < 8 || string(raw[:4]) != "MTCO" {
		return z, bench, fmt.Errorf("cpp meshtweenchain bad magic")
	}
	r := bytes.NewReader(raw[4:])
	if err := binary.Read(r, binary.LittleEndian, &z.loss); err != nil {
		return z, bench, err
	}
	var nCells int32
	if err := binary.Read(r, binary.LittleEndian, &nCells); err != nil {
		return z, bench, err
	}
	for i := int32(0); i < nCells; i++ {
		var o, inn int32
		if err := binary.Read(r, binary.LittleEndian, &o); err != nil {
			return z, bench, err
		}
		if err := binary.Read(r, binary.LittleEndian, &inn); err != nil {
			return z, bench, err
		}
		w := make([]float32, o*inn)
		for j := range w {
			if err := binary.Read(r, binary.LittleEndian, &w[j]); err != nil {
				return z, bench, err
			}
		}
		z.Ws = append(z.Ws, w)
	}
	return z, bench, nil
}

func runGoDenseMeshTweenChain(depth, rows, cols, lpc, batch, ticks int, cells []volCell, x, target []float32, lr float64) (meshtweenchainOut, error) {
	// MeshTweenChain ≡ MeshBP FamilyBP under mesh clock.
	mb, err := runGoDenseMeshBP(depth, rows, cols, lpc, batch, ticks, cells, x, target, lr)
	if err != nil {
		return meshtweenchainOut{}, err
	}
	return meshtweenchainOut{loss: mb.loss, Ws: mb.Ws}, nil
}

func compareMeshTweenChain(goO, cppO meshtweenchainOut, requireExact bool) error {
	if len(goO.Ws) != len(cppO.Ws) {
		return fmt.Errorf("n_cells go=%d cpp=%d", len(goO.Ws), len(cppO.Ws))
	}
	for i := range goO.Ws {
		m := suites.CompareF32Spectrum(goO.Ws[i], cppO.Ws[i])
		if requireExact && m.Band != suites.DetExactBit {
			return fmt.Errorf("W[%d]: %s", i, suites.FormatSpectrum(m))
		}
	}
	if goO.loss != cppO.loss {
		rel := math.Abs(goO.loss-cppO.loss) / math.Max(1e-30, math.Abs(goO.loss))
		if requireExact && rel > 0 {
			return fmt.Errorf("loss go=%g cpp=%g", goO.loss, cppO.loss)
		}
	}
	return nil
}

func runDenseMeshTweenChainCase(depth, rows, cols, lpc, batch, ticks, benchIters int) error {
	stack := depth * rows * cols * lpc
	dim := 4
	W, x, bias := makeSharedInit(dim, dim, batch)
	for i := range W {
		W[i] *= 0.05
	}
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, dim*batch)
	for i := range target {
		target[i] = float32(0.05 * float64(i%dim+1))
	}
	cells := make([]volCell, stack)
	for i := range cells {
		cells[i] = volCell{
			enabled: true, in: dim, out: dim, act: core.ActivationLinear,
			dtype: core.DTypeFloat32, W: append([]float32(nil), W...), bias: append([]float64(nil), bias...),
		}
	}
	goO, err := runGoDenseMeshTweenChain(depth, rows, cols, lpc, batch, ticks, cells, x, target, 1e-2)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeDenseMTC1(depth, rows, cols, lpc, batch, ticks, cells, x, target, 1e-2)
	if err != nil {
		return err
	}
	_, cppDur, err := runCppMTC1(payload, benchIters)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	cppO, _, err := runCppMTC1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp2: %w", err)
	}
	if err := compareMeshTweenChain(goO, cppO, true); err != nil {
		return err
	}
	t0 := time.Now()
	_, _ = runGoDenseMeshTweenChain(depth, rows, cols, lpc, batch, ticks, cells, x, target, 1e-2)
	goUS := float64(time.Since(t0).Nanoseconds()) / 1000
	bi := benchIters
	if bi < 1 {
		bi = 1
	}
	cppUS := float64(cppDur.Nanoseconds()) / float64(bi) / 1000
	spd := suites.SpeedPctDiff(goUS, cppUS)
	fmt.Printf("\n      MeshTweenChain %dx%dx%dx%d ticks=%d loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
		depth, rows, cols, lpc, ticks, goO.loss, goUS, cppUS, spd)
	return nil
}

func caseDenseMeshTweenChain111F32() error {
	return runDenseMeshTweenChainCase(1, 1, 1, 1, 1, 1, 40)
}
func caseDenseMeshTweenChain211F32() error {
	return runDenseMeshTweenChainCase(2, 1, 1, 1, 1, 2, 40)
}
func caseDenseMeshTweenChain222F32() error {
	return runDenseMeshTweenChainCase(2, 2, 2, 1, 1, 8, 20)
}
