package go_v_cuda

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os/exec"
	"time"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
)

// CasesMeshTweenChainTrain — Dense mesh-clock MeshTweenChain (= MeshBP FamilyBP) + TweenChain volume.
func casesMeshTweenChainBase() []Case {
	return []Case{
		{Name: "Dense MeshTweenChain mesh-clock 111 × all 34 dtypes", Run: func() error { return denseMeshCreditDTypes(true, 1) }},
		{Name: "Dense MeshTweenChain mesh-clock 211 × all 34 dtypes", Run: func() error { return denseMeshCreditDTypes(true, 2) }},
		{Name: "Dense MeshTweenChain mesh-clock 1×1×1×1 Float32 — exact tensors; CUDA loss band", Run: caseDenseMeshTweenChain111F32},
		{Name: "Dense MeshTweenChain mesh-clock 2×1×1×1 Float32 — exact tensors; CUDA loss band", Run: caseDenseMeshTweenChain211F32},
		{Name: "Dense MeshTweenChain mesh-clock 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseDenseMeshTweenChain222F32},
		// 3D volume credit (= ModeTweenChain volume wires)
		{Name: "Dense MeshTweenChain volume 1×1×1×1 Float32 — exact tensors; CUDA loss band", Run: caseDenseTweenChainVol111F32},
		{Name: "Dense MeshTweenChain volume 2×1×1×1 chain Float32 — exact tensors; CUDA loss band", Run: caseDenseTweenChainVol211F32},
		{Name: "Dense MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseDenseTweenChainVol222F32},
		{Name: "RMSNorm MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseRMSNormTweenChainVol222F32},
		{Name: "LayerNorm MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseLayerNormTweenChainVol222F32},
		{Name: "Softmax MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseSoftmaxTweenChainVol222F32},
		{Name: "SwiGLU MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseSwiGLUTweenChainVol222F32},
		{Name: "Sequential MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseSeqTweenChainVol222F32},
		{Name: "Residual MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseReTweenChainVol222F32},
		{Name: "Mamba MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseMbTweenChainVol222F32},
		{Name: "Metacognition MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseMcTweenChainVol222F32},

		{Name: "GDN MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseGdTweenChainVol222F32},
		{Name: "NestedStack MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseSkTweenChainVol222F32},
		{Name: "Embedding MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseEmbeddingTweenChainVol222F32},
		{Name: "MHA MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseMHATweenChainVol222F32},
		{Name: "CNN1 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: func() error { return caseCNNTweenChainVol222F32(1) }},
		{Name: "CNN2 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: func() error { return caseCNNTweenChainVol222F32(2) }},
		{Name: "CNN3 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: func() error { return caseCNNTweenChainVol222F32(3) }},
		{Name: "ConvT1 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: func() error { return caseConvTTweenChainVol222F32(1) }},
		{Name: "ConvT2 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: func() error { return caseConvTTweenChainVol222F32(2) }},
		{Name: "ConvT3 MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: func() error { return caseConvTTweenChainVol222F32(3) }},
		{Name: "RNN MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseRNNTweenChainVol222F32},
		{Name: "LSTM MeshTweenChain volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseLSTMTweenChainVol222F32},
	}
}

func RunMeshTweenChainTrain() error {
	var fails []string
	cs := CasesMeshTweenChainTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [meshtweenchain %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cuda_meshtweenchain_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cuda_meshtweenchain_train", c.Name, "PASS", "")
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

func runCUDAMTC1(payload []byte, benchIters int) (meshtweenchainOut, time.Duration, error) {
	var z meshtweenchainOut
	var bench time.Duration
	tool, err := findCUDAFwd()
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
		return z, 0, fmt.Errorf("cuda mtc1: %v: %s", err, stderr.String())
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
		return z, bench, fmt.Errorf("cuda meshtweenchain bad magic")
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

func compareMeshTweenChain(goO, cudaO meshtweenchainOut, requireExact bool) error {
	if len(goO.Ws) != len(cudaO.Ws) {
		return fmt.Errorf("n_cells go=%d cuda=%d", len(goO.Ws), len(cudaO.Ws))
	}
	for i := range goO.Ws {
		m := suites.CompareF32Spectrum(goO.Ws[i], cudaO.Ws[i])
		if requireExact && m.Band != suites.DetExactBit {
			return fmt.Errorf("W[%d]: %s", i, suites.FormatSpectrum(m))
		}
	}
	if !cudaLossWithinBand(goO.loss, cudaO.loss) {
		return fmt.Errorf("loss outside CUDA float2 band: go=%g cuda=%g", goO.loss, cudaO.loss)
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
	_, cudaDur, err := runCUDAMTC1(payload, benchIters)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	cudaO, _, err := runCUDAMTC1(payload, 0)
	if err != nil {
		return fmt.Errorf("cuda2: %w", err)
	}
	if err := compareMeshTweenChain(goO, cudaO, true); err != nil {
		return err
	}
	goUS, err := benchGoUS(func() error {
		_, _ = runGoDenseMeshTweenChain(depth, rows, cols, lpc, batch, ticks, cells, x, target, 1e-2)
		return nil
	}, defaultGoBenchIters)
	if err != nil {
		return err
	}
	bi := benchIters
	if bi < 1 {
		bi = 1
	}
	cudaUS := float64(cudaDur.Nanoseconds()) / float64(bi) / 1000
	spd := suites.SpeedPctDiff(goUS, cudaUS)
	fmt.Printf("\n      MeshTweenChain %dx%dx%dx%d ticks=%d loss=%.6g exact_bit  go=%s  cuda=%s  spd%%Δ=%+.1f%%",
		depth, rows, cols, lpc, ticks, goO.loss, FormatUS(goUS), FormatUS(cudaUS), spd)
	return nil
}

func caseDenseMeshTweenChain111F32() error {
	return runDenseMeshTweenChainCase(1, 1, 1, 1, 1, 1, defaultGoBenchIters)
}
func caseDenseMeshTweenChain211F32() error {
	return runDenseMeshTweenChainCase(2, 1, 1, 1, 1, 2, defaultGoBenchIters)
}
func caseDenseMeshTweenChain222F32() error {
	return runDenseMeshTweenChainCase(2, 2, 2, 1, 1, 8, defaultGoBenchIters)
}

func CasesMeshTweenChainTrain() []Case {
	cases := withTypedVolumes(casesMeshTweenChainBase(), CasesTweenChainTrain(), "MeshTweenChain ")
	cases = append(cases, kpVolumeCases(parallel.ModeTweenChain)...)
	return append(cases, CasesMeshTweenChainClockTrain()...)
}
