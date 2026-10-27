package go_v_cuda

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"time"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/quant"
)

func CasesSequential() []Case {
	return []Case{
		{Name: "Locate welvet.cuda tool", Run: caseLocateCUDA},
		{Name: "Locate welvet.cpp tool (optional speed)", Run: caseLocateCppOptional},
		{Name: "Sequential Dense Float32 [B,D] — exact bits vs Go", Run: caseSequential2D},
		{Name: "Sequential Dense Float32 [B,T,D] — exact bits vs Go", Run: caseSequential3D},
		{Name: "Sequential Dense FormatNone ×34 — exact_bit + Go/C++/CUDA timings", Run: caseSequentialAllDTypes},
	}
}

func RunSequential() error {
	var fails []string
	cs := CasesSequential()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [sq %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cuda_sequential", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cuda_sequential", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cuda sequential: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type sqCase struct {
	cfg     sequential.Config
	dtype   core.DType
	backend core.Backend
	batch   int
	seq     int // 1 → [B,D]
	packed  []float32
	x       []float32
}

func runGoSequential(fc sqCase) (pre, post []float32, err error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	l, err := sequential.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.packed)
	if err != nil {
		return nil, nil, err
	}
	l.Exec.Backend = fc.backend
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	for _, ch := range l.Children {
		if ch == nil {
			continue
		}
		ch.Exec.MultiCore = false
		ch.Core.MultiCore = false
		ch.Exec.Backend = fc.backend
	}
	var t *core.Tensor[float32]
	if fc.seq <= 1 {
		t = core.NewTensor[float32](fc.batch, cfg.Dim)
	} else {
		t = core.NewTensor[float32](fc.batch, fc.seq, cfg.Dim)
	}
	copy(t.Data, fc.x)
	preT, postT, err := sequential.Forward(l, t)
	if err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), preT.Data...), append([]float32(nil), postT.Data...), nil
}

func encodeSequential(fc sqCase) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	seq := fc.seq
	if seq <= 0 {
		seq = 1
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("WSQ1"))
	for _, v := range []int32{
		int32(fc.batch), int32(seq), int32(cfg.Dim), int32(cfg.Depth),
		int32(core.ActivationLinear), int32(fc.dtype), int32(fc.backend),
	} {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.packed {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	return in.Bytes(), nil
}

func decodeSequential(raw []byte) (pre, post []float32, bench time.Duration, err error) {
	if len(raw) >= 4+8+4 && string(raw[:4]) == "WVB1" {
		r := bytes.NewReader(raw[4:])
		var ns int64
		var iters int32
		if err := binary.Read(r, binary.LittleEndian, &ns); err != nil {
			return nil, nil, 0, err
		}
		if err := binary.Read(r, binary.LittleEndian, &iters); err != nil {
			return nil, nil, 0, err
		}
		rest, err := io.ReadAll(r)
		if err != nil {
			return nil, nil, 0, err
		}
		raw = rest
		bench = time.Duration(ns)
	}
	if len(raw) < 8 || string(raw[:4]) != "WSQO" {
		return nil, nil, bench, fmt.Errorf("sequential bad magic: %q", string(raw[:min(4, len(raw))]))
	}
	r := bytes.NewReader(raw[4:])
	var preN, postN int32
	if err := binary.Read(r, binary.LittleEndian, &preN); err != nil {
		return nil, nil, bench, err
	}
	pre = make([]float32, preN)
	for i := range pre {
		if err := binary.Read(r, binary.LittleEndian, &pre[i]); err != nil {
			return nil, nil, bench, err
		}
	}
	if err := binary.Read(r, binary.LittleEndian, &postN); err != nil {
		return nil, nil, bench, err
	}
	post = make([]float32, postN)
	for i := range post {
		if err := binary.Read(r, binary.LittleEndian, &post[i]); err != nil {
			return nil, nil, bench, err
		}
	}
	return pre, post, bench, nil
}

func runToolSequential(tool string, fc sqCase, benchIters int) (pre, post []float32, dur time.Duration, err error) {
	payload, err := encodeSequential(fc)
	if err != nil {
		return nil, nil, 0, err
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
		return nil, nil, 0, fmt.Errorf("%v: %s", err, stderr.String())
	}
	return decodeSequential(stdout.Bytes())
}

func runCudaSequential(fc sqCase) (pre, post []float32, err error) {
	tool, err := findCUDAFwd()
	if err != nil {
		return nil, nil, err
	}
	pre, post, _, err = runToolSequential(tool, fc, 0)
	return pre, post, err
}

func requireSequentialExact(fc sqCase) error {
	goPre, goPost, err := runGoSequential(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cPre, cPost, err := runCudaSequential(fc)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	preM := suites.CompareF32Spectrum(goPre, cPre)
	postM := suites.CompareF32Spectrum(goPost, cPost)
	fmt.Printf("\n      pre  %s", suites.FormatSpectrum(preM))
	fmt.Printf("\n      post %s", suites.FormatSpectrum(postM))
	if preM.Band != suites.DetExactBit {
		return fmt.Errorf("pre %s", suites.FormatSpectrum(preM))
	}
	if postM.Band != suites.DetExactBit {
		return fmt.Errorf("post %s", suites.FormatSpectrum(postM))
	}
	return nil
}

func caseSequential2D() error {
	cfg := sequential.Config{Dim: 8, SeqLen: 1, Depth: 3}
	return requireSequentialExact(sqCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch: 2, seq: 1,
		packed: randInit(1, cfg.WeightCount(), 0x53515101),
		x:      randInit(2, 8, 0x53515102),
	})
}

func caseSequential3D() error {
	cfg := sequential.Config{Dim: 6, SeqLen: 4, Depth: 2}
	return requireSequentialExact(sqCase{
		cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		batch: 2, seq: 4,
		packed: randInit(1, cfg.WeightCount(), 0x53515103),
		x:      randInit(2*4, 6, 0x53515104),
	})
}

func caseSequentialAllDTypes() error {
	const batch, seq, benchIters = 4, 16, 80
	cfg := sequential.Config{Dim: 64, SeqLen: seq, Depth: 3}
	if err := cfg.Validate(); err != nil {
		return err
	}
	if len(core.AllDTypes) != 34 {
		return fmt.Errorf("expected 34 dtypes, got %d", len(core.AllDTypes))
	}
	cudaTool, err := findCUDAFwd()
	if err != nil {
		return err
	}
	cppTool, cppErr := findCppFwd()
	haveCpp := cppErr == nil

	fmt.Printf("\n      Go vs CUDA Sequential Dense FormatNone×34 — exact_bit required")
	fmt.Printf("\n      shape batch=%d seq=%d dim=%d depth=%d  iters=%d",
		batch, seq, cfg.Dim, cfg.Depth, benchIters)
	if haveCpp {
		fmt.Printf("\n      speed: Go / C++ / CUDA — winner on warm Dense chain")
	}
	fmt.Printf("\n      %-12s %-12s %7s %9s %9s %9s %8s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "cuda_µs", "winner", "vsGo")
	fmt.Printf("      %s\n", strings.Repeat("-", 100))

	var fails []string
	var exactN, totalN int
	winCount := map[string]int{}
	packed := randInit(1, cfg.WeightCount(), 0xd7a4e0b1)
	x := randInit(batch*seq, cfg.Dim, 0xd7a4e0b2)

	for _, dt := range core.AllDTypes {
		fc := sqCase{
			cfg: cfg, dtype: dt, backend: core.BackendCPUTiled,
			batch: batch, seq: seq, packed: packed, x: x,
		}
		goPre, goPost, err := runGoSequential(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		cPre, cPost, err := runCudaSequential(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cuda: %v", dt, err))
			fmt.Printf("      %-12s FAIL cuda %v\n", dt, err)
			continue
		}
		preM := suites.CompareF32Spectrum(goPre, cPre)
		postM := suites.CompareF32Spectrum(goPost, cPost)
		m := postM
		if preM.Band < m.Band {
			m = preM
		}
		exactN += m.ExactBits
		totalN += m.N

		goUS, err := benchGoUS(func() error {
			_, _, e := runGoSequential(fc)
			return e
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}

		cppUS := 0.0
		cppStr := "—"
		if haveCpp {
			_, _, cppDur, err := runToolSequential(cppTool, fc, benchIters)
			if err != nil {
				cppStr = "err"
			} else {
				cppUS = perIterUS(cppDur, benchIters)
				cppStr = FormatUS(cppUS)
			}
		}

		_, _, cudaDur, err := runToolSequential(cudaTool, fc, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cuda bench: %v", dt, err))
			fmt.Printf("      %-12s %-12s %6.2f%% %9s %9s %9s %8s %s\n",
				dt, m.Band, m.ExactPct, FormatUS(goUS), cppStr, "?", "?", err)
			continue
		}
		cudaUS := perIterUS(cudaDur, benchIters)

		names := []string{"Go", "CUDA"}
		uss := []float64{goUS, cudaUS}
		if haveCpp && cppUS > 0 {
			names = []string{"Go", "C++", "CUDA"}
			uss = []float64{goUS, cppUS, cudaUS}
		}
		winner, _, vsGo := pickFastest(names, uss)
		winCount[winner]++
		vsGoStr := "baseline"
		if winner != "Go" {
			vsGoStr = fmt.Sprintf("%s %+0.1f%%", winner, vsGo)
		}
		fmt.Printf("      %-12s %-12s %6.2f%% %9s %9s %9s %8s %s\n",
			dt, m.Band, m.ExactPct, FormatUS(goUS), cppStr, FormatUS(cudaUS), winner, vsGoStr)
		if m.Band != suites.DetExactBit {
			fails = append(fails, fmt.Sprintf("%s: %s", dt, suites.FormatSpectrum(m)))
		}
	}
	overallExact := 0.0
	if totalN > 0 {
		overallExact = 100.0 * float64(exactN) / float64(totalN)
	}
	fmt.Printf("      %s\n", strings.Repeat("-", 100))
	fmt.Printf("      overall exact_bits=%.2f%%  (aim=100%% vs Go)\n", overallExact)
	parts := make([]string, 0, len(winCount))
	for k, v := range winCount {
		parts = append(parts, fmt.Sprintf("%s=%d", k, v))
	}
	fmt.Printf("      wins: %s", strings.Join(parts, " "))
	fmt.Printf("\n      note: Sequential = Depth Dense Dim→Dim chain (CUDA soft-double)")
	if len(fails) > 0 {
		return fmt.Errorf("%d sequential-dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}
