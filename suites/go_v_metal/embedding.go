package go_v_metal

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
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/quant"
)

func CasesEmbedding() []Case {
	return []Case{
		{Name: "Locate welvet.metal tool", Run: caseLocateMetal},
		{Name: "Locate welvet.cpp tool (optional speed)", Run: caseLocateCppOptional},
		{Name: "Embedding Float32 Linear [B,T] — exact bits vs Go", Run: caseEmbeddingSmoke},
		{Name: "Embedding Float32 Linear [T] rank1 — exact bits vs Go", Run: caseEmbeddingRank1},
		{Name: "Embedding Float32 Linear [B,T,1] rank3 — exact bits vs Go", Run: caseEmbeddingRank3},
		{Name: "Embedding FormatNone ×34 — exact_bit + Go/C++/Metal timings", Run: caseEmbeddingAllDTypes},
	}
}

func RunEmbedding() error {
	var fails []string
	cs := CasesEmbedding()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [emb %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_metal_embedding", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_metal_embedding", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_metal embedding: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type embCase struct {
	batch   int
	cfg     embedding.Config
	inRank  int // 1..3
	dtype   core.DType
	backend core.Backend
	act     core.ActivationType
	W       []float32
	x       []float32 // token ids as f32
}

func tokenIDs(n, vocab int, seed uint32) []float32 {
	out := make([]float32, n)
	s := seed
	v := uint32(vocab)
	for i := range out {
		s = s*1664525 + 1013904223
		out[i] = float32(s % v)
	}
	return out
}

func runGoEmbedding(fc embCase) (pre, post []float32, err error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	l, err := embedding.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.W)
	if err != nil {
		return nil, nil, err
	}
	l.Exec.Backend = fc.backend
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	l.Core.Activation = fc.act
	var t *core.Tensor[float32]
	switch fc.inRank {
	case 1:
		t = core.NewTensor[float32](cfg.SeqLen)
	case 3:
		t = core.NewTensor[float32](fc.batch, cfg.SeqLen, 1)
	default:
		t = core.NewTensor[float32](fc.batch, cfg.SeqLen)
	}
	copy(t.Data, fc.x)
	preT, postT, err := embedding.Forward(l, t)
	if err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), preT.Data...), append([]float32(nil), postT.Data...), nil
}

func encodeEmbedding(fc embCase) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	inLast := int32(1)
	inRank := int32(fc.inRank)
	if inRank < 1 || inRank > 3 {
		inRank = 2
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("WEM1"))
	for _, v := range []int32{
		int32(fc.batch), int32(cfg.VocabSize), int32(cfg.EmbeddingDim), int32(cfg.SeqLen),
		inRank, inLast, int32(fc.act), int32(fc.dtype), int32(fc.backend),
	} {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.W {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	return in.Bytes(), nil
}

func decodeEmbedding(raw []byte) (pre, post []float32, bench time.Duration, err error) {
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
	if len(raw) < 8 || string(raw[:4]) != "WEMO" {
		return nil, nil, bench, fmt.Errorf("embedding bad magic: %q", string(raw[:min(4, len(raw))]))
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

func runToolEmbedding(tool string, fc embCase, benchIters int) (pre, post []float32, dur time.Duration, err error) {
	payload, err := encodeEmbedding(fc)
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
	return decodeEmbedding(stdout.Bytes())
}

func runMetalEmbedding(fc embCase) (pre, post []float32, err error) {
	tool, err := findMetalFwd()
	if err != nil {
		return nil, nil, err
	}
	pre, post, _, err = runToolEmbedding(tool, fc, 0)
	return pre, post, err
}

func requireEmbeddingExact(fc embCase) error {
	goPre, goPost, err := runGoEmbedding(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	mPre, mPost, err := runMetalEmbedding(fc)
	if err != nil {
		return fmt.Errorf("metal: %w", err)
	}
	preM := suites.CompareF32Spectrum(goPre, mPre)
	postM := suites.CompareF32Spectrum(goPost, mPost)
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

func caseEmbeddingSmoke() error {
	cfg := embedding.Config{VocabSize: 32, EmbeddingDim: 8, SeqLen: 5}
	return requireEmbeddingExact(embCase{
		batch: 2, cfg: cfg, inRank: 2, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		act: core.ActivationLinear,
		W:   randInit(cfg.VocabSize, cfg.EmbeddingDim, 0x454d4201),
		x:   tokenIDs(2*cfg.SeqLen, cfg.VocabSize, 0x454d4202),
	})
}

func caseEmbeddingRank1() error {
	cfg := embedding.Config{VocabSize: 16, EmbeddingDim: 4, SeqLen: 7}
	return requireEmbeddingExact(embCase{
		batch: 1, cfg: cfg, inRank: 1, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		act: core.ActivationLinear,
		W:   randInit(cfg.VocabSize, cfg.EmbeddingDim, 0x454d4203),
		x:   tokenIDs(cfg.SeqLen, cfg.VocabSize, 0x454d4204),
	})
}

func caseEmbeddingRank3() error {
	cfg := embedding.Config{VocabSize: 20, EmbeddingDim: 6, SeqLen: 4}
	return requireEmbeddingExact(embCase{
		batch: 2, cfg: cfg, inRank: 3, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		act: core.ActivationLinear,
		W:   randInit(cfg.VocabSize, cfg.EmbeddingDim, 0x454d4205),
		x:   tokenIDs(2*cfg.SeqLen, cfg.VocabSize, 0x454d4206),
	})
}

func caseEmbeddingAllDTypes() error {
	const batch, benchIters = 4, 80
	cfg := embedding.Config{VocabSize: 1024, EmbeddingDim: 256, SeqLen: 64}
	if err := cfg.Validate(); err != nil {
		return err
	}
	if len(core.AllDTypes) != 34 {
		return fmt.Errorf("expected 34 dtypes, got %d", len(core.AllDTypes))
	}
	metalTool, err := findMetalFwd()
	if err != nil {
		return err
	}
	cppTool, cppErr := findCppFwd()
	haveCpp := cppErr == nil

	fmt.Printf("\n      Go vs Metal Embedding FormatNone×34 — exact_bit required")
	fmt.Printf("\n      shape batch=%d vocab=%d emb=%d seq=%d  iters=%d",
		batch, cfg.VocabSize, cfg.EmbeddingDim, cfg.SeqLen, benchIters)
	if haveCpp {
		fmt.Printf("\n      speed: Go / C++ / Metal — winner on warm gather")
	}
	fmt.Printf("\n      %-12s %-12s %7s %9s %9s %9s %8s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "mtl_µs", "winner", "vsGo")
	fmt.Printf("      %s\n", strings.Repeat("-", 100))

	var fails []string
	var exactN, totalN int
	winCount := map[string]int{}
	W := randInit(cfg.VocabSize, cfg.EmbeddingDim, 0xd7a4e081)
	x := tokenIDs(batch*cfg.SeqLen, cfg.VocabSize, 0xd7a4e082)

	for _, dt := range core.AllDTypes {
		fc := embCase{
			batch: batch, cfg: cfg, inRank: 2, dtype: dt, backend: core.BackendCPUTiled,
			act: core.ActivationLinear, W: W, x: x,
		}
		goPre, goPost, err := runGoEmbedding(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		mPre, mPost, err := runMetalEmbedding(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s metal: %v", dt, err))
			fmt.Printf("      %-12s FAIL metal %v\n", dt, err)
			continue
		}
		preM := suites.CompareF32Spectrum(goPre, mPre)
		postM := suites.CompareF32Spectrum(goPost, mPost)
		m := postM
		if preM.Band < m.Band {
			m = preM
		}
		exactN += m.ExactBits
		totalN += m.N

		goUS, err := benchGoUS(func() error {
			_, _, e := runGoEmbedding(fc)
			return e
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}

		cppUS := 0.0
		cppStr := "—"
		if haveCpp {
			_, _, cppDur, err := runToolEmbedding(cppTool, fc, benchIters)
			if err != nil {
				cppStr = "err"
			} else {
				cppUS = perIterUS(cppDur, benchIters)
				cppStr = FormatUS(cppUS)
			}
		}

		_, _, metalDur, err := runToolEmbedding(metalTool, fc, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s metal bench: %v", dt, err))
			fmt.Printf("      %-12s %-12s %6.2f%% %9s %9s %9s %8s %s\n",
				dt, m.Band, m.ExactPct, FormatUS(goUS), cppStr, "?", "?", err)
			continue
		}
		metalUS := perIterUS(metalDur, benchIters)

		names := []string{"Go", "Metal"}
		uss := []float64{goUS, metalUS}
		if haveCpp && cppUS > 0 {
			names = []string{"Go", "C++", "Metal"}
			uss = []float64{goUS, cppUS, metalUS}
		}
		winner, _, vsGo := pickFastest(names, uss)
		winCount[winner]++
		vsGoStr := "baseline"
		if winner != "Go" {
			vsGoStr = fmt.Sprintf("%s %+0.1f%%", winner, vsGo)
		}
		fmt.Printf("      %-12s %-12s %6.2f%% %9s %9s %9s %8s %s\n",
			dt, m.Band, m.ExactPct, FormatUS(goUS), cppStr, FormatUS(metalUS), winner, vsGoStr)

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
	parts := []string{}
	for _, n := range []string{"Go", "C++", "Metal"} {
		if winCount[n] > 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", n, winCount[n]))
		}
	}
	fmt.Printf("      wins: %s\n", strings.Join(parts, " "))
	fmt.Printf("      note: Vocab×Emb FormatNone×34 gather via decode_row_f64 (Go/cpp exact_bit)")

	if len(fails) > 0 {
		return fmt.Errorf("%d embedding-dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}
