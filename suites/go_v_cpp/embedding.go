package go_v_cpp

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
		{Name: "Embedding Float32 Linear — exact bits", Run: caseEmbeddingSmoke},
		{Name: "Embedding FormatNone × all 34 dtypes — spectrum + speed", Run: caseEmbeddingAllDTypes},
	}
}

func RunEmbedding() error {
	var fails []string
	cs := CasesEmbedding()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [emb %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_embedding", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_embedding", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp embedding: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type embCase struct {
	batch   int
	cfg     embedding.Config
	inRank  int // 2 → [B,T]
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
		return nil, nil, bench, fmt.Errorf("cpp embedding bad magic")
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

func runCppEmbedding(payload []byte, benchIters int) (pre, post []float32, dur time.Duration, err error) {
	tool, err := findDenseFwd()
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
		return nil, nil, 0, fmt.Errorf("cpp embedding: %v: %s", err, stderr.String())
	}
	return decodeEmbedding(stdout.Bytes())
}

func requireEmbeddingExact(goPre, goPost, cppPre, cppPost []float32) error {
	preM := suites.CompareF32Spectrum(goPre, cppPre)
	postM := suites.CompareF32Spectrum(goPost, cppPost)
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
	fc := embCase{
		batch: 2, cfg: cfg, inRank: 2, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		act: core.ActivationLinear,
		W:   randInit(cfg.VocabSize, cfg.EmbeddingDim, 0x454d4201),
		x:   tokenIDs(2*cfg.SeqLen, cfg.VocabSize, 0x454d4202),
	}
	goPre, goPost, err := runGoEmbedding(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeEmbedding(fc)
	if err != nil {
		return err
	}
	cppPre, cppPost, _, err := runCppEmbedding(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return requireEmbeddingExact(goPre, goPost, cppPre, cppPost)
}

func caseEmbeddingAllDTypes() error {
	const batch, benchIters = 1, defaultGoBenchIters
	cfg := embedding.Config{VocabSize: 24, EmbeddingDim: 6, SeqLen: 7}
	if err := cfg.Validate(); err != nil {
		return err
	}
	fmt.Printf("\n      Go CPU vs C++ CPU — Embedding FormatNone×34\n")
	fmt.Printf("      %-12s %-12s %8s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "spect%", "go_µs", "cpp_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 88))
	var exactN, totalN int
	worstBand := suites.DetExactBit
	var fails []string
	for _, dt := range core.AllDTypes {
		fc := embCase{
			batch: batch, cfg: cfg, inRank: 2, dtype: dt, backend: core.BackendCPUTiled,
			act: core.ActivationLinear,
			W:   randInit(cfg.VocabSize, cfg.EmbeddingDim, 0xd7a4e081),
			x:   tokenIDs(batch*cfg.SeqLen, cfg.VocabSize, 0xd7a4e082),
		}
		goPre, goPost, err := runGoEmbedding(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		payload, err := encodeEmbedding(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s enc: %v", dt, err))
			continue
		}
		cppPre, cppPost, _, err := runCppEmbedding(payload, 0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp: %v", dt, err))
			fmt.Printf("      %-12s FAIL cpp %v\n", dt, err)
			continue
		}
		preM := suites.CompareF32Spectrum(goPre, cppPre)
		postM := suites.CompareF32Spectrum(goPost, cppPost)
		m := postM
		if preM.Band < m.Band {
			m = preM
		}
		if m.Band < worstBand {
			worstBand = m.Band
		}
		exactN += m.ExactBits
		totalN += m.N
		goUS, err := benchGoUS(func() error {
			_, _, err := runGoEmbedding(fc)
			return err
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		_, _, cppDur, err := runCppEmbedding(payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp bench: %v", dt, err))
			fmt.Printf("      %-12s %-12s %7.2f%% %7.0f%% %10s %10s %9s %s\n",
				dt, m.Band, m.ExactPct, m.Band.Percent(), FormatUS(goUS), "?", "?", fmt.Sprintf("%.3g", m.MaxAbs))
			continue
		}
		cppUS := perIterUS(cppDur, benchIters)
		spdPct := suites.SpeedPctDiff(goUS, cppUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %7.0f%% %10s %10s %+8.1f%% %s\n",
			dt, m.Band, m.ExactPct, m.Band.Percent(), FormatUS(goUS), FormatUS(cppUS), spdPct, fmt.Sprintf("%.3g", m.MaxAbs))
		if m.Band != suites.DetExactBit {
			fails = append(fails, fmt.Sprintf("%s: %s", dt, suites.FormatSpectrum(m)))
		}
	}
	overallExact := 0.0
	if totalN > 0 {
		overallExact = 100.0 * float64(exactN) / float64(totalN)
	}
	fmt.Printf("      %s\n", strings.Repeat("-", 88))
	fmt.Printf("      overall exact_bits=%.2f%%  worst_band=%s (%.0f%%)  aim=exact_bit (100%%)\n",
		overallExact, worstBand, worstBand.Percent())
	fmt.Printf("      spd%%Δ = (cpp−go)/go×100  (negative ⇒ C++ faster)")
	if len(fails) > 0 {
		return fmt.Errorf("%d embedding-dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}
