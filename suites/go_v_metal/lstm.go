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
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/quant"
)

func CasesLSTM() []Case {
	return []Case{
		{Name: "Locate welvet.metal tool", Run: caseLocateMetal},
		{Name: "Locate welvet.cpp tool (optional speed)", Run: caseLocateCppOptional},
		{Name: "LSTM Float32 — exact bits vs Go", Run: caseLSTMSmoke},
		{Name: "LSTM FormatNone ×34 — ≥98% + Go/C++/Metal timings", Run: caseLSTMAllDTypes},
	}
}

func RunLSTM() error {
	var fails []string
	cs := CasesLSTM()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [lstm %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_metal_lstm", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_metal_lstm", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_metal lstm: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type lstmCase struct {
	batch   int
	cfg     lstm.Config
	dtype   core.DType
	backend core.Backend
	packed  []float32
	x       []float32
}

func disableLSTMGates(l *lstm.Layer, be core.Backend) {
	l.Exec.Backend = be
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	for _, g := range []*lstm.Gate{l.I, l.F, l.G, l.O} {
		if g == nil {
			continue
		}
		if g.IH != nil {
			g.IH.Exec.MultiCore = false
			g.IH.Core.MultiCore = false
			g.IH.Exec.Backend = be
		}
		if g.HH != nil {
			g.HH.Exec.MultiCore = false
			g.HH.Core.MultiCore = false
			g.HH.Exec.Backend = be
		}
	}
}

func runGoLSTM(fc lstmCase) (pre, post []float32, err error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	l, err := lstm.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.packed)
	if err != nil {
		return nil, nil, err
	}
	disableLSTMGates(l, fc.backend)
	t := core.NewTensor[float32](fc.batch, cfg.SeqLen, cfg.InputSize)
	copy(t.Data, fc.x)
	preT, postT, err := lstm.Forward(l, t)
	if err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), preT.Data...), append([]float32(nil), postT.Data...), nil
}

func encodeLSTM(fc lstmCase) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("WLS1"))
	for _, v := range []int32{
		int32(fc.batch), int32(cfg.InputSize), int32(cfg.HiddenSize), int32(cfg.SeqLen),
		int32(fc.dtype), int32(fc.backend),
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

func decodeLSTM(raw []byte) (pre, post []float32, bench time.Duration, err error) {
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
	if len(raw) < 8 || string(raw[:4]) != "WLSO" {
		return nil, nil, bench, fmt.Errorf("lstm bad magic: %q", string(raw[:min(4, len(raw))]))
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

func runToolLSTM(tool string, fc lstmCase, benchIters int) (pre, post []float32, dur time.Duration, err error) {
	payload, err := encodeLSTM(fc)
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
	return decodeLSTM(stdout.Bytes())
}

func runMetalLSTM(fc lstmCase) (pre, post []float32, err error) {
	tool, err := findMetalFwd()
	if err != nil {
		return nil, nil, err
	}
	pre, post, _, err = runToolLSTM(tool, fc, 0)
	return pre, post, err
}

func requireLSTMExact(fc lstmCase) error {
	goPre, goPost, err := runGoLSTM(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	mPre, mPost, err := runMetalLSTM(fc)
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

func caseLSTMSmoke() error {
	cfg := lstm.Config{InputSize: 4, HiddenSize: 3, SeqLen: 5}
	return requireLSTMExact(lstmCase{
		batch: 2, cfg: cfg, dtype: core.DTypeFloat32, backend: core.BackendCPUTiled,
		packed: randInit(1, cfg.WeightCount(), 0x4c535401),
		x:      randInit(2*cfg.SeqLen, cfg.InputSize, 0x4c535402),
	})
}

func caseLSTMAllDTypes() error {
	// 4 gates × IH/HH — slightly leaner than RNN mid so ×34 finishes; ≥99% ok.
	const batch, benchIters = 2, 60
	cfg := lstm.Config{InputSize: 48, HiddenSize: 48, SeqLen: 12}
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

	fmt.Printf("\n      Go vs Metal LSTM FormatNone×34 — exact_bit preferred (≥98%% ok; 4-gate chain)")
	fmt.Printf("\n      shape B=%d T=%d In=%d H=%d  iters=%d", batch, cfg.SeqLen, cfg.InputSize, cfg.HiddenSize, benchIters)
	if haveCpp {
		fmt.Printf("\n      speed: Go / C++ / Metal")
	}
	fmt.Printf("\n      %-12s %-12s %7s %9s %9s %9s %8s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "mtl_µs", "winner", "vsGo")
	fmt.Printf("      %s\n", strings.Repeat("-", 100))

	var fails, soft []string
	var exactN, totalN int
	winCount := map[string]int{}
	packed := randInit(1, cfg.WeightCount(), 0xd7a4e071)
	x := randInit(batch*cfg.SeqLen, cfg.InputSize, 0xd7a4e072)

	for _, dt := range core.AllDTypes {
		fc := lstmCase{
			batch: batch, cfg: cfg, dtype: dt, backend: core.BackendCPUTiled,
			packed: packed, x: x,
		}
		goPre, goPost, err := runGoLSTM(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		mPre, mPost, err := runMetalLSTM(fc)
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
			_, _, e := runGoLSTM(fc)
			return e
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}

		cppUS := 0.0
		cppStr := "—"
		if haveCpp {
			_, _, cppDur, err := runToolLSTM(cppTool, fc, benchIters)
			if err != nil {
				cppStr = "err"
			} else {
				cppUS = perIterUS(cppDur, benchIters)
				cppStr = FormatUS(cppUS)
			}
		}

		_, _, metalDur, err := runToolLSTM(metalTool, fc, benchIters)
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
			soft = append(soft, fmt.Sprintf("%s: %s", dt, suites.FormatSpectrum(m)))
		}
		if m.Band < suites.DetIndustry {
			fails = append(fails, fmt.Sprintf("%s below industry: %s", dt, suites.FormatSpectrum(m)))
		}
	}
	overallExact := 0.0
	if totalN > 0 {
		overallExact = 100.0 * float64(exactN) / float64(totalN)
	}
	fmt.Printf("      %s\n", strings.Repeat("-", 100))
	fmt.Printf("      overall exact_bits=%.2f%%  (aim≥98%% vs Go; floor=industry; soft-double 4-gate)\n", overallExact)
	parts := []string{}
	for _, n := range []string{"Go", "C++", "Metal"} {
		if winCount[n] > 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", n, winCount[n]))
		}
	}
	fmt.Printf("      wins: %s\n", strings.Join(parts, " "))
	if len(soft) > 0 {
		fmt.Printf("      soft (non-exact_bit, allowed if overall≥98%% + ≥industry): %s\n", strings.Join(soft, " | "))
	}
	fmt.Printf("      note: LSTM = 4×Dense soft-double + host f64 sigmoid/tanh")
	if len(fails) > 0 {
		return fmt.Errorf("%d lstm hard fails: %s", len(fails), strings.Join(fails, " | "))
	}
	if overallExact < 98.0 {
		return fmt.Errorf("overall exact_bits=%.2f%% < 98%%", overallExact)
	}
	return nil
}
