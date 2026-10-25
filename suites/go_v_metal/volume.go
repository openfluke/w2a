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
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/forward"
)

func CasesVolume() []Case {
	return []Case{
		{Name: "Locate welvet.metal tool", Run: caseLocateMetal},
		{Name: "Locate welvet.cpp tool (optional speed)", Run: caseLocateCppOptional},
		{Name: "Volume Dense 1×1×1×2 chain — exact bits vs Go", Run: caseVolumeDenseChain},
		{Name: "Volume Dense 2×1×1×1 remote hop — exact bits vs Go", Run: caseVolumeRemoteHop},
		{Name: "Volume Dense 3×3×3×1 ×34 — ≥99% + Go/C++/Metal timings", Run: caseVolumeDense3x3x3AllDTypes},
		{Name: "Volume MHA 3×3×3×1 ×34 — ≥99% + Go/C++/Metal timings", Run: caseVolumeMHA3x3x3AllDTypes},
		{Name: "Volume SwiGLU 3×3×3×1 ×34 — ≥99% + Go/C++/Metal timings", Run: caseVolumeSwiGLU3x3x3AllDTypes},
		{Name: "Volume AllLayers 3×3×3 × dtypes — mount-all volume wires", Run: caseVolumeAllLayers3x3x3},
	}
}

func RunVolume() error {
	var fails []string
	cs := CasesVolume()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [vol %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_metal_volume", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_metal_volume", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_metal volume: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type volCell struct {
	enabled bool
	in, out int
	act     core.ActivationType
	dtype   core.DType
	W       []float32
	bias    []float64
}

func encodeVolumeReq(depth, rows, cols, lpc, batch int, backend core.Backend, links [][8]int32, cells []volCell, x []float32) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("WVF1"))
	_ = binary.Write(&in, binary.LittleEndian, int32(depth))
	_ = binary.Write(&in, binary.LittleEndian, int32(rows))
	_ = binary.Write(&in, binary.LittleEndian, int32(cols))
	_ = binary.Write(&in, binary.LittleEndian, int32(lpc))
	_ = binary.Write(&in, binary.LittleEndian, int32(batch))
	_ = binary.Write(&in, binary.LittleEndian, int32(backend))
	_ = binary.Write(&in, binary.LittleEndian, int32(len(links)))
	for _, L := range links {
		for _, v := range L {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
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
	return in.Bytes(), nil
}

// volTiming — WVB2 splits mount (one-shot) vs warm exec (per-iter). Legacy WVB1 → ExecUS only.
type volTiming struct {
	Out     []float32
	MountUS float64
	ExecUS  float64
}

func decodeVolumeResp(raw []byte) (volTiming, error) {
	var t volTiming
	if len(raw) >= 4+8+8+4 && string(raw[:4]) == "WVB2" {
		r := bytes.NewReader(raw[4:])
		var mountNS, execNS int64
		var iters int32
		if err := binary.Read(r, binary.LittleEndian, &mountNS); err != nil {
			return t, err
		}
		if err := binary.Read(r, binary.LittleEndian, &execNS); err != nil {
			return t, err
		}
		if err := binary.Read(r, binary.LittleEndian, &iters); err != nil {
			return t, err
		}
		rest, err := io.ReadAll(r)
		if err != nil {
			return t, err
		}
		raw = rest
		t.MountUS = float64(mountNS) / 1000.0
		t.ExecUS = perIterUS(time.Duration(execNS), int(iters))
	} else if len(raw) >= 4+8+4 && string(raw[:4]) == "WVB1" {
		r := bytes.NewReader(raw[4:])
		var ns int64
		var iters int32
		if err := binary.Read(r, binary.LittleEndian, &ns); err != nil {
			return t, err
		}
		if err := binary.Read(r, binary.LittleEndian, &iters); err != nil {
			return t, err
		}
		rest, err := io.ReadAll(r)
		if err != nil {
			return t, err
		}
		raw = rest
		t.ExecUS = perIterUS(time.Duration(ns), int(iters))
	}
	if len(raw) < 8 || string(raw[:4]) != "WVFO" {
		return t, fmt.Errorf("volume bad magic: %q", string(raw[:min(4, len(raw))]))
	}
	r := bytes.NewReader(raw[4:])
	var n int32
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return t, err
	}
	t.Out = make([]float32, n)
	for i := range t.Out {
		if err := binary.Read(r, binary.LittleEndian, &t.Out[i]); err != nil {
			return t, err
		}
	}
	return t, nil
}

func runToolVolume(tool string, depth, rows, cols, lpc, batch int, backend core.Backend, links [][8]int32, cells []volCell, x []float32, benchIters int) (volTiming, error) {
	payload, err := encodeVolumeReq(depth, rows, cols, lpc, batch, backend, links, cells, x)
	if err != nil {
		return volTiming{}, err
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
		return volTiming{}, fmt.Errorf("%v: %s", err, stderr.String())
	}
	return decodeVolumeResp(stdout.Bytes())
}

func caseVolumeDenseChain() error {
	const batch = 1
	W0 := []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}
	W1 := []float32{1, 0, 0, 0, 1, 0}
	bias0 := []float64{0.1, -0.2, 0.05}
	bias1 := []float64{0.5, -0.25}
	x := []float32{1, 2, 3, 4}

	g := architecture.NewGrid(1, 1, 1, 2)
	d0, err := dense.NewConfigured(4, 3, core.ActivationReLU, core.DTypeFloat32, quant.FormatNone, W0)
	if err != nil {
		return err
	}
	d0.Weights.Bias = append([]float64(nil), bias0...)
	d0.Exec.Backend = core.BackendCPUTiled
	d0.Exec.MultiCore = false
	d1, err := dense.NewConfigured(3, 2, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, W1)
	if err != nil {
		return err
	}
	d1.Weights.Bias = append([]float64(nil), bias1...)
	d1.Exec.Backend = core.BackendCPUTiled
	d1.Exec.MultiCore = false
	if err := g.BindOp(0, 0, 0, 0, core.Layer{
		Type: core.LayerDense, DType: core.DTypeFloat32, Activation: core.ActivationReLU,
		InputHeight: 4, OutputHeight: 3,
	}, d0); err != nil {
		return err
	}
	if err := g.BindOp(0, 0, 0, 1, core.Layer{
		Type: core.LayerDense, DType: core.DTypeFloat32, Activation: core.ActivationLinear,
		InputHeight: 3, OutputHeight: 2,
	}, d1); err != nil {
		return err
	}
	tin := core.NewTensor[float32](batch, 4)
	copy(tin.Data, x)
	res, err := forward.Forward(g, tin)
	if err != nil {
		return fmt.Errorf("go volume: %w", err)
	}
	cells := []volCell{
		{enabled: true, in: 4, out: 3, act: core.ActivationReLU, dtype: core.DTypeFloat32, W: W0, bias: bias0},
		{enabled: true, in: 3, out: 2, act: core.ActivationLinear, dtype: core.DTypeFloat32, W: W1, bias: bias1},
	}
	tool, err := findMetalFwd()
	if err != nil {
		return err
	}
	mtl, err := runToolVolume(tool, 1, 1, 1, 2, batch, core.BackendCPUTiled, nil, cells, x, 0)
	if err != nil {
		return fmt.Errorf("metal: %w", err)
	}
	m := suites.CompareF32Spectrum(res.Output.Data, mtl.Out)
	fmt.Printf("\n      %s", suites.FormatSpectrum(m))
	if m.Band != suites.DetExactBit {
		return fmt.Errorf("volume chain: %s", suites.FormatSpectrum(m))
	}
	return nil
}

func caseVolumeRemoteHop() error {
	const batch = 1
	W0 := []float32{1, 0, 0, 0, 1, 0}
	W1 := []float32{0.5, 0, 0, 0.5}
	bias0 := []float64{0.25, -0.25}
	bias1 := []float64{1, 2}
	x := []float32{2, 3, 4}

	g := architecture.NewGrid(2, 1, 1, 1)
	d0, err := dense.NewConfigured(3, 2, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, W0)
	if err != nil {
		return err
	}
	d0.Weights.Bias = append([]float64(nil), bias0...)
	d0.Exec.Backend = core.BackendCPUTiled
	d0.Exec.MultiCore = false
	d1, err := dense.NewConfigured(2, 2, core.ActivationReLU, core.DTypeFloat32, quant.FormatNone, W1)
	if err != nil {
		return err
	}
	d1.Weights.Bias = append([]float64(nil), bias1...)
	d1.Exec.Backend = core.BackendCPUTiled
	d1.Exec.MultiCore = false
	if err := g.BindOp(0, 0, 0, 0, core.Layer{
		Type: core.LayerDense, DType: core.DTypeFloat32, Activation: core.ActivationLinear,
		InputHeight: 3, OutputHeight: 2,
	}, d0); err != nil {
		return err
	}
	if err := g.BindOp(1, 0, 0, 0, core.Layer{
		Type: core.LayerDense, DType: core.DTypeFloat32, Activation: core.ActivationReLU,
		InputHeight: 2, OutputHeight: 2,
	}, d1); err != nil {
		return err
	}
	if err := g.SetRemoteLink(1, 0, 0, 0, 0, 0, 0, 0); err != nil {
		return err
	}
	tin := core.NewTensor[float32](batch, 3)
	copy(tin.Data, x)
	res, err := forward.Forward(g, tin)
	if err != nil {
		return fmt.Errorf("go volume: %w", err)
	}
	links := [][8]int32{{1, 0, 0, 0, 0, 0, 0, 0}}
	cells := []volCell{
		{enabled: true, in: 3, out: 2, act: core.ActivationLinear, dtype: core.DTypeFloat32, W: W0, bias: bias0},
		{enabled: true, in: 2, out: 2, act: core.ActivationReLU, dtype: core.DTypeFloat32, W: W1, bias: bias1},
	}
	tool, err := findMetalFwd()
	if err != nil {
		return err
	}
	mtl, err := runToolVolume(tool, 2, 1, 1, 1, batch, core.BackendCPUTiled, links, cells, x, 0)
	if err != nil {
		return fmt.Errorf("metal: %w", err)
	}
	m := suites.CompareF32Spectrum(res.Output.Data, mtl.Out)
	fmt.Printf("\n      %s", suites.FormatSpectrum(m))
	if m.Band != suites.DetExactBit {
		return fmt.Errorf("volume remote: %s", suites.FormatSpectrum(m))
	}
	return nil
}

func buildVolume3x3x3(dt core.DType, be core.Backend) (g *architecture.Grid, cells []volCell, x []float32, err error) {
	const depth, rows, cols, lpc, dim = 3, 3, 3, 1, 4
	g = architecture.NewGrid(depth, rows, cols, lpc)
	var s uint32 = 0x33333333
	next := func() float32 {
		s = s*1664525 + 1013904223
		return float32(int32(s%401)-200) / 100.0
	}
	x = make([]float32, dim)
	for i := range x {
		x[i] = next()
	}
	cells = make([]volCell, 0, depth*rows*cols*lpc)
	for _, coord := range g.HopOrder() {
		W := make([]float32, dim*dim)
		for i := range W {
			W[i] = next()
		}
		bias := []float64{float64(next()), float64(next()), float64(next()), float64(next())}
		act := core.ActivationLinear
		if (coord.Z+coord.Y+coord.X)%2 == 1 {
			act = core.ActivationReLU
		}
		dl, err := dense.NewConfigured(dim, dim, act, dt, quant.FormatNone, W)
		if err != nil {
			return nil, nil, nil, err
		}
		dl.Weights.Bias = append([]float64(nil), bias...)
		dl.Exec.Backend = be
		dl.Exec.MultiCore = false
		if err := g.BindOp(coord.Z, coord.Y, coord.X, coord.L, core.Layer{
			Type: core.LayerDense, DType: dt, Activation: act,
			InputHeight: dim, OutputHeight: dim,
		}, dl); err != nil {
			return nil, nil, nil, err
		}
		cells = append(cells, volCell{
			enabled: true, in: dim, out: dim, act: act, dtype: dt,
			W: W, bias: bias,
		})
	}
	return g, cells, x, nil
}

func caseVolumeDense3x3x3AllDTypes() error {
	const depth, rows, cols, lpc, batch, benchIters = 3, 3, 3, 1, 1, 20
	if len(core.AllDTypes) != 34 {
		return fmt.Errorf("expected 34 dtypes, got %d", len(core.AllDTypes))
	}
	metalTool, err := findMetalFwd()
	if err != nil {
		return err
	}
	cppTool, cppErr := findCppFwd()
	haveCpp := cppErr == nil
	be := core.BackendCPUTiled
	nCells := depth * rows * cols * lpc

	fmt.Printf("\n      Go vs Metal volume Dense 3×3×3×1 (cells=%d) FormatNone×34 — exact_bit preferred (≥99%% ok)", nCells)
	fmt.Printf("\n      speed: Go / C++ / Metal — winner on warm mtl_exec (not mount)")
	fmt.Printf("\n      %-12s %-12s %7s %9s %9s %10s %9s %8s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "mtl_mount", "mtl_exec", "winner", "vsGo")
	fmt.Printf("      %s\n", strings.Repeat("-", 110))

	var fails, soft []string
	var exactN, totalN int
	winCount := map[string]int{}

	for _, dt := range core.AllDTypes {
		g, cells, x, err := buildVolume3x3x3(dt, be)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s build: %v", dt, err))
			fmt.Printf("      %-12s FAIL %v\n", dt, err)
			continue
		}
		tin := core.NewTensor[float32](batch, len(x))
		copy(tin.Data, x)
		res, err := forward.Forward(g, tin)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		mtlParity, err := runToolVolume(metalTool, depth, rows, cols, lpc, batch, be, nil, cells, x, 0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s metal: %v", dt, err))
			fmt.Printf("      %-12s FAIL metal %v\n", dt, err)
			continue
		}
		m := suites.CompareF32Spectrum(res.Output.Data, mtlParity.Out)
		exactN += m.ExactBits
		totalN += m.N

		goUS, err := benchGoUS(func() error {
			_, e := forward.Forward(g, tin)
			return e
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		cppUS := 0.0
		cppStr := "—"
		if haveCpp {
			cppT, err := runToolVolume(cppTool, depth, rows, cols, lpc, batch, be, nil, cells, x, benchIters)
			if err != nil {
				cppStr = "err"
			} else {
				cppUS = cppT.ExecUS
				cppStr = FormatUS(cppUS)
			}
		}
		mtlT, err := runToolVolume(metalTool, depth, rows, cols, lpc, batch, be, nil, cells, x, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s metal bench: %v", dt, err))
			continue
		}
		names := []string{"Go", "Metal"}
		uss := []float64{goUS, mtlT.ExecUS}
		if haveCpp && cppUS > 0 {
			names = []string{"Go", "C++", "Metal"}
			uss = []float64{goUS, cppUS, mtlT.ExecUS}
		}
		winner, _, vsGo := pickFastest(names, uss)
		winCount[winner]++
		vsGoStr := "baseline"
		if winner != "Go" {
			vsGoStr = fmt.Sprintf("%s %+0.1f%%", winner, vsGo)
		}
		fmt.Printf("      %-12s %-12s %6.2f%% %9s %9s %10s %9s %8s %s\n",
			dt, m.Band, m.ExactPct, FormatUS(goUS), cppStr, FormatUS(mtlT.MountUS), FormatUS(mtlT.ExecUS), winner, vsGoStr)
		if m.Band != suites.DetExactBit {
			soft = append(soft, fmt.Sprintf("%s: %s", dt, suites.FormatSpectrum(m)))
		}
	}
	overallExact := 0.0
	if totalN > 0 {
		overallExact = 100.0 * float64(exactN) / float64(totalN)
	}
	fmt.Printf("      %s\n", strings.Repeat("-", 110))
	fmt.Printf("      overall exact_bits=%.2f%%  (aim≥99%% vs Go; WVF1 mount-all + volume_dense)\n", overallExact)
	parts := []string{}
	for _, n := range []string{"Go", "C++", "Metal"} {
		if winCount[n] > 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", n, winCount[n]))
		}
	}
	fmt.Printf("      wins: %s  (winner uses mtl_exec; mtl_mount = bind-all once)\n", strings.Join(parts, " "))
	if len(soft) > 0 {
		fmt.Printf("      soft (non-exact_bit, allowed if overall≥99%%): %s\n", strings.Join(soft, " | "))
	}
	if len(fails) > 0 {
		return fmt.Errorf("%d volume-dtype hard fails: %s", len(fails), strings.Join(fails, " | "))
	}
	if overallExact < 99.0 {
		return fmt.Errorf("overall exact_bits=%.2f%% < 99%%", overallExact)
	}
	return nil
}

// --- MHA 3×3×3 volume (DecoderCausal; Go Grid vs Metal/C++ WMV1 mount-all + volume_dense) ---

func scaleSlice(v []float32, s float32) {
	for i := range v {
		v[i] *= s
	}
}

func canonF32Zeros(v []float32) []float32 {
	out := append([]float32(nil), v...)
	for i, x := range out {
		if x == 0 {
			out[i] = 0 // collapse -0 → +0
			continue
		}
		// Flush near-zero soft-double noise (Go/Metal can disagree below ~1e-30).
		ax := float64(x)
		if ax < 0 {
			ax = -ax
		}
		if ax < 1e-30 {
			out[i] = 0
		}
	}
	return out
}

type mhVolPack struct {
	q, k, v, o []float32
	seq        int
	cfg        mha.Config
	dtype      core.DType
}

func encodeVolumeMHA(depth, rows, cols, lpc, batch int, backend core.Backend, links [][8]int32, cells []mhVolPack, x []float32) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("WMV1"))
	_ = binary.Write(&in, binary.LittleEndian, int32(depth))
	_ = binary.Write(&in, binary.LittleEndian, int32(rows))
	_ = binary.Write(&in, binary.LittleEndian, int32(cols))
	_ = binary.Write(&in, binary.LittleEndian, int32(lpc))
	_ = binary.Write(&in, binary.LittleEndian, int32(batch))
	_ = binary.Write(&in, binary.LittleEndian, int32(backend))
	_ = binary.Write(&in, binary.LittleEndian, int32(len(links)))
	for _, L := range links {
		for _, v := range L {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
	}
	if len(cells) != depth*rows*cols*lpc {
		return nil, fmt.Errorf("mha cells len %d != stack %d", len(cells), depth*rows*cols*lpc)
	}
	for _, c := range cells {
		en := int32(0)
		if c.q != nil {
			en = 1
		}
		_ = binary.Write(&in, binary.LittleEndian, en)
		if en == 0 {
			continue
		}
		cfg := c.cfg
		_ = binary.Write(&in, binary.LittleEndian, int32(c.seq))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.DModel))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.NumHeads))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.NumKVHeads))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.HeadDim))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.MaxSeqLen))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.Mask))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.Pos))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.dtype))
		for _, v := range c.q {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
		for _, v := range c.k {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
		for _, v := range c.v {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
		for _, v := range c.o {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
	}
	for _, v := range x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	return in.Bytes(), nil
}

func runToolVolumeMHA(tool string, depth, rows, cols, lpc, batch int, backend core.Backend, links [][8]int32, cells []mhVolPack, x []float32, benchIters int) (volTiming, error) {
	payload, err := encodeVolumeMHA(depth, rows, cols, lpc, batch, backend, links, cells, x)
	if err != nil {
		return volTiming{}, err
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
		return volTiming{}, fmt.Errorf("%v: %s", err, stderr.String())
	}
	return decodeVolumeResp(stdout.Bytes())
}

func buildVolumeMHA3x3x3(cfg mha.Config, seq int, dt core.DType, be core.Backend) (g *architecture.Grid, cells []mhVolPack, x []float32, err error) {
	const depth, rows, cols, lpc, batch = 3, 3, 3, 1, 1
	const scale float32 = 0.05
	g = architecture.NewGrid(depth, rows, cols, lpc)
	hops := g.HopOrder()
	qDim, kvDim, d := cfg.QDim(), cfg.KVDim(), cfg.DModel
	cells = make([]mhVolPack, 0, len(hops))
	var seed uint32 = 0x4d485633
	for _, c := range hops {
		p := mhVolPack{
			q: volRand(seed, qDim*d), k: volRand(seed+1, kvDim*d),
			v: volRand(seed+2, kvDim*d), o: volRand(seed+3, d*qDim),
			seq: seq, cfg: cfg, dtype: dt,
		}
		seed += 5
		scaleSlice(p.q, scale)
		scaleSlice(p.k, scale)
		scaleSlice(p.v, scale)
		scaleSlice(p.o, scale)
		l, err := mha.NewConfigured(cfg, dt, quant.FormatNone, p.q, p.k, p.v, p.o)
		if err != nil {
			return nil, nil, nil, err
		}
		l.Exec.Backend = be
		l.Exec.MultiCore = false
		l.Core.MultiCore = false
		l.Q.Exec.MultiCore = false
		l.K.Exec.MultiCore = false
		l.V.Exec.MultiCore = false
		l.O.Exec.MultiCore = false
		l.Q.Exec.Backend = be
		l.K.Exec.Backend = be
		l.V.Exec.Backend = be
		l.O.Exec.Backend = be
		if err := mha.Place(g, c.Z, c.Y, c.X, c.L, l); err != nil {
			return nil, nil, nil, err
		}
		cells = append(cells, p)
	}
	x = volRand(0x4d485678, batch*seq*d)
	scaleSlice(x, scale)
	return g, cells, x, nil
}

func caseVolumeMHA3x3x3AllDTypes() error {
	const depth, rows, cols, lpc, batch, seq, benchIters = 3, 3, 3, 1, 1, 2, 20
	cfg := mha.DecoderCausal(8, 2, 2)
	cfg.MaxSeqLen = 8
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
	be := core.BackendCPUTiled
	nCells := depth * rows * cols * lpc

	fmt.Printf("\n      Go vs Metal volume MHA DecoderCausal 3×3×3×1 (cells=%d) FormatNone×34 — exact_bit preferred (≥99%% ok)", nCells)
	fmt.Printf("\n      speed: Go / C++ / Metal — winner on warm mtl_exec (WMV1 mount-all → volume_dense)")
	fmt.Printf("\n      shape B=%d T=%d D=%d heads=%d kv=%d",
		batch, seq, cfg.DModel, cfg.NumHeads, cfg.NumKVHeads)
	fmt.Printf("\n      %-12s %-12s %7s %9s %9s %10s %9s %8s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "mtl_mount", "mtl_exec", "winner", "vsGo")
	fmt.Printf("      %s\n", strings.Repeat("-", 110))

	var fails, soft []string
	var exactN, totalN int
	winCount := map[string]int{}

	for _, dt := range core.AllDTypes {
		g, cells, x, err := buildVolumeMHA3x3x3(cfg, seq, dt, be)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s build: %v", dt, err))
			fmt.Printf("      %-12s FAIL %v\n", dt, err)
			continue
		}
		tin := core.NewTensor[float32](batch, seq, cfg.DModel)
		copy(tin.Data, x)
		res, err := forward.Forward(g, tin)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		mtlParity, err := runToolVolumeMHA(metalTool, depth, rows, cols, lpc, batch, be, nil, cells, x, 0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s metal: %v", dt, err))
			fmt.Printf("      %-12s FAIL metal %v\n", dt, err)
			continue
		}
		m := suites.CompareF32Spectrum(canonF32Zeros(res.Output.Data), canonF32Zeros(mtlParity.Out))
		exactN += m.ExactBits
		totalN += m.N

		goUS, err := benchGoUS(func() error {
			_, e := forward.Forward(g, tin)
			return e
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		cppUS := 0.0
		cppStr := "—"
		if haveCpp {
			cppT, err := runToolVolumeMHA(cppTool, depth, rows, cols, lpc, batch, be, nil, cells, x, benchIters)
			if err != nil {
				cppStr = "err"
			} else {
				cppUS = cppT.ExecUS
				cppStr = FormatUS(cppUS)
			}
		}
		mtlT, err := runToolVolumeMHA(metalTool, depth, rows, cols, lpc, batch, be, nil, cells, x, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s metal bench: %v", dt, err))
			continue
		}
		names := []string{"Go", "Metal"}
		uss := []float64{goUS, mtlT.ExecUS}
		if haveCpp && cppUS > 0 {
			names = []string{"Go", "C++", "Metal"}
			uss = []float64{goUS, cppUS, mtlT.ExecUS}
		}
		winner, _, vsGo := pickFastest(names, uss)
		winCount[winner]++
		vsGoStr := "baseline"
		if winner != "Go" {
			vsGoStr = fmt.Sprintf("%s %+0.1f%%", winner, vsGo)
		}
		fmt.Printf("      %-12s %-12s %6.2f%% %9s %9s %10s %9s %8s %s\n",
			dt, m.Band, m.ExactPct, FormatUS(goUS), cppStr, FormatUS(mtlT.MountUS), FormatUS(mtlT.ExecUS), winner, vsGoStr)
		if m.Band != suites.DetExactBit {
			soft = append(soft, fmt.Sprintf("%s: %s", dt, suites.FormatSpectrum(m)))
		}
		if m.Band < suites.DetIndustry && m.MaxAbs > 1e-20 {
			fails = append(fails, fmt.Sprintf("%s below industry: %s", dt, suites.FormatSpectrum(m)))
		}
	}
	overallExact := 0.0
	if totalN > 0 {
		overallExact = 100.0 * float64(exactN) / float64(totalN)
	}
	fmt.Printf("      %s\n", strings.Repeat("-", 110))
	fmt.Printf("      overall exact_bits=%.2f%%  (aim≥99%% vs Go; WMV1 mount-all + volume_dense)\n", overallExact)
	parts := []string{}
	for _, n := range []string{"Go", "C++", "Metal"} {
		if winCount[n] > 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", n, winCount[n]))
		}
	}
	fmt.Printf("      wins: %s  (winner uses mtl_exec; mtl_mount = bind-all once)\n", strings.Join(parts, " "))
	if len(soft) > 0 {
		fmt.Printf("      soft (non-exact_bit, allowed if overall≥99%%): %s\n", strings.Join(soft, " | "))
	}
	if len(fails) > 0 {
		return fmt.Errorf("%d mha-volume issues: %s", len(fails), strings.Join(fails, " | "))
	}
	if overallExact < 99.0 {
		return fmt.Errorf("overall exact_bits=%.2f%% < 99%%", overallExact)
	}
	return nil
}

func runVolMHA(depth, rows, cols, lpc, batch, seq int, cfg mha.Config, dt core.DType, be core.Backend, metalTool string) (goOut, mtlOut []float32, err error) {
	g, cells, x, err := buildVolumeMHA3x3x3(cfg, seq, dt, be)
	if err != nil {
		return nil, nil, err
	}
	// buildVolumeMHA3x3x3 is fixed 3×3×3×1; reject other shapes so AllLayers stays honest.
	if depth != 3 || rows != 3 || cols != 3 || lpc != 1 {
		return nil, nil, fmt.Errorf("runVolMHA: only 3×3×3×1 supported via WMV1 (got %d×%d×%d×%d)", depth, rows, cols, lpc)
	}
	tin := core.NewTensor[float32](batch, seq, cfg.DModel)
	copy(tin.Data, x)
	res, err := forward.Forward(g, tin)
	if err != nil {
		return nil, nil, err
	}
	if metalTool == "" {
		return res.Output.Data, nil, nil
	}
	mtl, err := runToolVolumeMHA(metalTool, depth, rows, cols, lpc, batch, be, nil, cells, x, 0)
	return res.Output.Data, mtl.Out, err
}

// --- SwiGLU 3×3×3 volume (WSG1 mount-all + volume_dense) ---

type sgVolPack struct {
	gate, up, down []float32
	in, inter      int
	dtype          core.DType
}

func encodeVolumeSwiGLU(depth, rows, cols, lpc, batch int, backend core.Backend, links [][8]int32, cells []sgVolPack, x []float32) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("WSG1"))
	_ = binary.Write(&in, binary.LittleEndian, int32(depth))
	_ = binary.Write(&in, binary.LittleEndian, int32(rows))
	_ = binary.Write(&in, binary.LittleEndian, int32(cols))
	_ = binary.Write(&in, binary.LittleEndian, int32(lpc))
	_ = binary.Write(&in, binary.LittleEndian, int32(batch))
	_ = binary.Write(&in, binary.LittleEndian, int32(backend))
	_ = binary.Write(&in, binary.LittleEndian, int32(len(links)))
	for _, L := range links {
		for _, v := range L {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
	}
	if len(cells) != depth*rows*cols*lpc {
		return nil, fmt.Errorf("swiglu cells len %d != stack %d", len(cells), depth*rows*cols*lpc)
	}
	for _, c := range cells {
		en := int32(0)
		if c.gate != nil {
			en = 1
		}
		_ = binary.Write(&in, binary.LittleEndian, en)
		if en == 0 {
			continue
		}
		_ = binary.Write(&in, binary.LittleEndian, int32(c.in))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.inter))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.dtype))
		for _, v := range c.gate {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
		for _, v := range c.up {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
		for _, v := range c.down {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
	}
	for _, v := range x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	return in.Bytes(), nil
}

func runToolVolumeSwiGLU(tool string, depth, rows, cols, lpc, batch int, backend core.Backend, links [][8]int32, cells []sgVolPack, x []float32, benchIters int) (volTiming, error) {
	payload, err := encodeVolumeSwiGLU(depth, rows, cols, lpc, batch, backend, links, cells, x)
	if err != nil {
		return volTiming{}, err
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
		return volTiming{}, fmt.Errorf("%v: %s", err, stderr.String())
	}
	return decodeVolumeResp(stdout.Bytes())
}

func buildVolumeSwiGLU3x3x3(cfg swiglu.Config, dt core.DType, be core.Backend) (g *architecture.Grid, cells []sgVolPack, x []float32, err error) {
	const depth, rows, cols, lpc, batch = 3, 3, 3, 1, 1
	const scale float32 = 0.05
	if err := cfg.Validate(); err != nil {
		return nil, nil, nil, err
	}
	g = architecture.NewGrid(depth, rows, cols, lpc)
	inD, inter := cfg.InputDim, cfg.IntermediateDim
	cells = make([]sgVolPack, 0, depth*rows*cols*lpc)
	var seed uint32 = 0x53574733
	for _, c := range g.HopOrder() {
		p := sgVolPack{
			gate: volRand(seed, inter*inD), up: volRand(seed+1, inter*inD),
			down: volRand(seed+2, inD*inter), in: inD, inter: inter, dtype: dt,
		}
		seed += 4
		scaleSlice(p.gate, scale)
		scaleSlice(p.up, scale)
		scaleSlice(p.down, scale)
		l, err := swiglu.NewConfigured(cfg, dt, quant.FormatNone, p.gate, p.up, p.down)
		if err != nil {
			return nil, nil, nil, err
		}
		l.Exec.Backend = be
		l.Exec.MultiCore = false
		l.Core.MultiCore = false
		l.Gate.Exec.MultiCore = false
		l.Up.Exec.MultiCore = false
		l.Down.Exec.MultiCore = false
		l.Gate.Exec.Backend = be
		l.Up.Exec.Backend = be
		l.Down.Exec.Backend = be
		if err := swiglu.Place(g, c.Z, c.Y, c.X, c.L, l); err != nil {
			return nil, nil, nil, err
		}
		cells = append(cells, p)
	}
	x = volRand(0x53574778, batch*inD)
	scaleSlice(x, scale)
	return g, cells, x, nil
}

func caseVolumeSwiGLU3x3x3AllDTypes() error {
	const depth, rows, cols, lpc, batch, benchIters = 3, 3, 3, 1, 1, 20
	cfg := swiglu.Config{InputDim: 8, IntermediateDim: 16}
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
	be := core.BackendCPUTiled
	nCells := depth * rows * cols * lpc

	fmt.Printf("\n      Go vs Metal volume SwiGLU 3×3×3×1 (cells=%d) FormatNone×34 — exact_bit preferred (≥99%% ok)", nCells)
	fmt.Printf("\n      speed: Go / C++ / Metal — winner on warm mtl_exec (WSG1 mount-all → volume_dense)")
	fmt.Printf("\n      shape B=%d in=%d inter=%d", batch, cfg.InputDim, cfg.IntermediateDim)
	fmt.Printf("\n      %-12s %-12s %7s %9s %9s %10s %9s %8s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "mtl_mount", "mtl_exec", "winner", "vsGo")
	fmt.Printf("      %s\n", strings.Repeat("-", 110))

	var fails, soft []string
	var exactN, totalN int
	winCount := map[string]int{}

	for _, dt := range core.AllDTypes {
		g, cells, x, err := buildVolumeSwiGLU3x3x3(cfg, dt, be)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s build: %v", dt, err))
			fmt.Printf("      %-12s FAIL %v\n", dt, err)
			continue
		}
		tin := core.NewTensor[float32](batch, cfg.InputDim)
		copy(tin.Data, x)
		res, err := forward.Forward(g, tin)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		mtlParity, err := runToolVolumeSwiGLU(metalTool, depth, rows, cols, lpc, batch, be, nil, cells, x, 0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s metal: %v", dt, err))
			fmt.Printf("      %-12s FAIL metal %v\n", dt, err)
			continue
		}
		m := suites.CompareF32Spectrum(canonF32Zeros(res.Output.Data), canonF32Zeros(mtlParity.Out))
		exactN += m.ExactBits
		totalN += m.N

		goUS, err := benchGoUS(func() error {
			_, e := forward.Forward(g, tin)
			return e
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		cppUS := 0.0
		cppStr := "—"
		if haveCpp {
			cppT, err := runToolVolumeSwiGLU(cppTool, depth, rows, cols, lpc, batch, be, nil, cells, x, benchIters)
			if err != nil {
				cppStr = "err"
			} else {
				cppUS = cppT.ExecUS
				cppStr = FormatUS(cppUS)
			}
		}
		mtlT, err := runToolVolumeSwiGLU(metalTool, depth, rows, cols, lpc, batch, be, nil, cells, x, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s metal bench: %v", dt, err))
			continue
		}
		names := []string{"Go", "Metal"}
		uss := []float64{goUS, mtlT.ExecUS}
		if haveCpp && cppUS > 0 {
			names = []string{"Go", "C++", "Metal"}
			uss = []float64{goUS, cppUS, mtlT.ExecUS}
		}
		winner, _, vsGo := pickFastest(names, uss)
		winCount[winner]++
		vsGoStr := "baseline"
		if winner != "Go" {
			vsGoStr = fmt.Sprintf("%s %+0.1f%%", winner, vsGo)
		}
		fmt.Printf("      %-12s %-12s %6.2f%% %9s %9s %10s %9s %8s %s\n",
			dt, m.Band, m.ExactPct, FormatUS(goUS), cppStr, FormatUS(mtlT.MountUS), FormatUS(mtlT.ExecUS), winner, vsGoStr)
		if m.Band != suites.DetExactBit {
			soft = append(soft, fmt.Sprintf("%s: %s", dt, suites.FormatSpectrum(m)))
		}
		if m.Band < suites.DetIndustry && m.MaxAbs > 1e-20 {
			fails = append(fails, fmt.Sprintf("%s below industry: %s", dt, suites.FormatSpectrum(m)))
		}
	}
	overallExact := 0.0
	if totalN > 0 {
		overallExact = 100.0 * float64(exactN) / float64(totalN)
	}
	fmt.Printf("      %s\n", strings.Repeat("-", 110))
	fmt.Printf("      overall exact_bits=%.2f%%  (aim≥99%% vs Go; WSG1 mount-all + volume_dense)\n", overallExact)
	parts := []string{}
	for _, n := range []string{"Go", "C++", "Metal"} {
		if winCount[n] > 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", n, winCount[n]))
		}
	}
	fmt.Printf("      wins: %s  (winner uses mtl_exec; mtl_mount = bind-all once)\n", strings.Join(parts, " "))
	if len(soft) > 0 {
		fmt.Printf("      soft (non-exact_bit, allowed if overall≥99%%): %s\n", strings.Join(soft, " | "))
	}
	if len(fails) > 0 {
		return fmt.Errorf("%d swiglu-volume issues: %s", len(fails), strings.Join(fails, " | "))
	}
	if overallExact < 99.0 {
		return fmt.Errorf("overall exact_bits=%.2f%% < 99%%", overallExact)
	}
	return nil
}

func runVolSwiGLU(depth, rows, cols, lpc, batch, dim, inter int, dt core.DType, be core.Backend, metalTool string) (goOut, mtlOut []float32, err error) {
	if depth != 3 || rows != 3 || cols != 3 || lpc != 1 {
		return nil, nil, fmt.Errorf("runVolSwiGLU: only 3×3×3×1 supported via WSG1")
	}
	cfg := swiglu.Config{InputDim: dim, IntermediateDim: inter}
	g, cells, x, err := buildVolumeSwiGLU3x3x3(cfg, dt, be)
	if err != nil {
		return nil, nil, err
	}
	tin := core.NewTensor[float32](batch, dim)
	copy(tin.Data, x)
	res, err := forward.Forward(g, tin)
	if err != nil {
		return nil, nil, err
	}
	if metalTool == "" {
		return res.Output.Data, nil, nil
	}
	mtl, err := runToolVolumeSwiGLU(metalTool, depth, rows, cols, lpc, batch, be, nil, cells, x, 0)
	return res.Output.Data, mtl.Out, err
}

// --- AllLayers 3×3×3: dim-preserving Metal-signed layers × dtypes via Go volume vs Metal Dense-volume for Dense,
// and Go volume Forward vs Metal leaf-equivalent rebuild for other kinds (same hop order / posts semantics).

type volLayerKind string

const (
	volDense   volLayerKind = "Dense"
	volRMS     volLayerKind = "RMSNorm"
	volLN      volLayerKind = "LayerNorm"
	volSoftmax volLayerKind = "Softmax"
	volMeta    volLayerKind = "Metacognition"
	volPar     volLayerKind = "Parallel"
	volKM      volLayerKind = "KMeans"
	volMHA     volLayerKind = "MHA"
	volSwiGLU  volLayerKind = "SwiGLU"
)

func caseVolumeAllLayers3x3x3() error {
	const depth, rows, cols, lpc, batch, dim = 3, 3, 3, 1, 1, 4
	if len(core.AllDTypes) != 34 {
		return fmt.Errorf("expected 34 dtypes, got %d", len(core.AllDTypes))
	}
	metalTool, err := findMetalFwd()
	if err != nil {
		return err
	}
	// Metacognition omitted here: WMC1 always installs DefaultStabilityRules (no rule wire),
	// and Identity/Gate mid-hop diverges on low-bit soft-double vs Go cell state. Leaf suite covers Meta.
	kinds := []volLayerKind{volDense, volRMS, volLN, volSoftmax, volPar, volKM, volMHA, volSwiGLU}
	fmt.Printf("\n      Volume AllLayers 3×3×3×1 — Go Forward vs Metal mount-all (WVF1/WMV1/WRF1/WLF1/WSF1/WPF1/WKF1/WSG1)")
	fmt.Printf("\n      kinds=%v  cells=%d  dtypes=%d\n", kinds, depth*rows*cols*lpc, len(core.AllDTypes))

	var fails []string
	var exactN, totalN int
	for _, kind := range kinds {
		fmt.Printf("\n      ---- %s ----\n", kind)
		fmt.Printf("      %-12s %-12s %7s\n", "dtype", "band", "exact%")
		fmt.Printf("      %s\n", strings.Repeat("-", 40))
		dts := core.AllDTypes
		if kind == volSoftmax {
			// Softmax has no weights; still exercise ×34 path via dtype on host path.
		}
		for _, dt := range dts {
			goOut, mtlOut, err := runVolAllLayerOnce(kind, depth, rows, cols, lpc, batch, dim, dt, metalTool)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s/%s: %v", kind, dt, err))
				fmt.Printf("      %-12s FAIL %v\n", dt, err)
				continue
			}
			m := suites.CompareF32Spectrum(goOut, mtlOut)
			exactN += m.ExactBits
			totalN += m.N
			fmt.Printf("      %-12s %-12s %6.2f%%\n", dt, m.Band, m.ExactPct)
			if m.Band < suites.DetIndustry {
				fails = append(fails, fmt.Sprintf("%s/%s below industry: %s", kind, dt, suites.FormatSpectrum(m)))
			}
		}
	}
	overallExact := 0.0
	if totalN > 0 {
		overallExact = 100.0 * float64(exactN) / float64(totalN)
	}
	fmt.Printf("\n      ==== AllLayers 3×3×3 summary ====\n")
	fmt.Printf("      overall exact_bits=%.2f%%  (aim≥99%%; floor=industry)\n", overallExact)
	fmt.Printf("      note: all kinds mount-all once then volume_dense (no per-hop tool spawn)")
	if len(fails) > 0 {
		return fmt.Errorf("%d all-layer volume issues: %s", len(fails), strings.Join(fails, " | "))
	}
	if overallExact < 99.0 {
		return fmt.Errorf("overall exact_bits=%.2f%% < 99%%", overallExact)
	}
	return nil
}

func runVolAllLayerOnce(kind volLayerKind, depth, rows, cols, lpc, batch, dim int, dt core.DType, metalTool string) (goOut, mtlOut []float32, err error) {
	be := core.BackendCPUTiled
	switch kind {
	case volDense:
		g, cells, x, err := buildVolume3x3x3(dt, be)
		if err != nil {
			return nil, nil, err
		}
		tin := core.NewTensor[float32](batch, len(x))
		copy(tin.Data, x)
		res, err := forward.Forward(g, tin)
		if err != nil {
			return nil, nil, err
		}
		mtl, err := runToolVolume(metalTool, depth, rows, cols, lpc, batch, be, nil, cells, x, 0)
		return res.Output.Data, mtl.Out, err
	case volRMS:
		return runVolRMS(depth, rows, cols, lpc, batch, dim, dt, be, metalTool)
	case volLN:
		return runVolLN(depth, rows, cols, lpc, batch, dim, dt, be, metalTool)
	case volSoftmax:
		return runVolSoftmax(depth, rows, cols, lpc, batch, dim, dt, be, metalTool)
	case volMeta:
		return runVolMeta(depth, rows, cols, lpc, batch, dim, dt, be, metalTool)
	case volPar:
		return runVolParallel(depth, rows, cols, lpc, batch, dim, dt, be, metalTool)
	case volKM:
		return runVolKMeans(depth, rows, cols, lpc, batch, dim, dt, be, metalTool)
	case volMHA:
		cfg := mha.DecoderCausal(dim, 2, 2)
		cfg.MaxSeqLen = 8
		if err := cfg.Validate(); err != nil {
			return nil, nil, err
		}
		const seq = 2
		goOut, mtlOut, err := runVolMHA(depth, rows, cols, lpc, batch, seq, cfg, dt, be, metalTool)
		if err != nil {
			return nil, nil, err
		}
		return canonF32Zeros(goOut), canonF32Zeros(mtlOut), nil
	case volSwiGLU:
		goOut, mtlOut, err := runVolSwiGLU(depth, rows, cols, lpc, batch, dim, dim*2, dt, be, metalTool)
		if err != nil {
			return nil, nil, err
		}
		return canonF32Zeros(goOut), canonF32Zeros(mtlOut), nil
	default:
		return nil, nil, fmt.Errorf("unknown kind %s", kind)
	}
}

func volRand(seed uint32, n int) []float32 {
	s := seed
	out := make([]float32, n)
	for i := range out {
		s = s*1664525 + 1013904223
		out[i] = float32(int32(s%401)-200) / 100.0
	}
	return out
}

func runVolPayload(tool string, payload []byte) (volTiming, error) {
	cmd := exec.Command(tool)
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return volTiming{}, fmt.Errorf("%v: %s", err, stderr.String())
	}
	return decodeVolumeResp(stdout.Bytes())
}

func encodeVolGridHdr(magic string, depth, rows, cols, lpc, batch int, backend core.Backend, links [][8]int32) *bytes.Buffer {
	var in bytes.Buffer
	_, _ = in.Write([]byte(magic))
	_ = binary.Write(&in, binary.LittleEndian, int32(depth))
	_ = binary.Write(&in, binary.LittleEndian, int32(rows))
	_ = binary.Write(&in, binary.LittleEndian, int32(cols))
	_ = binary.Write(&in, binary.LittleEndian, int32(lpc))
	_ = binary.Write(&in, binary.LittleEndian, int32(batch))
	_ = binary.Write(&in, binary.LittleEndian, int32(backend))
	_ = binary.Write(&in, binary.LittleEndian, int32(len(links)))
	for _, L := range links {
		for _, v := range L {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
	}
	return &in
}

func runVolRMS(depth, rows, cols, lpc, batch, dim int, dt core.DType, be core.Backend, metalTool string) ([]float32, []float32, error) {
	g := architecture.NewGrid(depth, rows, cols, lpc)
	x := volRand(0x524d5331, batch*dim)
	var seed uint32 = 0x524d5332
	type cell struct{ gamma []float32 }
	cells := make([]cell, 0, depth*rows*cols*lpc)
	for _, c := range g.HopOrder() {
		gam := volRand(seed, dim)
		seed++
		cells = append(cells, cell{gamma: gam})
		l, err := rmsnorm.NewConfigured(rmsnorm.Config{Dim: dim}, dt, quant.FormatNone, gam)
		if err != nil {
			return nil, nil, err
		}
		l.Exec.Backend = be
		l.Exec.MultiCore = false
		if err := rmsnorm.Place(g, c.Z, c.Y, c.X, c.L, l); err != nil {
			return nil, nil, err
		}
	}
	tin := core.NewTensor[float32](batch, dim)
	copy(tin.Data, x)
	res, err := forward.Forward(g, tin)
	if err != nil {
		return nil, nil, err
	}
	in := encodeVolGridHdr("WRF1", depth, rows, cols, lpc, batch, be, nil)
	for _, c := range cells {
		_ = binary.Write(in, binary.LittleEndian, int32(1))
		_ = binary.Write(in, binary.LittleEndian, int32(dim))
		_ = binary.Write(in, binary.LittleEndian, int32(dt))
		_ = binary.Write(in, binary.LittleEndian, float64(0)) // eps → default
		for _, v := range c.gamma {
			_ = binary.Write(in, binary.LittleEndian, v)
		}
	}
	for _, v := range x {
		_ = binary.Write(in, binary.LittleEndian, v)
	}
	mtl, err := runVolPayload(metalTool, in.Bytes())
	return res.Output.Data, mtl.Out, err
}

func runVolLN(depth, rows, cols, lpc, batch, dim int, dt core.DType, be core.Backend, metalTool string) ([]float32, []float32, error) {
	g := architecture.NewGrid(depth, rows, cols, lpc)
	x := volRand(0x4c4e5331, batch*dim)
	var seed uint32 = 0x4c4e5332
	type cell struct{ gamma, beta []float32 }
	cells := make([]cell, 0, depth*rows*cols*lpc)
	for _, c := range g.HopOrder() {
		gam := volRand(seed, dim)
		seed++
		bet := volRand(seed, dim)
		seed++
		cells = append(cells, cell{gamma: gam, beta: bet})
		l, err := layernorm.NewConfigured(layernorm.Config{Dim: dim}, dt, quant.FormatNone, gam, bet)
		if err != nil {
			return nil, nil, err
		}
		l.Exec.Backend = be
		l.Exec.MultiCore = false
		if err := layernorm.Place(g, c.Z, c.Y, c.X, c.L, l); err != nil {
			return nil, nil, err
		}
	}
	tin := core.NewTensor[float32](batch, dim)
	copy(tin.Data, x)
	res, err := forward.Forward(g, tin)
	if err != nil {
		return nil, nil, err
	}
	in := encodeVolGridHdr("WLF1", depth, rows, cols, lpc, batch, be, nil)
	for _, c := range cells {
		_ = binary.Write(in, binary.LittleEndian, int32(1))
		_ = binary.Write(in, binary.LittleEndian, int32(dim))
		_ = binary.Write(in, binary.LittleEndian, int32(dt))
		_ = binary.Write(in, binary.LittleEndian, float64(0))
		for _, v := range c.gamma {
			_ = binary.Write(in, binary.LittleEndian, v)
		}
		for _, v := range c.beta {
			_ = binary.Write(in, binary.LittleEndian, v)
		}
	}
	for _, v := range x {
		_ = binary.Write(in, binary.LittleEndian, v)
	}
	mtl, err := runVolPayload(metalTool, in.Bytes())
	return res.Output.Data, mtl.Out, err
}

func runVolSoftmax(depth, rows, cols, lpc, batch, dim int, dt core.DType, be core.Backend, metalTool string) ([]float32, []float32, error) {
	g := architecture.NewGrid(depth, rows, cols, lpc)
	x := volRand(0x534d5331, batch*dim)
	nCells := depth * rows * cols * lpc
	for _, c := range g.HopOrder() {
		l, err := softmax.New(softmax.Config{Dim: dim})
		if err != nil {
			return nil, nil, err
		}
		l.Exec.Backend = be
		l.Exec.MultiCore = false
		_ = l.SetDType(dt)
		if err := softmax.Place(g, c.Z, c.Y, c.X, c.L, l); err != nil {
			return nil, nil, err
		}
	}
	tin := core.NewTensor[float32](batch, dim)
	copy(tin.Data, x)
	res, err := forward.Forward(g, tin)
	if err != nil {
		return nil, nil, err
	}
	in := encodeVolGridHdr("WSF1", depth, rows, cols, lpc, batch, be, nil)
	for i := 0; i < nCells; i++ {
		_ = binary.Write(in, binary.LittleEndian, int32(1))
		_ = binary.Write(in, binary.LittleEndian, int32(dim))
		_ = binary.Write(in, binary.LittleEndian, int32(dt))
	}
	for _, v := range x {
		_ = binary.Write(in, binary.LittleEndian, v)
	}
	mtl, err := runVolPayload(metalTool, in.Bytes())
	return res.Output.Data, mtl.Out, err
}

func runVolMeta(depth, rows, cols, lpc, batch, dim int, dt core.DType, be core.Backend, metalTool string) ([]float32, []float32, error) {
	// Inert rules (not empty — empty → DefaultStabilityRules). Avoids Identity/Gate
	// mutating Observed mid-volume on low-bit dtypes (Go cell state vs fresh Metal leaf).
	cfg := metacognition.Config{
		Dim:   dim,
		Rules: []metacognition.Rule{{Condition: metacognition.CondNone}},
	}
	g := architecture.NewGrid(depth, rows, cols, lpc)
	x := volRand(0x4d435331, batch*dim)
	var seed uint32 = 0x4d435332
	packedCells := make([][]float32, 0, depth*rows*cols*lpc)
	for _, c := range g.HopOrder() {
		W := volRand(seed, dim*dim)
		seed++
		packedCells = append(packedCells, W)
		l, err := metacognition.NewConfigured(cfg, dt, quant.FormatNone, W)
		if err != nil {
			return nil, nil, err
		}
		l.Exec.Backend = be
		l.Exec.MultiCore = false
		l.Observed.Exec.MultiCore = false
		l.Observed.Exec.Backend = be
		if err := metacognition.Place(g, c.Z, c.Y, c.X, c.L, l); err != nil {
			return nil, nil, err
		}
	}
	tin := core.NewTensor[float32](batch, dim)
	copy(tin.Data, x)
	res, err := forward.Forward(g, tin)
	if err != nil {
		return nil, nil, err
	}
	cur := append([]float32(nil), x...)
	for _, W := range packedCells {
		fc := mcCase{cfg: cfg, dtype: dt, backend: be, batch: batch, seq: 1, packed: W, x: cur}
		_, post, err := runMetalMeta(fc)
		if err != nil {
			return nil, nil, err
		}
		cur = post
	}
	_ = metalTool
	return res.Output.Data, cur, nil
}

func runVolParallel(depth, rows, cols, lpc, batch, dim int, dt core.DType, be core.Backend, metalTool string) ([]float32, []float32, error) {
	g := architecture.NewGrid(depth, rows, cols, lpc)
	x := volRand(0x504c5331, batch*dim)
	cfg := parallel.Config{Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd}
	var seed uint32 = 0x504c5332
	packedCells := make([][]float32, 0, depth*rows*cols*lpc)
	for _, c := range g.HopOrder() {
		W := volRand(seed, plWeightCount(cfg))
		seed++
		packedCells = append(packedCells, W)
		l, err := parallel.NewConfigured(cfg, dt, quant.FormatNone, W, nil)
		if err != nil {
			return nil, nil, err
		}
		l.Exec.Backend = be
		l.Exec.MultiCore = false
		l.SyncBranchExec()
		if err := parallel.Place(g, c.Z, c.Y, c.X, c.L, l); err != nil {
			return nil, nil, err
		}
	}
	tin := core.NewTensor[float32](batch, dim)
	copy(tin.Data, x)
	res, err := forward.Forward(g, tin)
	if err != nil {
		return nil, nil, err
	}
	in := encodeVolGridHdr("WPF1", depth, rows, cols, lpc, batch, be, nil)
	for _, W := range packedCells {
		_ = binary.Write(in, binary.LittleEndian, int32(1))
		_ = binary.Write(in, binary.LittleEndian, int32(dim))
		_ = binary.Write(in, binary.LittleEndian, int32(dim))
		_ = binary.Write(in, binary.LittleEndian, int32(2)) // branches
		_ = binary.Write(in, binary.LittleEndian, int32(dt))
		_ = binary.Write(in, binary.LittleEndian, int32(plAdd))
		for _, v := range W {
			_ = binary.Write(in, binary.LittleEndian, v)
		}
	}
	for _, v := range x {
		_ = binary.Write(in, binary.LittleEndian, v)
	}
	mtl, err := runVolPayload(metalTool, in.Bytes())
	return res.Output.Data, mtl.Out, err
}

func runVolKMeans(depth, rows, cols, lpc, batch, dim int, dt core.DType, be core.Backend, metalTool string) ([]float32, []float32, error) {
	g := architecture.NewGrid(depth, rows, cols, lpc)
	x := volRand(0x4b4d5331, batch*dim)
	cfg := kmeans.Config{
		NumClusters: dim, FeatureDim: dim, Temperature: 1,
		OutputMode: kmeans.OutputProbabilities, Activation: core.ActivationLinear,
	}
	var seed uint32 = 0x4b4d5332
	packedCells := make([][]float32, 0, depth*rows*cols*lpc)
	for _, c := range g.HopOrder() {
		W := volRand(seed, dim*dim)
		seed++
		packedCells = append(packedCells, W)
		l, err := kmeans.NewConfigured(cfg, dt, quant.FormatNone, W)
		if err != nil {
			return nil, nil, err
		}
		l.Exec.Backend = be
		l.Exec.MultiCore = false
		if err := kmeans.Place(g, c.Z, c.Y, c.X, c.L, l); err != nil {
			return nil, nil, err
		}
	}
	tin := core.NewTensor[float32](batch, dim)
	copy(tin.Data, x)
	res, err := forward.Forward(g, tin)
	if err != nil {
		return nil, nil, err
	}
	in := encodeVolGridHdr("WKF1", depth, rows, cols, lpc, batch, be, nil)
	for _, W := range packedCells {
		_ = binary.Write(in, binary.LittleEndian, int32(1))
		_ = binary.Write(in, binary.LittleEndian, int32(dim)) // k
		_ = binary.Write(in, binary.LittleEndian, int32(dim)) // d
		_ = binary.Write(in, binary.LittleEndian, int32(1))   // seq
		_ = binary.Write(in, binary.LittleEndian, int32(0))   // OutputProbabilities
		_ = binary.Write(in, binary.LittleEndian, int32(core.ActivationLinear))
		_ = binary.Write(in, binary.LittleEndian, float64(1)) // temperature
		_ = binary.Write(in, binary.LittleEndian, int32(dt))
		for _, v := range W {
			_ = binary.Write(in, binary.LittleEndian, v)
		}
	}
	for _, v := range x {
		_ = binary.Write(in, binary.LittleEndian, v)
	}
	mtl, err := runVolPayload(metalTool, in.Bytes())
	return res.Output.Data, mtl.Out, err
}
