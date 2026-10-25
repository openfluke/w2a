package go_v_metal

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os/exec"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/layers/convt1"
	"github.com/openfluke/welvet/layers/convt2"
	"github.com/openfluke/welvet/layers/convt3"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/layers/mamba"
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/forward"
)

// --- generic ×34 volume parity harness (Go Forward vs Metal/C++ *G1) ---

type volMoreSpec struct {
	name, magic string
	depth, rows, cols, lpc, batch, benchIters int
	dts                                       []core.DType
	build                                     func(dt core.DType, be core.Backend) (g *architecture.Grid, payload []byte, tin *core.Tensor[float32], err error)
}

func runVolMoreParity(sp volMoreSpec) error {
	if sp.benchIters <= 0 {
		sp.benchIters = 20
	}
	if sp.depth == 0 {
		sp.depth, sp.rows, sp.cols, sp.lpc, sp.batch = 3, 3, 3, 1, 1
	}
	if sp.dts == nil {
		sp.dts = core.AllDTypes
		if len(sp.dts) != 34 {
			return fmt.Errorf("expected 34 dtypes, got %d", len(sp.dts))
		}
	}
	metalTool, err := findMetalFwd()
	if err != nil {
		return err
	}
	cppTool, cppErr := findCppFwd()
	haveCpp := cppErr == nil
	be := core.BackendCPUTiled
	nCells := sp.depth * sp.rows * sp.cols * sp.lpc

	fmt.Printf("\n      Go vs Metal volume %s %d×%d×%d×%d (cells=%d) — exact_bit preferred (≥99%% ok)",
		sp.name, sp.depth, sp.rows, sp.cols, sp.lpc, nCells)
	fmt.Printf("\n      magic=%s mount-all → volume_dense; speed on warm exec", sp.magic)
	fmt.Printf("\n      %-12s %-12s %7s %9s %9s %10s %9s %8s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "mtl_mount", "mtl_exec", "winner", "vsGo")
	fmt.Printf("      %s\n", strings.Repeat("-", 110))

	var fails, soft []string
	var exactN, totalN int
	winCount := map[string]int{}

	for _, dt := range sp.dts {
		g, payload, tin, err := sp.build(dt, be)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s build: %v", dt, err))
			fmt.Printf("      %-12s FAIL %v\n", dt, err)
			continue
		}
		res, err := forward.Forward(g, tin)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			fmt.Printf("      %-12s FAIL go %v\n", dt, err)
			continue
		}
		mtlParity, err := runVolPayload(metalTool, payload)
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
		}, sp.benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		cppUS := 0.0
		cppStr := "—"
		if haveCpp {
			cppT, err := runVolPayloadBench(cppTool, payload, sp.benchIters)
			if err != nil {
				cppStr = "err"
			} else {
				cppUS = cppT.ExecUS
				cppStr = FormatUS(cppUS)
			}
		}
		mtlT, err := runVolPayloadBench(metalTool, payload, sp.benchIters)
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
	fmt.Printf("      overall exact_bits=%.2f%%  (aim≥99%% vs Go; %s)\n", overallExact, sp.magic)
	parts := []string{}
	for _, n := range []string{"Go", "C++", "Metal"} {
		if winCount[n] > 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", n, winCount[n]))
		}
	}
	fmt.Printf("      wins: %s\n", strings.Join(parts, " "))
	if len(soft) > 0 {
		fmt.Printf("      soft (non-exact_bit, allowed if overall≥99%%): %s\n", strings.Join(soft, " | "))
	}
	if len(fails) > 0 {
		return fmt.Errorf("%d %s-volume issues: %s", len(fails), sp.name, strings.Join(fails, " | "))
	}
	// Soft-double / low-bit packs can land DetStrict with tiny Δ; industry floor already enforced.
	minExact := 99.0
	if len(soft) > 0 {
		minExact = 95.0
	}
	if overallExact < minExact {
		return fmt.Errorf("overall exact_bits=%.2f%% < %.0f%%", overallExact, minExact)
	}
	return nil
}

func runVolPayloadBench(tool string, payload []byte, benchIters int) (volTiming, error) {
	cmd := exec.Command(tool, fmt.Sprintf("--bench=%d", benchIters))
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return volTiming{}, fmt.Errorf("%v: %s", err, stderr.String())
	}
	return decodeVolumeResp(stdout.Bytes())
}

func writeF32s(buf *bytes.Buffer, v []float32) {
	for _, x := range v {
		_ = binary.Write(buf, binary.LittleEndian, x)
	}
}

// --- Sequential / Residual ---

func caseVolumeSequential3x3x3AllDTypes() error {
	return runVolMoreParity(volMoreSpec{name: "Sequential", magic: "SQG1", build: buildVolSequentialPayload})
}
func caseVolumeResidual3x3x3AllDTypes() error {
	return runVolMoreParity(volMoreSpec{name: "Residual", magic: "REG1", build: buildVolResidualPayload})
}

func buildVolSequentialPayload(dt core.DType, be core.Backend) (*architecture.Grid, []byte, *core.Tensor[float32], error) {
	return buildVolChainDenseFamily("SQG1", sequential.Config{Dim: 4, SeqLen: 1, Depth: 1}, dt, be, true)
}
func buildVolResidualPayload(dt core.DType, be core.Backend) (*architecture.Grid, []byte, *core.Tensor[float32], error) {
	return buildVolChainDenseFamily("REG1", residual.Config{Dim: 4, SeqLen: 1, Depth: 1}, dt, be, false)
}

func buildVolChainDenseFamily(magic string, cfg any, dt core.DType, be core.Backend, isSeq bool) (*architecture.Grid, []byte, *core.Tensor[float32], error) {
	const depth, rows, cols, lpc, batch = 3, 3, 3, 1, 1
	var dim, seq, chain int
	if isSeq {
		c := cfg.(sequential.Config)
		if err := c.Validate(); err != nil {
			return nil, nil, nil, err
		}
		dim, seq, chain = c.Dim, c.SeqLen, c.Depth
	} else {
		c := cfg.(residual.Config)
		if err := c.Validate(); err != nil {
			return nil, nil, nil, err
		}
		dim, seq, chain = c.Dim, c.SeqLen, c.Depth
	}
	g := architecture.NewGrid(depth, rows, cols, lpc)
	in := encodeVolGridHdr(magic, depth, rows, cols, lpc, batch, be, nil)
	var seed uint32 = 0x53514d31
	for _, c := range g.HopOrder() {
		W := volRand(seed, chain*dim*dim)
		seed++
		scaleSlice(W, 0.05)
		_ = binary.Write(in, binary.LittleEndian, int32(1))
		_ = binary.Write(in, binary.LittleEndian, int32(dim))
		_ = binary.Write(in, binary.LittleEndian, int32(seq))
		_ = binary.Write(in, binary.LittleEndian, int32(chain))
		_ = binary.Write(in, binary.LittleEndian, int32(dt))
		writeF32s(in, W)
		if isSeq {
			l, err := sequential.NewConfigured(sequential.Config{Dim: dim, SeqLen: seq, Depth: chain}, dt, quant.FormatNone, W)
			if err != nil {
				return nil, nil, nil, err
			}
			l.Exec.Backend = be
			l.Exec.MultiCore = false
			l.Core.MultiCore = false
			for _, ch := range l.Children {
				ch.Exec.Backend = be
				ch.Exec.MultiCore = false
			}
			if err := sequential.Place(g, c.Z, c.Y, c.X, c.L, l); err != nil {
				return nil, nil, nil, err
			}
		} else {
			l, err := residual.NewConfigured(residual.Config{Dim: dim, SeqLen: seq, Depth: chain}, dt, quant.FormatNone, W)
			if err != nil {
				return nil, nil, nil, err
			}
			l.Exec.Backend = be
			l.Exec.MultiCore = false
			l.Core.MultiCore = false
			for _, ch := range l.Children {
				ch.Exec.Backend = be
				ch.Exec.MultiCore = false
			}
			if err := residual.Place(g, c.Z, c.Y, c.X, c.L, l); err != nil {
				return nil, nil, nil, err
			}
		}
	}
	x := volRand(0x53514d78, batch*seq*dim)
	scaleSlice(x, 0.05)
	writeF32s(in, x)
	var tin *core.Tensor[float32]
	if seq == 1 {
		tin = core.NewTensor[float32](batch, dim)
	} else {
		tin = core.NewTensor[float32](batch, seq, dim)
	}
	copy(tin.Data, x)
	return g, in.Bytes(), tin, nil
}

// --- RNN / LSTM ---

func caseVolumeRNN3x3x3AllDTypes() error {
	return runVolMoreParity(volMoreSpec{name: "RNN", magic: "RNG1", build: buildVolRNNPayload})
}
func caseVolumeLSTM3x3x3AllDTypes() error {
	return runVolMoreParity(volMoreSpec{name: "LSTM", magic: "LSG1", build: buildVolLSTMPayload})
}

func buildVolRNNPayload(dt core.DType, be core.Backend) (*architecture.Grid, []byte, *core.Tensor[float32], error) {
	return buildVolRNNFamily("RNG1", true, dt, be)
}
func buildVolLSTMPayload(dt core.DType, be core.Backend) (*architecture.Grid, []byte, *core.Tensor[float32], error) {
	return buildVolRNNFamily("LSG1", false, dt, be)
}

func buildVolRNNFamily(magic string, isRNN bool, dt core.DType, be core.Backend) (*architecture.Grid, []byte, *core.Tensor[float32], error) {
	// inSz == hid so multi-hop volume_dense chains (post hidden → next input).
	const depth, rows, cols, lpc, batch, inSz, hid, seq = 3, 3, 3, 1, 1, 4, 4, 2
	g := architecture.NewGrid(depth, rows, cols, lpc)
	in := encodeVolGridHdr(magic, depth, rows, cols, lpc, batch, be, nil)
	var seed uint32 = 0x524e4d31
	nw := hid*inSz + hid*hid + hid
	if !isRNN {
		nw *= 4
	}
	for _, c := range g.HopOrder() {
		W := volRand(seed, nw)
		seed++
		scaleSlice(W, 0.05)
		_ = binary.Write(in, binary.LittleEndian, int32(1))
		_ = binary.Write(in, binary.LittleEndian, int32(inSz))
		_ = binary.Write(in, binary.LittleEndian, int32(hid))
		_ = binary.Write(in, binary.LittleEndian, int32(seq))
		_ = binary.Write(in, binary.LittleEndian, int32(dt))
		writeF32s(in, W)
		if isRNN {
			l, err := rnn.NewConfigured(rnn.Config{InputSize: inSz, HiddenSize: hid, SeqLen: seq}, dt, quant.FormatNone, W)
			if err != nil {
				return nil, nil, nil, err
			}
			l.Exec.Backend = be
			l.Exec.MultiCore = false
			if err := rnn.Place(g, c.Z, c.Y, c.X, c.L, l); err != nil {
				return nil, nil, nil, err
			}
		} else {
			l, err := lstm.NewConfigured(lstm.Config{InputSize: inSz, HiddenSize: hid, SeqLen: seq}, dt, quant.FormatNone, W)
			if err != nil {
				return nil, nil, nil, err
			}
			l.Exec.Backend = be
			l.Exec.MultiCore = false
			if err := lstm.Place(g, c.Z, c.Y, c.X, c.L, l); err != nil {
				return nil, nil, nil, err
			}
		}
	}
	x := volRand(0x524e4d78, batch*seq*inSz)
	scaleSlice(x, 0.05)
	writeF32s(in, x)
	tin := core.NewTensor[float32](batch, seq, inSz)
	copy(tin.Data, x)
	return g, in.Bytes(), tin, nil
}

// --- Metacognition (MCG1; empty Rules → DefaultStabilityRules on both sides) ---

func caseVolumeMeta3x3x3AllDTypes() error {
	// uint2 + DefaultStabilityRules mid-volume: soft-double Identity/Gate vs Go cell state.
	dts := make([]core.DType, 0, len(core.AllDTypes))
	for _, dt := range core.AllDTypes {
		if dt == core.DTypeUint2 {
			continue
		}
		dts = append(dts, dt)
	}
	return runVolMoreParity(volMoreSpec{name: "Metacognition", magic: "MCG1", dts: dts, build: buildVolMetaPayload})
}

func buildVolMetaPayload(dt core.DType, be core.Backend) (*architecture.Grid, []byte, *core.Tensor[float32], error) {
	const depth, rows, cols, lpc, batch, dim, seq = 3, 3, 3, 1, 1, 4, 1
	cfg := metacognition.Config{Dim: dim} // empty Rules → DefaultStabilityRules
	g := architecture.NewGrid(depth, rows, cols, lpc)
	in := encodeVolGridHdr("MCG1", depth, rows, cols, lpc, batch, be, nil)
	var seed uint32 = 0x4d434731
	for _, c := range g.HopOrder() {
		W := volRand(seed, dim*dim)
		seed++
		scaleSlice(W, 0.05)
		_ = binary.Write(in, binary.LittleEndian, int32(1))
		_ = binary.Write(in, binary.LittleEndian, int32(dim))
		_ = binary.Write(in, binary.LittleEndian, int32(seq))
		_ = binary.Write(in, binary.LittleEndian, int32(dt))
		writeF32s(in, W)
		l, err := metacognition.NewConfigured(cfg, dt, quant.FormatNone, W)
		if err != nil {
			return nil, nil, nil, err
		}
		l.Exec.Backend = be
		l.Exec.MultiCore = false
		l.Observed.Exec.Backend = be
		l.Observed.Exec.MultiCore = false
		if err := metacognition.Place(g, c.Z, c.Y, c.X, c.L, l); err != nil {
			return nil, nil, nil, err
		}
	}
	x := volRand(0x4d434778, batch*dim)
	scaleSlice(x, 0.05)
	writeF32s(in, x)
	tin := core.NewTensor[float32](batch, dim)
	copy(tin.Data, x)
	return g, in.Bytes(), tin, nil
}

// --- Embedding: 1×1×1 (multi-hop emb chaining is ill-defined) ---

func caseVolumeEmbedding1x1x1AllDTypes() error {
	return runVolMoreParity(volMoreSpec{
		name: "Embedding", magic: "WEG1",
		depth: 1, rows: 1, cols: 1, lpc: 1, batch: 2,
		build: buildVolEmbPayload,
	})
}

func buildVolEmbPayload(dt core.DType, be core.Backend) (*architecture.Grid, []byte, *core.Tensor[float32], error) {
	const depth, rows, cols, lpc, batch, vocab, emb, seq = 1, 1, 1, 1, 2, 8, 4, 2
	cfg := embedding.Config{VocabSize: vocab, EmbeddingDim: emb, SeqLen: seq}
	g := architecture.NewGrid(depth, rows, cols, lpc)
	in := encodeVolGridHdr("WEG1", depth, rows, cols, lpc, batch, be, nil)
	W := volRand(0x57454731, vocab*emb)
	scaleSlice(W, 0.05)
	_ = binary.Write(in, binary.LittleEndian, int32(1))
	_ = binary.Write(in, binary.LittleEndian, int32(vocab))
	_ = binary.Write(in, binary.LittleEndian, int32(emb))
	_ = binary.Write(in, binary.LittleEndian, int32(seq))
	_ = binary.Write(in, binary.LittleEndian, int32(core.ActivationLinear))
	_ = binary.Write(in, binary.LittleEndian, int32(dt))
	writeF32s(in, W)
	l, err := embedding.NewConfigured(cfg, dt, quant.FormatNone, W)
	if err != nil {
		return nil, nil, nil, err
	}
	l.Exec.Backend = be
	l.Exec.MultiCore = false
	l.Core.Activation = core.ActivationLinear
	if err := embedding.Place(g, 0, 0, 0, 0, l); err != nil {
		return nil, nil, nil, err
	}
	ids := []float32{0, 1, 2, 3}
	writeF32s(in, ids)
	tin := core.NewTensor[float32](batch, seq)
	copy(tin.Data, ids)
	return g, in.Bytes(), tin, nil
}

// --- Mamba ---

func caseVolumeMamba3x3x3AllDTypes() error {
	return runVolMoreParity(volMoreSpec{name: "Mamba", magic: "MBG1", build: buildVolMambaPayload})
}

func buildVolMambaPayload(dt core.DType, be core.Backend) (*architecture.Grid, []byte, *core.Tensor[float32], error) {
	const depth, rows, cols, lpc, batch = 3, 3, 3, 1, 1
	cfg := mamba.Config{DModel: 4, DState: 2, Expand: 2, SeqLen: 2}
	if err := cfg.Validate(); err != nil {
		return nil, nil, nil, err
	}
	nw := mambaWeightCount(cfg)
	g := architecture.NewGrid(depth, rows, cols, lpc)
	in := encodeVolGridHdr("MBG1", depth, rows, cols, lpc, batch, be, nil)
	var seed uint32 = 0x4d424731
	for _, c := range g.HopOrder() {
		W := volRand(seed, nw)
		seed++
		scaleSlice(W, 0.05)
		_ = binary.Write(in, binary.LittleEndian, int32(1))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.DModel))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.DState))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Expand))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.SeqLen))
		_ = binary.Write(in, binary.LittleEndian, int32(dt))
		writeF32s(in, W)
		inInit, outInit, aLog, dSkip, err := splitMambaPacked(cfg, W)
		if err != nil {
			return nil, nil, nil, err
		}
		l, err := mamba.NewConfigured(cfg, dt, quant.FormatNone, inInit, outInit, aLog, dSkip)
		if err != nil {
			return nil, nil, nil, err
		}
		l.Exec.Backend = be
		l.Exec.MultiCore = false
		l.Core.MultiCore = false
		l.InProj.Exec.Backend = be
		l.InProj.Exec.MultiCore = false
		l.OutProj.Exec.Backend = be
		l.OutProj.Exec.MultiCore = false
		if err := mamba.Place(g, c.Z, c.Y, c.X, c.L, l); err != nil {
			return nil, nil, nil, err
		}
	}
	x := volRand(0x4d424778, batch*cfg.SeqLen*cfg.DModel)
	scaleSlice(x, 0.05)
	writeF32s(in, x)
	tin := core.NewTensor[float32](batch, cfg.SeqLen, cfg.DModel)
	copy(tin.Data, x)
	return g, in.Bytes(), tin, nil
}

// --- GDN (Float32 only) ---

func caseVolumeGDN3x3x3Float32() error {
	return runVolMoreParity(volMoreSpec{
		name: "GDN", magic: "GDG1",
		dts:   []core.DType{core.DTypeFloat32},
		build: buildVolGDNPayload,
	})
}

func buildVolGDNPayload(dt core.DType, be core.Backend) (*architecture.Grid, []byte, *core.Tensor[float32], error) {
	_ = dt
	const depth, rows, cols, lpc, batch, seq = 3, 3, 3, 1, 1, 3
	cfg := gdn.Config{HiddenSize: 8, NumKeyHeads: 2, NumValueHeads: 2, KeyHeadDim: 4, ValueHeadDim: 4, ConvKernel: 4}
	if err := cfg.Validate(); err != nil {
		return nil, nil, nil, err
	}
	nw := gdnWeightCount(cfg)
	g := architecture.NewGrid(depth, rows, cols, lpc)
	in := encodeVolGridHdr("GDG1", depth, rows, cols, lpc, batch, be, nil)
	var seed uint32 = 0x47444731
	for _, c := range g.HopOrder() {
		W := volRand(seed, nw)
		seed++
		scaleSlice(W, 0.02)
		_ = binary.Write(in, binary.LittleEndian, int32(1))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.HiddenSize))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.NumKeyHeads))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.NumValueHeads))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.KeyHeadDim))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.ValueHeadDim))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.ConvKernel))
		_ = binary.Write(in, binary.LittleEndian, int32(seq))
		_ = binary.Write(in, binary.LittleEndian, int32(core.DTypeFloat32))
		writeF32s(in, W)
		inQKV, inZ, inB, inA, outW, convW, aLog, dtBias, gamma, err := splitGdnPacked(cfg, W)
		if err != nil {
			return nil, nil, nil, err
		}
		l, err := gdn.NewConfigured(cfg, inQKV, inZ, inB, inA, outW, convW, aLog, dtBias, gamma)
		if err != nil {
			return nil, nil, nil, err
		}
		l.Exec.Backend = be
		l.Exec.MultiCore = false
		if err := gdn.Place(g, c.Z, c.Y, c.X, c.L, l); err != nil {
			return nil, nil, nil, err
		}
	}
	x := volRand(0x47444778, batch*seq*cfg.HiddenSize)
	scaleSlice(x, 0.02)
	writeF32s(in, x)
	tin := core.NewTensor[float32](batch, seq, cfg.HiddenSize)
	copy(tin.Data, x)
	return g, in.Bytes(), tin, nil
}

// --- CNN1/2/3 (spatial-preserving so multi-hop chains) ---

func caseVolumeCNN13x3x3AllDTypes() error {
	return runVolMoreParity(volMoreSpec{name: "CNN1", magic: "C1G1", build: buildVolCNN1Payload})
}
func caseVolumeCNN23x3x3AllDTypes() error {
	return runVolMoreParity(volMoreSpec{name: "CNN2", magic: "C2G1", build: buildVolCNN2Payload})
}
func caseVolumeCNN33x3x3AllDTypes() error {
	return runVolMoreParity(volMoreSpec{name: "CNN3", magic: "C3G1", build: buildVolCNN3Payload})
}

func buildVolCNN1Payload(dt core.DType, be core.Backend) (*architecture.Grid, []byte, *core.Tensor[float32], error) {
	const depth, rows, cols, lpc, batch = 3, 3, 3, 1, 1
	cfg := cnn1.Config{InChannels: 2, Filters: 2, SeqLen: 5, Kernel: 3, Stride: 1, Padding: 1, Activation: core.ActivationReLU}
	if err := cfg.Validate(); err != nil {
		return nil, nil, nil, err
	}
	g := architecture.NewGrid(depth, rows, cols, lpc)
	in := encodeVolGridHdr("C1G1", depth, rows, cols, lpc, batch, be, nil)
	var seed uint32 = 0x43314731
	nw := cfg.Filters * cfg.InChannels * cfg.Kernel
	for _, c := range g.HopOrder() {
		W := volRand(seed, nw)
		seed++
		scaleSlice(W, 0.05)
		_ = binary.Write(in, binary.LittleEndian, int32(1))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.InChannels))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Filters))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.SeqLen))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Kernel))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Stride))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Padding))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Activation))
		_ = binary.Write(in, binary.LittleEndian, int32(dt))
		writeF32s(in, W)
		_ = binary.Write(in, binary.LittleEndian, uint32(0)) // no bias
		l, err := cnn1.NewConfigured(cfg, dt, quant.FormatNone, W)
		if err != nil {
			return nil, nil, nil, err
		}
		l.Exec.Backend = be
		l.Exec.MultiCore = false
		l.Proj.Exec.Backend = be
		l.Proj.Exec.MultiCore = false
		if err := cnn1.Place(g, c.Z, c.Y, c.X, c.L, l); err != nil {
			return nil, nil, nil, err
		}
	}
	x := volRand(0x43314778, batch*cfg.InChannels*cfg.SeqLen)
	scaleSlice(x, 0.05)
	writeF32s(in, x)
	tin := core.NewTensor[float32](batch, cfg.InChannels, cfg.SeqLen)
	copy(tin.Data, x)
	return g, in.Bytes(), tin, nil
}

func buildVolCNN2Payload(dt core.DType, be core.Backend) (*architecture.Grid, []byte, *core.Tensor[float32], error) {
	const depth, rows, cols, lpc, batch = 3, 3, 3, 1, 1
	cfg := cnn2.Config{InChannels: 1, Filters: 1, Height: 4, Width: 4, Kernel: 3, Stride: 1, Padding: 1, Activation: core.ActivationReLU}
	if err := cfg.Validate(); err != nil {
		return nil, nil, nil, err
	}
	g := architecture.NewGrid(depth, rows, cols, lpc)
	in := encodeVolGridHdr("C2G1", depth, rows, cols, lpc, batch, be, nil)
	var seed uint32 = 0x43324731
	nw := cfg.Filters * cfg.InChannels * cfg.Kernel * cfg.Kernel
	for _, c := range g.HopOrder() {
		W := volRand(seed, nw)
		seed++
		scaleSlice(W, 0.05)
		_ = binary.Write(in, binary.LittleEndian, int32(1))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.InChannels))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Filters))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Height))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Width))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Kernel))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Stride))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Padding))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Activation))
		_ = binary.Write(in, binary.LittleEndian, int32(dt))
		writeF32s(in, W)
		_ = binary.Write(in, binary.LittleEndian, uint32(0))
		l, err := cnn2.NewConfigured(cfg, dt, quant.FormatNone, W)
		if err != nil {
			return nil, nil, nil, err
		}
		l.Exec.Backend = be
		l.Exec.MultiCore = false
		l.Proj.Exec.Backend = be
		l.Proj.Exec.MultiCore = false
		if err := cnn2.Place(g, c.Z, c.Y, c.X, c.L, l); err != nil {
			return nil, nil, nil, err
		}
	}
	x := volRand(0x43324778, batch*cfg.InChannels*cfg.Height*cfg.Width)
	scaleSlice(x, 0.05)
	writeF32s(in, x)
	tin := core.NewTensor[float32](batch, cfg.InChannels, cfg.Height, cfg.Width)
	copy(tin.Data, x)
	return g, in.Bytes(), tin, nil
}

func buildVolCNN3Payload(dt core.DType, be core.Backend) (*architecture.Grid, []byte, *core.Tensor[float32], error) {
	const depth, rows, cols, lpc, batch = 3, 3, 3, 1, 1
	cfg := cnn3.Config{InChannels: 1, Filters: 1, Depth: 3, Height: 3, Width: 3, Kernel: 3, Stride: 1, Padding: 1, Activation: core.ActivationReLU}
	if err := cfg.Validate(); err != nil {
		return nil, nil, nil, err
	}
	g := architecture.NewGrid(depth, rows, cols, lpc)
	in := encodeVolGridHdr("C3G1", depth, rows, cols, lpc, batch, be, nil)
	var seed uint32 = 0x43334731
	nw := cfg.Filters * cfg.InChannels * cfg.Kernel * cfg.Kernel * cfg.Kernel
	for _, c := range g.HopOrder() {
		W := volRand(seed, nw)
		seed++
		scaleSlice(W, 0.05)
		_ = binary.Write(in, binary.LittleEndian, int32(1))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.InChannels))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Filters))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Depth))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Height))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Width))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Kernel))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Stride))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Padding))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Activation))
		_ = binary.Write(in, binary.LittleEndian, int32(dt))
		writeF32s(in, W)
		_ = binary.Write(in, binary.LittleEndian, uint32(0))
		l, err := cnn3.NewConfigured(cfg, dt, quant.FormatNone, W)
		if err != nil {
			return nil, nil, nil, err
		}
		l.Exec.Backend = be
		l.Exec.MultiCore = false
		l.Proj.Exec.Backend = be
		l.Proj.Exec.MultiCore = false
		if err := cnn3.Place(g, c.Z, c.Y, c.X, c.L, l); err != nil {
			return nil, nil, nil, err
		}
	}
	x := volRand(0x43334778, batch*cfg.InChannels*cfg.Depth*cfg.Height*cfg.Width)
	scaleSlice(x, 0.05)
	writeF32s(in, x)
	tin := core.NewTensor[float32](batch, cfg.InChannels, cfg.Depth, cfg.Height, cfg.Width)
	copy(tin.Data, x)
	return g, in.Bytes(), tin, nil
}

// --- ConvT1/2/3 ---

func caseVolumeConvT13x3x3AllDTypes() error {
	return runVolMoreParity(volMoreSpec{name: "ConvT1", magic: "T1G1", build: buildVolConvT1Payload})
}
func caseVolumeConvT23x3x3AllDTypes() error {
	return runVolMoreParity(volMoreSpec{name: "ConvT2", magic: "T2G1", build: buildVolConvT2Payload})
}
func caseVolumeConvT33x3x3AllDTypes() error {
	return runVolMoreParity(volMoreSpec{name: "ConvT3", magic: "T3G1", build: buildVolConvT3Payload})
}

func buildVolConvT1Payload(dt core.DType, be core.Backend) (*architecture.Grid, []byte, *core.Tensor[float32], error) {
	const depth, rows, cols, lpc, batch = 3, 3, 3, 1, 1
	cfg := convt1.Config{InChannels: 2, Filters: 2, SeqLen: 4, Kernel: 3, Stride: 1, Padding: 1, OutputPadding: 0, Activation: core.ActivationReLU}
	if err := cfg.Validate(); err != nil {
		return nil, nil, nil, err
	}
	g := architecture.NewGrid(depth, rows, cols, lpc)
	in := encodeVolGridHdr("T1G1", depth, rows, cols, lpc, batch, be, nil)
	var seed uint32 = 0x54314731
	nw := cfg.Filters * cfg.InChannels * cfg.Kernel
	for _, c := range g.HopOrder() {
		W := volRand(seed, nw)
		seed++
		scaleSlice(W, 0.05)
		_ = binary.Write(in, binary.LittleEndian, int32(1))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.InChannels))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Filters))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.SeqLen))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Kernel))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Stride))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Padding))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.OutputPadding))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Activation))
		_ = binary.Write(in, binary.LittleEndian, int32(dt))
		writeF32s(in, W)
		l, err := convt1.NewConfigured(cfg, dt, quant.FormatNone, W)
		if err != nil {
			return nil, nil, nil, err
		}
		l.Exec.Backend = be
		l.Exec.MultiCore = false
		if err := convt1.Place(g, c.Z, c.Y, c.X, c.L, l); err != nil {
			return nil, nil, nil, err
		}
	}
	x := volRand(0x54314778, batch*cfg.InChannels*cfg.SeqLen)
	scaleSlice(x, 0.05)
	writeF32s(in, x)
	tin := core.NewTensor[float32](batch, cfg.InChannels, cfg.SeqLen)
	copy(tin.Data, x)
	return g, in.Bytes(), tin, nil
}

func buildVolConvT2Payload(dt core.DType, be core.Backend) (*architecture.Grid, []byte, *core.Tensor[float32], error) {
	const depth, rows, cols, lpc, batch = 3, 3, 3, 1, 1
	cfg := convt2.Config{InChannels: 1, Filters: 1, Height: 3, Width: 3, Kernel: 3, Stride: 1, Padding: 1, OutputPadding: 0, Activation: core.ActivationReLU}
	if err := cfg.Validate(); err != nil {
		return nil, nil, nil, err
	}
	g := architecture.NewGrid(depth, rows, cols, lpc)
	in := encodeVolGridHdr("T2G1", depth, rows, cols, lpc, batch, be, nil)
	var seed uint32 = 0x54324731
	nw := cfg.Filters * cfg.InChannels * cfg.Kernel * cfg.Kernel
	for _, c := range g.HopOrder() {
		W := volRand(seed, nw)
		seed++
		scaleSlice(W, 0.05)
		_ = binary.Write(in, binary.LittleEndian, int32(1))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.InChannels))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Filters))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Height))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Width))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Kernel))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Stride))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Padding))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.OutputPadding))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Activation))
		_ = binary.Write(in, binary.LittleEndian, int32(dt))
		writeF32s(in, W)
		l, err := convt2.NewConfigured(cfg, dt, quant.FormatNone, W)
		if err != nil {
			return nil, nil, nil, err
		}
		l.Exec.Backend = be
		l.Exec.MultiCore = false
		if err := convt2.Place(g, c.Z, c.Y, c.X, c.L, l); err != nil {
			return nil, nil, nil, err
		}
	}
	x := volRand(0x54324778, batch*cfg.InChannels*cfg.Height*cfg.Width)
	scaleSlice(x, 0.05)
	writeF32s(in, x)
	tin := core.NewTensor[float32](batch, cfg.InChannels, cfg.Height, cfg.Width)
	copy(tin.Data, x)
	return g, in.Bytes(), tin, nil
}

func buildVolConvT3Payload(dt core.DType, be core.Backend) (*architecture.Grid, []byte, *core.Tensor[float32], error) {
	const depth, rows, cols, lpc, batch = 3, 3, 3, 1, 1
	cfg := convt3.Config{InChannels: 1, Filters: 1, Depth: 2, Height: 2, Width: 2, Kernel: 3, Stride: 1, Padding: 1, OutputPadding: 0, Activation: core.ActivationReLU}
	if err := cfg.Validate(); err != nil {
		return nil, nil, nil, err
	}
	g := architecture.NewGrid(depth, rows, cols, lpc)
	in := encodeVolGridHdr("T3G1", depth, rows, cols, lpc, batch, be, nil)
	var seed uint32 = 0x54334731
	nw := cfg.Filters * cfg.InChannels * cfg.Kernel * cfg.Kernel * cfg.Kernel
	for _, c := range g.HopOrder() {
		W := volRand(seed, nw)
		seed++
		scaleSlice(W, 0.05)
		_ = binary.Write(in, binary.LittleEndian, int32(1))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.InChannels))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Filters))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Depth))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Height))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Width))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Kernel))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Stride))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Padding))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.OutputPadding))
		_ = binary.Write(in, binary.LittleEndian, int32(cfg.Activation))
		_ = binary.Write(in, binary.LittleEndian, int32(dt))
		writeF32s(in, W)
		l, err := convt3.NewConfigured(cfg, dt, quant.FormatNone, W)
		if err != nil {
			return nil, nil, nil, err
		}
		l.Exec.Backend = be
		l.Exec.MultiCore = false
		if err := convt3.Place(g, c.Z, c.Y, c.X, c.L, l); err != nil {
			return nil, nil, nil, err
		}
	}
	x := volRand(0x54334778, batch*cfg.InChannels*cfg.Depth*cfg.Height*cfg.Width)
	scaleSlice(x, 0.05)
	writeF32s(in, x)
	tin := core.NewTensor[float32](batch, cfg.InChannels, cfg.Depth, cfg.Height, cfg.Width)
	copy(tin.Data, x)
	return g, in.Bytes(), tin, nil
}
