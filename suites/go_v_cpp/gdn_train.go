package go_v_cpp

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
)

// CasesGDNTrain — NormalBP GDN Go↔C++ (leaf + volumetric). f32-only (no ×34).
func CasesGDNTrain() []Case {
	return []Case{
		{Name: "GDN NormalBP leaf Float32 — exact bits (dX/dW/W/loss)", Run: caseGdNBPLeafF32},
		{Name: "GDN NormalBP volume 1×1×1×1 Float32 — exact bits", Run: caseGdNBPVol1x1x1},
		{Name: "GDN NormalBP volume 1×1×1×2 chain Float32 — exact bits", Run: caseGdNBPVolChain},
		{Name: "GDN NormalBP volume 2×2×2×1 Float32 — exact bits", Run: caseGdNBPVol222F32},
	}
}

func RunGDNTrain() error {
	var fails []string
	cs := CasesGDNTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [gdnbp %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_gdn_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_gdn_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp gdn_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type gdNBPLeaf struct {
	batch  int
	cfg    gdn.Config
	W      []float32
	x      []float32
	target []float32
	lr     float64
}

type gdVolCell struct {
	enabled bool
	cfg     gdn.Config
	seq     int
	W       []float32
}

// Smoke dims match WGD1 forward suite (small hidden/heads).
func smokeGdCfg() gdn.Config {
	return gdn.Config{
		HiddenSize: 8, NumKeyHeads: 2, NumValueHeads: 4,
		KeyHeadDim: 4, ValueHeadDim: 4, ConvKernel: 4, Eps: 1e-6,
	}
}

func volGdCfg() gdn.Config {
	return gdn.Config{
		HiddenSize: 6, NumKeyHeads: 1, NumValueHeads: 2,
		KeyHeadDim: 4, ValueHeadDim: 4, ConvKernel: 4, Eps: 1e-6,
	}
}

func disableGDNTrain(l *gdn.Layer) {
	be := core.BackendCPUTiled
	l.Exec.Backend = be
	l.Exec.MultiCore = false
}

func packGDNWeightsF32(l *gdn.Layer) ([]float32, error) {
	if l == nil {
		return nil, fmt.Errorf("gdn: nil layer")
	}
	unpack := func(b *quant.Blob) ([]float32, error) {
		if b == nil {
			return nil, fmt.Errorf("gdn: nil blob")
		}
		return quant.Unpack(b)
	}
	inQKV, err := unpack(l.InQKV)
	if err != nil {
		return nil, err
	}
	inZ, err := unpack(l.InZ)
	if err != nil {
		return nil, err
	}
	inB, err := unpack(l.InB)
	if err != nil {
		return nil, err
	}
	inA, err := unpack(l.InA)
	if err != nil {
		return nil, err
	}
	outW, err := unpack(l.Out)
	if err != nil {
		return nil, err
	}
	out := make([]float32, 0, l.GradWSize())
	out = append(out, inQKV...)
	out = append(out, inZ...)
	out = append(out, inB...)
	out = append(out, inA...)
	out = append(out, outW...)
	out = append(out, l.ConvWeight...)
	out = append(out, l.ALog...)
	out = append(out, l.DtBias...)
	out = append(out, l.NormGamma...)
	return out, nil
}

func runGoGdNBP(fc gdNBPLeaf) (nbpLeafOut, error) {
	var z nbpLeafOut
	cfg := fc.cfg
	inQKV, inZ, inB, inA, outW, convW, aLog, dtBias, gamma, err := splitGdnPacked(cfg, fc.W)
	if err != nil {
		return z, err
	}
	l, err := gdn.NewConfigured(cfg, inQKV, inZ, inB, inA, outW, convW, aLog, dtBias, gamma)
	if err != nil {
		return z, err
	}
	disableGDNTrain(l)
	seq := len(fc.x) / (fc.batch * cfg.HiddenSize)
	xin := core.NewTensor[float32](fc.batch, seq, cfg.HiddenSize)
	copy(xin.Data, fc.x)
	preT, postT, err := gdn.Forward(l, xin)
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
	dX, dW, err := gdn.Backward(l, gy, xin, preT)
	if err != nil {
		return z, err
	}
	if fc.lr != 0 {
		if err := gdn.ApplyGradSGD(l, dW, fc.lr); err != nil {
			return z, err
		}
	}
	wa, err := packGDNWeightsF32(l)
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

func encodeGdNBP(fc gdNBPLeaf) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	seq := len(fc.x) / (fc.batch * cfg.HiddenSize)
	var in bytes.Buffer
	_, _ = in.Write([]byte("GDB1"))
	for _, v := range []int32{
		int32(fc.batch), int32(seq), int32(cfg.HiddenSize),
		int32(cfg.NumKeyHeads), int32(cfg.NumValueHeads),
		int32(cfg.KeyHeadDim), int32(cfg.ValueHeadDim), int32(cfg.ConvKernel),
		int32(core.DTypeFloat32), 0,
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
	return in.Bytes(), nil
}

func decodeGdNBP(raw []byte) (nbpLeafOut, time.Duration, error) {
	return decodeRecNBPLeaf(raw, "GDBO")
}

func runCppGdNBP(fc gdNBPLeaf) (nbpLeafOut, error) {
	out, _, err := runCppGdNBPBench(fc, 0)
	return out, err
}

func runCppGdNBPBench(fc gdNBPLeaf, benchIters int) (nbpLeafOut, time.Duration, error) {
	tool, err := findDenseFwd()
	if err != nil {
		return nbpLeafOut{}, 0, err
	}
	payload, err := encodeGdNBP(fc)
	if err != nil {
		return nbpLeafOut{}, 0, err
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
		return nbpLeafOut{}, 0, fmt.Errorf("cpp gdn nbp: %v: %s", err, stderr.String())
	}
	return decodeGdNBP(stdout.Bytes())
}

func makeGdLeafCase(cfg gdn.Config, batch, seq int, seed uint32) (gdNBPLeaf, error) {
	if err := cfg.Validate(); err != nil {
		return gdNBPLeaf{}, err
	}
	fc := gdNBPLeaf{
		batch: batch, cfg: cfg, lr: 1e-2,
		W: randInit(1, gdnWeightCount(cfg), seed),
		x: randInit(batch*seq, cfg.HiddenSize, seed^0x9e3779b9),
	}
	fc.target = make([]float32, batch*seq*cfg.HiddenSize)
	for i := range fc.target {
		fc.target[i] = float32(0.05 * float64(i%7+1))
	}
	return fc, nil
}

func caseGdNBPLeafF32() error {
	const benchIters = 40
	cfg := smokeGdCfg()
	fc, err := makeGdLeafCase(cfg, 2, 3, 0x47444250)
	if err != nil {
		return err
	}
	goO, err := runGoGdNBP(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppO, err := runCppGdNBP(fc)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareNBP(goO, cppO, true); err != nil {
		return err
	}
	for i := 0; i < 3; i++ {
		if _, err := runGoGdNBP(fc); err != nil {
			return err
		}
	}
	t0 := time.Now()
	for i := 0; i < benchIters; i++ {
		if _, err := runGoGdNBP(fc); err != nil {
			return err
		}
	}
	goDur := time.Since(t0)
	_, cppDur, err := runCppGdNBPBench(fc, benchIters)
	if err != nil {
		return fmt.Errorf("cpp bench: %w", err)
	}
	goUS := float64(goDur.Nanoseconds()) / float64(benchIters) / 1000
	cppUS := float64(cppDur.Nanoseconds()) / float64(benchIters) / 1000
	spdPct := suites.SpeedPctDiff(goUS, cppUS)
	fmt.Printf("\n      loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%  (neg ⇒ C++ faster)",
		goO.loss, goUS, cppUS, spdPct)
	return nil
}

func runGoGdVolNBP(depth, rows, cols, lpc, batch int, cells []gdVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
	hops := g.HopOrder()
	if len(cells) != len(hops) {
		return z, fmt.Errorf("cells %d != hops %d", len(cells), len(hops))
	}
	firstH, firstSeq := -1, -1
	for i, c := range hops {
		vc := cells[i]
		if !vc.enabled {
			cell := g.At(c.Z, c.Y, c.X, c.L)
			if cell != nil {
				cell.Layer.IsDisabled = true
			}
			continue
		}
		if firstH < 0 {
			firstH, firstSeq = vc.cfg.HiddenSize, vc.seq
		}
		inQKV, inZ, inB, inA, outW, convW, aLog, dtBias, gamma, err := splitGdnPacked(vc.cfg, vc.W)
		if err != nil {
			return z, err
		}
		gl, err := gdn.NewConfigured(vc.cfg, inQKV, inZ, inB, inA, outW, convW, aLog, dtBias, gamma)
		if err != nil {
			return z, err
		}
		disableGDNTrain(gl)
		if err := gdn.Place(g, c.Z, c.Y, c.X, c.L, gl); err != nil {
			return z, err
		}
	}
	xin := core.NewTensor[float32](batch, firstSeq, firstH)
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
		gl, ok := cell.Op.(*gdn.Layer)
		if !ok {
			return z, fmt.Errorf("expected gdn at %v", c)
		}
		wa, err := packGDNWeightsF32(gl)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, wa)
	}
	return z, nil
}

func encodeGdVolNBP(depth, rows, cols, lpc, batch int, cells []gdVolCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("GDV1"))
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
		cfg := c.cfg
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.HiddenSize))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.NumKeyHeads))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.NumValueHeads))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.KeyHeadDim))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.ValueHeadDim))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.ConvKernel))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.seq))
		_ = binary.Write(&in, binary.LittleEndian, int32(core.DTypeFloat32))
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

func decodeGdVolNBP(raw []byte) (nbpVolOut, time.Duration, error) {
	return decodeRecVolNBP(raw, "GDVO")
}

func runCppGdVolNBP(payload []byte, benchIters int) (nbpVolOut, time.Duration, error) {
	tool, err := findDenseFwd()
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
		return nbpVolOut{}, 0, fmt.Errorf("cpp gdn vol nbp: %v: %s", err, stderr.String())
	}
	return decodeGdVolNBP(stdout.Bytes())
}

func benchGdVolNBP(depth, rows, cols, lpc, batch int, cells []gdVolCell, x, target []float32, lr float64, benchIters int) (goUS, cppUS, spdPct float64, err error) {
	payload, err := encodeGdVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return 0, 0, 0, err
	}
	for i := 0; i < 3; i++ {
		if _, err := runGoGdVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr); err != nil {
			return 0, 0, 0, err
		}
	}
	t0 := time.Now()
	for i := 0; i < benchIters; i++ {
		if _, err := runGoGdVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr); err != nil {
			return 0, 0, 0, err
		}
	}
	goDur := time.Since(t0)
	_, cppDur, err := runCppGdVolNBP(payload, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	goUS = float64(goDur.Nanoseconds()) / float64(benchIters) / 1000
	cppUS = float64(cppDur.Nanoseconds()) / float64(benchIters) / 1000
	spdPct = suites.SpeedPctDiff(goUS, cppUS)
	return goUS, cppUS, spdPct, nil
}

func gdVolPostLen(batch, seq int, cfg gdn.Config) int {
	return batch * seq * cfg.HiddenSize
}

func caseGdNBPVol1x1x1() error {
	const batch, seq, benchIters, lr = 2, 2, 40, 1e-2
	cfg := volGdCfg()
	W := randInit(1, gdnWeightCount(cfg), 0x47445631)
	cells := []gdVolCell{{
		enabled: true, cfg: cfg, seq: seq, W: W,
	}}
	x := randInit(batch*seq, cfg.HiddenSize, 0x47445632)
	target := make([]float32, gdVolPostLen(batch, seq, cfg))
	for i := range target {
		target[i] = float32(0.03 * float64(i%4+1))
	}
	goO, err := runGoGdVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, _ := encodeGdVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	cppO, _, err := runCppGdVolNBP(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchGdVolNBP(1, 1, 1, 1, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×1 loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
		goO.loss, goUS, cppUS, spdPct)
	return nil
}

func caseGdNBPVolChain() error {
	const batch, seq, benchIters, lr = 1, 2, 40, 1e-2
	cfg := volGdCfg()
	W0 := randInit(1, gdnWeightCount(cfg), 0x47445633)
	W1 := randInit(1, gdnWeightCount(cfg), 0x47445634)
	cells := []gdVolCell{
		{enabled: true, cfg: cfg, seq: seq, W: W0},
		{enabled: true, cfg: cfg, seq: seq, W: W1},
	}
	x := randInit(batch*seq, cfg.HiddenSize, 0x47445635)
	target := make([]float32, gdVolPostLen(batch, seq, cfg))
	for i := range target {
		target[i] = float32(0.02 * float64(i%5+1))
	}
	goO, err := runGoGdVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, _ := encodeGdVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	cppO, _, err := runCppGdVolNBP(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchGdVolNBP(1, 1, 1, 2, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×2 loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
		goO.loss, goUS, cppUS, spdPct)
	return nil
}

func caseGdNBPVol222F32() error {
	const depth, rows, cols, lpc, batch, seq, benchIters = 2, 2, 2, 1, 1, 2, 20
	stack := depth * rows * cols * lpc
	const lr = 1e-2
	cfg := volGdCfg()
	W := randInit(1, gdnWeightCount(cfg), 0x47445638)
	for i := range W {
		W[i] *= 0.05
	}
	x := randInit(batch*seq, cfg.HiddenSize, 0x47445639)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, gdVolPostLen(batch, seq, cfg))
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := make([]gdVolCell, stack)
	for i := range cells {
		cells[i] = gdVolCell{
			enabled: true, cfg: cfg, seq: seq, W: append([]float32(nil), W...),
		}
	}
	goO, err := runGoGdVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, _ := encodeGdVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	cppO, _, err := runCppGdVolNBP(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchGdVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      2×2×2×1 loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
		goO.loss, goUS, cppUS, spdPct)
	return nil
}
