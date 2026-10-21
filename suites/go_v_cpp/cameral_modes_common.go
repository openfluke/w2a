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
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

// Shared Parallel Dense Add + BranchModes harness for Adversarial / Memory / Shadow.

type cameralModesOut struct {
	loss   float64
	WAfter [][]float32
}

func encodeCameralModes(magic string, batch, dim, outFeat, branches int, dtype core.DType,
	packed, x, target []float32, lr float64, modes []parallel.TrainMode,
	shadowCoef, surpriseThresh float64) []byte {
	return encodeCameralModesParent(magic, batch, dim, outFeat, branches, dtype, packed, x, target, lr,
		modes, shadowCoef, surpriseThresh, parallel.ModeNormalBP, false)
}

// encodeCameralModesParent — ADB1 shape + optional trailing parent_mode i32 (FZB1).
func encodeCameralModesParent(magic string, batch, dim, outFeat, branches int, dtype core.DType,
	packed, x, target []float32, lr float64, modes []parallel.TrainMode,
	shadowCoef, surpriseThresh float64, parent parallel.TrainMode, writeParent bool) []byte {
	var in bytes.Buffer
	_, _ = in.Write([]byte(magic))
	for _, v := range []int32{int32(batch), int32(dim), int32(outFeat), int32(branches), int32(dtype)} {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range packed {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range target {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	_ = binary.Write(&in, binary.LittleEndian, lr)
	for _, m := range modes {
		_ = binary.Write(&in, binary.LittleEndian, int32(m))
	}
	_ = binary.Write(&in, binary.LittleEndian, shadowCoef)
	_ = binary.Write(&in, binary.LittleEndian, surpriseThresh)
	if writeParent {
		_ = binary.Write(&in, binary.LittleEndian, int32(parent))
	}
	return in.Bytes()
}

func runCppCameralModes(payload []byte, outMagic string, benchIters int) (cameralModesOut, time.Duration, error) {
	var z cameralModesOut
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
		return z, 0, fmt.Errorf("cpp cameral: %v: %s", err, stderr.String())
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
	if len(raw) < 8 || string(raw[:4]) != outMagic {
		return z, bench, fmt.Errorf("cpp cameral bad magic want %s got %q", outMagic, string(raw[:min(4, len(raw))]))
	}
	r := bytes.NewReader(raw[4:])
	if err := binary.Read(r, binary.LittleEndian, &z.loss); err != nil {
		return z, bench, err
	}
	var nBranches int32
	if err := binary.Read(r, binary.LittleEndian, &nBranches); err != nil {
		return z, bench, err
	}
	z.WAfter = make([][]float32, nBranches)
	for i := 0; i < int(nBranches); i++ {
		var n int32
		if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
			return z, bench, err
		}
		v := make([]float32, n)
		for j := range v {
			if err := binary.Read(r, binary.LittleEndian, &v[j]); err != nil {
				return z, bench, err
			}
		}
		z.WAfter[i] = v
	}
	return z, bench, nil
}

func compareCameralModes(goO, cppO cameralModesOut, requireExact bool) error {
	if len(goO.WAfter) != len(cppO.WAfter) {
		return fmt.Errorf("n_branches go=%d cpp=%d", len(goO.WAfter), len(cppO.WAfter))
	}
	for i := range goO.WAfter {
		m := suites.CompareF32Spectrum(goO.WAfter[i], cppO.WAfter[i])
		if requireExact && m.Band != suites.DetExactBit {
			return fmt.Errorf("W[%d]: %s", i, suites.FormatSpectrum(m))
		}
		if !requireExact && (m.Band == suites.DetBroken || m.Band == suites.DetBallpark) {
			return fmt.Errorf("W[%d]: %s", i, suites.FormatSpectrum(m))
		}
	}
	if goO.loss != cppO.loss {
		rel := math.Abs(goO.loss-cppO.loss) / math.Max(1e-30, math.Abs(goO.loss))
		if requireExact && rel > 0 {
			return fmt.Errorf("loss go=%g cpp=%g", goO.loss, cppO.loss)
		}
		if !requireExact && rel > 1e-5 {
			return fmt.Errorf("loss go=%g cpp=%g", goO.loss, cppO.loss)
		}
	}
	return nil
}

func runGoCameralModes(dtype core.DType, batch, dim, outFeat, branches int,
	packed, x, target []float32, lr float64, modes []parallel.TrainMode,
	kit parallel.CamKit) (cameralModesOut, error) {
	return runGoCameralModesParent(dtype, batch, dim, outFeat, branches, packed, x, target, lr, modes,
		kit, parallel.ModeNormalBP)
}

func runGoCameralModesParent(dtype core.DType, batch, dim, outFeat, branches int,
	packed, x, target []float32, lr float64, modes []parallel.TrainMode,
	kit parallel.CamKit, parent parallel.TrainMode) (cameralModesOut, error) {
	var z cameralModesOut
	cfg := parallel.Config{Dim: dim, OutFeat: outFeat, Branches: branches, Combine: parallel.CombineAdd}
	l, err := parallel.NewConfigured[float32](cfg, dtype, quant.FormatNone, packed, nil)
	if err != nil {
		return z, err
	}
	if len(modes) > 0 {
		l.SetBranchModes(modes...)
	}
	l.SetCamKit(kit)
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	tgt := core.NewTensor[float32](batch, outFeat)
	copy(tgt.Data, target)
	loss, err := parallel.TrainMSE(l, xin, tgt, parent, lr)
	if err != nil {
		return z, err
	}
	z.loss = loss
	z.WAfter = make([][]float32, branches)
	for i := 0; i < branches; i++ {
		dl := l.Branches[i].(*dense.Layer)
		Wa, err := weights.MatrixF32(dl.Weights)
		if err != nil {
			return z, err
		}
		z.WAfter[i] = append([]float32(nil), Wa...)
	}
	return z, nil
}

func runCameralModesCase(magic, outMagic string, modes []parallel.TrainMode, kit parallel.CamKit,
	dtype core.DType, exact bool, benchIters int) error {
	return runCameralModesCaseParent(magic, outMagic, modes, kit, parallel.ModeNormalBP, false,
		dtype, exact, benchIters)
}

// runCameralModesCaseParent — FZB1-style: optional trailing parent_mode on wire.
func runCameralModesCaseParent(magic, outMagic string, modes []parallel.TrainMode, kit parallel.CamKit,
	parent parallel.TrainMode, writeParent bool, dtype core.DType, exact bool, benchIters int) error {
	const batch, dim, outFeat, branches = 2, 4, 4, 2
	const lr = 1e-2
	packed := randInit(1, branches*outFeat*dim, 0x43414d01)
	for i := range packed {
		packed[i] *= 0.05
	}
	x := randInit(batch, dim, 0x43414d02)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, batch*outFeat)
	for i := range target {
		target[i] = float32(0.05 * float64(i%outFeat+1))
	}
	goO, err := runGoCameralModesParent(dtype, batch, dim, outFeat, branches, packed, x, target, lr,
		modes, kit, parent)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload := encodeCameralModesParent(magic, batch, dim, outFeat, branches, dtype, packed, x, target, lr,
		modes, kit.ShadowCoef, kit.SurpriseThresh, parent, writeParent)
	if benchIters > 0 {
		_, _, err = runCppCameralModes(payload, outMagic, benchIters)
		if err != nil {
			return fmt.Errorf("cpp-bench: %w", err)
		}
	}
	cppO, _, err := runCppCameralModes(payload, outMagic, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareCameralModes(goO, cppO, exact)
}

func runCameralModesSpectrum(label, magic, outMagic string, modes []parallel.TrainMode, kit parallel.CamKit) error {
	return runCameralModesSpectrumParent(label, magic, outMagic, modes, kit, parallel.ModeNormalBP, false)
}

func runCameralModesSpectrumParent(label, magic, outMagic string, modes []parallel.TrainMode,
	kit parallel.CamKit, parent parallel.TrainMode, writeParent bool) error {
	const benchIters = 20
	fmt.Printf("\n      [%s cameral ×34]\n", label)
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s\n", "dtype", "band", "exact%W", "goµs", "cppµs", "spd%Δ")
	var fails []string
	for _, dt := range core.AllDTypes {
		t0 := time.Now()
		err := runCameralModesCaseParent(magic, outMagic, modes, kit, parent, writeParent, dt, true, 0)
		goDur := time.Since(t0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s:%v", dt, err))
			fmt.Printf("      %-12s %-12s %7s %10.2f %10s %8s  %v\n", dt, suites.DetBroken, "—",
				float64(goDur.Nanoseconds())/1000, "—", "—", err)
			continue
		}
		const batch, dim, outFeat, branches = 2, 4, 4, 2
		const lr = 1e-2
		packed := randInit(1, branches*outFeat*dim, 0x43414d01)
		for i := range packed {
			packed[i] *= 0.05
		}
		x := randInit(batch, dim, 0x43414d02)
		for i := range x {
			x[i] *= 0.05
		}
		target := make([]float32, batch*outFeat)
		for i := range target {
			target[i] = float32(0.05 * float64(i%outFeat+1))
		}
		payload := encodeCameralModesParent(magic, batch, dim, outFeat, branches, dt, packed, x, target, lr,
			modes, kit.ShadowCoef, kit.SurpriseThresh, parent, writeParent)
		_, cppDur, err := runCppCameralModes(payload, outMagic, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s bench:%v", dt, err))
			continue
		}
		goO, _ := runGoCameralModesParent(dt, batch, dim, outFeat, branches, packed, x, target, lr,
			modes, kit, parent)
		cppO, _, _ := runCppCameralModes(payload, outMagic, 0)
		m := suites.CompareF32Spectrum(goO.WAfter[0], cppO.WAfter[0])
		goUS := float64(goDur.Nanoseconds()) / 1000
		cppUS := float64(cppDur.Nanoseconds()) / float64(benchIters) / 1000
		spd := suites.SpeedPctDiff(goUS, cppUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %10.2f %10.2f %8.1f%%\n",
			dt, m.Band, m.ExactPct, goUS, cppUS, spd)
	}
	if len(fails) > 0 {
		return fmt.Errorf("dtype fails: %s", joinFails(fails))
	}
	return nil
}

func joinFails(fails []string) string {
	if len(fails) == 0 {
		return ""
	}
	s := fails[0]
	for i := 1; i < len(fails) && i < 8; i++ {
		s += "; " + fails[i]
	}
	if len(fails) > 8 {
		s += fmt.Sprintf("; …+%d", len(fails)-8)
	}
	return s
}
