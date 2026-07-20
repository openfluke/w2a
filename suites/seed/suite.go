package seed

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/stub/seed"
)

type Case struct {
	Name string
	Run  func() error
}

func Cases() []Case {
	return []Case{
		{Name: "SeedFrom deterministic", Run: fromDet},
		{Name: "He-init round-trip invert", Run: heInvert},
		{Name: "Dense manifest build+grid", Run: denseManifest},
		{Name: "Infinite dense override round-trip", Run: infiniteRT},
		{Name: "Infinite MHA manifest round-trip", Run: infiniteMHART},
		{Name: "Infinite SwiGLU manifest round-trip", Run: infiniteSwiGLURT},
		{Name: "Infinite CNN1 manifest round-trip", Run: infiniteCNN1RT},
		{Name: "InitGrid + GridFingerprint on mixed grid (mha/swiglu/cnn1)", Run: initGridMixed},
	}
}

func RunAll() error {
	var fails []string
	for i, c := range Cases() {
		suites.BeginCase()
		fmt.Printf("  [%d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("seed", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("seed", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("seed: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func RunOne(n int) error {
	cs := Cases()
	if n < 1 || n > len(cs) {
		return fmt.Errorf("seed: case %d out of range 1..%d", n, len(cs))
	}
	suites.BeginCase()
	fmt.Printf("  [%d] %s … ", n, cs[n-1].Name)
	if err := cs[n-1].Run(); err != nil {
		suites.EndCase("seed", cs[n-1].Name, "FAIL", err.Error())
		fmt.Printf("FAIL\n      %v\n", err)
		return err
	}
	suites.EndCase("seed", cs[n-1].Name, "PASS", "")
	fmt.Println("PASS")
	return nil
}

func fromDet() error {
	a := seed.From("welvet", 42, true)
	b := seed.SeedFrom("welvet", 42, true)
	if a != b {
		return fmt.Errorf("mismatch")
	}
	return nil
}

func heInvert() error {
	w := make([]float32, 16)
	seed.InitFloat32He(w, 4, 0xabc)
	got, ok := seed.FindLayerSeed(w, 4, 0xabc)
	if !ok || got != 0xabc {
		return fmt.Errorf("invert got %x ok=%v", got, ok)
	}
	return nil
}

func denseManifest() error {
	m, err := seed.BuildDense(99, []int{4, 4, 4}, nil)
	if err != nil {
		return err
	}
	g, err := seed.BuildDenseGrid(m)
	if err != nil {
		return err
	}
	if g.LayersPerCell != 2 {
		return fmt.Errorf("layers %d", g.LayersPerCell)
	}
	return nil
}

func infiniteRT() error {
	m, err := seed.BuildDense(7, []int{8, 8}, nil)
	if err != nil {
		return err
	}
	g, err := seed.BuildDenseGrid(m)
	if err != nil {
		return err
	}
	op := g.At(0, 0, 0, 0).Op.(*dense.Layer)
	w, _ := op.Weights.FlattenF32()
	w[0] += 0.5
	_ = op.Weights.SetFromF32(w)
	inf, err := seed.ManifestFromDense(op, m.Layers[0].LayerSeed)
	if err != nil {
		return err
	}
	if inf.OverrideCount() == 0 {
		return fmt.Errorf("expected overrides")
	}
	rebuilt, err := seed.BuildDenseFromInfinite(inf)
	if err != nil {
		return err
	}
	w2, _ := rebuilt.Weights.FlattenF32()
	if w2[0] != w[0] {
		return fmt.Errorf("override not applied")
	}
	return nil
}

func mhaCfg() mha.Config {
	return mha.Config{DModel: 8, NumHeads: 2, HeadDim: 4}
}

func infiniteMHART() error {
	l, err := mha.New(mhaCfg())
	if err != nil {
		return err
	}
	// Non-zero so the manifest carries real override chunks, not an all-zero baseline.
	qw, _ := l.Q.Weights.FlattenF32()
	for i := range qw {
		qw[i] = 0.1 * float32(i%5)
	}
	if err := l.Q.Weights.SetFromF32(qw); err != nil {
		return err
	}
	m, err := seed.ManifestFromMHA(l, 123)
	if err != nil {
		return err
	}
	if !m.IsComposite() || m.Kind != "mha" {
		return fmt.Errorf("bad manifest kind=%q composite=%v", m.Kind, m.IsComposite())
	}
	rebuilt, err := seed.BuildMHAFromInfinite(m, mhaCfg())
	if err != nil {
		return err
	}
	qw2, _ := rebuilt.Q.Weights.FlattenF32()
	for i := range qw {
		if qw[i] != qw2[i] {
			return fmt.Errorf("Q mismatch at %d: %v != %v", i, qw[i], qw2[i])
		}
	}
	return nil
}

func swigluCfg() swiglu.Config {
	return swiglu.Config{InputDim: 8, IntermediateDim: 6}
}

func infiniteSwiGLURT() error {
	l, err := swiglu.New(swigluCfg())
	if err != nil {
		return err
	}
	gw, _ := l.Gate.Weights.FlattenF32()
	for i := range gw {
		gw[i] = 0.05 * float32(i%7)
	}
	if err := l.Gate.Weights.SetFromF32(gw); err != nil {
		return err
	}
	m, err := seed.ManifestFromSwiGLU(l, 456)
	if err != nil {
		return err
	}
	rebuilt, err := seed.BuildSwiGLUFromInfinite(m, swigluCfg())
	if err != nil {
		return err
	}
	gw2, _ := rebuilt.Gate.Weights.FlattenF32()
	for i := range gw {
		if gw[i] != gw2[i] {
			return fmt.Errorf("Gate mismatch at %d: %v != %v", i, gw[i], gw2[i])
		}
	}
	return nil
}

func cnn1Cfg() cnn1.Config {
	return cnn1.Config{InChannels: 2, Filters: 4, SeqLen: 10, Kernel: 3, Activation: core.ActivationLinear}
}

func infiniteCNN1RT() error {
	l, err := cnn1.New(cnn1Cfg())
	if err != nil {
		return err
	}
	pw, _ := l.Proj.Weights.FlattenF32()
	for i := range pw {
		pw[i] = 0.02 * float32(i%9)
	}
	if err := l.Proj.Weights.SetFromF32(pw); err != nil {
		return err
	}
	m, err := seed.ManifestFromCNN1(l, 789)
	if err != nil {
		return err
	}
	rebuilt, err := seed.BuildCNN1FromInfinite(m, cnn1Cfg())
	if err != nil {
		return err
	}
	pw2, _ := rebuilt.Proj.Weights.FlattenF32()
	for i := range pw {
		if pw[i] != pw2[i] {
			return fmt.Errorf("Proj mismatch at %d: %v != %v", i, pw[i], pw2[i])
		}
	}
	return nil
}

func initGridMixed() error {
	g := architecture.NewGrid(1, 1, 1, 3)
	mhaL, err := mha.New(mhaCfg())
	if err != nil {
		return err
	}
	if err := mha.Place(g, 0, 0, 0, 0, mhaL); err != nil {
		return err
	}
	swigL, err := swiglu.New(swigluCfg())
	if err != nil {
		return err
	}
	if err := swiglu.Place(g, 0, 0, 0, 1, swigL); err != nil {
		return err
	}
	cnnL, err := cnn1.New(cnn1Cfg())
	if err != nil {
		return err
	}
	if err := cnn1.Place(g, 0, 0, 0, 2, cnnL); err != nil {
		return err
	}
	if err := seed.InitGrid(g, 0xf00d); err != nil {
		return err
	}
	fp1 := seed.GridFingerprint(g)
	if fp1 == 0 {
		return fmt.Errorf("fingerprint zero after init")
	}
	// Deterministic: re-init with the same seed reproduces the same weights/fingerprint.
	if err := seed.InitGrid(g, 0xf00d); err != nil {
		return err
	}
	fp2 := seed.GridFingerprint(g)
	if fp1 != fp2 {
		return fmt.Errorf("InitGrid not deterministic: %x != %x", fp1, fp2)
	}
	// A different seed must change at least one weight (fingerprint moves).
	if err := seed.InitGrid(g, 0xbeef); err != nil {
		return err
	}
	fp3 := seed.GridFingerprint(g)
	if fp3 == fp1 {
		return fmt.Errorf("InitGrid with different seed produced identical fingerprint")
	}
	return nil
}
