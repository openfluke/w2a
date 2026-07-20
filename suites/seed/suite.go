package seed

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/layers/dense"
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
