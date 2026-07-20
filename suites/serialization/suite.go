package serialization

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/stub/serialization"
	"github.com/openfluke/welvet/weights"
)

type Case struct {
	Name string
	Run  func() error
}

func Cases() []Case {
	return []Case{
		{Name: "Serialize/Deserialize Dense round-trip forward", Run: roundTripDense},
		{Name: "Native encode stable", Run: nativeStable},
		{Name: "FormatNone × all 34 dtypes bit-stable JSON", Run: allDTypesNative},
		{Name: "Packable quants bit-stable JSON (Dense)", Run: allQuantsNative},
		{Name: "Multi-layer Ops JSON round-trip", Run: multiLayerJSON},
		{Name: "ENTITY Save/Load Dense + forward", Run: entityDense},
		{Name: "ENTITY multi-layer bit-stable", Run: entityMulti},
	}
}

func RunAll() error {
	var fails []string
	for i, c := range Cases() {
		suites.BeginCase()
		fmt.Printf("  [%d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("serialization", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("serialization", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("serialization: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func RunOne(n int) error {
	cs := Cases()
	if n < 1 || n > len(cs) {
		return fmt.Errorf("serialization: case %d out of range 1..%d", n, len(cs))
	}
	suites.BeginCase()
	fmt.Printf("  [%d] %s … ", n, cs[n-1].Name)
	if err := cs[n-1].Run(); err != nil {
		suites.EndCase("serialization", cs[n-1].Name, "FAIL", err.Error())
		fmt.Printf("FAIL\n      %v\n", err)
		return err
	}
	suites.EndCase("serialization", cs[n-1].Name, "PASS", "")
	fmt.Println("PASS")
	return nil
}

func makeDenseGrid(dt core.DType, format quant.Format) (*architecture.Grid, error) {
	w := make([]float32, 16)
	for i := range w {
		w[i] = float32(i+1) * 0.01
	}
	l, err := dense.NewConfigured[float32](4, 4, core.ActivationLinear, dt, format, w)
	if err != nil {
		return nil, err
	}
	g := architecture.NewGrid(1, 1, 1, 1)
	return g, dense.Place(g, 0, 0, 0, 0, l)
}

func roundTripDense() error {
	g, err := makeDenseGrid(core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		return err
	}
	x := core.NewTensor[float32](1, 4)
	for i := range x.Data {
		x.Data[i] = 0.1
	}
	fwd1, err := forward.Forward(g, x)
	if err != nil {
		return err
	}
	raw, err := serialization.SerializeGrid(g)
	if err != nil {
		return err
	}
	g2, err := serialization.DeserializeGrid(raw)
	if err != nil {
		return err
	}
	fwd2, err := forward.Forward(g2, x)
	if err != nil {
		return err
	}
	for i := range fwd1.Output.Data {
		if fwd1.Output.Data[i] != fwd2.Output.Data[i] {
			return fmt.Errorf("forward mismatch at %d", i)
		}
	}
	return nil
}

func nativeStable() error {
	w := []float32{1, 2, 3, 4}
	a := serialization.EncodeF32LE(w)
	b := serialization.EncodeNativeWeightsRaw(w)
	if string(a) != string(b) {
		return fmt.Errorf("encode mismatch")
	}
	got, err := serialization.DecodeNativeF32(a, 4)
	if err != nil {
		return err
	}
	if !serialization.NativeWeightsEqual(w, got) {
		return fmt.Errorf("decode mismatch")
	}
	return nil
}

func snapshotEqual(a, b *weights.Store) error {
	sa, err := weights.TakeSnapshot(a)
	if err != nil {
		return err
	}
	sb, err := weights.TakeSnapshot(b)
	if err != nil {
		return err
	}
	if sa.DType != sb.DType || sa.Format != sb.Format || sa.Rows != sb.Rows || sa.Cols != sb.Cols {
		return fmt.Errorf("meta mismatch %v vs %v", sa, sb)
	}
	if sa.Scale != sb.Scale {
		return fmt.Errorf("scale %v vs %v", sa.Scale, sb.Scale)
	}
	if string(sa.Raw) != string(sb.Raw) {
		return fmt.Errorf("raw bytes differ (%d vs %d)", len(sa.Raw), len(sb.Raw))
	}
	return nil
}

func allDTypesNative() error {
	for _, dt := range core.AllDTypes {
		g, err := makeDenseGrid(dt, quant.FormatNone)
		if err != nil {
			return fmt.Errorf("%s: %w", dt, err)
		}
		raw, err := serialization.SerializeGrid(g)
		if err != nil {
			return fmt.Errorf("%s serialize: %w", dt, err)
		}
		g2, err := serialization.DeserializeGrid(raw)
		if err != nil {
			return fmt.Errorf("%s deserialize: %w", dt, err)
		}
		c1 := g.At(0, 0, 0, 0).Op.(*dense.Layer)
		c2 := g2.At(0, 0, 0, 0).Op.(*dense.Layer)
		if err := snapshotEqual(c1.Weights, c2.Weights); err != nil {
			return fmt.Errorf("%s: %w", dt, err)
		}
	}
	fmt.Printf("(%d dtypes) ", len(core.AllDTypes))
	return nil
}

func allQuantsNative() error {
	ok, gap := 0, 0
	for _, f := range serialization.PackableFormats() {
		if f == quant.FormatNone {
			continue
		}
		g, err := makeDenseGrid(core.DTypeFloat32, f)
		if err != nil {
			gap++
			continue
		}
		raw, err := serialization.SerializeGrid(g)
		if err != nil {
			return fmt.Errorf("%s serialize: %w", f, err)
		}
		g2, err := serialization.DeserializeGrid(raw)
		if err != nil {
			return fmt.Errorf("%s deserialize: %w", f, err)
		}
		c1 := g.At(0, 0, 0, 0).Op.(*dense.Layer)
		c2 := g2.At(0, 0, 0, 0).Op.(*dense.Layer)
		if err := snapshotEqual(c1.Weights, c2.Weights); err != nil {
			return fmt.Errorf("%s: %w", f, err)
		}
		ok++
	}
	fmt.Printf("(%d ok, %d gap) ", ok, gap)
	return nil
}

func multiLayerJSON() error {
	g := architecture.NewGrid(1, 1, 1, 6)
	d, err := dense.NewConfigured[float32](4, 4, core.ActivationReLU, core.DTypeFloat32, quant.FormatNone, ones(16))
	if err != nil {
		return err
	}
	if err := dense.Place(g, 0, 0, 0, 0, d); err != nil {
		return err
	}
	m, err := mha.New(mha.Config{DModel: 8, NumHeads: 2, MaxSeqLen: 4})
	if err != nil {
		return err
	}
	if err := mha.Place(g, 0, 0, 0, 1, m); err != nil {
		return err
	}
	sg, err := swiglu.New(swiglu.Config{InputDim: 8, IntermediateDim: 16})
	if err != nil {
		return err
	}
	if err := swiglu.Place(g, 0, 0, 0, 2, sg); err != nil {
		return err
	}
	rn, err := rmsnorm.New(rmsnorm.Config{Dim: 8})
	if err != nil {
		return err
	}
	if err := rmsnorm.Place(g, 0, 0, 0, 3, rn); err != nil {
		return err
	}
	ln, err := layernorm.New(layernorm.Config{Dim: 8})
	if err != nil {
		return err
	}
	if err := layernorm.Place(g, 0, 0, 0, 4, ln); err != nil {
		return err
	}
	sm, err := softmax.New(softmax.Config{Dim: 8, SeqLen: 1})
	if err != nil {
		return err
	}
	if err := softmax.Place(g, 0, 0, 0, 5, sm); err != nil {
		return err
	}
	raw, err := serialization.SerializeGrid(g)
	if err != nil {
		return err
	}
	g2, err := serialization.DeserializeGrid(raw)
	if err != nil {
		return err
	}
	raw2, err := serialization.SerializeGrid(g2)
	if err != nil {
		return err
	}
	if string(raw) != string(raw2) {
		return fmt.Errorf("JSON not bit-stable after reload")
	}
	return nil
}

func entityDense() error {
	g, err := makeDenseGrid(core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		return err
	}
	dir, err := os.MkdirTemp("", "welvet-entity-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "net.entity")
	if err := serialization.SaveEntity(path, g); err != nil {
		return err
	}
	g2, err := serialization.LoadEntity(path)
	if err != nil {
		return err
	}
	x := core.NewTensor[float32](1, 4)
	for i := range x.Data {
		x.Data[i] = 0.25
	}
	fwd1, err := forward.Forward(g, x)
	if err != nil {
		return err
	}
	fwd2, err := forward.Forward(g2, x)
	if err != nil {
		return err
	}
	for i := range fwd1.Output.Data {
		if fwd1.Output.Data[i] != fwd2.Output.Data[i] {
			return fmt.Errorf("entity forward mismatch at %d", i)
		}
	}
	return nil
}

func entityMulti() error {
	g := architecture.NewGrid(1, 1, 1, 3)
	d, err := dense.NewConfigured[float32](4, 4, core.ActivationLinear, core.DTypeInt8, quant.FormatNone, ones(16))
	if err != nil {
		return err
	}
	if err := dense.Place(g, 0, 0, 0, 0, d); err != nil {
		return err
	}
	c, err := cnn1.NewConfigured[float32](cnn1.Config{
		InChannels: 1, Filters: 2, SeqLen: 4, Kernel: 2, Stride: 1, Padding: 0,
		Activation: core.ActivationLinear,
	}, core.DTypeFloat32, quant.FormatQ8_0, ones(4))
	if err != nil {
		return err
	}
	if err := cnn1.Place(g, 0, 0, 0, 1, c); err != nil {
		return err
	}
	e, err := embedding.NewConfigured[float32](embedding.Config{
		VocabSize: 8, EmbeddingDim: 4, SeqLen: 2,
	}, core.DTypeFloat32, quant.FormatNone, ones(32))
	if err != nil {
		return err
	}
	if err := embedding.Place(g, 0, 0, 0, 2, e); err != nil {
		return err
	}
	wire, err := serialization.SerializeEntity(g)
	if err != nil {
		return err
	}
	g2, err := serialization.DeserializeEntity(wire)
	if err != nil {
		return err
	}
	wire2, err := serialization.SerializeEntity(g2)
	if err != nil {
		return err
	}
	if string(wire) != string(wire2) {
		return fmt.Errorf("ENTITY not bit-stable after reload")
	}
	return nil
}

func ones(n int) []float32 {
	w := make([]float32, n)
	for i := range w {
		w[i] = 0.1 + float32(i%7)*0.01
	}
	return w
}
