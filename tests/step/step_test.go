package step_test

import (
	"testing"

	"github.com/openfluke/w2a/suites"
	stepsuite "github.com/openfluke/w2a/suites/step"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/step"
	"github.com/openfluke/welvet/runtime/training"
)

func TestStepSuite(t *testing.T) {
	restore, err := suites.BeginLog()
	if err != nil {
		t.Fatalf("suite log: %v", err)
	}
	defer func() {
		suites.PrintReport()
		restore()
	}()

	for i, c := range stepsuite.Cases() {
		c := c
		i := i
		t.Run(c.Name, func(t *testing.T) {
			suites.BeginCase()
			err := c.Run()
			if err != nil {
				suites.EndCase("step", c.Name, "FAIL", err.Error())
				t.Fatalf("[%d] %v", i+1, err)
			}
			suites.EndCase("step", c.Name, "PASS", "")
		})
	}
}

func TestStepMeshSmoke(t *testing.T) {
	g := architecture.NewGrid(1, 1, 1, 1)
	w := make([]float32, 16)
	for i := 0; i < 4; i++ {
		w[i*4+i] = 1
	}
	l, err := dense.NewConfigured(4, 4, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, w)
	if err != nil {
		t.Fatal(err)
	}
	if err := dense.Place(g, 0, 0, 0, 0, l); err != nil {
		t.Fatal(err)
	}
	x := core.NewTensor[float32](1, 4)
	target := core.NewTensor[float32](1, 4)
	for i := 0; i < 4; i++ {
		x.Data[i] = float32(i + 1)
		target.Data[i] = float32(i+1) * 1.5
	}
	if _, _, err := training.StepMesh(g, x, target, 1, 0.02); err != nil {
		t.Fatal(err)
	}
	st := step.New[float32](g)
	st.SetInput(x)
	if _, err := step.Forward(g, st, true); err != nil {
		t.Fatal(err)
	}
	gy := core.NewTensor[float32](1, 4)
	gy.Data[0] = 1
	if _, _, err := step.Backward(g, st, gy); err != nil {
		t.Fatal(err)
	}
}
