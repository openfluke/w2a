package telemetry_test

import (
	"testing"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/telemetry"
)

func TestBlueprintDenseParams(t *testing.T) {
	g := architecture.NewGrid(1, 1, 1, 1)
	l0, err := dense.New(4, 8, core.ActivationReLU, core.DTypeFloat32)
	if err != nil {
		t.Fatal(err)
	}
	if err := dense.Place(g, 0, 0, 0, 0, l0); err != nil {
		t.Fatal(err)
	}
	bp := telemetry.ExtractNetworkBlueprint(g, "smoke")
	if bp.TotalLayers != 1 {
		t.Fatalf("layers=%d", bp.TotalLayers)
	}
	if bp.TotalParams != 4*8 {
		t.Fatalf("params=%d want %d", bp.TotalParams, 32)
	}
	if len(bp.Layers) != 1 || bp.Layers[0].Type != "Dense" {
		t.Fatalf("layer meta=%+v", bp.Layers)
	}
}
