package go_v_metal_test

import (
	"fmt"
	"testing"

	"github.com/openfluke/w2a/suites"
	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalVolume — volumetric Grid forward (Dense WVF1 + MHA + AllLayers 3×3×3 × dtypes).
// Run: go test ./tests/go_v_metal/ -v -run '^TestGoVMetalVolume$'
func TestGoVMetalVolume(t *testing.T) {
	if err := gvm.RunVolume(); err != nil {
		t.Fatal(err)
	}
}

// TestGoVMetalVolumeMHA — MHA DecoderCausal 3×3×3×1 ×34 only.
// Run: go test ./tests/go_v_metal/ -v -run '^TestGoVMetalVolumeMHA$'
func TestGoVMetalVolumeMHA(t *testing.T) {
	for _, c := range gvm.CasesVolume() {
		if c.Name != "Volume MHA 3×3×3×1 ×34 — ≥99% + Go/C++/Metal timings" &&
			c.Name != "Locate welvet.metal tool" &&
			c.Name != "Locate welvet.cpp tool (optional speed)" {
			continue
		}
		suites.BeginCase()
		t.Logf("[mha-vol] %s …", c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_metal_volume_mha", c.Name, "FAIL", err.Error())
			t.Fatal(fmt.Errorf("%s: %w", c.Name, err))
		}
		suites.EndCase("go_v_metal_volume_mha", c.Name, "PASS", "")
	}
}

// TestGoVMetalVolumeSwiGLU — SwiGLU 3×3×3×1 ×34 only.
// Run: go test ./tests/go_v_metal/ -v -run '^TestGoVMetalVolumeSwiGLU$'
func TestGoVMetalVolumeSwiGLU(t *testing.T) {
	for _, c := range gvm.CasesVolume() {
		if c.Name != "Volume SwiGLU 3×3×3×1 ×34 — ≥99% + Go/C++/Metal timings" &&
			c.Name != "Locate welvet.metal tool" &&
			c.Name != "Locate welvet.cpp tool (optional speed)" {
			continue
		}
		suites.BeginCase()
		t.Logf("[swiglu-vol] %s …", c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_metal_volume_swiglu", c.Name, "FAIL", err.Error())
			t.Fatal(fmt.Errorf("%s: %w", c.Name, err))
		}
		suites.EndCase("go_v_metal_volume_swiglu", c.Name, "PASS", "")
	}
}

// TestGoVMetalVolumeAllLayers — mount-all volume wires × kinds ×34.
// Run: go test ./tests/go_v_metal/ -v -run '^TestGoVMetalVolumeAllLayers$'
func TestGoVMetalVolumeAllLayers(t *testing.T) {
	for _, c := range gvm.CasesVolume() {
		if c.Name != "Volume AllLayers 3×3×3 × dtypes — mount-all volume wires" &&
			c.Name != "Locate welvet.metal tool" {
			continue
		}
		suites.BeginCase()
		t.Logf("[all-vol] %s …", c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_metal_volume_all", c.Name, "FAIL", err.Error())
			t.Fatal(fmt.Errorf("%s: %w", c.Name, err))
		}
		suites.EndCase("go_v_metal_volume_all", c.Name, "PASS", "")
	}
}
