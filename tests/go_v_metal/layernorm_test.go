package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalLayerNorm — Go↔Metal LayerNorm forward only (exact_bit + timings).
// Run: go test ./tests/go_v_metal/ -v -run '^TestGoVMetalLayerNorm$'
func TestGoVMetalLayerNorm(t *testing.T) {
	if err := gvm.RunLayerNorm(); err != nil {
		t.Fatal(err)
	}
}
