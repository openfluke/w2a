package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalMamba — Go↔Metal Mamba (exact_bit smoke + ×34 ≥99% + timings).
// Run: go test ./tests/go_v_metal/ -v -run '^TestGoVMetalMamba$'
func TestGoVMetalMamba(t *testing.T) {
	if err := gvm.RunMamba(); err != nil {
		t.Fatal(err)
	}
}
