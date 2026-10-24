package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalSequential — Go↔Metal Sequential Dense chain (exact_bit ×34 + timings).
// Run: go test ./tests/go_v_metal/ -v -run '^TestGoVMetalSequential$'
func TestGoVMetalSequential(t *testing.T) {
	if err := gvm.RunSequential(); err != nil {
		t.Fatal(err)
	}
}
