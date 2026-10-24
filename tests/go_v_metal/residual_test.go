package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalResidual — Go↔Metal Residual F(x)+x (exact_bit ×34 + timings).
// Run: go test ./tests/go_v_metal/ -v -run '^TestGoVMetalResidual$'
func TestGoVMetalResidual(t *testing.T) {
	if err := gvm.RunResidual(); err != nil {
		t.Fatal(err)
	}
}
