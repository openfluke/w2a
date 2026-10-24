package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalRNN — Go↔Metal RNN (exact_bit smoke + ×34 ≥99% + timings).
// Run: go test ./tests/go_v_metal/ -v -run '^TestGoVMetalRNN$'
func TestGoVMetalRNN(t *testing.T) {
	if err := gvm.RunRNN(); err != nil {
		t.Fatal(err)
	}
}
