package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalLSTM — Go↔Metal LSTM (exact_bit smoke + ×34 ≥99% + timings).
// Run: go test ./tests/go_v_metal/ -v -run '^TestGoVMetalLSTM$'
func TestGoVMetalLSTM(t *testing.T) {
	if err := gvm.RunLSTM(); err != nil {
		t.Fatal(err)
	}
}
