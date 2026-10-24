package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalMHA — Go↔Metal MHA ModeSelf forward only (exact_bit + timings).
// Run: go test ./tests/go_v_metal/ -v -run '^TestGoVMetalMHA$'
func TestGoVMetalMHA(t *testing.T) {
	if err := gvm.RunMHA(); err != nil {
		t.Fatal(err)
	}
}
