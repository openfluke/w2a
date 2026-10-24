package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalSoftmax — Go↔Metal Softmax forward (all Kind ×34 exact_bit + timings).
// Run: go test ./tests/go_v_metal/ -v -run '^TestGoVMetalSoftmax$'
func TestGoVMetalSoftmax(t *testing.T) {
	if err := gvm.RunSoftmax(); err != nil {
		t.Fatal(err)
	}
}
