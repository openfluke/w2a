package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalCNN — Go↔Metal CNN1/2/3 (exact_bit ×34 + timings).
func TestGoVMetalCNN(t *testing.T) {
	if err := gvm.RunCNN(); err != nil {
		t.Fatal(err)
	}
}
