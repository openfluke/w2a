package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalRMSNorm — Go↔Metal RMSNorm forward only (exact_bit + timings).
// Run: go test ./tests/go_v_metal/ -v -run '^TestGoVMetalRMSNorm$'
func TestGoVMetalRMSNorm(t *testing.T) {
	if err := gvm.RunRMSNorm(); err != nil {
		t.Fatal(err)
	}
}
