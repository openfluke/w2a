package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppGDN runs Go↔C++ GDN (Gated DeltaNet) Float32 cases.
func TestGoVCppGDN(t *testing.T) {
	if err := gvc.RunGDN(); err != nil {
		t.Fatal(err)
	}
}
