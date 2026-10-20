package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCpp runs the full Go↔C++ suite (dense + architecture + transformer + CNN1/2/3).
func TestGoVCpp(t *testing.T) {
	if err := gvc.RunAll(); err != nil {
		t.Fatal(err)
	}
}
