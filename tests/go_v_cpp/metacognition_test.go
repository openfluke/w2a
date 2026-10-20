package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppMetacognition runs Go↔C++ Metacognition Observed+rules cases.
func TestGoVCppMetacognition(t *testing.T) {
	if err := gvc.RunMetacognition(); err != nil {
		t.Fatal(err)
	}
}
