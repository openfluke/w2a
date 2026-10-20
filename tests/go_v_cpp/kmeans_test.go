package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppKMeans runs Go↔C++ KMeans soft assignment cases.
func TestGoVCppKMeans(t *testing.T) {
	if err := gvc.RunKMeans(); err != nil {
		t.Fatal(err)
	}
}
