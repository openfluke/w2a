package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppAllTrain runs every train suite with a section timing table + totals.
// Use for OS/CPU sign-off: linux/amd64, linux/arm64, darwin/arm64, windows/*.
func TestGoVCppAllTrain(t *testing.T) {
	if err := gvc.RunAllTrain(); err != nil {
		t.Fatal(err)
	}
}
