package go_v_cpp_test

import (
	"testing"

	"github.com/openfluke/w2a/suites/go_v_cpp"
)

func TestGoVCppFreezeTrain(t *testing.T) {
	if err := go_v_cpp.RunFreezeTrain(); err != nil {
		t.Fatal(err)
	}
}
