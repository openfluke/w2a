package go_v_cpp_test

import (
	"testing"

	"github.com/openfluke/w2a/suites/go_v_cpp"
)

func TestGoVCppTweenSplitHeadProxyAsyncTrain(t *testing.T) {
	if err := go_v_cpp.RunTweenSplitHeadProxyAsyncTrain(); err != nil {
		t.Fatal(err)
	}
}
