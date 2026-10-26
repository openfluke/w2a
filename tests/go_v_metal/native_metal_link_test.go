//go:build welvet_metal

package go_v_metal_test

import (
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/nativemetal"
	"strings"
	"testing"
)

func TestGoNativeMetalIndependentLink(t *testing.T) {
	if nativecpp.NativeCPUAvailable() {
		t.Fatal("Metal-only build enabled CPU sibling")
	}
	_, model, x, _ := metalLifecycleModel(t)
	s, e := nativemetal.Mount(model)
	if e != nil {
		t.Fatal(e)
	}
	defer s.Close()
	if !strings.HasPrefix(s.Identity(), "metal:") {
		t.Fatalf("wrong backend: %s", s.Identity())
	}
	if _, e = s.Forward(x); e != nil {
		t.Fatal(e)
	}
}
