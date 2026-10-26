package go_v_metal_test

import (
	gvm "github.com/openfluke/w2a/suites/go_v_metal"
	"testing"
)

func TestGoNativeMetalIntegration(t *testing.T) {
	for _, c := range gvm.CasesNativeMetalIntegration() {
		t.Run(c.Name, func(t *testing.T) {
			if e := c.Run(); e != nil {
				t.Fatal(e)
			}
		})
	}
}

func TestGoNativeMetalMesh(t *testing.T) {
	for _, c := range gvm.CasesNativeMetalMesh() {
		t.Run(c.Name, func(t *testing.T) {
			if e := c.Run(); e != nil {
				t.Fatal(e)
			}
		})
	}
}
