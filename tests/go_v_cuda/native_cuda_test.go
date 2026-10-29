package go_v_cuda_test

import (
	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
	"testing"
)

func TestGoNativeCudaIntegration(t *testing.T) {
	for _, c := range gvc.CasesNativeCudaIntegration() {
		t.Run(c.Name, func(t *testing.T) {
			if e := c.Run(); e != nil {
				t.Fatal(e)
			}
		})
	}
}

func TestGoNativeCudaMesh(t *testing.T) {
	for _, c := range gvc.CasesNativeCudaMesh() {
		t.Run(c.Name, func(t *testing.T) {
			if e := c.Run(); e != nil {
				t.Fatal(e)
			}
		})
	}
}

func TestGoNativeCudaRoutingLayers(t *testing.T) {
	for _, c := range gvc.CasesNativeCudaRouting() {
		t.Run(c.Name, func(t *testing.T) {
			if e := c.Run(); e != nil {
				t.Fatal(e)
			}
		})
	}
}
func TestGoNativeCudaRoutingModes(t *testing.T) {
	for _, c := range gvc.CasesNativeCudaRoutedModes() {
		t.Run(c.Name, func(t *testing.T) {
			if e := c.Run(); e != nil {
				t.Fatal(e)
			}
		})
	}
}
