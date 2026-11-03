package go_v_vulkan_test

import (
	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
	"testing"
)

func TestGoVVulkanCameralTrain(t *testing.T) {
	for _, c := range gvr.CasesCameralTrain() {
		t.Run(c.Name, func(t *testing.T) {
			if e := c.Run(); e != nil {
				t.Fatal(e)
			}
		})
	}
}
func TestGoVVulkanCamSync(t *testing.T) {
	for _, c := range gvr.CasesCamSync() {
		t.Run(c.Name, func(t *testing.T) {
			if e := c.Run(); e != nil {
				t.Fatal(e)
			}
		})
	}
}
