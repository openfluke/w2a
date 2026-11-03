package go_v_rocm_test

import (
	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
	"testing"
)

func TestGoVRocmCameralTrain(t *testing.T) {
	for _, c := range gvr.CasesCameralTrain() {
		t.Run(c.Name, func(t *testing.T) {
			if e := c.Run(); e != nil {
				t.Fatal(e)
			}
		})
	}
}
func TestGoVRocmCamSync(t *testing.T) {
	for _, c := range gvr.CasesCamSync() {
		t.Run(c.Name, func(t *testing.T) {
			if e := c.Run(); e != nil {
				t.Fatal(e)
			}
		})
	}
}
