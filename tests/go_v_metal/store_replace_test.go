package go_v_metal_test

import (
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
	"reflect"
	"testing"
)

// CamSync/SetFromF32 must invalidate both cached wire representations.
func TestGoStoreReplaceInvalidatesWires(t *testing.T) {
	for _, dt := range []core.DType{core.DTypeFloat32, core.DTypeFloat64, core.DTypeFloat16, core.DTypeInt32, core.DTypeNF4} {
		t.Run(dt.String(), func(t *testing.T) {
			before := []float32{.1, .2, -.3, .4}
			after := []float32{-.4, .3, .2, -.1}
			s, e := weights.New(1, 4, before, dt, quant.FormatNone)
			if e != nil {
				t.Fatal(e)
			}
			if _, e = s.WireF64(); e != nil {
				t.Fatal(e)
			}
			if _, e = s.GPUWireF32(); e != nil {
				t.Fatal(e)
			}
			if e = s.SetFromF32(after); e != nil {
				t.Fatal(e)
			}
			fresh, e := weights.New(1, 4, after, dt, quant.FormatNone)
			if e != nil {
				t.Fatal(e)
			}
			actual, e := s.WireF64()
			if e != nil {
				t.Fatal(e)
			}
			expected, e := fresh.WireF64()
			if e != nil {
				t.Fatal(e)
			}
			if !reflect.DeepEqual(actual, expected) {
				t.Fatalf("stale float64 wire: %v != %v", actual, expected)
			}
			f32, e := s.GPUWireF32()
			if e != nil {
				t.Fatal(e)
			}
			f32ref, e := fresh.GPUWireF32()
			if e != nil {
				t.Fatal(e)
			}
			if !reflect.DeepEqual(f32, f32ref) {
				t.Fatalf("stale float32 wire: %v != %v", f32, f32ref)
			}
		})
	}
}
