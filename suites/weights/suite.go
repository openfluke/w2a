// Package weights holds w2a cases for welvet/weights (native SGD, storage truth).
package weights

import (
	"fmt"
	"math"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

type Case struct {
	Name string
	Run  func() error
}

func Cases() []Case {
	return []Case{
		{Name: "Native SGD — FormatNone × all 34 dtypes (no retained f32 scratch)", Run: applySGDNativeNoMaster},
		{Name: "Native SGD — float64 in-dtype ALU (no f32 hop)", Run: applySGDF64NativeALU},
	}
}

func RunAll() error {
	var fails []string
	for i, c := range Cases() {
		suites.BeginCase()
		fmt.Printf("  [%d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("weights", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("weights", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("weights: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func RunOne(n int) error {
	cs := Cases()
	if n < 1 || n > len(cs) {
		return fmt.Errorf("weights: case %d out of range 1..%d", n, len(cs))
	}
	suites.BeginCase()
	fmt.Printf("  [%d] %s … ", n, cs[n-1].Name)
	if err := cs[n-1].Run(); err != nil {
		suites.EndCase("weights", cs[n-1].Name, "FAIL", err.Error())
		fmt.Printf("FAIL\n      %v\n", err)
		return err
	}
	suites.EndCase("weights", cs[n-1].Name, "PASS", "")
	fmt.Println("PASS")
	return nil
}

func applySGDNativeNoMaster() error {
	const rows, cols = 8, 8
	src := make([]float32, rows*cols)
	for i := range src {
		src[i] = float32(math.Sin(float64(i)*0.17)) * 0.5
	}
	dW := make([]float64, rows*cols)
	for i := range dW {
		dW[i] = 0.01 * float64((i%5)-2)
	}
	const lr = 0.1

	var fails []string
	okN := 0
	for _, dt := range core.AllDTypes {
		s, err := weights.New(rows, cols, src, core.DTypeFloat32, quant.FormatNone)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s new: %v", dt, err))
			continue
		}
		if err := s.SetDType(dt); err != nil {
			fails = append(fails, fmt.Sprintf("%s SetDType: %v", dt, err))
			continue
		}
		before, err := s.FlattenF32()
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s flatten: %v", dt, err))
			continue
		}
		if err := s.ApplySGD(dW, lr); err != nil {
			fails = append(fails, fmt.Sprintf("%s ApplySGD: %v", dt, err))
			continue
		}
		if dt != core.DTypeFloat32 {
			if s.F32BufferLen() != 0 {
				fails = append(fails, fmt.Sprintf("%s retained f32 buffer len=%d", dt, s.F32BufferLen()))
				continue
			}
			if s.RetainsF32Master() {
				fails = append(fails, fmt.Sprintf("%s RetainsF32Master", dt))
				continue
			}
			if len(s.Native) == 0 {
				fails = append(fails, fmt.Sprintf("%s empty Native", dt))
				continue
			}
		} else if !s.RetainsF32Master() {
			fails = append(fails, "float32 should retain f32 payload")
			continue
		}
		after, err := s.FlattenF32()
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s flatten after: %v", dt, err))
			continue
		}
		moved := false
		bad := false
		for i := range after {
			if math.IsNaN(float64(after[i])) || math.IsInf(float64(after[i]), 0) {
				fails = append(fails, fmt.Sprintf("%s non-finite[%d]=%v", dt, i, after[i]))
				bad = true
				break
			}
			if after[i] != before[i] {
				moved = true
			}
		}
		if bad {
			continue
		}
		switch dt {
		case core.DTypeFP4, core.DTypeBinary, core.DTypeTernary, core.DTypeNF4,
			core.DTypeInt4, core.DTypeUint4, core.DTypeInt2, core.DTypeUint2,
			core.DTypeFP6, core.DTypeInt6, core.DTypeUint6,
			core.DTypeInt5, core.DTypeUint5, core.DTypeInt3, core.DTypeUint3:
			// coarse codebooks may round Δ away
		default:
			if !moved {
				fails = append(fails, fmt.Sprintf("%s no visible weight delta", dt))
				continue
			}
		}
		okN++
	}
	fmt.Printf("(%d/%d dtypes) ", okN, len(core.AllDTypes))
	if len(fails) > 0 {
		return fmt.Errorf("%s", strings.Join(fails, " | "))
	}
	return nil
}

func applySGDF64NativeALU() error {
	src := []float64{1, 2, 3, 4}
	s, err := weights.New(2, 2, src, core.DTypeFloat64, quant.FormatNone)
	if err != nil {
		return err
	}
	dW := []float64{1, 0, 0, 1}
	if err := s.ApplySGD(dW, 0.5); err != nil {
		return err
	}
	if s.F32BufferLen() != 0 {
		return fmt.Errorf("float64 SGD retained f32 buffer len=%d", s.F32BufferLen())
	}
	got, err := s.FlattenF32()
	if err != nil {
		return err
	}
	if math.Abs(float64(got[0])-0.5) > 1e-6 || math.Abs(float64(got[3])-3.5) > 1e-6 {
		return fmt.Errorf("got %v want [0.5 2 3 3.5]", got)
	}
	return nil
}
