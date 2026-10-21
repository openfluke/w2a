package go_v_cpp

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
)

// CasesShadowTrain — Parallel Dense Add Shadow teacher + NormalBP student + KD.
func CasesShadowTrain() []Case {
	return []Case{
		{Name: "Parallel Shadow∥NormalBP KD Float32 — exact bits", Run: caseShadowKDF32},
		{Name: "Parallel Shadow∥NormalBP KD coef=0.5 Float32 — exact bits", Run: caseShadowKDCoefF32},
		{Name: "Parallel Shadow∥NormalBP KD × all 34 dtypes — spectrum", Run: caseShadowKDAllDTypes},
	}
}

func RunShadowTrain() error {
	var fails []string
	cs := CasesShadowTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [shadow %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_shadow_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_shadow_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp shadow_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func caseShadowKDF32() error {
	modes := []parallel.TrainMode{parallel.ModeShadow, parallel.ModeNormalBP}
	kit := parallel.CamKit{ShadowCoef: 1.0}
	return runCameralModesCase("SHB1", "SHBO", modes, kit, core.DTypeFloat32, true)
}

func caseShadowKDCoefF32() error {
	modes := []parallel.TrainMode{parallel.ModeShadow, parallel.ModeNormalBP}
	kit := parallel.CamKit{ShadowCoef: 0.5}
	return runCameralModesCase("SHB1", "SHBO", modes, kit, core.DTypeFloat32, true)
}

func caseShadowKDAllDTypes() error {
	modes := []parallel.TrainMode{parallel.ModeShadow, parallel.ModeNormalBP}
	kit := parallel.CamKit{ShadowCoef: 1.0}
	return runCameralModesSpectrum("Shadow", "SHB1", "SHBO", modes, kit)
}
