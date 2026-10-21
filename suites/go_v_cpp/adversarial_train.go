package go_v_cpp

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
)

// CasesAdversarialTrain — Parallel Dense Add with BranchMode Adversarial (negated LR).
func CasesAdversarialTrain() []Case {
	return []Case{
		{Name: "Parallel Adversarial∥NormalBP Float32 — exact bits", Run: caseAdversarialMixedF32},
		{Name: "Parallel Adversarial∥Adversarial Float32 — exact bits", Run: caseAdversarialBothF32},
		{Name: "Parallel Adversarial∥NormalBP × all 34 dtypes — spectrum", Run: caseAdversarialMixedAllDTypes},
	}
}

func RunAdversarialTrain() error {
	var fails []string
	cs := CasesAdversarialTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [adversarial %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_adversarial_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_adversarial_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp adversarial_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func caseAdversarialMixedF32() error {
	modes := []parallel.TrainMode{parallel.ModeNormalBP, parallel.ModeAdversarial}
	return runCameralModesCase("ADB1", "ADBO", modes, parallel.CamKit{}, core.DTypeFloat32, true)
}

func caseAdversarialBothF32() error {
	modes := []parallel.TrainMode{parallel.ModeAdversarial, parallel.ModeAdversarial}
	return runCameralModesCase("ADB1", "ADBO", modes, parallel.CamKit{}, core.DTypeFloat32, true)
}

func caseAdversarialMixedAllDTypes() error {
	modes := []parallel.TrainMode{parallel.ModeNormalBP, parallel.ModeAdversarial}
	return runCameralModesSpectrum("Adversarial", "ADB1", "ADBO", modes, parallel.CamKit{})
}
