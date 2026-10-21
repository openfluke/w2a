package go_v_cpp

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
)

// CasesMemoryTrain — Parallel Dense Add + ModeMemory surprise gate via CamKit.
func CasesMemoryTrain() []Case {
	return []Case{
		{Name: "Parallel Memory gate open (thresh=0) Float32 — exact bits", Run: caseMemoryAlwaysF32},
		{Name: "Parallel Memory gate closed (high thresh) Float32 — exact bits", Run: caseMemoryBlockedF32},
		{Name: "Parallel Memory gate open × all 34 dtypes — spectrum", Run: caseMemoryAlwaysAllDTypes},
	}
}

func RunMemoryTrain() error {
	var fails []string
	cs := CasesMemoryTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [memory %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_memory_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_memory_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp memory_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func caseMemoryAlwaysF32() error {
	modes := []parallel.TrainMode{parallel.ModeMemory, parallel.ModeNormalBP}
	kit := parallel.CamKit{SurpriseThresh: 0} // always train
	return runCameralModesCase("MMB1", "MMBO", modes, kit, core.DTypeFloat32, true)
}

func caseMemoryBlockedF32() error {
	// Huge thresh ⇒ LastLoss < thresh ⇒ Memory cam skips; NormalBP cam still updates.
	modes := []parallel.TrainMode{parallel.ModeMemory, parallel.ModeNormalBP}
	kit := parallel.CamKit{SurpriseThresh: 1e9}
	return runCameralModesCase("MMB1", "MMBO", modes, kit, core.DTypeFloat32, true)
}

func caseMemoryAlwaysAllDTypes() error {
	modes := []parallel.TrainMode{parallel.ModeMemory, parallel.ModeNormalBP}
	kit := parallel.CamKit{SurpriseThresh: 0}
	return runCameralModesSpectrum("Memory", "MMB1", "MMBO", modes, kit)
}
