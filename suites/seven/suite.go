package seven

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites"
)

// Case is one Lucy-[7]-style harness entry.
type Case struct {
	Name string
	Run  func() error
}

// Cases returns the seven-style suite (determinism, SC↔MC, train, ENTITY, tiers).
func Cases() []Case {
	return []Case{
		{Name: "Dense repeat-forward determinism (CPU/SIMD/WebGPU)", Run: denseRepeatDet},
		{Name: "Dense SC↔MC fwd+bwd × all 34 FormatNone", Run: denseSCMCFormatNone},
		{Name: "Dense SC↔MC fwd+bwd × all quants (f32 pack)", Run: denseSCMCQuants},
		{Name: "Dense multi-epoch train loss↓ + SC/MC/SIMD train parity", Run: denseTrainParity},
		{Name: "Dense ENTITY save/load before+after train", Run: denseEntityTrain},
		{Name: "Dense S/M/L shape tiers (fwd+bwd+1-step train)", Run: denseShapeTiers},
		{Name: "Volumetric 7-Dense/cell × 1³/2³/3³ SC↔MC + short train", Run: volumetricSevenDense},
		{Name: "Peak layers repeat-det + SC↔MC + short train (f32 + Q8_0)", Run: peakLayersBlock},
		{Name: "Seven-style summary table", Run: printSevenSummary},
	}
}

func RunAll() error {
	var fails []string
	for i, c := range Cases() {
		suites.BeginCase()
		fmt.Printf("  [%d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("seven", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("seven", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("seven: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func RunOne(n int) error {
	cs := Cases()
	if n < 1 || n > len(cs) {
		return fmt.Errorf("seven: case %d out of range 1..%d", n, len(cs))
	}
	suites.BeginCase()
	fmt.Printf("  [%d] %s … ", n, cs[n-1].Name)
	if err := cs[n-1].Run(); err != nil {
		suites.EndCase("seven", cs[n-1].Name, "FAIL", err.Error())
		fmt.Printf("FAIL\n      %v\n", err)
		return err
	}
	suites.EndCase("seven", cs[n-1].Name, "PASS", "")
	fmt.Println("PASS")
	return nil
}
