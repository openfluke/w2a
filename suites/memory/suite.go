package memory

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/stub/memory"
)

type Case struct {
	Name string
	Run  func() error
}

func Cases() []Case {
	return []Case{
		{Name: "Footprint FromGrid", Run: footprint},
		{Name: "History record when enabled", Run: history},
		{Name: "ReleaseTransient", Run: release},
	}
}

func RunAll() error {
	var fails []string
	for i, c := range Cases() {
		suites.BeginCase()
		fmt.Printf("  [%d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("memory", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("memory", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("memory: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func RunOne(n int) error {
	cs := Cases()
	if n < 1 || n > len(cs) {
		return fmt.Errorf("memory: case %d out of range 1..%d", n, len(cs))
	}
	suites.BeginCase()
	fmt.Printf("  [%d] %s … ", n, cs[n-1].Name)
	if err := cs[n-1].Run(); err != nil {
		suites.EndCase("memory", cs[n-1].Name, "FAIL", err.Error())
		fmt.Printf("FAIL\n      %v\n", err)
		return err
	}
	suites.EndCase("memory", cs[n-1].Name, "PASS", "")
	fmt.Println("PASS")
	return nil
}

func footprint() error {
	l, err := dense.NewConfigured[float32](8, 8, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, nil)
	if err != nil {
		return err
	}
	g := architecture.NewGrid(1, 1, 1, 1)
	_ = dense.Place(g, 0, 0, 0, 0, l)
	fp := memory.FromGrid(g)
	if fp.HostWeightsMB <= 0 {
		return fmt.Errorf("expected host weights > 0 got %v", fp)
	}
	_ = fp.FormatOneLine()
	return nil
}

func history() error {
	memory.SetMemoryHistoryRecording(true)
	defer memory.ResetMemoryHistoryRecording()
	h := &memory.MemoryHistory{}
	h.BeginSession("test")
	h.Record("a", memory.FromBytes(1024*1024, 0, 0), 0)
	if len(h.Samples()) != 1 {
		return fmt.Errorf("samples %d", len(h.Samples()))
	}
	return nil
}

func release() error {
	memory.InitScavenger()
	memory.ReleaseTransient()
	memory.ReleaseAggressive()
	return nil
}
