package hardware

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/stub/hardware"
)

type Case struct {
	Name string
	Run  func() error
}

func Cases() []Case {
	return []Case{
		{Name: "Audit non-empty OS/CPU", Run: auditSmoke},
		{Name: "ToJSON parses", Run: jsonSmoke},
	}
}

func RunAll() error {
	var fails []string
	for i, c := range Cases() {
		suites.BeginCase()
		fmt.Printf("  [%d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("hardware", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("hardware", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("hardware: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func RunOne(n int) error {
	cs := Cases()
	if n < 1 || n > len(cs) {
		return fmt.Errorf("hardware: case %d out of range 1..%d", n, len(cs))
	}
	suites.BeginCase()
	fmt.Printf("  [%d] %s … ", n, cs[n-1].Name)
	if err := cs[n-1].Run(); err != nil {
		suites.EndCase("hardware", cs[n-1].Name, "FAIL", err.Error())
		fmt.Printf("FAIL\n      %v\n", err)
		return err
	}
	suites.EndCase("hardware", cs[n-1].Name, "PASS", "")
	fmt.Println("PASS")
	return nil
}

func auditSmoke() error {
	a := hardware.Audit()
	if a == nil || a.OS.Platform == "" || a.CPU.Logical < 1 {
		return fmt.Errorf("sparse audit %+v", a)
	}
	_ = hardware.Description()
	return nil
}

func jsonSmoke() error {
	s := hardware.Audit().ToJSON()
	var m map[string]any
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		return err
	}
	return nil
}
