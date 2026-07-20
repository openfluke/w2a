package fountain

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/stub/fountain"
)

type Case struct {
	Name string
	Run  func() error
}

func Cases() []Case {
	return []Case{
		{Name: "LT peel recovers sources (no loss)", Run: ltPeel},
		{Name: "RecoverWeightBlobs under 10% loss", Run: recoverLossy},
	}
}

func RunAll() error {
	var fails []string
	for i, c := range Cases() {
		suites.BeginCase()
		fmt.Printf("  [%d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("fountain", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("fountain", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("fountain: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func RunOne(n int) error {
	cs := Cases()
	if n < 1 || n > len(cs) {
		return fmt.Errorf("fountain: case %d out of range 1..%d", n, len(cs))
	}
	suites.BeginCase()
	fmt.Printf("  [%d] %s … ", n, cs[n-1].Name)
	if err := cs[n-1].Run(); err != nil {
		suites.EndCase("fountain", cs[n-1].Name, "FAIL", err.Error())
		fmt.Printf("FAIL\n      %v\n", err)
		return err
	}
	suites.EndCase("fountain", cs[n-1].Name, "PASS", "")
	fmt.Println("PASS")
	return nil
}

func makeBlocks(k, sz int) [][]byte {
	out := make([][]byte, k)
	for i := range out {
		out[i] = make([]byte, sz)
		for j := range out[i] {
			out[i][j] = byte(i*31 + j)
		}
	}
	return out
}

func ltPeel() error {
	src := makeBlocks(8, 32)
	enc, err := fountain.NewLTEncoder(src, 42)
	if err != nil {
		return err
	}
	dec := fountain.NewLTDecoder(8, 32)
	for !dec.Done() {
		dec.Catch(enc.Spray())
		if dec.KnownCount() > 0 && dec.KnownCount()%3 == 0 {
			dec.TryResidualGE(0)
		}
	}
	if !fountain.BlocksEqual(src, dec.Recovered) {
		return fmt.Errorf("mismatch")
	}
	return nil
}

func recoverLossy() error {
	src := makeBlocks(6, 16)
	rec, _, _, err := fountain.RecoverWeightBlobs(src, 7, 0.1, 3)
	if err != nil {
		return err
	}
	if !fountain.BlocksEqual(src, rec) {
		return fmt.Errorf("recover mismatch")
	}
	return nil
}
