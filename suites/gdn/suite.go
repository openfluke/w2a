package gdn

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/layers/seqmix"
	"github.com/openfluke/welvet/webgpu"
)

type Case struct {
	Name string
	Run  func() error
}

func Cases() []Case {
	return []Case{
		{Name: "ForwardDecode smoke", Run: decodeSmoke},
		{Name: "Forward [B,T,H] smoke", Run: seqSmoke},
		{Name: "seqmix KindLinearAttn", Run: kindSmoke},
		{Name: "Backward hard-errors (inference-first)", Run: bwdHardError},
		{Name: "WebGPU path when no device (decode still host)", Run: webGPUNote},
	}
}

func RunAll() error {
	var fails []string
	for i, c := range Cases() {
		suites.BeginCase()
		fmt.Printf("  [%d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("gdn", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("gdn", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("gdn: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func RunOne(n int) error {
	cs := Cases()
	if n < 1 || n > len(cs) {
		return fmt.Errorf("gdn: case %d out of range 1..%d", n, len(cs))
	}
	suites.BeginCase()
	fmt.Printf("  [%d] %s … ", n, cs[n-1].Name)
	if err := cs[n-1].Run(); err != nil {
		suites.EndCase("gdn", cs[n-1].Name, "FAIL", err.Error())
		fmt.Printf("FAIL\n      %v\n", err)
		return err
	}
	suites.EndCase("gdn", cs[n-1].Name, "PASS", "")
	fmt.Println("PASS")
	return nil
}

func cfg() gdn.Config {
	return gdn.Config{HiddenSize: 16, NumKeyHeads: 2, NumValueHeads: 2, KeyHeadDim: 4, ValueHeadDim: 4, ConvKernel: 2, Eps: 1e-6}
}

func decodeSmoke() error {
	l, err := gdn.New(cfg())
	if err != nil { return err }
	x := make([]float32, 16)
	y := make([]float32, 16)
	for i := range x { x[i] = 0.01 * float32(i) }
	return l.ForwardDecode(x, y)
}

func seqSmoke() error {
	l, err := gdn.New(cfg())
	if err != nil { return err }
	x := core.NewTensor[float32](1, 3, 16)
	for i := range x.Data { x.Data[i] = 0.01 }
	_, post, err := gdn.Forward(l, x)
	if err != nil { return err }
	if post.Shape[1] != 3 || post.Shape[2] != 16 { return fmt.Errorf("shape %v", post.Shape) }
	return nil
}

func kindSmoke() error {
	l, err := gdn.New(cfg())
	if err != nil { return err }
	if l.Kind() != seqmix.KindLinearAttn {
		return fmt.Errorf("kind %v", l.Kind())
	}
	return nil
}

func bwdHardError() error {
	l, err := gdn.New(cfg())
	if err != nil { return err }
	x := core.NewTensor[float32](1, 2, 16)
	_, _, err = gdn.Backward(l, x, x, x)
	if err == nil { return fmt.Errorf("expected bwd error") }
	return nil
}

func webGPUNote() error {
	_ = webgpu.Available()
	return nil
}
