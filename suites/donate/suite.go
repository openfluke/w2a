package donate

import (
	"bytes"
	"fmt"
	"strings"
	"time"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/stub/donate"
)

type Case struct {
	Name string
	Run  func() error
}

func Cases() []Case {
	return []Case{
		{Name: "Frame round-trip", Run: frameRT},
		{Name: "TCP PutModel + Infer stub echo", Run: tcpInfer},
	}
}

func RunAll() error {
	var fails []string
	for i, c := range Cases() {
		suites.BeginCase()
		fmt.Printf("  [%d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("donate", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("donate", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("donate: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func RunOne(n int) error {
	cs := Cases()
	if n < 1 || n > len(cs) {
		return fmt.Errorf("donate: case %d out of range 1..%d", n, len(cs))
	}
	suites.BeginCase()
	fmt.Printf("  [%d] %s … ", n, cs[n-1].Name)
	if err := cs[n-1].Run(); err != nil {
		suites.EndCase("donate", cs[n-1].Name, "FAIL", err.Error())
		fmt.Printf("FAIL\n      %v\n", err)
		return err
	}
	suites.EndCase("donate", cs[n-1].Name, "PASS", "")
	fmt.Println("PASS")
	return nil
}

func frameRT() error {
	var buf bytes.Buffer
	msg := donate.DonateHello{V: 1, Type: donate.DonateMsgHello, Mode: "model_push", Role: "server"}
	if err := donate.WriteFrame(&buf, msg); err != nil {
		return err
	}
	var got donate.DonateHello
	if err := donate.ReadFrame(&buf, &got); err != nil {
		return err
	}
	if got.Type != donate.DonateMsgHello || got.Mode != "model_push" {
		return fmt.Errorf("bad frame %+v", got)
	}
	return nil
}

func tcpInfer() error {
	ln, err := donate.ServeTCP(donate.ServerOptions{
		Addr: "127.0.0.1:0", Mode: donate.ServerModelPush, QueueCapacity: 8, WorkerCount: 1,
	})
	if err != nil {
		return err
	}
	defer ln.Close()
	addr := ln.Addr().String()
	time.Sleep(20 * time.Millisecond)
	cli, hi, err := donate.Dial(addr)
	if err != nil {
		return err
	}
	defer cli.Close()
	if hi.Type != donate.DonateMsgHello {
		return fmt.Errorf("hello type %q", hi.Type)
	}
	weights := []byte("hello-weights")
	if err := cli.PutModel(`{"name":"t"}`, weights); err != nil {
		return err
	}
	res, err := cli.EnqueueInfer("j1", []int32{1, 2, 3}, 4)
	if err != nil {
		return err
	}
	if !res.OK {
		return fmt.Errorf("infer not ok: %s", res.Error)
	}
	return nil
}
