package go_v_metal

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os/exec"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/layers/mamba"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/seqmix"
)

func CasesSeqmix() []Case {
	return []Case{
		{Name: "Locate welvet.metal tool", Run: caseLocateMetal},
		{Name: "seqmix Kind iota + String — Go", Run: caseSeqmixGoKinds},
		{Name: "seqmix mixer tags — Go MHA/Mamba/GDN", Run: caseSeqmixGoTags},
		{Name: "seqmix Kind names + tags — Go vs Metal WSX1", Run: caseSeqmixMetalWire},
	}
}

func RunSeqmix() error {
	var fails []string
	cs := CasesSeqmix()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [sx %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_metal_seqmix", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_metal_seqmix", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_metal seqmix: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func caseSeqmixGoKinds() error {
	want := []struct {
		k seqmix.Kind
		s string
	}{
		{seqmix.KindAttention, "attention"},
		{seqmix.KindSSM, "ssm"},
		{seqmix.KindLinearAttn, "linear_attn"},
		{seqmix.KindConvMix, "conv_mix"},
	}
	for i, w := range want {
		if int(w.k) != i {
			return fmt.Errorf("kind iota: %v want %d", w.k, i)
		}
		if w.k.String() != w.s {
			return fmt.Errorf("String(%v)=%q want %q", w.k, w.k.String(), w.s)
		}
	}
	fmt.Printf("\n      kinds 0..3 = attention/ssm/linear_attn/conv_mix")
	return nil
}

func caseSeqmixGoTags() error {
	mb, err := mamba.New(mamba.Config{DModel: 4, DState: 2, Expand: 2, SeqLen: 2})
	if err != nil {
		return err
	}
	if mb.Kind() != seqmix.KindSSM {
		return fmt.Errorf("mamba.Kind=%v want KindSSM", mb.Kind())
	}
	gd, err := gdn.New(gdn.Config{
		HiddenSize: 4, NumKeyHeads: 1, NumValueHeads: 1,
		KeyHeadDim: 4, ValueHeadDim: 4, ConvKernel: 4,
	})
	if err != nil {
		return err
	}
	if gd.Kind() != seqmix.KindLinearAttn {
		return fmt.Errorf("gdn.Kind=%v want KindLinearAttn", gd.Kind())
	}
	mh, err := mha.New(mha.DecoderCausal(8, 2, 2))
	if err != nil {
		return err
	}
	if mh.MixerKind() != seqmix.KindAttention {
		return fmt.Errorf("mha.MixerKind=%v want KindAttention", mh.MixerKind())
	}
	fmt.Printf("\n      tags: mha=attention mamba=ssm gdn=linear_attn")
	return nil
}

func caseSeqmixMetalWire() error {
	kinds := []int32{
		int32(seqmix.KindAttention),
		int32(seqmix.KindSSM),
		int32(seqmix.KindLinearAttn),
		int32(seqmix.KindConvMix),
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("WSX1"))
	_ = binary.Write(&in, binary.LittleEndian, int32(len(kinds)))
	for _, k := range kinds {
		_ = binary.Write(&in, binary.LittleEndian, k)
	}
	tool, err := findMetalFwd()
	if err != nil {
		return err
	}
	cmd := exec.Command(tool)
	cmd.Stdin = bytes.NewReader(in.Bytes())
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("metal seqmix: %v: %s", err, stderr.String())
	}
	raw := stdout.Bytes()
	if len(raw) < 8 || string(raw[:4]) != "WSXO" {
		return fmt.Errorf("metal seqmix bad magic")
	}
	r := bytes.NewReader(raw[4:])
	var n int32
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return err
	}
	if int(n) != len(kinds) {
		return fmt.Errorf("n=%d want %d", n, len(kinds))
	}
	wantNames := []string{"attention", "ssm", "linear_attn", "conv_mix"}
	for i, want := range wantNames {
		var ln int32
		if err := binary.Read(r, binary.LittleEndian, &ln); err != nil {
			return err
		}
		buf := make([]byte, ln)
		if _, err := r.Read(buf); err != nil {
			return err
		}
		got := string(buf)
		if got != want {
			return fmt.Errorf("kind[%d] name %q want %q", i, got, want)
		}
	}
	var nTags int32
	if err := binary.Read(r, binary.LittleEndian, &nTags); err != nil {
		return err
	}
	if nTags != 3 {
		return fmt.Errorf("n_tags=%d want 3", nTags)
	}
	tags := make([]int32, 3)
	for i := range tags {
		if err := binary.Read(r, binary.LittleEndian, &tags[i]); err != nil {
			return err
		}
	}
	wantTags := []int32{
		int32(seqmix.KindAttention),
		int32(seqmix.KindSSM),
		int32(seqmix.KindLinearAttn),
	}
	for i := range wantTags {
		if tags[i] != wantTags[i] {
			return fmt.Errorf("tag[%d]=%d want %d", i, tags[i], wantTags[i])
		}
	}
	fmt.Printf("\n      WSX1 names+tags match Go iota (Metal)")
	return nil
}
