package go_v_qnn

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/openfluke/welvet/nativeqnn"
)

// README HTP dtype columns (bf16 capped off on V73).
var matrixDtypes = []string{"float32", "float16", "int8", "uint8", "int16", "uint16", "bfloat16"}

type cell struct {
	skip   bool // caps —
	pass   bool
	fail   bool
	execUS float64 // DLL mono_ns around graphExecute only (welvet.qnn)
	callUS float64 // Go wall around full C ABI (cgo + pack + exec + unpack)
	finUS  float64
	err    string
}

func caseLayerDtypeMatrix() error {
	sess, err := nativeqnn.DefaultSession()
	if err != nil {
		return err
	}
	layers := []string{
		"dense", "softmax", "rmsnorm", "layernorm",
		"cnn2", "cnn3", "convt2", "convt3",
		"lstm", "embedding", "residual", "swiglu", "mha",
	}
	grid := map[string]map[string]cell{}
	hosts := map[string]hostTiming{}
	for _, layer := range layers {
		grid[layer] = map[string]cell{}
		for _, dt := range matrixDtypes {
			grid[layer][dt] = measureCell(sess, layer, dt)
			if grid[layer][dt].fail {
				nativeqnn.CloseDefaultSession()
				sess, err = nativeqnn.DefaultSession()
				if err != nil {
					return err
				}
			}
		}
		hosts[layer] = measureHosts(layer)
	}
	printMatrix(layers, grid)
	printHosts(layers, hosts, grid)
	return writeResultsMD(layers, grid, hosts)
}

func timedABI(fn func() error) (callUS float64, err error) {
	t0 := time.Now()
	err = fn()
	return float64(time.Since(t0).Nanoseconds()) / 1e3, err
}

func finishCell(sess *nativeqnn.Session, callUS float64, err error) cell {
	ph := sess.LastPhase()
	c := cell{
		pass:   err == nil,
		fail:   err != nil,
		execUS: ph.ExecuteUS(),
		callUS: callUS,
		finUS:  ph.FinalizeUS(),
	}
	if err != nil {
		c.err = err.Error()
	}
	if c.pass && c.execUS < 1 && c.finUS < 1 {
		c.execUS = 0
	}
	// Go timer can under-resolve; call cannot be faster than DLL exec.
	if c.pass && c.execUS > 0 && (c.callUS < c.execUS || c.callUS < 1) {
		c.callUS = c.execUS
	}
	return c
}

func measureCell(sess *nativeqnn.Session, layer, dtype string) cell {
	if !nativeqnn.LayerSupported(layer, dtype) {
		return cell{skip: true}
	}
	switch layer {
	case "dense":
		b, in, out := int32(mx.DenseB), int32(mx.DenseIn), int32(mx.DenseOut)
		W := make([]float32, int(out*in))
		bias := make([]float32, out)
		x := make([]float32, int(b*in))
		y := make([]float32, int(b*out))
		fillRand(W, 1, 0.02)
		fillRand(x, 2, 0.1)
		if err := sess.DensePrepare(b, in, out, W, bias, dtype); err != nil {
			return cell{fail: true, err: err.Error()}
		}
		_ = sess.DenseExecuteF32(x, y)
		callUS, err := timedABI(func() error { return sess.DenseExecuteF32(x, y) })
		return finishCell(sess, callUS, err)
	case "softmax":
		rows, cols := int32(mx.SoftRows), int32(mx.SoftCols)
		x := make([]float32, rows*cols)
		y := make([]float32, len(x))
		fillRand(x, 3, 1)
		callUS, err := timedABI(func() error { return sess.SoftmaxForward(rows, cols, x, y, dtype) })
		return finishCell(sess, callUS, err)
	case "rmsnorm":
		b, seq, dim := int32(mx.NormB), int32(mx.NormSeq), int32(mx.NormDim)
		g := make([]float32, dim)
		x := make([]float32, b*seq*dim)
		y := make([]float32, len(x))
		for i := range g {
			g[i] = 1
		}
		fillRand(x, 4, 0.5)
		callUS, err := timedABI(func() error { return sess.RmsNormForward(b, seq, dim, 1e-6, g, x, y, dtype) })
		return finishCell(sess, callUS, err)
	case "layernorm":
		b, seq, dim := int32(mx.NormB), int32(mx.NormSeq), int32(mx.NormDim)
		g := make([]float32, dim)
		beta := make([]float32, dim)
		x := make([]float32, b*seq*dim)
		y := make([]float32, len(x))
		for i := range g {
			g[i] = 1
		}
		fillRand(x, 5, 0.4)
		callUS, err := timedABI(func() error {
			return sess.LayerNormForward(b, seq, dim, 1e-5, g, beta, x, y, dtype)
		})
		return finishCell(sess, callUS, err)
	case "cnn2":
		n, h, w, cin, cout := int32(mx.Cnn2N), int32(mx.Cnn2H), int32(mx.Cnn2W), int32(mx.Cnn2Cin), int32(mx.Cnn2Cout)
		W := make([]float32, cout*cin)
		bias := make([]float32, cout)
		x := make([]float32, n*h*w*cin)
		y := make([]float32, n*h*w*cout)
		fillRand(W, 6, 0.05)
		fillRand(x, 7, 0.1)
		callUS, err := timedABI(func() error {
			return sess.Cnn2Forward(n, h, w, cin, cout, W, bias, x, y, dtype)
		})
		return finishCell(sess, callUS, err)
	case "cnn3":
		n, d, h, w, cin, cout := int32(mx.Cnn3N), int32(mx.Cnn3D), int32(mx.Cnn3H), int32(mx.Cnn3W), int32(mx.Cnn3Cin), int32(mx.Cnn3Cout)
		W := make([]float32, cout*cin)
		bias := make([]float32, cout)
		x := make([]float32, n*d*h*w*cin)
		y := make([]float32, n*d*h*w*cout)
		fillRand(W, 8, 0.05)
		fillRand(x, 9, 0.1)
		callUS, err := timedABI(func() error {
			return sess.Cnn3Forward(n, d, h, w, cin, cout, W, bias, x, y, dtype)
		})
		return finishCell(sess, callUS, err)
	case "convt2":
		n, h, w, cin, cout := int32(mx.ConvT2N), int32(mx.ConvT2H), int32(mx.ConvT2W), int32(mx.ConvT2Cin), int32(mx.ConvT2Cout)
		W := make([]float32, cout*cin)
		bias := make([]float32, cout)
		x := make([]float32, n*h*w*cin)
		y := make([]float32, n*h*w*cout)
		fillRand(W, 10, 0.05)
		fillRand(x, 11, 0.1)
		callUS, err := timedABI(func() error {
			return sess.ConvT2Forward(n, h, w, cin, cout, W, bias, x, y, dtype)
		})
		return finishCell(sess, callUS, err)
	case "convt3":
		n, d, h, w, cin, cout := int32(mx.ConvT3N), int32(mx.ConvT3D), int32(mx.ConvT3H), int32(mx.ConvT3W), int32(mx.ConvT3Cin), int32(mx.ConvT3Cout)
		W := make([]float32, cout*cin)
		bias := make([]float32, cout)
		x := make([]float32, n*d*h*w*cin)
		y := make([]float32, n*d*h*w*cout)
		fillRand(W, 12, 0.05)
		fillRand(x, 13, 0.1)
		callUS, err := timedABI(func() error {
			return sess.ConvT3Forward(n, d, h, w, cin, cout, W, bias, x, y, dtype)
		})
		return finishCell(sess, callUS, err)
	case "lstm":
		seq, b, inSz, hid := int32(mx.LstmT), int32(mx.LstmB), int32(mx.LstmIn), int32(mx.LstmHid)
		wIn := make([]float32, 4*hid*inSz)
		wRec := make([]float32, 4*hid*hid)
		bias := make([]float32, 4*hid)
		x := make([]float32, seq*b*inSz)
		y := make([]float32, seq*b*hid)
		fillRand(wIn, 14, 0.02)
		fillRand(wRec, 15, 0.02)
		fillRand(x, 16, 0.1)
		callUS, err := timedABI(func() error {
			return sess.LstmForward(seq, b, inSz, hid, wIn, wRec, bias, x, y, dtype)
		})
		return finishCell(sess, callUS, err)
	case "embedding":
		vocab, dim, n := int32(mx.EmbVocab), int32(mx.EmbDim), int32(mx.EmbN)
		table := make([]float32, vocab*dim)
		ids := make([]int32, n)
		y := make([]float32, n*dim)
		fillRand(table, 17, 0.05)
		for i := range ids {
			ids[i] = int32(i % int(vocab))
		}
		callUS, err := timedABI(func() error {
			return sess.EmbeddingForward(vocab, dim, n, table, ids, y, dtype)
		})
		return finishCell(sess, callUS, err)
	case "residual":
		n := mx.ResN
		a := make([]float32, n)
		b := make([]float32, n)
		y := make([]float32, n)
		fillRand(a, 18, 0.5)
		fillRand(b, 19, 0.5)
		callUS, err := timedABI(func() error { return sess.ResidualForward(int32(n), a, b, y, dtype) })
		return finishCell(sess, callUS, err)
	case "swiglu":
		b, in, hid := int32(mx.SwiB), int32(mx.SwiIn), int32(mx.SwiHid)
		wG := make([]float32, hid*in)
		wU := make([]float32, hid*in)
		wD := make([]float32, in*hid)
		x := make([]float32, b*in)
		y := make([]float32, b*in)
		fillRand(wG, 20, 0.02)
		fillRand(wU, 21, 0.02)
		fillRand(wD, 22, 0.02)
		fillRand(x, 23, 0.1)
		callUS, err := timedABI(func() error {
			return sess.SwiGluForward(b, in, hid, wG, wU, wD, x, y, dtype)
		})
		return finishCell(sess, callUS, err)
	case "mha":
		b, seq, dim, heads := int32(mx.MhaB), int32(mx.MhaSeq), int32(mx.MhaDim), int32(mx.MhaHeads)
		wQ := make([]float32, dim*dim)
		wK := make([]float32, dim*dim)
		wV := make([]float32, dim*dim)
		wO := make([]float32, dim*dim)
		x := make([]float32, b*seq*dim)
		y := make([]float32, len(x))
		for i := 0; i < int(dim); i++ {
			wQ[i*int(dim)+i], wK[i*int(dim)+i], wV[i*int(dim)+i], wO[i*int(dim)+i] = 1, 1, 1, 1
		}
		fillRand(x, 24, 0.05)
		callUS, err := timedABI(func() error {
			return sess.MhaForward(b, seq, dim, heads, wQ, wK, wV, wO, x, y, dtype)
		})
		return finishCell(sess, callUS, err)
	default:
		return cell{skip: true}
	}
}

func cellMark(c cell) string {
	switch {
	case c.skip:
		return "—"
	case c.fail:
		return "❌"
	case c.pass:
		return "✅"
	default:
		return "?"
	}
}

func cellExec(c cell) string {
	switch {
	case c.skip:
		return "—"
	case c.fail:
		return "FAIL"
	case c.execUS >= 1000:
		return fmt.Sprintf("%.1fms", c.execUS/1000)
	case c.execUS >= 1:
		return fmt.Sprintf("%.0fµs", c.execUS)
	case c.pass:
		return "ok"
	default:
		return "?"
	}
}

func cellCall(c cell) string {
	switch {
	case c.skip:
		return "—"
	case c.fail:
		return "FAIL"
	case c.callUS >= 1000:
		return fmt.Sprintf("%.1fms", c.callUS/1000)
	case c.callUS >= 1:
		return fmt.Sprintf("%.0fµs", c.callUS)
	case c.callUS > 0:
		return fmt.Sprintf("%.2fµs", c.callUS)
	case c.pass:
		return "?"
	default:
		return "?"
	}
}

func cellOverhead(c cell) string {
	if c.skip || c.fail || c.callUS < 1 || c.execUS < 1 {
		return "—"
	}
	oh := c.callUS - c.execUS
	if oh < 0 {
		oh = 0
	}
	if oh >= 1000 {
		return fmt.Sprintf("%.1fms", oh/1000)
	}
	return fmt.Sprintf("%.0fµs", oh)
}

func printMatrix(layers []string, grid map[string]map[string]cell) {
	fmt.Println("      HTP status (layer × dtype):")
	fmt.Printf("      %-12s", "layer")
	for _, dt := range matrixDtypes {
		fmt.Printf(" %8s", shortDT(dt))
	}
	fmt.Println()
	for _, layer := range layers {
		fmt.Printf("      %-12s", layer)
		for _, dt := range matrixDtypes {
			fmt.Printf(" %8s", cellMark(grid[layer][dt]))
		}
		fmt.Println()
	}
	fmt.Println("      NPU DLL exec (graphExecute only):")
	fmt.Printf("      %-12s", "layer")
	for _, dt := range matrixDtypes {
		fmt.Printf(" %8s", shortDT(dt))
	}
	fmt.Println()
	for _, layer := range layers {
		fmt.Printf("      %-12s", layer)
		for _, dt := range matrixDtypes {
			fmt.Printf(" %8s", cellExec(grid[layer][dt]))
		}
		fmt.Println()
	}
	fmt.Println("      Go→NPU call (cgo+pack+exec+unpack):")
	fmt.Printf("      %-12s", "layer")
	for _, dt := range matrixDtypes {
		fmt.Printf(" %8s", shortDT(dt))
	}
	fmt.Println()
	for _, layer := range layers {
		fmt.Printf("      %-12s", layer)
		for _, dt := range matrixDtypes {
			fmt.Printf(" %8s", cellCall(grid[layer][dt]))
		}
		fmt.Println()
	}
}

func formatHostUS(us float64, ok bool) string {
	if !ok {
		return "—"
	}
	if us >= 1000 {
		return fmt.Sprintf("%.2fms", us/1000)
	}
	if us < 10 {
		return fmt.Sprintf("%.2fµs", us)
	}
	return fmt.Sprintf("%.0fµs", us)
}

func pctFaster(slow, fast float64) float64 {
	if slow <= 0 || fast <= 0 {
		return 0
	}
	return (slow - fast) / slow * 100
}

func hostOK(us float64, ok bool) bool {
	return ok && us > 0
}

func npuWinsHost(h hostTiming, npu cell) bool {
	if !npu.pass || npu.execUS <= 0 {
		return false
	}
	goOK, natOK := hostOK(h.goUS, h.goOK), hostOK(h.natUS, h.natOK)
	if !goOK && !natOK {
		return false
	}
	if goOK && npu.execUS >= h.goUS {
		return false
	}
	if natOK && npu.execUS >= h.natUS {
		return false
	}
	return true
}

func fastestName(h hostTiming, npu cell) string {
	goOK, natOK := hostOK(h.goUS, h.goOK), hostOK(h.natUS, h.natOK)
	npuOK := npu.pass && npu.execUS > 0
	best, name := 1e99, "—"
	if goOK && h.goUS < best {
		best, name = h.goUS, "Go"
	}
	if natOK && h.natUS < best {
		best, name = h.natUS, "native"
	}
	if npuOK && npu.execUS < best {
		name = "NPU"
	}
	return name
}

func vsHost(hostUS float64, hostOK bool, npuUS float64, npuOK bool) string {
	if !hostOK || !npuOK {
		return "—"
	}
	if npuUS < hostUS {
		return fmt.Sprintf("NPU **%.0f%%**", pctFaster(hostUS, npuUS))
	}
	return fmt.Sprintf("host **%.0f%%**", pctFaster(npuUS, hostUS))
}

func printHosts(layers []string, hosts map[string]hostTiming, grid map[string]map[string]cell) {
	fmt.Println("      Go vs native vs NPU (all layer×dtype; NPU=DLL exec):")
	fmt.Printf("      %-12s %6s %10s %10s %10s %10s  %s\n", "layer", "dtype", "go", "native", "npu", "call", "fastest")
	for _, layer := range layers {
		h := hosts[layer]
		goOK, natOK := hostOK(h.goUS, h.goOK), hostOK(h.natUS, h.natOK)
		for _, dt := range matrixDtypes {
			npu := grid[layer][dt]
			if npu.skip {
				continue
			}
			fmt.Printf("      %-12s %6s %10s %10s %10s %10s  %s\n",
				layer, shortDT(dt),
				formatHostUS(h.goUS, goOK),
				formatHostUS(h.natUS, natOK),
				cellExec(npu), cellCall(npu),
				fastestName(h, npu))
		}
	}
}

func shortDT(dt string) string {
	switch dt {
	case "float32":
		return "f32"
	case "float16":
		return "f16"
	case "bfloat16":
		return "bf16"
	case "int8":
		return "i8"
	case "uint8":
		return "u8"
	case "int16":
		return "i16"
	case "uint16":
		return "u16"
	default:
		return dt
	}
}

func writeDtypeHeader(b *strings.Builder) {
	b.WriteString("| Layer | f32 | f16 | i8 | u8 | i16 | u16 | bf16 |\n")
	b.WriteString("|-------|-----|-----|----|----|-----|-----|------|\n")
}

func writeResultsMD(layers []string, grid map[string]map[string]cell, hosts map[string]hostTiming) error {
	path := resultsPath()
	var b strings.Builder
	b.WriteString("# go_v_qnn results — layer × dtype (Hexagon V73 HTP)\n\n")
	b.WriteString("**Host:** Windows ARM64 · Snapdragon X · `nativeqnn` + QAIRT 2.42  \n")
	b.WriteString(fmt.Sprintf("**Run:** %s · **large** matrix shapes (same geometry Go / native / NPU)  \n", time.Now().Format("2006-01-02")))
	b.WriteString("**Shapes:** " + mxGeomNote() + "\n\n")
	b.WriteString("### Timing (read this)\n\n")
	b.WriteString("- **NPU DLL exec** — `mono_ns` *inside* `welvet.qnn` around `graphExecute` only (excludes cgo, f32↔dtype pack/unpack, bias epilogue).\n")
	b.WriteString("- **Go→NPU call** — Go wall around full C ABI (cgo + pack + DLL + unpack). Online layers also rebuild the graph each call.\n")
	b.WriteString("- **Dense** = prepare-once + warmed execute. Other layers = online compose today.\n")
	b.WriteString("- **Go / native** = **f32 CPU only** (one column). Welvet host does not run HTP quant kernels — dtype matrix is **NPU-only**. Compare each NPU dtype row against that f32 host baseline.\n\n")
	b.WriteString("Legend: ✅ pass · ❌ fail · — caps skip (incl. bf16 = V81+)\n\n")

	b.WriteString("## Status (layer × dtype)\n\n")
	writeDtypeHeader(&b)
	for _, layer := range layers {
		b.WriteString("| " + layer)
		for _, dt := range matrixDtypes {
			b.WriteString(" | " + cellMark(grid[layer][dt]))
		}
		b.WriteString(" |\n")
	}

	b.WriteString("\n## Go / native exec (f32 CPU, same shapes as NPU)\n\n")
	b.WriteString("One number per layer — host has no per-dtype HTP path. `—` = not wired yet.\n\n")
	b.WriteString("| Layer | Go f32 | Native f32 |\n|-------|--------|------------|\n")
	for _, layer := range layers {
		h := hosts[layer]
		b.WriteString(fmt.Sprintf("| %s | %s | %s |\n",
			layer,
			formatHostUS(h.goUS, hostOK(h.goUS, h.goOK)),
			formatHostUS(h.natUS, hostOK(h.natUS, h.natOK))))
	}

	b.WriteString("\n## NPU DLL exec (layer × dtype)\n\n")
	b.WriteString("`graphExecute` only, measured in `welvet.qnn`. **This** is where dtypes differ.\n\n")
	writeDtypeHeader(&b)
	for _, layer := range layers {
		b.WriteString("| " + layer)
		for _, dt := range matrixDtypes {
			b.WriteString(" | " + cellExec(grid[layer][dt]))
		}
		b.WriteString(" |\n")
	}

	b.WriteString("\n## Go→NPU call (layer × dtype)\n\n")
	b.WriteString("Full ABI from Go. Includes create/finalize for online layers.\n\n")
	writeDtypeHeader(&b)
	for _, layer := range layers {
		b.WriteString("| " + layer)
		for _, dt := range matrixDtypes {
			b.WriteString(" | " + cellCall(grid[layer][dt]))
		}
		b.WriteString(" |\n")
	}

	b.WriteString("\n## Go→NPU overhead = call − DLL exec\n\n")
	b.WriteString("cgo + pack/unpack (+ graph build on online layers).\n\n")
	writeDtypeHeader(&b)
	for _, layer := range layers {
		b.WriteString("| " + layer)
		for _, dt := range matrixDtypes {
			b.WriteString(" | " + cellOverhead(grid[layer][dt]))
		}
		b.WriteString(" |\n")
	}

	b.WriteString("\n## Go vs native vs NPU — every layer × dtype\n\n")
	b.WriteString("NPU column = DLL exec. **%** = `(slower−faster)/slower×100` vs that host.\n\n")
	b.WriteString("| Layer | dtype | Go | Native | NPU | call | vs Go | vs Native | Fastest |\n")
	b.WriteString("|-------|-------|-----|--------|-----|------|-------|-----------|--------|\n")
	for _, layer := range layers {
		h := hosts[layer]
		goOK, natOK := hostOK(h.goUS, h.goOK), hostOK(h.natUS, h.natOK)
		for _, dt := range matrixDtypes {
			npu := grid[layer][dt]
			if npu.skip {
				continue
			}
			npuOK := npu.pass && npu.execUS > 0
			b.WriteString(fmt.Sprintf("| %s | %s | %s | %s | %s | %s | %s | %s | **%s** |\n",
				layer, shortDT(dt),
				formatHostUS(h.goUS, goOK),
				formatHostUS(h.natUS, natOK),
				cellExec(npu), cellCall(npu),
				vsHost(h.goUS, goOK, npu.execUS, npuOK),
				vsHost(h.natUS, natOK, npu.execUS, npuOK),
				fastestName(h, npu)))
		}
	}

	b.WriteString("\n## Stack total — all layers on Go vs native vs NPU\n\n")
	b.WriteString("Sum of per-layer exec (skip missing). Go/native summed once per layer (f32). NPU summed per dtype.\n\n")
	b.WriteString("| Backend | f32 | f16 | i8 | u8 | i16 | u16 | bf16 |\n")
	b.WriteString("|---------|-----|-----|----|----|-----|-----|------|\n")
	var goSum, natSum float64
	goN, natN := 0, 0
	npuSum := map[string]float64{}
	npuN := map[string]int{}
	for _, layer := range layers {
		h := hosts[layer]
		if hostOK(h.goUS, h.goOK) {
			goSum += h.goUS
			goN++
		}
		if hostOK(h.natUS, h.natOK) {
			natSum += h.natUS
			natN++
		}
		for _, dt := range matrixDtypes {
			c := grid[layer][dt]
			if c.pass && c.execUS > 0 {
				npuSum[dt] += c.execUS
				npuN[dt]++
			}
		}
	}
	writeStackRow := func(name string, perDt func(dt string) string) {
		b.WriteString("| " + name)
		for _, dt := range matrixDtypes {
			b.WriteString(" | " + perDt(dt))
		}
		b.WriteString(" |\n")
	}
	fmtSum := func(us float64, n int) string {
		if n == 0 {
			return "—"
		}
		return formatHostUS(us, true) + fmt.Sprintf(" (%d)", n)
	}
	writeStackRow(fmt.Sprintf("Go f32 (%d layers)", goN), func(string) string { return fmtSum(goSum, goN) })
	writeStackRow(fmt.Sprintf("native f32 (%d layers)", natN), func(string) string { return fmtSum(natSum, natN) })
	writeStackRow("NPU DLL", func(dt string) string { return fmtSum(npuSum[dt], npuN[dt]) })
	writeStackRow("NPU call", func(dt string) string {
		var s float64
		n := 0
		for _, layer := range layers {
			c := grid[layer][dt]
			if c.pass && c.callUS > 0 {
				s += c.callUS
				n++
			}
		}
		return fmtSum(s, n)
	})

	b.WriteString("\n## NPU wins only (subset)\n\n")
	b.WriteString("| Layer | dtype | Go | Native | NPU | vs Go | vs Native |\n")
	b.WriteString("|-------|-------|-----|--------|-----|-------|----------|\n")
	anyWin := false
	for _, layer := range layers {
		h := hosts[layer]
		goOK, natOK := hostOK(h.goUS, h.goOK), hostOK(h.natUS, h.natOK)
		for _, dt := range matrixDtypes {
			npu := grid[layer][dt]
			if !npuWinsHost(h, npu) {
				continue
			}
			anyWin = true
			b.WriteString(fmt.Sprintf("| %s | %s | %s | %s | %s | %s | %s |\n",
				layer, shortDT(dt),
				formatHostUS(h.goUS, goOK),
				formatHostUS(h.natUS, natOK),
				cellExec(npu),
				vsHost(h.goUS, goOK, npu.execUS, true),
				vsHost(h.natUS, natOK, npu.execUS, true)))
		}
	}
	if !anyWin {
		b.WriteString("| _none_ | — | — | — | — | — | — |\n")
	}

	b.WriteString("\n## Failures\n\n")
	any := false
	for _, layer := range layers {
		for _, dt := range matrixDtypes {
			c := grid[layer][dt]
			if c.fail {
				any = true
				b.WriteString(fmt.Sprintf("- **%s/%s**: `%s`\n", layer, dt, c.err))
			}
		}
	}
	if !any {
		b.WriteString("_None._\n")
	}
	b.WriteString("\n## Re-run\n\n```bash\ncd w2a\ngo test ./tests/go_v_qnn -v -count=1 -run TestGoVQNNMatrix -timeout 180m\n```\n")
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func resultsPath() string {
	return filepath.Join(`c:\git\w2a\suites\go_v_qnn`, "RESULTS.md")
}
