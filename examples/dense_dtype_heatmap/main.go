// Dense dtype conversion pipeline (volumetric) + ASCII eval heatmap.
//
// Train a small 2×2×2 Dense grid in FP32, snapshot .entity, then for every
// core.AllDTypes: deserialize → Convert FormatNone → forward-eval vs the FP32
// reference outputs. Prints a heatmap of relative MSE / max|Δ|.
//
//	cd apps/w2a && go run ./examples/dense_dtype_heatmap
package main

import (
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/stub/serialization"
	"github.com/openfluke/welvet/weights"
)

const (
	depth, rows, cols, lpc = 2, 2, 2, 1 // 8 volumetric Dense cells
	dim                    = 8
	epochs                 = 80
	lr                     = 0.08
	evalN                  = 16
)

type row struct {
	dtype     core.DType
	name      string
	status    string // OK / FAIL / SKIP
	note      string
	mse       float64 // output drift vs FP32 reference outputs
	maxAbs    float64
	taskMSE   float64 // MSE vs ground-truth targets
	qlossPct  float64 // % quality loss vs FP32 task MSE: 100*(task-fp32)/fp32
	bytes     int
	shrink    float64 // entity bytes / fp32 entity bytes
}

func main() {
	fmt.Println("=== volumetric Dense FP32 train → .entity → convert all dtypes ===")
	fmt.Printf("grid %d×%d×%d×%d  dense %d→%d  epochs=%d\n\n", depth, rows, cols, lpc, dim, dim, epochs)

	g := buildFP32Volumetric()
	xTrain, yTrain := toyBatch(32, 1)
	trainFP32(g, xTrain, yTrain)

	entity, err := serialization.SerializeEntity(g)
	must(err)
	fmt.Printf("FP32 .entity snapshot: %d bytes\n", len(entity))

	// Eval against the *reloaded* FP32 entity (fair convert baseline).
	gRef, err := serialization.DeserializeEntity(entity)
	must(err)
	xEval, yEval := toyBatch(evalN, 99)
	refOut := make([][]float32, evalN)
	var fp32TaskMSE float64
	for i := 0; i < evalN; i++ {
		out := mustFwd(gRef, xEval[i])
		refOut[i] = append([]float32(nil), out.Data...)
		loss, _ := training.MSE(out, yEval[i])
		fp32TaskMSE += loss
	}
	fp32TaskMSE /= float64(evalN)
	fmt.Printf("FP32 hold-out task MSE (reloaded .entity): %.6f\n\n", fp32TaskMSE)

	rowsOut := make([]row, 0, len(core.AllDTypes))
	for _, dt := range core.AllDTypes {
		r := evalConverted(entity, dt, xEval, yEval, refOut, len(entity), fp32TaskMSE)
		rowsOut = append(rowsOut, r)
	}

	printTable(rowsOut, fp32TaskMSE)
	printHeatmap(rowsOut)
	fmt.Println("\ndone — train FP32, ship .entity, Convert(dtype, FormatNone) per cell, re-eval.")
}

func buildFP32Volumetric() *architecture.Grid {
	g := architecture.NewGrid(depth, rows, cols, lpc)
	g.Exec.Backend = core.BackendCPUTiled
	g.Exec.MultiCore = false

	for z := 0; z < depth; z++ {
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				init := make([]float32, dim*dim)
				// mild identity + noise so convert error is visible
				for i := 0; i < dim; i++ {
					init[i*dim+i] = 0.5
				}
				seed := z*100 + y*10 + x
				for i := range init {
					init[i] += 0.02 * float32(((seed+i)*17)%7-3)
				}
				l, err := dense.NewConfigured(dim, dim, core.ActivationReLU, core.DTypeFloat32, quant.FormatNone, init)
				must(err)
				must(dense.Place(g, z, y, x, 0, l))
			}
		}
	}
	// one spatial hop so the volume isn't a pure chain (cell at z=1,y=0,x=0
	// also peeks at the first cell's activation)
	_ = g.SetRemoteLink(1, 0, 0, 0, 0, 0, 0, 0)
	return g
}

func trainFP32(g *architecture.Grid, xs, ys []*core.Tensor[float32]) {
	for ep := 0; ep < epochs; ep++ {
		var lossSum float64
		for i := range xs {
			fwd, err := forward.Forward(g, xs[i])
			must(err)
			loss, err := training.Step(fwd, ys[i], lr)
			must(err)
			lossSum += loss
		}
		if ep == 0 || ep == epochs-1 || ep%20 == 19 {
			fmt.Printf("  epoch %3d  mean SGD loss=%.6f\n", ep+1, lossSum/float64(len(xs)))
		}
	}
}

func toyBatch(n, seed int) (xs, ys []*core.Tensor[float32]) {
	xs = make([]*core.Tensor[float32], n)
	ys = make([]*core.Tensor[float32], n)
	for i := 0; i < n; i++ {
		x := core.NewTensor[float32](1, dim)
		y := core.NewTensor[float32](1, dim)
		for j := 0; j < dim; j++ {
			// smooth target: soft one-hot-ish bump + scaled input
			v := float32(math.Sin(float64(seed+i*3+j)*0.37))
			x.Data[j] = 0.5*v + 0.1*float32(j)
			y.Data[j] = 0.5 + 0.4*float32(math.Tanh(float64(v)*1.2))
		}
		xs[i], ys[i] = x, y
	}
	return xs, ys
}

func evalConverted(
	entity []byte,
	dt core.DType,
	xs, ys []*core.Tensor[float32],
	ref [][]float32,
	fp32Bytes int,
	fp32TaskMSE float64,
) row {
	r := row{dtype: dt, name: dt.String(), status: "OK"}
	g, err := serialization.DeserializeEntity(entity)
	if err != nil {
		r.status, r.note = "FAIL", "deserialize: "+err.Error()
		return r
	}
	if err := convertAllDense(g, dt); err != nil {
		r.status, r.note = "FAIL", err.Error()
		return r
	}
	ent2, err := serialization.SerializeEntity(g)
	if err == nil {
		r.bytes = len(ent2)
		if fp32Bytes > 0 {
			r.shrink = float64(r.bytes) / float64(fp32Bytes)
		}
	}

	var driftSum, maxAbs, taskSum float64
	okN := 0
	for i := range xs {
		out, err := forward.Forward(g, xs[i])
		if err != nil {
			r.status, r.note = "FAIL", fmt.Sprintf("forward: %v", err)
			return r
		}
		m, mx := compare(out.Output.Data, ref[i])
		driftSum += m
		if mx > maxAbs {
			maxAbs = mx
		}
		tl, err := training.MSE(out.Output, ys[i])
		if err != nil {
			r.status, r.note = "FAIL", fmt.Sprintf("task MSE: %v", err)
			return r
		}
		taskSum += tl
		okN++
	}
	r.mse = driftSum / float64(okN)
	r.maxAbs = maxAbs
	r.taskMSE = taskSum / float64(okN)
	r.qlossPct = qualityLossPct(r.taskMSE, fp32TaskMSE)
	return r
}

// qualityLossPct is how much worse task MSE is vs FP32: 100 * (task - fp32) / fp32.
// FP32 row → 0%. Better-than-FP32 (rare) → negative. Broken → huge positive.
func qualityLossPct(taskMSE, fp32TaskMSE float64) float64 {
	base := fp32TaskMSE
	if base < 1e-12 {
		base = 1e-12
	}
	return 100 * (taskMSE - fp32TaskMSE) / base
}

func convertAllDense(g *architecture.Grid, dt core.DType) error {
	for _, c := range g.HopOrder() {
		cell := g.At(c.Z, c.Y, c.X, c.L)
		if cell == nil || cell.Op == nil {
			continue
		}
		dl, ok := cell.Op.(*dense.Layer)
		if !ok || dl == nil || dl.Weights == nil {
			continue
		}
		if err := weights.Convert(dl.Weights, weights.ConvertOpts{
			DType:  dt,
			Format: quant.FormatNone,
		}); err != nil {
			return fmt.Errorf("%s @ %v: %w", dt, c, err)
		}
	}
	return nil
}

func compare(a, b []float32) (mse, maxAbs float64) {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		d := float64(a[i] - b[i])
		mse += d * d
		if ad := math.Abs(d); ad > maxAbs {
			maxAbs = ad
		}
	}
	if n > 0 {
		mse /= float64(n)
	}
	return mse, maxAbs
}

func mustFwd(g *architecture.Grid, x *core.Tensor[float32]) *core.Tensor[float32] {
	fwd, err := forward.Forward(g, x)
	must(err)
	return fwd.Output
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func printTable(rows []row, fp32TaskMSE float64) {
	fmt.Printf("Hold-out eval  (FP32 task MSE=%.6f)\n", fp32TaskMSE)
	fmt.Printf("qloss%% = 100*(task_mse - fp32_task_mse)/fp32_task_mse  (0%% = same quality as FP32)\n")
	fmt.Printf("%-12s %-6s %10s %9s %12s %10s %8s %8s  %s\n",
		"dtype", "stat", "task_mse", "qloss%", "mse_vs_fp32", "max|d|", "bytes", "shrink", "note")
	for _, r := range rows {
		ql := "—"
		tm := "—"
		if r.status == "OK" {
			ql = fmt.Sprintf("%8.2f%%", r.qlossPct)
			tm = fmt.Sprintf("%10.6g", r.taskMSE)
		}
		fmt.Printf("%-12s %-6s %10s %9s %12.6g %10.4f %8d %7.2fx  %s\n",
			r.name, r.status, tm, ql, r.mse, r.maxAbs, r.bytes, r.shrink, r.note)
	}
	fmt.Println()
}

func printHeatmap(rows []row) {
	fmt.Println("Heatmap (lighter = better; denser = worse; X = fail)")
	fmt.Println("  qloss% vs FP32:")
	printStrip(rows, func(r row) float64 {
		if r.status != "OK" {
			return math.Inf(1)
		}
		if r.qlossPct < 0 {
			return 0
		}
		return r.qlossPct
	})
	fmt.Println("  mse_vs_fp32:")
	printStrip(rows, func(r row) float64 {
		if r.status != "OK" {
			return math.Inf(1)
		}
		return r.mse
	})
	fmt.Println("  max|d|:")
	printStrip(rows, func(r row) float64 {
		if r.status != "OK" {
			return math.Inf(1)
		}
		return r.maxAbs
	})
	fmt.Println()
	var b strings.Builder
	b.WriteString("  ")
	for _, r := range rows {
		short := r.name
		if len(short) > 4 {
			short = short[:4]
		}
		b.WriteString(fmt.Sprintf("%-5s", short))
	}
	fmt.Println(b.String())
}

func printStrip(rows []row, val func(row) float64) {
	// ASCII-only so terminals don't mojibake block glyphs
	glyphs := []byte(" .:+*#@")
	vals := make([]float64, len(rows))
	var max float64
	for i, r := range rows {
		v := val(r)
		if math.IsInf(v, 1) || math.IsNaN(v) {
			vals[i] = -1
			continue
		}
		vals[i] = v
		if v > max {
			max = v
		}
	}
	if max <= 0 {
		max = 1e-12
	}
	var b strings.Builder
	b.WriteString("  ")
	for _, v := range vals {
		if v < 0 {
			b.WriteString("X ")
			continue
		}
		t := math.Log10(1+v*1e6) / math.Log10(1+max*1e6)
		idx := int(t * float64(len(glyphs)-1))
		if idx < 0 {
			idx = 0
		}
		if idx >= len(glyphs) {
			idx = len(glyphs) - 1
		}
		b.WriteByte(glyphs[idx])
		b.WriteByte(' ')
	}
	fmt.Println(b.String())
}
