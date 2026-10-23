// Package go_v_qnn times Go CPU vs Hexagon QNN on every README HTP layer.
//
// Protocol (per layer):
//  1. Build identical weights / activations in Go
//  2. Transfer — materialize / push weights (timed, outside exec)
//  3. Graph prep — create + finalize separately (LastPhase / DensePrepare)
//  4. Exec — graphExecute only after weights are transferred (headline number)
//  5. E2E = transfer + create + finalize + exec
//  6. Accuracy — go↔qnn exact%, max|Δ|, RMSE (where Go ref is wired)
package go_v_qnn

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/nativeqnn"
)

type Case struct {
	Name string
	Run  func() error
}

func Cases() []Case {
	return []Case{
		{Name: "Matrix — all README layers × dtypes (writes RESULTS.md)", Run: caseLayerDtypeMatrix},
		{Name: "Caps — README HTP layers × float32", Run: caseCapsInventory},
		{Name: "Dense f32 B=64 in=1024 out=1024 — transfer/prep/exec/e2e + accuracy", Run: caseLayerDense},
		{Name: "Softmax f32 128×8192 — transfer/prep/exec/e2e + accuracy", Run: caseLayerSoftmax},
		{Name: "RMSNorm f32 [16,256,1024] — transfer/prep/exec/e2e + accuracy", Run: caseLayerRMSNorm},
		{Name: "LayerNorm f32 [16,256,1024] — transfer/prep/exec/e2e + accuracy", Run: caseLayerLayerNorm},
		{Name: "CNN2 f32 8×128×128×64→64 — transfer/prep/exec/e2e + accuracy", Run: caseLayerCnn2},
		{Name: "CNN3 f32 2×16³×32→32 — transfer/prep/exec/e2e", Run: caseLayerCnn3},
		{Name: "ConvT2 f32 8×64×64×64→64 — transfer/prep/exec/e2e", Run: caseLayerConvT2},
		{Name: "ConvT3 f32 2×16³×32→32 — transfer/prep/exec/e2e", Run: caseLayerConvT3},
		{Name: "LSTM f32 T=8 B=8 in=128 hid=128 — transfer/prep/exec/e2e", Run: caseLayerLstm},
		{Name: "Embedding f32 vocab=16k dim=1024 n=1024 — transfer/prep/exec/e2e + accuracy", Run: caseLayerEmbedding},
		{Name: "Residual f32 n=256k — transfer/prep/exec/e2e + accuracy", Run: caseLayerResidual},
		{Name: "SwiGLU f32 B=16 in=512 hidden=1408 — transfer/prep/exec/e2e", Run: caseLayerSwiGlu},
		{Name: "MHA f32 B=2 S=64 D=256 H=4 — transfer/prep/exec/e2e", Run: caseLayerMha},
		{Name: "Sequential Dense tower dim=128 depth=3 — go/native/qnn exec-only", Run: caseSequentialBig},
	}
}

func RunAll() error {
	if !nativeqnn.Available() {
		fmt.Println("  SKIP go_v_qnn: libwelvet_qnn not available (need cgo + windows/arm64 + DLL on PATH)")
		return nil
	}
	fmt.Printf("  nativeqnn=%s  nativecpp=%v  layers=%v\n",
		nativeqnn.Version(), nativecpp.NativeCPUAvailable(), nativeqnn.SupportedLayers())
	fmt.Println("  measure: transfer → create/finalize → exec-only (after weights) + e2e + go↔qnn exact%/max|Δ|/RMSE")
	var fails []string
	for i, c := range Cases() {
		nativeqnn.CloseDefaultSession()
		suites.BeginCase()
		fmt.Printf("  [%d] %s …\n", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_qnn", c.Name, "FAIL", err.Error())
			fmt.Printf("      FAIL: %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_qnn", c.Name, "PASS", "")
		fmt.Println("      PASS")
	}
	nativeqnn.CloseDefaultSession()
	if len(fails) > 0 {
		return fmt.Errorf("go_v_qnn: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func RunOne(n int) error {
	if !nativeqnn.Available() {
		return fmt.Errorf("go_v_qnn: libwelvet_qnn not available")
	}
	cs := Cases()
	if n < 1 || n > len(cs) {
		return fmt.Errorf("go_v_qnn: case %d out of range 1..%d", n, len(cs))
	}
	suites.BeginCase()
	fmt.Printf("  [%d] %s …\n", n, cs[n-1].Name)
	if err := cs[n-1].Run(); err != nil {
		suites.EndCase("go_v_qnn", cs[n-1].Name, "FAIL", err.Error())
		fmt.Printf("      FAIL: %v\n", err)
		return err
	}
	suites.EndCase("go_v_qnn", cs[n-1].Name, "PASS", "")
	fmt.Println("      PASS")
	return nil
}

func openSess() (*nativeqnn.Session, error) {
	return nativeqnn.DefaultSession()
}

func runLayer(fn func(*nativeqnn.Session) LayerReport) error {
	sess, err := openSess()
	if err != nil {
		return err
	}
	r := fn(sess)
	printReport(r)
	return gateReport(r)
}

func caseLayerDense() error      { return runLayer(measureDenseBig) }
func caseLayerSoftmax() error    { return runLayer(measureSoftmaxBig) }
func caseLayerRMSNorm() error    { return runLayer(measureRmsNormBig) }
func caseLayerLayerNorm() error  { return runLayer(measureLayerNormBig) }
func caseLayerCnn2() error       { return runLayer(measureCnn2Big) }
func caseLayerCnn3() error       { return runLayer(measureCnn3Big) }
func caseLayerConvT2() error     { return runLayer(measureConvT2Big) }
func caseLayerConvT3() error     { return runLayer(measureConvT3Big) }
func caseLayerLstm() error       { return runLayer(measureLstmBig) }
func caseLayerEmbedding() error  { return runLayer(measureEmbeddingBig) }
func caseLayerResidual() error   { return runLayer(measureResidualBig) }
func caseLayerSwiGlu() error     { return runLayer(measureSwiGluBig) }
func caseLayerMha() error        { return runLayer(measureMhaBig) }

func caseCapsInventory() error {
	want := []string{
		"dense", "softmax", "rmsnorm", "layernorm",
		"cnn2", "cnn3", "convt2", "convt3",
		"lstm", "embedding", "residual", "swiglu", "mha",
	}
	caps := map[string]bool{}
	for _, n := range nativeqnn.SupportedLayers() {
		caps[n] = true
	}
	fmt.Println("      HTP caps (library) — suite times all README f32 rows:")
	for _, n := range want {
		ok := nativeqnn.LayerSupported(n, "float32")
		mark := "  ← timed"
		if !caps[n] {
			mark = "  MISSING from SupportedLayers"
		}
		fmt.Printf("        %-12s f32=%v%s\n", n, ok, mark)
		if !caps[n] || !ok {
			return fmt.Errorf("%s/float32 not supported on HTP", n)
		}
	}
	return nil
}
