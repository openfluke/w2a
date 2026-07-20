package helpers

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/stub/clustering"
	"github.com/openfluke/welvet/stub/ensemble"
	"github.com/openfluke/welvet/stub/evaluation"
	"github.com/openfluke/welvet/stub/grafting"
	"github.com/openfluke/welvet/stub/grouping"
	"github.com/openfluke/welvet/stub/introspection"
	"github.com/openfluke/welvet/stub/observer"
	"github.com/openfluke/welvet/stub/pipeline"
	"github.com/openfluke/welvet/stub/templates"
	"github.com/openfluke/welvet/stub/universal"
)

type Case struct {
	Name string
	Run  func() error
}

func Cases() []Case {
	return []Case{
		{Name: "Grafting GraftGrids Dense branches", Run: graftSmoke},
		{Name: "Templates ChatML BuildPrompt", Run: templateSmoke},
		{Name: "Ensemble MajorityVote + matches", Run: ensembleSmoke},
		{Name: "Clustering KMeans + grouping DetectMHA", Run: clusterGroupSmoke},
		{Name: "Observer stats + BufferObserver", Run: observerSmoke},
		{Name: "Introspection GetMethods on Grid", Run: introspectionSmoke},
		{Name: "Evaluation EvaluatePrediction", Run: evaluationSmoke},
		{Name: "Universal ProbeDeepGeometry + Mount", Run: universalSmoke},
		{Name: "Pipeline TokenTimelineSummary", Run: pipelineSmoke},
	}
}

func RunAll() error {
	var fails []string
	for i, c := range Cases() {
		suites.BeginCase()
		fmt.Printf("  [%d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("helpers", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("helpers", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("helpers: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func RunOne(n int) error {
	cs := Cases()
	if n < 1 || n > len(cs) {
		return fmt.Errorf("helpers: case %d out of range 1..%d", n, len(cs))
	}
	suites.BeginCase()
	fmt.Printf("  [%d] %s … ", n, cs[n-1].Name)
	if err := cs[n-1].Run(); err != nil {
		suites.EndCase("helpers", cs[n-1].Name, "FAIL", err.Error())
		fmt.Printf("FAIL\n      %v\n", err)
		return err
	}
	suites.EndCase("helpers", cs[n-1].Name, "PASS", "")
	fmt.Println("PASS")
	return nil
}

func tinyDenseGrid(in, out int) (*architecture.Grid, error) {
	g := architecture.NewGrid(1, 1, 1, 1)
	l, err := dense.New(in, out, core.ActivationLinear, core.DTypeFloat32)
	if err != nil {
		return nil, err
	}
	return g, dense.Place(g, 0, 0, 0, 0, l)
}

func graftSmoke() error {
	g1, err := tinyDenseGrid(4, 4)
	if err != nil {
		return err
	}
	g2, err := tinyDenseGrid(4, 4)
	if err != nil {
		return err
	}
	par, err := grafting.GraftGrids([]*architecture.Grid{g1, g2}, parallel.CombineConcat)
	if err != nil {
		return err
	}
	if len(par.Branches) != 2 {
		return fmt.Errorf("want 2 branches got %d", len(par.Branches))
	}
	return nil
}

func templateSmoke() error {
	p := templates.ChatML.BuildPrompt(nil, "be helpful", "hello")
	if !strings.Contains(p, "hello") || !strings.Contains(p, "system") {
		return fmt.Errorf("unexpected prompt: %q", p)
	}
	seg := templates.ChatML.BuildNextTurnSegment("next")
	if !strings.Contains(seg, "next") {
		return fmt.Errorf("bad segment: %q", seg)
	}
	return nil
}

func ensembleSmoke() error {
	vote := ensemble.MajorityVote([][]int{{0, 1, 0}, {0, 2, 0}, {1, 1, 0}})
	if len(vote) != 3 || vote[0] != 0 {
		return fmt.Errorf("vote=%v", vote)
	}
	models := []ensemble.ModelPerformance{
		{ModelID: "a", Mask: []bool{true, false, true}},
		{ModelID: "b", Mask: []bool{false, true, true}},
	}
	matches := ensemble.FindComplementaryMatches(models, 0.9)
	if len(matches) == 0 {
		return fmt.Errorf("expected complementary match")
	}
	return nil
}

func clusterGroupSmoke() error {
	pts := []*core.Tensor[float32]{
		core.NewTensor[float32](3),
		core.NewTensor[float32](3),
		core.NewTensor[float32](3),
	}
	for i, t := range pts {
		for j := range t.Data {
			t.Data[j] = float32(i + j)
		}
	}
	_, assign := clustering.KMeansCluster(pts, 2, 5, false)
	if len(assign) != 3 {
		return fmt.Errorf("assignments=%v", assign)
	}
	tensors := []grouping.DetectedTensor{
		{Name: "layers.0.self_attn.q_proj.weight"},
		{Name: "layers.0.self_attn.k_proj.weight"},
		{Name: "layers.0.self_attn.v_proj.weight"},
		{Name: "layers.0.self_attn.o_proj.weight"},
	}
	ok, hint := grouping.DetectMHA("layers.0.self_attn", tensors, 64, 8)
	if !ok || hint.Type != core.LayerMultiHeadAttention {
		return fmt.Errorf("DetectMHA ok=%v hint=%v", ok, hint)
	}
	return nil
}

func observerSmoke() error {
	t := core.NewTensor[float32](1, 2, 3)
	for i := range t.Data {
		t.Data[i] = float32(i) * 0.1
	}
	st := observer.ComputeLayerStats(t)
	if st.Total != 6 {
		return fmt.Errorf("total=%d", st.Total)
	}
	buf := observer.NewBufferObserver(2)
	buf.OnForward(observer.LayerEvent{Stats: st})
	buf.OnForward(observer.LayerEvent{Stats: st})
	if len(buf.History) != 1 {
		return fmt.Errorf("history=%d", len(buf.History))
	}
	return nil
}

func introspectionSmoke() error {
	g := architecture.NewGrid(1, 1, 1, 1)
	methods, err := introspection.GetMethods(g)
	if err != nil {
		return err
	}
	if len(methods) == 0 {
		return fmt.Errorf("no methods")
	}
	sig, err := introspection.GetMethodSignature(g, methods[0].MethodName)
	if err != nil || sig == "" {
		return fmt.Errorf("sig=%q err=%v", sig, err)
	}
	return nil
}

func evaluationSmoke() error {
	r := evaluation.EvaluatePrediction(0, 1.0, 1.0)
	if r.Bucket != "0-10%" {
		return fmt.Errorf("bucket=%q", r.Bucket)
	}
	dm := evaluation.NewDeviationMetrics()
	dm.UpdateMetrics(r)
	dm.ComputeFinalMetrics()
	if dm.Accuracy != 100 {
		return fmt.Errorf("accuracy=%v", dm.Accuracy)
	}
	return nil
}

func universalSmoke() error {
	geoms := []universal.TensorMeta{
		{Idx: 0, Shape: []int{8, 8}, Rank: 2, MeanAbs: 0.1, OriginalDType: core.DTypeFloat32},
		{Idx: 1, Shape: []int{8}, Rank: 1, MeanAbs: 0.5, OriginalDType: core.DTypeFloat32},
	}
	archs, missed := universal.ProbeDeepGeometry(geoms)
	if len(archs) == 0 {
		return fmt.Errorf("no archetypes missed=%v", missed)
	}
	g := universal.MountGeometrically(archs, geoms)
	if g == nil || g.StackLayerCount() < 1 {
		return fmt.Errorf("nil/empty grid")
	}
	if _, err := universal.LoadUniversal("model.safetensors"); err == nil {
		return fmt.Errorf("expected safetensors error")
	}
	return nil
}

func pipelineSmoke() error {
	stats := pipeline.PipelineForwardStats{TokenDoneTick: []int{1, 3, 5, 7}}
	sum := stats.SummarizeTokenTimeline()
	if sum.NumTokens != 4 || sum.TickSpread != 6 {
		return fmt.Errorf("summary=%+v", sum)
	}
	out := sum.FormatComparison(1.2, 10)
	if !strings.Contains(out, "Pipeline") {
		return fmt.Errorf("format missing pipeline text")
	}
	return nil
}
