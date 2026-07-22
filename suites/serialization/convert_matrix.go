package serialization

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/w2a/suites/polyops"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/stub/serialization"
	"github.com/openfluke/welvet/systems/dna"
	"github.com/openfluke/welvet/weights"
)

func rec(layer, op, dt, format, status, note string) {
	suites.RecordCell(suites.Cell{
		Layer:   layer,
		Op:      op,
		DType:   dt,
		Format:  format,
		Backend: "cpu_tiled",
		Grid:    "1x1x1x1",
		Status:  status,
		Note:    note,
	})
}

func cfgDTFormat(cfg storageCfg) (dt, format string) {
	if cfg.Format == quant.FormatNone {
		return cfg.DType.String(), "none"
	}
	return "float32", cfg.Format.String()
}

// storageCfg is one FormatNone dtype or one packed quant target.
type storageCfg struct {
	Label  string
	DType  core.DType
	Format quant.Format
}

func allStorageCfgs() []storageCfg {
	out := make([]storageCfg, 0, len(core.AllDTypes)+len(quant.AllFormats))
	for _, dt := range core.AllDTypes {
		out = append(out, storageCfg{
			Label:  "none/" + dt.String(),
			DType:  dt,
			Format: quant.FormatNone,
		})
	}
	for _, f := range quant.AllFormats {
		if f == quant.FormatNone {
			continue
		}
		out = append(out, storageCfg{
			Label:  f.String(),
			DType:  core.DTypeFloat32,
			Format: f,
		})
	}
	return out
}

func gridStores(g *architecture.Grid) []*weights.Store {
	if g == nil {
		return nil
	}
	var out []*weights.Store
	for i := range g.Cells {
		if g.Cells[i].Op == nil {
			continue
		}
		out = append(out, dna.CollectStores(g.Cells[i].Op)...)
	}
	return out
}

func convertGridStores(g *architecture.Grid, cfg storageCfg) error {
	for _, s := range gridStores(g) {
		if s == nil {
			continue
		}
		if err := weights.Convert(s, weights.ConvertOpts{DType: cfg.DType, Format: cfg.Format}); err != nil {
			return err
		}
	}
	return nil
}

func verifyStoreCfg(g *architecture.Grid, cfg storageCfg) error {
	stores := gridStores(g)
	if len(stores) == 0 {
		return nil
	}
	for _, s := range stores {
		if s == nil {
			continue
		}
		if s.Format != cfg.Format {
			return fmt.Errorf("format got %s want %s", s.Format, cfg.Format)
		}
		if cfg.Format == quant.FormatNone && s.DType != cfg.DType {
			return fmt.Errorf("dtype got %s want %s", s.DType, cfg.DType)
		}
		snap, err := weights.TakeSnapshot(s)
		if err != nil {
			return err
		}
		if len(snap.Raw) == 0 {
			return fmt.Errorf("empty storage payload for %s/%s", s.DType, s.Format)
		}
		// FormatNone non-f32 must persist as Native bytes (not an f32-only path).
		if cfg.Format == quant.FormatNone && cfg.DType != core.DTypeFloat32 && len(s.Native) == 0 {
			return fmt.Errorf("%s missing Native payload (want native dtype storage)", cfg.Label)
		}
		if cfg.Format != quant.FormatNone && s.Packed == nil {
			return fmt.Errorf("%s missing Packed blob", cfg.Label)
		}
	}
	return nil
}

func payloadKB(g *architecture.Grid) float64 {
	var n int
	for _, s := range gridStores(g) {
		if s == nil {
			continue
		}
		snap, err := weights.TakeSnapshot(s)
		if err != nil {
			continue
		}
		n += len(snap.Raw)
	}
	return float64(n) / 1024.0
}

func fileSizesKB(g *architecture.Grid) (jsonKB, entKB float64, err error) {
	rawJSON, err := serialization.SerializeGrid(g)
	if err != nil {
		return 0, 0, fmt.Errorf("json: %w", err)
	}
	rawEnt, err := serialization.SerializeEntity(g)
	if err != nil {
		return 0, 0, fmt.Errorf("entity: %w", err)
	}
	return float64(len(rawJSON)) / 1024.0, float64(len(rawEnt)) / 1024.0, nil
}

func trainShort(g *architecture.Grid, x, y *core.Tensor[float32], epochs int) (first, last float64, err error) {
	first, last = math.NaN(), math.NaN()
	for e := 0; e < epochs; e++ {
		fwd, err := forward.Forward(g, x)
		if err != nil {
			return first, last, err
		}
		loss, err := training.Step(fwd, y, 1e-2)
		if err != nil {
			return first, last, err
		}
		if e == 0 {
			first = loss
		}
		last = loss
	}
	return first, last, nil
}

func saveReloadBoth(dir, tag string, g *architecture.Grid, x *core.Tensor[float32]) error {
	jsonPath := filepath.Join(dir, tag+".json")
	entPath := filepath.Join(dir, tag+".entity")
	if err := serialization.SaveGridJSON(jsonPath, g); err != nil {
		return fmt.Errorf("save json: %w", err)
	}
	if err := serialization.SaveEntity(entPath, g); err != nil {
		return fmt.Errorf("save entity: %w", err)
	}
	gJ, err := serialization.LoadGridJSON(jsonPath)
	if err != nil {
		return fmt.Errorf("load json: %w", err)
	}
	gE, err := serialization.LoadEntity(entPath)
	if err != nil {
		return fmt.Errorf("load entity: %w", err)
	}
	fwd0, err := forward.Forward(g, x)
	if err != nil {
		return err
	}
	fwdJ, err := forward.Forward(gJ, x)
	if err != nil {
		return fmt.Errorf("json fwd: %w", err)
	}
	fwdE, err := forward.Forward(gE, x)
	if err != nil {
		return fmt.Errorf("entity fwd: %w", err)
	}
	if d := maxAbs(fwd0.Output.Data, fwdJ.Output.Data); d > 1e-3 {
		return fmt.Errorf("json reload fwd Δ=%g", d)
	}
	if d := maxAbs(fwd0.Output.Data, fwdE.Output.Data); d > 1e-3 {
		return fmt.Errorf("entity reload fwd Δ=%g", d)
	}
	// Reloaded stores must match storage truth meta.
	if err := storesMetaEqual(g, gJ); err != nil {
		return fmt.Errorf("json meta: %w", err)
	}
	if err := storesMetaEqual(g, gE); err != nil {
		return fmt.Errorf("entity meta: %w", err)
	}
	return nil
}

func storesMetaEqual(a, b *architecture.Grid) error {
	sa, sb := gridStores(a), gridStores(b)
	if len(sa) != len(sb) {
		return fmt.Errorf("store count %d vs %d", len(sa), len(sb))
	}
	for i := range sa {
		if sa[i] == nil || sb[i] == nil {
			continue
		}
		if sa[i].DType != sb[i].DType || sa[i].Format != sb[i].Format {
			return fmt.Errorf("[%d] %s/%s vs %s/%s", i, sa[i].DType, sa[i].Format, sb[i].DType, sb[i].Format)
		}
		snapA, err := weights.TakeSnapshot(sa[i])
		if err != nil {
			return err
		}
		snapB, err := weights.TakeSnapshot(sb[i])
		if err != nil {
			return err
		}
		if len(snapA.Raw) != len(snapB.Raw) {
			return fmt.Errorf("[%d] raw len %d vs %d", i, len(snapA.Raw), len(snapB.Raw))
		}
	}
	return nil
}

func maxAbs(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var m float64
	for i := 0; i < n; i++ {
		d := math.Abs(float64(a[i]) - float64(b[i]))
		if d > m {
			m = d
		}
	}
	return m
}

func fingerprintStores(g *architecture.Grid) string {
	var b strings.Builder
	for _, s := range gridStores(g) {
		if s == nil {
			continue
		}
		snap, err := weights.TakeSnapshot(s)
		if err != nil {
			continue
		}
		// Cheap change detector: length + first/last bytes + scale.
		fmt.Fprintf(&b, "%d:%g:", len(snap.Raw), snap.Scale)
		if len(snap.Raw) > 0 {
			fmt.Fprintf(&b, "%x", snap.Raw[0])
			fmt.Fprintf(&b, "%x", snap.Raw[len(snap.Raw)-1])
		}
		b.WriteByte('|')
	}
	return b.String()
}

// createSizeTrainReload builds every layer × every dtype/quant, reports KB,
// trains briefly, then save/reloads JSON + .entity in native storage form.
func createSizeTrainReload() error {
	cfgs := allStorageCfgs()
	dir, err := os.MkdirTemp("", "w2a-ser-create-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)

	var fails []string
	okN, gapN, trainGapN := 0, 0, 0
	fmt.Printf("\n")
	fmt.Printf("    %-14s %-18s %10s %10s %10s  %s\n", "layer", "storage", "payloadKB", "jsonKB", "entityKB", "train")
	for _, kind := range polyops.AllKinds() {
		for _, cfg := range cfgs {
			dtName, fmtName := cfgDTFormat(cfg)
			g, err := kind.Make(cfg.DType, cfg.Format)
			if err != nil {
				gapN++
				rec(kind.Name, "ser_create", dtName, fmtName, "GAP", err.Error())
				fmt.Printf("    %-14s %-18s  GAP (%v)\n", kind.Name, cfg.Label, err)
				continue
			}
			g.Exec.Backend = core.BackendCPUTiled
			g.Exec.MultiCore = false
			if err := verifyStoreCfg(g, cfg); err != nil {
				fails = append(fails, fmt.Sprintf("%s/%s verify: %v", kind.Name, cfg.Label, err))
				rec(kind.Name, "ser_create", dtName, fmtName, "FAIL", "verify: "+err.Error())
				fmt.Printf("    %-14s %-18s  FAIL verify: %v\n", kind.Name, cfg.Label, err)
				continue
			}
			payKB := payloadKB(g)
			jsonKB, entKB, err := fileSizesKB(g)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s/%s size: %v", kind.Name, cfg.Label, err))
				rec(kind.Name, "ser_create", dtName, fmtName, "FAIL", "size: "+err.Error())
				fmt.Printf("    %-14s %-18s  FAIL size: %v\n", kind.Name, cfg.Label, err)
				continue
			}
			x, y := polyops.MakeIO(kind, 1.0)
			before := fingerprintStores(g)
			trainNote := "n/a"
			status := "OK"
			if len(gridStores(g)) > 0 {
				first, last, err := trainShort(g, x, y, 3)
				switch {
				case err != nil:
					fails = append(fails, fmt.Sprintf("%s/%s train: %v", kind.Name, cfg.Label, err))
					rec(kind.Name, "ser_create", dtName, fmtName, "FAIL", "train: "+err.Error())
					fmt.Printf("    %-14s %-18s  FAIL train: %v\n", kind.Name, cfg.Label, err)
					continue
				case math.IsNaN(first) || math.IsNaN(last) || math.IsInf(first, 0) || math.IsInf(last, 0):
					// Extreme low-bit packs can explode on some Ops (e.g. mamba+binary).
					// Still require create/size/native save-reload on a fresh model.
					trainGapN++
					status = "GAP"
					trainNote = fmt.Sprintf("train GAP non-finite %g→%g; save untrained", first, last)
					g, err = kind.Make(cfg.DType, cfg.Format)
					if err != nil {
						fails = append(fails, fmt.Sprintf("%s/%s rebuild: %v", kind.Name, cfg.Label, err))
						rec(kind.Name, "ser_create", dtName, fmtName, "FAIL", "rebuild: "+err.Error())
						continue
					}
					g.Exec.Backend = core.BackendCPUTiled
					g.Exec.MultiCore = false
				default:
					after := fingerprintStores(g)
					if before == after {
						trainNote = fmt.Sprintf("loss %.4g→%.4g (codes unchanged)", first, last)
					} else {
						trainNote = fmt.Sprintf("loss %.4g→%.4g", first, last)
					}
					if err := verifyStoreCfg(g, cfg); err != nil {
						fails = append(fails, fmt.Sprintf("%s/%s post-train storage: %v", kind.Name, cfg.Label, err))
						rec(kind.Name, "ser_create", dtName, fmtName, "FAIL", "post-train: "+err.Error())
						fmt.Printf("    %-14s %-18s  FAIL post-train: %v\n", kind.Name, cfg.Label, err)
						continue
					}
				}
			}
			tag := sanitize(kind.Name + "_" + cfg.Label)
			if err := saveReloadBoth(dir, tag, g, x); err != nil {
				fails = append(fails, fmt.Sprintf("%s/%s reload: %v", kind.Name, cfg.Label, err))
				rec(kind.Name, "ser_create", dtName, fmtName, "FAIL", "reload: "+err.Error())
				fmt.Printf("    %-14s %-18s  FAIL reload: %v\n", kind.Name, cfg.Label, err)
				continue
			}
			note := fmt.Sprintf("payload=%.2fKB json=%.2fKB entity=%.2fKB %s", payKB, jsonKB, entKB, trainNote)
			rec(kind.Name, "ser_create", dtName, fmtName, status, note)
			if status == "OK" {
				okN++
			}
			fmt.Printf("    %-14s %-18s %10.2f %10.2f %10.2f  %s\n",
				kind.Name, cfg.Label, payKB, jsonKB, entKB, trainNote)
		}
	}
	fmt.Printf("    create/size/train/reload: ok=%d build_gap=%d train_gap=%d fail=%d\n", okN, gapN, trainGapN, len(fails))
	if len(fails) > 0 {
		return fmt.Errorf("%d failures (first: %s)", len(fails), fails[0])
	}
	return nil
}

// convertAllPermutations converts every viable storage cfg into every other cfg
// per layer and prints the new JSON/.entity KB sizes. Full JSON/.entity
// save+reload is checked once per destination (from float32), so the O(n²)
// size table stays tractable.
func convertAllPermutations() error {
	cfgs := allStorageCfgs()
	dir, err := os.MkdirTemp("", "w2a-ser-cvt-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)

	var fails []string
	okN, gapN, reloadN := 0, 0, 0
	fmt.Printf("\n")
	fmt.Printf("    %-14s %-18s → %-18s %10s %10s %10s\n", "layer", "from", "to", "payloadKB", "jsonKB", "entityKB")
	for _, kind := range polyops.AllKinds() {
		if len(gridStoresMust(kind)) == 0 {
			rec(kind.Name, "ser_convert", "-", "-", "GAP", "weightless — skip convert permutations")
			fmt.Printf("    %-14s (weightless — skip convert permutations)\n", kind.Name)
			continue
		}
		var viable []storageCfg
		for _, cfg := range cfgs {
			g, err := kind.Make(cfg.DType, cfg.Format)
			if err != nil {
				continue
			}
			if err := verifyStoreCfg(g, cfg); err != nil {
				continue
			}
			viable = append(viable, cfg)
		}
		reloaded := map[string]bool{}
		for _, src := range viable {
			for _, dst := range viable {
				if src.Label == dst.Label {
					continue
				}
				_, srcFmt := cfgDTFormat(src)
				dstDT, dstFmt := cfgDTFormat(dst)
				// Cell dtype/format = destination storage (what we converted into).
				cellNote := src.Label + "→" + dst.Label
				g, err := kind.Make(src.DType, src.Format)
				if err != nil {
					gapN++
					rec(kind.Name, "ser_convert", dstDT, dstFmt, "GAP", cellNote+": "+err.Error())
					continue
				}
				g.Exec.Backend = core.BackendCPUTiled
				if err := convertGridStores(g, dst); err != nil {
					gapN++
					rec(kind.Name, "ser_convert", dstDT, dstFmt, "GAP", cellNote+": "+err.Error())
					fmt.Printf("    %-14s %-18s → %-18s  GAP (%v)\n", kind.Name, src.Label, dst.Label, err)
					continue
				}
				if err := verifyStoreCfg(g, dst); err != nil {
					fails = append(fails, fmt.Sprintf("%s %s→%s: %v", kind.Name, src.Label, dst.Label, err))
					rec(kind.Name, "ser_convert", dstDT, dstFmt, "FAIL", cellNote+": "+err.Error())
					fmt.Printf("    %-14s %-18s → %-18s  FAIL %v\n", kind.Name, src.Label, dst.Label, err)
					continue
				}
				payKB := payloadKB(g)
				jsonKB, entKB, err := fileSizesKB(g)
				if err != nil {
					fails = append(fails, fmt.Sprintf("%s %s→%s size: %v", kind.Name, src.Label, dst.Label, err))
					rec(kind.Name, "ser_convert", dstDT, dstFmt, "FAIL", cellNote+": size "+err.Error())
					continue
				}

				// One full save/reload per destination from float32 source.
				if src.Label == "none/float32" && !reloaded[dst.Label] {
					x, _ := polyops.MakeIO(kind, 1.0)
					tag := sanitize(kind.Name + "_to_" + dst.Label)
					if err := saveReloadBoth(dir, tag, g, x); err != nil {
						fails = append(fails, fmt.Sprintf("%s →%s reload: %v", kind.Name, dst.Label, err))
						rec(kind.Name, "ser_convert", dstDT, dstFmt, "FAIL", cellNote+": reload "+err.Error())
						fmt.Printf("      reload FAIL %s→%s: %v\n", kind.Name, dst.Label, err)
						continue
					}
					reloaded[dst.Label] = true
					reloadN++
				}
				okN++
				rec(kind.Name, "ser_convert", dstDT, dstFmt, "OK",
					fmt.Sprintf("%s srcFmt=%s payload=%.2fKB json=%.2fKB entity=%.2fKB",
						cellNote, srcFmt, payKB, jsonKB, entKB))
				fmt.Printf("    %-14s %-18s → %-18s %10.2f %10.2f %10.2f\n",
					kind.Name, src.Label, dst.Label, payKB, jsonKB, entKB)
			}
		}
	}
	fmt.Printf("    convert permutations: ok=%d gap=%d reload=%d fail=%d\n", okN, gapN, reloadN, len(fails))
	if len(fails) > 0 {
		return fmt.Errorf("%d failures (first: %s)", len(fails), fails[0])
	}
	return nil
}

func gridStoresMust(kind polyops.Kind) []*weights.Store {
	g, err := kind.Make(core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		return nil
	}
	return gridStores(g)
}

func sanitize(s string) string {
	s = strings.ReplaceAll(s, "/", "_")
	s = strings.ReplaceAll(s, " ", "_")
	return s
}
