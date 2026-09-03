import { LAYER_DEFS } from "./catalog.js";
import { SuiteReport, isErr, classifyOpError } from "./report.js";

type DnaCmp = { OverallOverlap?: number };

function placeLayer(method: string, spec: Record<string, unknown>) {
  const g = createWelvetGrid(JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 }));
  const fn = (g as unknown as Record<string, (s: string) => unknown>)[method];
  if (typeof fn !== "function") throw new Error(`missing ${method}`);
  const p = fn.call(g, JSON.stringify(spec));
  if (isErr(p)) throw new Error(p.error);
  return g;
}

/** DNA extract/compare/drift + light layer×dtype self-overlap (portable). */
export function runDnaHonestyMatrix(report?: SuiteReport, opts?: { fullDtype?: boolean }): SuiteReport {
  const r = report ?? new SuiteReport();
  const fullDtype = !!opts?.fullDtype;
  r.log("\n## dna honesty");

  const cell = (name: string, fn: () => void) => {
    const t0 = performance.now();
    try {
      fn();
      r.record("dna", name, "OK", "", performance.now() - t0);
    } catch (e) {
      r.record("dna", name, classifyOpError(String(e)), String(e), performance.now() - t0);
    }
  };

  cell("extract_immutable_self_overlap", () => {
    const g = placeLayer("placeDense", { in: 4, out: 4, act: "linear", dtype: 1 });
    const w0 = g.getDenseWeights();
    if (isErr(w0) || !(w0 instanceof Float32Array)) throw new Error("weights");
    const before = new Float32Array(w0);
    const dnaA = ExtractDNA(g._id);
    if (!dnaA || dnaA.includes('"error"')) throw new Error(dnaA || "empty");
    const w1 = g.getDenseWeights();
    if (isErr(w1) || !(w1 instanceof Float32Array)) throw new Error("weights after");
    for (let i = 0; i < before.length; i++) {
      if (before[i] !== w1[i]) throw new Error("ExtractDNA mutated weights");
    }
    const cmp = JSON.parse(CompareDNA(dnaA, dnaA)) as DnaCmp;
    if ((cmp.OverallOverlap ?? 0) < 0.999) throw new Error(`overlap=${cmp.OverallOverlap}`);
    g.free?.();
  });

  cell("detect_weight_drift", () => {
    const g1 = placeLayer("placeDense", { in: 4, out: 4, act: "linear", dtype: 1 });
    const g2 = placeLayer("placeDense", { in: 4, out: 4, act: "linear", dtype: 1 });
    const w = g1.getDenseWeights();
    if (isErr(w) || !(w instanceof Float32Array)) throw new Error("w");
    for (let i = 0; i < w.length; i++) w[i] = (i + 1) * 0.05;
    g1.setDenseWeights(w);
    const mut = new Float32Array(w);
    for (let i = 0; i < mut.length; i++) mut[i] = i % 2 === 0 ? -mut[i]! : mut[i]! * 0.25;
    g2.setDenseWeights(mut);
    const cmp = JSON.parse(CompareDNA(ExtractDNA(g1._id), ExtractDNA(g2._id))) as DnaCmp;
    if ((cmp.OverallOverlap ?? 1) > 0.999) throw new Error(`expected drift, overlap=${cmp.OverallOverlap}`);
    g1.free?.();
    g2.free?.();
  });

  cell("multi_layer_ops_dna", () => {
    const g = createWelvetGrid(JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 4 }));
    const places: Array<[string, Record<string, unknown>]> = [
      ["placeDense", { z: 0, y: 0, x: 0, l: 0, in: 8, out: 8, act: "linear", dtype: 1 }],
      ["placeRMSNorm", { z: 0, y: 0, x: 0, l: 1, dim: 8, dtype: 1 }],
      ["placeSwiGLU", { z: 0, y: 0, x: 0, l: 2, dim: 8, InputDim: 8, IntermediateDim: 16, dtype: 1 }],
      ["placeLayerNorm", { z: 0, y: 0, x: 0, l: 3, dim: 8, dtype: 1 }],
    ];
    for (const [method, spec] of places) {
      const fn = (g as unknown as Record<string, (s: string) => unknown>)[method];
      const p = fn.call(g, JSON.stringify(spec));
      if (isErr(p)) throw new Error(`${method}: ${p.error}`);
    }
    const dna = JSON.parse(ExtractDNA(g._id)) as unknown[];
    if (!Array.isArray(dna) || dna.length < 4) throw new Error(`sig len=${Array.isArray(dna) ? dna.length : "?"}`);
    const cmp = JSON.parse(CompareDNA(JSON.stringify(dna), JSON.stringify(dna))) as DnaCmp;
    if ((cmp.OverallOverlap ?? 0) < 0.999) throw new Error(`overlap=${cmp.OverallOverlap}`);
    g.free?.();
  });

  const dtypes: Array<{ id: number; name: string }> = fullDtype
    ? (JSON.parse(listWelvetDTypes()) as Array<{ id: number; name: string }>)
    : [{ id: 1, name: "float32" }, { id: 0, name: "float64" }, { id: 9, name: "int8" }];

  for (const dt of dtypes) {
    const name = `self_overlap/${dt.name}`;
    const t0 = performance.now();
    try {
      const g = placeLayer("placeDense", { in: 4, out: 4, act: "linear", dtype: dt.id });
      const dna = ExtractDNA(g._id);
      const cmp = JSON.parse(CompareDNA(dna, dna)) as DnaCmp;
      if ((cmp.OverallOverlap ?? 0) < 0.999) throw new Error(`overlap=${cmp.OverallOverlap}`);
      r.record("dna", name, "OK", "", performance.now() - t0);
      g.free?.();
    } catch (e) {
      r.record("dna", name, classifyOpError(String(e)), String(e), performance.now() - t0);
    }
  }

  // Light layer census: one DNA extract per placeable layer (f32).
  for (const layer of LAYER_DEFS.slice(0, fullDtype ? LAYER_DEFS.length : 8)) {
    const name = `layer_extract/${layer.id}`;
    const t0 = performance.now();
    try {
      const g = placeLayer(layer.method, { ...layer.spec, dtype: 1 });
      const dna = ExtractDNA(g._id);
      if (!dna || dna.includes('"error"')) throw new Error(dna || "empty");
      const cmp = JSON.parse(CompareDNA(dna, dna)) as DnaCmp;
      if ((cmp.OverallOverlap ?? 0) < 0.999) throw new Error(`overlap=${cmp.OverallOverlap}`);
      r.record("dna", name, "OK", "", performance.now() - t0);
      g.free?.();
    } catch (e) {
      r.record("dna", name, classifyOpError(String(e)), String(e), performance.now() - t0);
    }
  }

  return r;
}
