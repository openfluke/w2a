import { LAYER_DEFS } from "./catalog.js";
import { SuiteReport, isErr, classifyOpError } from "./report.js";

function placeDense(): ReturnType<typeof createWelvetGrid> {
  const g = createWelvetGrid(JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 }));
  const p = g.placeDense(JSON.stringify({ in: 4, out: 4, act: "linear", dtype: 1 }));
  if (isErr(p)) throw new Error(p.error);
  return g;
}

/** Evolution portables: splice / NEAT pop / clone+splice per layer (f32). */
export function runEvolutionHonestyMatrix(report?: SuiteReport, opts?: { fullLayers?: boolean }): SuiteReport {
  const r = report ?? new SuiteReport();
  const fullLayers = !!opts?.fullLayers;
  r.log("\n## evolution honesty");

  const cell = (name: string, fn: () => void) => {
    const t0 = performance.now();
    try {
      fn();
      r.record("evolution", name, "OK", "", performance.now() - t0);
    } catch (e) {
      r.record("evolution", name, classifyOpError(String(e)), String(e), performance.now() - t0);
    }
  };

  cell("splice_blend_smoke", () => {
    const a = placeDense();
    const b = placeDense();
    const out = SpliceDNA(a._id, b._id);
    if (isErr(out)) throw new Error(out.error);
    out.free?.();
    a.free?.();
    b.free?.();
  });

  cell("neat_population_one_gen", () => {
    const g = placeDense();
    const pop = createWelvetNEATPopulation(g._id, 4);
    if (isErr(pop)) throw new Error(pop.error);
    if (typeof pop.size !== "number" || pop.size < 1) throw new Error("bad size");
    const fits = new Float64Array(pop.size);
    for (let i = 0; i < fits.length; i++) fits[i] = 1 / (1 + i);
    const ev = pop.evolveWithFitnesses(fits);
    if (isErr(ev)) throw new Error(ev.error);
    const best = pop.bestFitness();
    if (typeof best !== "number") throw new Error("bestFitness");
    const sum = pop.summary(1);
    if (typeof sum !== "string" || !sum.length) throw new Error("summary");
    g.free?.();
  });

  cell("clone_then_splice", () => {
    const g = placeDense();
    const c = CloneGrid(g._id);
    if (isErr(c)) throw new Error(c.error);
    const out = SpliceDNA(g._id, c._id);
    if (isErr(out)) throw new Error(out.error);
    out.free?.();
    c.free?.();
    g.free?.();
  });

  const layers = fullLayers ? LAYER_DEFS : LAYER_DEFS.filter((l) =>
    ["dense", "rmsnorm", "layernorm", "swiglu", "softmax", "sequential", "residual", "metacognition"].includes(l.id),
  );

  for (const layer of layers) {
    const name = `clone_splice/${layer.id}`;
    const t0 = performance.now();
    try {
      const g = createWelvetGrid(JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 }));
      const fn = (g as unknown as Record<string, (s: string) => unknown>)[layer.method];
      const p = fn.call(g, JSON.stringify({ ...layer.spec, dtype: 1 }));
      if (isErr(p)) throw new Error(p.error);
      const c = CloneGrid(g._id);
      if (isErr(c)) throw new Error(c.error);
      const out = SpliceDNA(g._id, c._id);
      if (isErr(out)) throw new Error(out.error);
      r.record("evolution", name, "OK", "", performance.now() - t0);
      out.free?.();
      c.free?.();
      g.free?.();
    } catch (e) {
      r.record("evolution", name, classifyOpError(String(e)), String(e), performance.now() - t0);
    }
  }

  return r;
}
