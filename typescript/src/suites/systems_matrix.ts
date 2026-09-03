import { SuiteReport, isErr, classifyOpError } from "./report.js";

export function runSystemsMatrix(report?: SuiteReport): SuiteReport {
  const r = report ?? new SuiteReport();
  r.log("\n## systems / model / lucy");

  const g = createWelvetGrid(JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 }));
  g.placeDense(JSON.stringify({ in: 4, out: 4, act: "relu", dtype: 1 }));

  const cell = (name: string, fn: () => void) => {
    const t0 = performance.now();
    try {
      fn();
      r.record("systems", name, "OK", "", performance.now() - t0);
    } catch (e) {
      const msg = String(e);
      r.record("systems", name, classifyOpError(msg), msg, performance.now() - t0);
    }
  };

  cell("ExtractDNA", () => {
    const dna = ExtractDNA(g._id);
    if (!dna || dna.includes('"error"')) throw new Error(dna || "empty");
  });
  cell("ExtractNetworkBlueprint", () => {
    const bp = ExtractNetworkBlueprint(g._id, "matrix");
    if (!bp || bp.includes('"error"')) throw new Error(bp || "empty");
  });
  cell("CloneGrid", () => {
    const c = CloneGrid(g._id);
    if (isErr(c)) throw new Error(c.error);
    c.free?.();
  });
  cell("SerializeDeserializeGrid", () => {
    const b = SerializeGrid(g._id);
    if (isErr(b) || !(b instanceof Uint8Array)) throw new Error(isErr(b) ? b.error : "bad");
    const back = DeserializeGrid(b);
    if (isErr(back)) throw new Error(back.error);
    back.free?.();
  });
  cell("serializeEntity", () => {
    const ent = g.serializeEntity();
    if (isErr(ent) || !(ent instanceof Uint8Array)) throw new Error(isErr(ent) ? ent.error : "bad");
    const back = DeserializeGrid(ent);
    if (isErr(back)) throw new Error(back.error);
    back.free?.();
  });
  cell("defaultSpliceConfig", () => {
    const s = defaultSpliceConfig();
    if (!s) throw new Error("empty");
  });
  cell("defaultNEATConfig", () => {
    const s = defaultNEATConfig(8);
    if (!s) throw new Error("empty");
  });
  cell("SpliceDNA", () => {
    const g2 = createWelvetGrid(JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 }));
    g2.placeDense(JSON.stringify({ in: 4, out: 4, dtype: 1 }));
    const out = SpliceDNA(g._id, g2._id);
    if (isErr(out)) throw new Error(out.error);
    out.free?.();
    g2.free?.();
  });
  cell("NEATMutate", () => {
    const out = NEATMutate(g._id);
    if (isErr(out)) throw new Error(out.error);
    out.free?.();
  });
  cell("ArgMax", () => {
    if (ArgMax(new Float32Array([0.1, 0.9, 0.2])) !== 1) throw new Error("argmax");
  });
  cell("SampleTopK", () => {
    const i = SampleTopK(new Float32Array([0.1, 0.9, 0.2]), 2, 1);
    if (typeof i !== "number") throw new Error("topk");
  });
  cell("LucyAvailability", () => {
    if (typeof LucyAvailability(1, 2) !== "number") throw new Error("avail");
  });
  cell("LucyScore", () => {
    if (typeof LucyScore(10, 50, 0.9) !== "number") throw new Error("score");
  });
  cell("LucySoftAccBatch", () => {
    if (typeof LucySoftAccBatch(new Float32Array([1, 0]), new Float32Array([1, 0])) !== "number") {
      throw new Error("soft");
    }
  });
  cell("BuildTransformerPrompt", () => {
    const p = BuildTransformerPrompt("hi", "sys");
    if (typeof p !== "string" || !p.includes("hi")) throw new Error(p);
  });
  cell("createWelvetStepState", () => {
    const st = createWelvetStepState(g._id);
    if (isErr(st)) throw new Error(st.error);
    st.setInput(new Float32Array([0.1, 0.2, 0.3, 0.4]));
    const s = st.step(false);
    if (isErr(s)) throw new Error(s.error);
    st.free?.();
  });
  cell("createWelvetTweenState", () => {
    const st = createWelvetTweenState(g._id);
    if (isErr(st)) throw new Error(st.error);
    st.free?.();
  });
  cell("TrainStep", () => {
    const tr = TrainStep(g._id, new Float32Array([0.1, 0.2, 0.3, 0.4]), new Float32Array([1, 0, 0, 0]), 0.05);
    if (isErr(tr)) throw new Error(tr.error);
  });
  cell("TrainStepTween", () => {
    const tr = TrainStepTween(g._id, new Float32Array([0.1, 0.2, 0.3, 0.4]), new Float32Array([1, 0, 0, 0]), 0.05);
    if (isErr(tr)) throw new Error(tr.error);
  });
  cell("TrainStepMesh", () => {
    const tr = TrainStepMesh(g._id, new Float32Array([0.1, 0.2, 0.3, 0.4]), new Float32Array([1, 0, 0, 0]), 2, 0.05);
    if (isErr(tr)) throw new Error(tr.error);
  });
  cell("ConfigureTanhi", () => {
    const o = ConfigureTanhi(g._id, JSON.stringify({ Enabled: false }));
    if (isErr(o)) throw new Error(o.error);
  });
  cell("welvetGC", () => {
    welvetGC();
  });

  g.free?.();
  return r;
}

/** Seven Dense ops in one cell (layers_per_cell=7) — Lucy [7] style. */
export function runSevenMatrix(report?: SuiteReport): SuiteReport {
  const r = report ?? new SuiteReport();
  r.log("\n## seven dense stack");
  const t0 = performance.now();
  try {
    const g = createWelvetGrid(JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 7 }));
    for (let l = 0; l < 7; l++) {
      const out = l === 6 ? 4 : 8;
      const p = g.placeDense(JSON.stringify({
        z: 0, y: 0, x: 0, l, in: 8, out, act: l === 6 ? "linear" : "relu", dtype: 1,
      }));
      if (isErr(p)) throw new Error(`place l=${l}: ${p.error}`);
    }
    const inp = new Float32Array(8).fill(0.1);
    const fwd = g.forward(inp);
    if (isErr(fwd)) throw new Error(fwd.error);
    const tr = g.trainSGD(inp, new Float32Array([1, 0, 0, 0]), 0.05);
    if (isErr(tr)) throw new Error(tr.error);
    const ent = g.serializeEntity();
    if (isErr(ent) || !(ent instanceof Uint8Array)) throw new Error("entity");
    const back = DeserializeGrid(ent);
    if (isErr(back)) throw new Error(back.error);
    r.record("seven", "dense7/train/entity", "OK", `loss=${tr.loss} steps=${fwd.steps}`, performance.now() - t0);
    back.free?.();
    g.free?.();
  } catch (e) {
    r.record("seven", "dense7/train/entity", classifyOpError(String(e)), String(e), performance.now() - t0);
  }
  return r;
}
