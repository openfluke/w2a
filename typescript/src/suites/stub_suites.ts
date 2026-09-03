import { SuiteReport, isErr, classifyOpError } from "./report.js";

const COARSE = new Set([
  "fp4", "binary", "ternary", "nf4", "int4", "uint4", "int2", "uint2",
  "fp6", "int6", "uint6", "int5", "uint5", "int3", "uint3",
]);

/** Native ApplySGD — FormatNone × all dtypes, no retained f32 scratch. */
export function runWeightsMatrix(report?: SuiteReport): SuiteReport {
  const r = report ?? new SuiteReport();
  r.log("\n## weights ApplySGD × dtypes");
  const dtypes = JSON.parse(listWelvetDTypes()) as { id: number; name: string }[];
  const rows = 8, cols = 8, n = rows * cols;
  const src = new Float32Array(n);
  for (let i = 0; i < n; i++) src[i] = Math.sin(i * 0.17) * 0.5;
  const dW = new Float64Array(n);
  for (let i = 0; i < n; i++) dW[i] = 0.01 * ((i % 5) - 2);

  for (const dt of dtypes) {
    const t0 = performance.now();
    try {
      const s = createWelvetStore(rows, cols, 1, 0, src); // start f32 then SetDType
      if (isErr(s)) {
        r.record("weights", `ApplySGD/${dt.name}`, classifyOpError(s.error), s.error, performance.now() - t0);
        continue;
      }
      if (dt.id !== 1) {
        const sd = s.setDType(dt.id);
        if (isErr(sd)) {
          r.record("weights", `ApplySGD/${dt.name}`, classifyOpError(sd.error), sd.error, performance.now() - t0);
          s.free?.();
          continue;
        }
      }
      const before = s.flattenF32();
      if (isErr(before) || !(before instanceof Float32Array)) {
        r.record("weights", `ApplySGD/${dt.name}`, "GAP", "flatten before", performance.now() - t0);
        s.free?.();
        continue;
      }
      const ap = s.applySGD(dW, 0.1);
      if (isErr(ap)) {
        r.record("weights", `ApplySGD/${dt.name}`, classifyOpError(ap.error), ap.error, performance.now() - t0);
        s.free?.();
        continue;
      }
      if (dt.id !== 1 /* float32 */) {
        if (s.f32BufferLen() !== 0 || s.retainsF32Master()) {
          r.record("weights", `ApplySGD/${dt.name}`, "FAIL", "retained f32 scratch", performance.now() - t0);
          s.free?.();
          continue;
        }
      }
      const after = s.flattenF32();
      if (isErr(after) || !(after instanceof Float32Array)) {
        r.record("weights", `ApplySGD/${dt.name}`, "GAP", "flatten after", performance.now() - t0);
        s.free?.();
        continue;
      }
      let moved = false;
      let bad = false;
      for (let i = 0; i < after.length; i++) {
        if (!Number.isFinite(after[i])) { bad = true; break; }
        if (after[i] !== before[i]) moved = true;
      }
      if (bad) {
        r.record("weights", `ApplySGD/${dt.name}`, "FAIL", "non-finite", performance.now() - t0);
      } else if (!moved && !COARSE.has(dt.name.toLowerCase())) {
        r.record("weights", `ApplySGD/${dt.name}`, "FAIL", "no weight delta", performance.now() - t0);
      } else {
        r.record("weights", `ApplySGD/${dt.name}`, "OK", "", performance.now() - t0);
      }
      s.free?.();
    } catch (e) {
      r.record("weights", `ApplySGD/${dt.name}`, "FAIL", String(e), performance.now() - t0);
    }
  }

  // float64 native ALU smoke
  {
    const t0 = performance.now();
    try {
      const s = createWelvetStore(2, 2, 0, 0, new Float32Array([1, 2, 3, 4]));
      if (isErr(s)) throw new Error(s.error);
      const ap = s.applySGD(new Float64Array([0.1, 0.1, 0.1, 0.1]), 1);
      if (isErr(ap)) throw new Error(ap.error);
      r.record("weights", "ApplySGD/float64_alu", "OK", "", performance.now() - t0);
      s.free?.();
    } catch (e) {
      r.record("weights", "ApplySGD/float64_alu", classifyOpError(String(e)), String(e), performance.now() - t0);
    }
  }
  return r;
}

/** Seed / fountain / helpers / memory portable Cases. */
export function runStubSuitesMatrix(report?: SuiteReport): SuiteReport {
  const r = report ?? new SuiteReport();
  r.log("\n## seed / fountain / helpers / memory");

  const cell = (suite: string, name: string, fn: () => void) => {
    const t0 = performance.now();
    try {
      fn();
      r.record(suite, name, "OK", "", performance.now() - t0);
    } catch (e) {
      r.record(suite, name, classifyOpError(String(e)), String(e), performance.now() - t0);
    }
  };

  cell("seed", "SeedFrom_det", () => {
    const a = SeedFrom(JSON.stringify(["welvet", 42, true]));
    const b = SeedFrom(JSON.stringify(["welvet", 42, true]));
    if (String(a) !== String(b) || isErr(a)) throw new Error(`mismatch ${a}`);
  });
  cell("seed", "BuildDense_InitGrid_fp", () => {
    const man = BuildDenseManifest("42", JSON.stringify([4, 4, 4]));
    if (isErr(man) || typeof man !== "string") throw new Error(String(man));
    const g = BuildDenseGridFromManifest(man);
    if (isErr(g)) throw new Error(g.error);
    const ig = InitGrid(g._id, "99");
    if (isErr(ig)) throw new Error(ig.error);
    const fp = GridFingerprint(g._id);
    if (isErr(fp) || !fp) throw new Error(String(fp));
    g.free?.();
  });

  cell("fountain", "LT_roundtrip", () => {
    const blobs = [
      new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]),
      new Uint8Array([9, 10, 11, 12, 13, 14, 15, 16]),
      new Uint8Array([17, 18, 19, 20, 21, 22, 23, 24]),
    ];
    const rt = FountainLTRoundTrip(blobs, "7");
    if (isErr(rt) || !rt.ok) throw new Error(JSON.stringify(rt));
  });
  cell("fountain", "PackUnpack_weights", () => {
    const g = createWelvetGrid(JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 }));
    g.placeDense(JSON.stringify({ in: 4, out: 4, dtype: 1 }));
    const blob = PackGridWeights(g._id);
    if (isErr(blob) || !(blob instanceof Uint8Array)) throw new Error(String(blob));
    const u = UnpackGridWeights(g._id, blob);
    if (isErr(u)) throw new Error(u.error);
    g.free?.();
  });

  cell("helpers", "GraftGrids", () => {
    const g1 = createWelvetGrid(JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 }));
    const g2 = createWelvetGrid(JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 }));
    g1.placeDense(JSON.stringify({ in: 4, out: 4, dtype: 1 }));
    g2.placeDense(JSON.stringify({ in: 4, out: 4, dtype: 1 }));
    const p = GraftGrids([g1._id, g2._id], "concat");
    if (isErr(p)) throw new Error(p.error);
    p.free?.();
    g1.free?.();
    g2.free?.();
  });
  cell("helpers", "TemplateBuildPrompt", () => {
    const p = TemplateBuildPrompt("chatml", "hi", "sys");
    if (typeof p !== "string" || !p.includes("hi")) throw new Error(p);
  });
  cell("helpers", "EnsembleMajorityVote", () => {
    const v = EnsembleMajorityVote(JSON.stringify([[0, 1], [0, 2], [0, 1]]));
    if (isErr(v)) throw new Error(String(v));
    const arr = JSON.parse(v as string) as number[];
    if (!Array.isArray(arr) || arr[0] !== 0) throw new Error(v as string);
  });
  cell("helpers", "EvaluatePrediction", () => {
    const j = EvaluatePrediction(0, 1, 0.9);
    if (isErr(j) || typeof j !== "string") throw new Error(String(j));
  });
  cell("helpers", "IntrospectGrid", () => {
    const g = createWelvetGrid(JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 }));
    g.placeDense(JSON.stringify({ in: 4, out: 4, dtype: 1 }));
    const m = IntrospectGrid(g._id);
    if (isErr(m) || typeof m !== "string") throw new Error(String(m));
    g.free?.();
  });

  cell("memory", "FromGrid_Release", () => {
    SetMemoryHistoryRecording(true);
    const g = createWelvetGrid(JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 }));
    g.placeDense(JSON.stringify({ in: 8, out: 8, dtype: 1 }));
    const fp = MemoryFromGrid(g._id);
    if (isErr(fp) || typeof fp !== "string") throw new Error(String(fp));
    const o = JSON.parse(fp) as { HostWeightsMB?: number };
    if (typeof o.HostWeightsMB !== "number") throw new Error(fp);
    ReleaseTransient();
    SetMemoryHistoryRecording(false);
    g.free?.();
  });

  cell("serialization", "BuildSpec_Packable", () => {
    const g = createWelvetGrid(JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 }));
    g.placeDense(JSON.stringify({ in: 4, out: 4, dtype: 1 }));
    const spec = BuildSpec(g._id);
    if (isErr(spec) || typeof spec !== "string") throw new Error(String(spec));
    const pf = PackableFormats();
    if (typeof pf !== "string" || !pf.includes("[")) throw new Error(String(pf));
    const ent = SerializeEntity(g._id);
    if (isErr(ent) || !(ent instanceof Uint8Array)) throw new Error(String(ent));
    const back = DeserializeEntity(ent);
    if (isErr(back)) throw new Error(back.error);
    back.free?.();
    g.free?.();
  });

  // Dense convert permutations (CPU): FormatNone×dtypes + packable quants@f32
  {
    const dtypes = JSON.parse(listWelvetDTypes()) as { id: number; name: string }[];
    for (const dt of dtypes) {
      cell("serialization", `convert_dtype/${dt.name}`, () => {
        const g = createWelvetGrid(JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 }));
        const p = g.placeDense(JSON.stringify({ in: 8, out: 8, dtype: 1 }));
        if (isErr(p)) throw new Error(p.error);
        const c = g.convertDense(dt.id, 0);
        if (isErr(c)) throw new Error(c.error);
        const ent = g.serializeEntity();
        if (isErr(ent) || !(ent instanceof Uint8Array)) throw new Error("entity");
        const back = DeserializeGrid(ent);
        if (isErr(back)) throw new Error(back.error);
        back.free?.();
        g.free?.();
      });
    }
    const formats = JSON.parse(PackableFormats()) as { id: number; name: string }[];
    for (const fmt of formats) {
      if (fmt.id === 0) continue;
      cell("serialization", `convert_quant/${fmt.name}`, () => {
        const g = createWelvetGrid(JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 }));
        // AffinePacked needs cols%64==0
        const dim = fmt.name.toLowerCase().includes("affine") ? 64 : 8;
        const p = g.placeDense(JSON.stringify({ in: dim, out: dim, dtype: 1 }));
        if (isErr(p)) throw new Error(p.error);
        const c = g.convertDense(1, fmt.id);
        if (isErr(c)) throw new Error(c.error);
        g.free?.();
      });
    }
  }

  return r;
}
