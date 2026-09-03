import { SuiteReport, isErr, classifyOpError } from "./report.js";

const DET_TOL_FWD = 1e-5;
const DET_TOL_BWD = 1e-4;
const FD_TOL = 5e-2;

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = Math.min(a.length, b.length);
  let max = 0;
  for (let i = 0; i < n; i++) {
    const d = Math.abs(a[i]! - b[i]!);
    if (d > max) max = d;
  }
  return max;
}

function fillInput(n: number): Float32Array {
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = (i % 5) * 0.25 + 0.1;
  return x;
}

function makeDense(inDim = 8, outDim = 8, act = "linear"): ReturnType<typeof createWelvetGrid> {
  const g = createWelvetGrid(JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 }));
  const p = g.placeDense(JSON.stringify({ in: inDim, out: outDim, act, dtype: 1 }));
  if (isErr(p)) throw new Error(p.error);
  return g;
}

function seedWeights(g: ReturnType<typeof createWelvetGrid>, inDim: number, outDim: number): Float32Array {
  const w = g.getDenseWeights();
  if (isErr(w) || !(w instanceof Float32Array)) throw new Error(isErr(w) ? w.error : "no weights");
  for (let i = 0; i < w.length; i++) w[i] = ((i % 13) - 6) * 0.1;
  const s = g.setDenseWeights(w);
  if (isErr(s)) throw new Error(s.error);
  return w;
}

/** Portable CPU honesty: det / SC↔MC / loss↓ / gradIn finite-diff / shape tiers. */
export function runHonestyMatrix(report?: SuiteReport): SuiteReport {
  const r = report ?? new SuiteReport();
  r.log("\n## honesty (portable CPU)");

  const cell = (name: string, fn: () => void) => {
    const t0 = performance.now();
    try {
      fn();
      r.record("honesty", name, "OK", "", performance.now() - t0);
    } catch (e) {
      r.record("honesty", name, classifyOpError(String(e)), String(e), performance.now() - t0);
    }
  };

  cell("repeat_forward_det", () => {
    const inDim = 16,
      outDim = 8;
    const x = fillInput(inDim);
    const g0 = makeDense(inDim, outDim);
    const w = seedWeights(g0, inDim, outDim);
    const base = g0.forward(x);
    if (isErr(base) || !base.output) throw new Error(isErr(base) ? base.error : "fwd");
    g0.free?.();
    let max = 0;
    for (let i = 0; i < 3; i++) {
      const g = makeDense(inDim, outDim);
      const s = g.setDenseWeights(w);
      if (isErr(s)) throw new Error(s.error);
      const f = g.forward(x);
      if (isErr(f) || !f.output) throw new Error(isErr(f) ? f.error : "fwd");
      max = Math.max(max, maxAbsDiff(base.output, f.output));
      g.free?.();
    }
    if (max > DET_TOL_FWD) throw new Error(`repeat Δ=${max} > ${DET_TOL_FWD}`);
  });

  cell("sc_mc_fwd_bwd", () => {
    const inDim = 16,
      outDim = 8;
    const x = fillInput(inDim);
    const gSeed = makeDense(inDim, outDim);
    const w = seedWeights(gSeed, inDim, outDim);
    gSeed.free?.();

    const run = (mc: boolean) => {
      const g = makeDense(inDim, outDim);
      const s = g.setDenseWeights(w);
      if (isErr(s)) throw new Error(s.error);
      const sm = g.setMultiCore?.(mc);
      if (sm && isErr(sm)) throw new Error(sm.error);
      const f = g.forward(x);
      if (isErr(f) || !f.output) throw new Error(isErr(f) ? f.error : "fwd");
      const ones = new Float32Array(f.output.length).fill(1);
      const b = g.backward(ones);
      if (isErr(b) || !b.gradIn) throw new Error(isErr(b) ? b.error : "no gradIn");
      const post = new Float32Array(f.output);
      const gin = new Float32Array(b.gradIn);
      g.free?.();
      return { post, gin };
    };
    const sc = run(false);
    const mc = run(true);
    const dP = maxAbsDiff(sc.post, mc.post);
    const dG = maxAbsDiff(sc.gin, mc.gin);
    if (dP > DET_TOL_FWD) throw new Error(`fwd SC↔MC Δ=${dP}`);
    if (dG > DET_TOL_BWD) throw new Error(`bwd SC↔MC Δ=${dG}`);
  });

  cell("train_loss_decreases", () => {
    const g = makeDense(8, 4, "relu");
    seedWeights(g, 8, 4);
    const x = fillInput(8);
    const y = new Float32Array([1, 0, 0, 0]);
    let first = NaN;
    let last = NaN;
    for (let i = 0; i < 12; i++) {
      const tr = g.trainSGD(x, y, 0.1);
      if (isErr(tr) || typeof tr.loss !== "number") throw new Error(isErr(tr) ? tr.error : "loss");
      if (i === 0) first = tr.loss;
      last = tr.loss;
    }
    if (!(last < first - 1e-4)) throw new Error(`loss not ↓: first=${first} last=${last}`);
    g.free?.();
  });

  cell("gradIn_finite_diff", () => {
    const inDim = 8,
      outDim = 4;
    const g = makeDense(inDim, outDim, "linear");
    seedWeights(g, inDim, outDim);
    const x = fillInput(inDim);
    const f = g.forward(x);
    if (isErr(f) || !f.output) throw new Error(isErr(f) ? f.error : "fwd");
    const ones = new Float32Array(f.output.length).fill(1);
    const b = g.backward(ones);
    if (isErr(b) || !b.gradIn) throw new Error(isErr(b) ? b.error : "gradIn");
    const eps = 1e-3;
    let worst = 0;
    for (let i = 0; i < Math.min(4, inDim); i++) {
      const xp = new Float32Array(x);
      const xm = new Float32Array(x);
      xp[i]! += eps;
      xm[i]! -= eps;
      const fp = g.forward(xp);
      const fm = g.forward(xm);
      if (isErr(fp) || isErr(fm) || !fp.output || !fm.output) throw new Error("fd fwd");
      let sumP = 0,
        sumM = 0;
      for (let j = 0; j < fp.output.length; j++) {
        sumP += fp.output[j]!;
        sumM += fm.output[j]!;
      }
      const num = (sumP - sumM) / (2 * eps);
      const ana = b.gradIn[i]!;
      worst = Math.max(worst, Math.abs(num - ana));
    }
    g.free?.();
    if (worst > FD_TOL) throw new Error(`gradIn FD Δ=${worst} > ${FD_TOL}`);
  });

  cell("shape_tiers_SML", () => {
    for (const dim of [32, 64, 128]) {
      const g = makeDense(dim, dim, "relu");
      const x = fillInput(dim);
      const f = g.forward(x);
      if (isErr(f)) throw new Error(`fwd ${dim}: ${f.error}`);
      const ones = new Float32Array(f.output?.length ?? dim).fill(1);
      const b = g.backward(ones);
      if (isErr(b)) throw new Error(`bwd ${dim}: ${b.error}`);
      const y = new Float32Array(dim);
      y[0] = 1;
      const tr = g.trainSGD(x, y, 0.01);
      if (isErr(tr)) throw new Error(`train ${dim}: ${tr.error}`);
      g.free?.();
    }
  });

  cell("entity_before_after_train", () => {
    const g = makeDense(8, 4, "relu");
    seedWeights(g, 8, 4);
    const x = fillInput(8);
    const y = new Float32Array([1, 0, 0, 0]);
    const before = g.serializeEntity();
    if (isErr(before) || !(before instanceof Uint8Array)) throw new Error("entity before");
    const tr = g.trainSGD(x, y, 0.1);
    if (isErr(tr)) throw new Error(tr.error);
    const after = g.serializeEntity();
    if (isErr(after) || !(after instanceof Uint8Array)) throw new Error("entity after");
    let same = before.length === after.length;
    if (same) {
      same = false;
      for (let i = 0; i < before.length; i++) {
        if (before[i] !== after[i]) {
          same = true; // found difference — good
          break;
        }
      }
      if (!same) throw new Error("entity bytes identical after train");
    }
    const back = DeserializeGrid(after);
    if (isErr(back)) throw new Error(back.error);
    const f = back.forward(x);
    if (isErr(f)) throw new Error(f.error);
    back.free?.();
    g.free?.();
  });

  return r;
}
