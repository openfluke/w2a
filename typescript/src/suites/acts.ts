import { SuiteReport, isErr, classifyOpError } from "./report.js";

const DEFAULT_ACTS = ["linear", "relu", "silu", "gelu", "tanh", "sigmoid", "leaky_relu"];

/** Dense activation sweep — place + forward + one train step. */
export function runActsMatrix(report?: SuiteReport): SuiteReport {
  const r = report ?? new SuiteReport();
  r.log("\n## activations");

  let acts = DEFAULT_ACTS;
  if (typeof listWelvetActivations === "function") {
    try {
      acts = JSON.parse(listWelvetActivations()) as string[];
    } catch {
      /* keep defaults */
    }
  }

  for (const act of acts) {
    const t0 = performance.now();
    try {
      const g = createWelvetGrid(JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 }));
      const p = g.placeDense(JSON.stringify({ in: 8, out: 8, act, dtype: 1 }));
      if (isErr(p)) throw new Error(p.error);
      const x = new Float32Array(8).fill(0.2);
      const f = g.forward(x);
      if (isErr(f)) throw new Error(f.error);
      const tr = g.trainSGD(x, new Float32Array(8).fill(0.1), 0.05);
      if (isErr(tr)) throw new Error(tr.error);
      r.record("acts", act, "OK", `loss=${tr.loss}`, performance.now() - t0);
      g.free?.();
    } catch (e) {
      r.record("acts", act, classifyOpError(String(e)), String(e), performance.now() - t0);
    }
  }
  return r;
}
