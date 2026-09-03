import { SuiteReport, isErr, classifyOpError } from "./report.js";

/** Extra step / tween / remote-link smokes beyond systems_matrix. */
export function runStepTweenHonesty(report?: SuiteReport): SuiteReport {
  const r = report ?? new SuiteReport();
  r.log("\n## step/tween honesty");

  const cell = (name: string, fn: () => void) => {
    const t0 = performance.now();
    try {
      fn();
      r.record("step_tween", name, "OK", "", performance.now() - t0);
    } catch (e) {
      r.record("step_tween", name, classifyOpError(String(e)), String(e), performance.now() - t0);
    }
  };

  cell("step_forward_backward", () => {
    const g = createWelvetGrid(JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 }));
    const p = g.placeDense(JSON.stringify({ in: 4, out: 4, act: "relu", dtype: 1 }));
    if (isErr(p)) throw new Error(p.error);
    const st = createWelvetStepState(g._id);
    if (isErr(st)) throw new Error(st.error);
    st.setInput(new Float32Array([0.1, 0.2, 0.3, 0.4]));
    const s = st.step(true);
    if (isErr(s)) throw new Error(s.error);
    st.free?.();
    g.free?.();
  });

  cell("tween_train_gap_smoke", () => {
    const g = createWelvetGrid(JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 }));
    const p = g.placeDense(JSON.stringify({ in: 4, out: 4, act: "linear", dtype: 1 }));
    if (isErr(p)) throw new Error(p.error);
    const x = new Float32Array([0.2, 0.1, 0.3, 0.4]);
    const y = new Float32Array([1, 0, 0, 0]);
    const a = g.trainTween(x, y, 0.05);
    if (isErr(a)) throw new Error(a.error);
    const b = g.trainTween(x, y, 0.05);
    if (isErr(b)) throw new Error(b.error);
    // Prefer loss not exploding; gap reduce is soft on WASM.
    if (!Number.isFinite(a.loss) || !Number.isFinite(b.loss)) throw new Error("non-finite loss");
    g.free?.();
  });

  cell("mesh_train_smoke", () => {
    const g = createWelvetGrid(JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 }));
    const p = g.placeDense(JSON.stringify({ in: 4, out: 4, act: "relu", dtype: 1 }));
    if (isErr(p)) throw new Error(p.error);
    const tr = g.trainMesh(new Float32Array(4).fill(0.2), new Float32Array([1, 0, 0, 0]), 2, 0.05);
    if (isErr(tr)) throw new Error(tr.error);
    g.free?.();
  });

  cell("remote_link_2cell", () => {
    // Match Go step/remoteLinkSmoke: two layers in one cell, remote hop + 2× step.Forward
    const g = createWelvetGrid(JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 2 }));
    const p0 = g.placeDense(JSON.stringify({ z: 0, y: 0, x: 0, l: 0, in: 4, out: 4, act: "linear", dtype: 1 }));
    const p1 = g.placeDense(JSON.stringify({ z: 0, y: 0, x: 0, l: 1, in: 4, out: 4, act: "linear", dtype: 1 }));
    if (isErr(p0)) throw new Error(p0.error);
    if (isErr(p1)) throw new Error(p1.error);
    const link = g.setRemoteLink(0, 0, 0, 1, 0, 0, 0, 0);
    if (isErr(link)) throw new Error(link.error);
    const st = createWelvetStepState(g._id);
    if (isErr(st)) throw new Error(st.error);
    st.setInput(new Float32Array([1, 2, 3, 4]));
    const s0 = st.step(false);
    if (isErr(s0)) throw new Error(s0.error);
    const s1 = st.step(false);
    if (isErr(s1)) throw new Error(s1.error);
    st.free?.();
    g.free?.();
  });

  cell("serialize_roundtrip_forward", () => {
    const g = createWelvetGrid(JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 }));
    const p = g.placeDense(JSON.stringify({ in: 4, out: 4, act: "linear", dtype: 1 }));
    if (isErr(p)) throw new Error(p.error);
    const x = new Float32Array([0.1, 0.2, 0.3, 0.4]);
    const f0 = g.forward(x);
    if (isErr(f0) || !f0.output) throw new Error(isErr(f0) ? f0.error : "fwd");
    const blob = SerializeGrid(g._id);
    if (isErr(blob) || !(blob instanceof Uint8Array)) throw new Error(isErr(blob) ? blob.error : "ser");
    const back = DeserializeGrid(blob);
    if (isErr(back)) throw new Error(back.error);
    const f1 = back.forward(x);
    if (isErr(f1) || !f1.output) throw new Error(isErr(f1) ? f1.error : "fwd2");
    let max = 0;
    for (let i = 0; i < f0.output.length; i++) max = Math.max(max, Math.abs(f0.output[i]! - f1.output[i]!));
    if (max > 1e-4) throw new Error(`roundtrip Δ=${max}`);
    back.free?.();
    g.free?.();
  });

  return r;
}
