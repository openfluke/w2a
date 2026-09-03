import { SuiteReport, isErr, classifyOpError } from "./report.js";

export type TrainModesOpts = {
  /** Also sweep FormatNone × all dtypes on bicameral NormalBP + each named mode @ f32. */
  dtypeSweep?: boolean;
};

/** Every named TrainMode on bicameral Stack + Parallel (mirrors Test49 / credit_perm). */
export function runTrainModesMatrix(report?: SuiteReport, opts: TrainModesOpts = {}): SuiteReport {
  const r = report ?? new SuiteReport();
  const modes = JSON.parse(listWelvetNamedTrainModes()) as string[];
  r.log(`\n## train_modes named=${modes.length}`);

  const inp = new Float32Array([0.1, 0.2, 0.3, 0.4]);
  const tgt = new Float32Array([1, 0, 0, 0]);

  for (const mode of modes) {
    const t0 = performance.now();
    try {
      const st = createWelvetBicameral(JSON.stringify({ in: 4, hidden: 4, out: 4 }));
      if (isErr(st)) {
        r.record("train_modes", mode, classifyOpError(st.error), st.error, performance.now() - t0);
        continue;
      }
      st.setChildModes(JSON.stringify([mode, mode]));
      const tr = st.trainStackMSE(inp, tgt, mode, 0.05);
      if (isErr(tr)) {
        const status = /grid|mesh|unimplemented|not support/i.test(tr.error)
          ? "GAP"
          : classifyOpError(tr.error);
        r.record("train_modes", `stack/${mode}`, status, tr.error, performance.now() - t0);
      } else {
        r.record("train_modes", `stack/${mode}`, "OK", `loss=${tr.loss}`, performance.now() - t0);
      }
      st.free?.();
    } catch (e) {
      r.record("train_modes", `stack/${mode}`, "FAIL", String(e), performance.now() - t0);
    }
  }

  for (const mode of modes) {
    const t0 = performance.now();
    try {
      const p = createWelvetParallel(JSON.stringify({ Dim: 4, OutFeat: 4, Branches: 2, Combine: "add" }));
      if (isErr(p)) {
        r.record("train_modes", `parallel/${mode}`, classifyOpError(p.error), p.error, performance.now() - t0);
        continue;
      }
      p.setBranchModes(JSON.stringify([mode, mode]));
      const tr = p.trainMSE(inp, tgt, mode, 0.05);
      if (isErr(tr)) {
        const status = /grid|mesh|unimplemented|not support/i.test(tr.error)
          ? "GAP"
          : classifyOpError(tr.error);
        r.record("train_modes", `parallel/${mode}`, status, tr.error, performance.now() - t0);
      } else {
        r.record("train_modes", `parallel/${mode}`, "OK", `loss=${tr.loss}`, performance.now() - t0);
      }
      p.free?.();
    } catch (e) {
      r.record("train_modes", `parallel/${mode}`, "FAIL", String(e), performance.now() - t0);
    }
  }

  if (opts.dtypeSweep) {
    const dtypes = JSON.parse(listWelvetDTypes()) as { id: number; name: string }[];
    r.log(`\n## train_modes × dtypes (bicameral NormalBP) n=${dtypes.length}`);
    for (const dt of dtypes) {
      const t0 = performance.now();
      try {
        const st = createWelvetBicameral(JSON.stringify({ in: 4, hidden: 4, out: 4, dtype: dt.id }));
        if (isErr(st)) {
          r.record("train_modes", `dtype/${dt.name}`, classifyOpError(st.error), st.error, performance.now() - t0);
          continue;
        }
        const tr = st.trainStackMSE(inp, tgt, "NormalBP", 0.05);
        if (isErr(tr)) {
          r.record("train_modes", `dtype/${dt.name}`, classifyOpError(tr.error), tr.error, performance.now() - t0);
        } else {
          r.record("train_modes", `dtype/${dt.name}`, "OK", `loss=${tr.loss}`, performance.now() - t0);
        }
        st.free?.();
      } catch (e) {
        r.record("train_modes", `dtype/${dt.name}`, "FAIL", String(e), performance.now() - t0);
      }
    }

    // Named modes × float32 already covered; also sample concrete modes × a few quants
    const formats = JSON.parse(listWelvetFormats()) as { id: number; name: string }[];
    const concrete = typeof listWelvetConcreteTrainModes === "function"
      ? JSON.parse(listWelvetConcreteTrainModes()) as string[]
      : ["NormalBP", "Tween", "StepTween"];
    for (const mode of concrete) {
      for (const fmt of formats) {
        if (fmt.id === 0) continue;
        const t0 = performance.now();
        const name = `concrete/${mode}/fmt=${fmt.name}`;
        try {
          const st = createWelvetBicameral(JSON.stringify({
            in: fmt.name.toLowerCase().includes("affine") ? 64 : 4,
            hidden: fmt.name.toLowerCase().includes("affine") ? 64 : 4,
            out: fmt.name.toLowerCase().includes("affine") ? 64 : 4,
            dtype: 1,
            format: fmt.id,
          }));
          if (isErr(st)) {
            r.record("train_modes", name, classifyOpError(st.error), st.error, performance.now() - t0);
            continue;
          }
          const dim = fmt.name.toLowerCase().includes("affine") ? 64 : 4;
          const tr = st.trainStackMSE(new Float32Array(dim).fill(0.1), new Float32Array(dim).fill(0), mode, 0.05);
          if (isErr(tr)) {
            const status = /grid|mesh|unimplemented|not support/i.test(tr.error)
              ? "GAP"
              : classifyOpError(tr.error);
            r.record("train_modes", name, status, tr.error, performance.now() - t0);
          } else {
            r.record("train_modes", name, "OK", `loss=${tr.loss}`, performance.now() - t0);
          }
          st.free?.();
        } catch (e) {
          r.record("train_modes", name, "FAIL", String(e), performance.now() - t0);
        }
      }
    }
  }

  return r;
}

/** CamSync / CamKit / Tanhi / Combine accessibility. */
export function runCameralMatrix(report?: SuiteReport): SuiteReport {
  const r = report ?? new SuiteReport();
  r.log("\n## cameral features");
  const inp = new Float32Array([0.1, 0.2, 0.3, 0.4]);
  const tgt = new Float32Array([1, 0, 0, 0]);
  const combines = ["add", "avg", "max"]; // concat changes outFeat — covered as GAP below
  for (const combine of ["concat"]) {
    const t0 = performance.now();
    const p = createWelvetHemispheres(JSON.stringify({ dim: 4, n: 2, combine }));
    if (isErr(p)) {
      r.record("cameral", `combine/${combine}`, classifyOpError(p.error), p.error, performance.now() - t0);
    } else {
      // concat → out 8; train with matching target
      const tr = p.trainMSE(inp, new Float32Array(8).fill(0), "NormalBP", 0.05);
      if (isErr(tr)) r.record("cameral", `combine/${combine}`, classifyOpError(tr.error), tr.error, performance.now() - t0);
      else r.record("cameral", `combine/${combine}`, "OK", `loss=${tr.loss}`, performance.now() - t0);
      p.free?.();
    }
  }

  for (const combine of combines) {
    const t0 = performance.now();
    try {
      const p = createWelvetHemispheres(JSON.stringify({ dim: 4, n: 2, combine }));
      if (isErr(p)) {
        r.record("cameral", `combine/${combine}`, classifyOpError(p.error), p.error, performance.now() - t0);
        continue;
      }
      p.setCamSync(JSON.stringify({ Enabled: true, Alpha: 1.0 }));
      p.setCamKit(JSON.stringify({ ShadowCoef: 1, DNAReg: 0, SurpriseThresh: 0 }));
      p.setTanhi(JSON.stringify({ Enabled: false }));
      const tr = p.trainMSE(inp, tgt, "NormalBP", 0.05);
      if (isErr(tr)) {
        r.record("cameral", `combine/${combine}`, classifyOpError(tr.error), tr.error, performance.now() - t0);
      } else {
        r.record("cameral", `combine/${combine}`, "OK", `loss=${tr.loss}`, performance.now() - t0);
      }
      p.free?.();
    } catch (e) {
      r.record("cameral", `combine/${combine}`, "FAIL", String(e), performance.now() - t0);
    }
  }

  // Grid tanhi
  {
    const t0 = performance.now();
    const g = createWelvetGrid(JSON.stringify({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 }));
    g.placeDense(JSON.stringify({ in: 4, out: 4, dtype: 1 }));
    const cfg = g.configureTanhi(JSON.stringify({ Enabled: false, Host: "127.0.0.1", Port: DefaultTanhiUDPPort() }));
    if (isErr(cfg)) r.record("cameral", "grid/tanhi", classifyOpError(cfg.error), cfg.error, performance.now() - t0);
    else r.record("cameral", "grid/tanhi", "OK", "", performance.now() - t0);
    EmitSweep("matrix");
    r.record("cameral", "EmitSweep", "OK");
    g.free?.();
  }

  return r;
}
