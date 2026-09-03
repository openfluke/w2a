import { LAYER_DEFS, type LayerPlaceDef } from "./catalog.js";
import { SuiteReport, isErr, classifyOpError, type CellStatus } from "./report.js";

const BACKENDS = [
  { id: 0, name: "cpu_tiled", json: "cpu_tiled" },
  { id: 1, name: "simd", json: "simd" },
  { id: 2, name: "webgpu", json: "webgpu" },
] as const;

function fill(n: number, v = 0.1): Float32Array {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = v * ((i % 7) + 1);
  return a;
}

function placeBody(def: LayerPlaceDef, dtype: number, format: number, z = 0, y = 0, x = 0, l = 0): string {
  // AffinePacked needs cols % 64 == 0 — bump dense dims when needed
  const spec = { ...def.spec };
  if (format === 20 /* AffinePacked */ && def.id === "dense") {
    spec.in = 64;
    spec.out = 64;
  }
  return JSON.stringify({ z, y, x, l, dtype, format, act: "relu", ...spec });
}

function inOutLen(def: LayerPlaceDef, format: number): { inLen: number; outLen: number } {
  if (format === 20 && def.id === "dense") return { inLen: 64, outLen: 64 };
  return { inLen: def.inLen, outLen: def.outLen };
}

export type MegaOpts = {
  report?: SuiteReport;
  /** Log every N cells (default 1000). */
  progressEvery?: number;
  layers?: LayerPlaceDef[];
  /** Include cameral arch×mode×dtype/format×backend (default true). */
  cameral?: boolean;
  /** Include dense volumetric 1/2/3 (default true). */
  volumetric?: boolean;
  /** Record place/fwd/train/entity as separate cells (×4, closer to Go census). */
  splitOps?: boolean;
};

/**
 * Full W2A-scale WASM matrix:
 *   layers × dtypes × formats × backends × {place,fwd,train,entity}
 *   + cameral arches × modes × (dtypes|formats) × backends
 *   + dense volumetric grids × (dtypes|formats) × backends
 *
 * Quiet by default — only FAILs + periodic progress are logged.
 */
export function runMegaW2AMatrix(opts: MegaOpts = {}): SuiteReport {
  const report = opts.report ?? new SuiteReport(console.log, { quiet: true, progressEvery: opts.progressEvery ?? 1000 });
  const layers = opts.layers ?? LAYER_DEFS;
  const dtypes = JSON.parse(listWelvetDTypes()) as { id: number; name: string }[];
  const formats = JSON.parse(listWelvetFormats()) as { id: number; name: string }[];
  const split = opts.splitOps !== false;

  const layerCells = layers.length * dtypes.length * formats.length * BACKENDS.length * (split ? 4 : 1);
  report.log(`\n## MEGA layers×dtype×format×backend${split ? "×ops(4)" : ""} → ~${layerCells} cells`);

  for (const layer of layers) {
    for (const be of BACKENDS) {
      for (const dt of dtypes) {
        for (const fmt of formats) {
          const base = `${layer.id}/dt=${dt.name}/fmt=${fmt.name}/${be.name}`;
          if (typeof welvetPermutationOK === "function" && !welvetPermutationOK(layer.id, dt.id, fmt.id, be.id)) {
            report.record("mega_layers", base, "SKIP", "PermutationOK=false");
            continue;
          }
          runLayerCell(report, layer, dt, fmt, be.json, base, split);
        }
      }
    }
  }

  if (opts.cameral !== false) {
    runCameralMega(report, dtypes, formats);
  }
  if (opts.volumetric !== false) {
    runVolumetricDense(report, dtypes, formats);
  }

  return report;
}

function rec(
  report: SuiteReport,
  suite: string,
  name: string,
  status: CellStatus,
  note: string,
  split: boolean,
  op?: string,
) {
  report.record(suite, split && op ? `${name}/${op}` : name, status, note);
}

function runLayerCell(
  report: SuiteReport,
  layer: LayerPlaceDef,
  dt: { id: number; name: string },
  fmt: { id: number; name: string },
  backend: string,
  base: string,
  split: boolean,
) {
  const { inLen, outLen } = inOutLen(layer, fmt.id);
  let g: ReturnType<typeof createWelvetGrid> | null = null;
  try {
    g = createWelvetGrid(JSON.stringify({
      depth: 1, rows: 1, cols: 1, layers_per_cell: 1, backend,
    }));
    const fn = (g as unknown as Record<string, (s: string) => unknown>)[layer.method];
    if (typeof fn !== "function") {
      rec(report, "mega_layers", base, "FAIL", `missing ${layer.method}`, split, "place");
      return;
    }
    const placed = fn.call(g, placeBody(layer, dt.id, fmt.id));
    if (isErr(placed)) {
      rec(report, "mega_layers", base, classifyOpError(placed.error), `place: ${placed.error}`, split, "place");
      if (split) {
        rec(report, "mega_layers", base, "SKIP", "place failed", split, "forward");
        rec(report, "mega_layers", base, "SKIP", "place failed", split, "train");
        rec(report, "mega_layers", base, "SKIP", "place failed", split, "entity");
      }
      return;
    }
    rec(report, "mega_layers", base, "OK", "", split, "place");

    const inp = fill(inLen);
    const fwd = layer.shape && fmt.id !== 20
      ? g.forward(inp, JSON.stringify(layer.shape))
      : g.forward(inp);
    let trainOut = outLen;
    if (isErr(fwd)) {
      rec(report, "mega_layers", base, classifyOpError(fwd.error), `fwd: ${fwd.error}`, split, "forward");
    } else {
      rec(report, "mega_layers", base, "OK", "", split, "forward");
      if (fwd.output && fwd.output.length > 0) trainOut = fwd.output.length;
    }

    if (layer.skipTrain) {
      rec(report, "mega_layers", base, "SKIP", "skipTrain", split, "train");
    } else {
      const tr = layer.shape
        ? g.trainSGD(inp, JSON.stringify(layer.shape), fill(trainOut, 0), 0.01)
        : g.trainSGD(inp, fill(trainOut, 0), 0.01);
      if (isErr(tr)) {
        rec(report, "mega_layers", base, classifyOpError(tr.error), `train: ${tr.error}`, split, "train");
      } else {
        rec(report, "mega_layers", base, "OK", "", split, "train");
      }
    }

    const ent = g.serializeEntity();
    if (isErr(ent) || !(ent instanceof Uint8Array) || ent.length < 4) {
      const msg = isErr(ent) ? ent.error : "bad entity";
      rec(report, "mega_layers", base, classifyOpError(msg), `entity: ${msg}`, split, "entity");
    } else {
      const back = DeserializeGrid(ent);
      if (isErr(back)) {
        rec(report, "mega_layers", base, classifyOpError(back.error), `deser: ${back.error}`, split, "entity");
      } else {
        rec(report, "mega_layers", base, "OK", "", split, "entity");
        back.free?.();
      }
    }
  } catch (e) {
    rec(report, "mega_layers", base, "FAIL", String(e), split);
  } finally {
    g?.free?.();
  }
}

const CAMERAL_ARCHES = ["bicameral", "hemi2", "hemi3", "parallel"] as const;

function runCameralMega(
  report: SuiteReport,
  dtypes: { id: number; name: string }[],
  formats: { id: number; name: string }[],
) {
  const modes = JSON.parse(listWelvetNamedTrainModes()) as string[];
  const fnCells = CAMERAL_ARCHES.length * modes.length * dtypes.length * BACKENDS.length;
  const qCells = CAMERAL_ARCHES.length * modes.length * formats.length * BACKENDS.length;
  report.log(`\n## MEGA cameral FormatNone ~${fnCells} + quant ~${qCells}`);

  // FormatNone × dtypes
  for (const arch of CAMERAL_ARCHES) {
    for (const mode of modes) {
      for (const dt of dtypes) {
        for (const be of BACKENDS) {
          const name = `${arch}/${mode}/dt=${dt.name}/none/${be.name}`;
          runCameralCell(report, arch, mode, dt.id, 0, be.json, name);
        }
      }
    }
  }
  // Quants @ float32
  for (const arch of CAMERAL_ARCHES) {
    for (const mode of modes) {
      for (const fmt of formats) {
        for (const be of BACKENDS) {
          const name = `${arch}/${mode}/dt=float32/fmt=${fmt.name}/${be.name}`;
          runCameralCell(report, arch, mode, 1, fmt.id, be.json, name);
        }
      }
    }
  }
}

function runCameralCell(
  report: SuiteReport,
  arch: string,
  mode: string,
  dtype: number,
  format: number,
  _backend: string,
  name: string,
) {
  try {
    const dim = format === 20 ? 64 : 4;
    let obj: {
      trainStackMSE?: (a: Float32Array, b: Float32Array, m: string, lr: number) => { error?: string; loss?: number };
      trainMSE?: (a: Float32Array, b: Float32Array, m: string, lr: number) => { error?: string; loss?: number };
      setChildModes?: (s: string) => unknown;
      setBranchModes?: (s: string) => unknown;
      free?: () => unknown;
    } | null = null;
    if (arch === "bicameral") {
      obj = createWelvetBicameral(JSON.stringify({ in: dim, hidden: dim, out: dim, dtype, format }));
    } else if (arch === "hemi2") {
      obj = createWelvetHemispheres(JSON.stringify({ dim, n: 2, combine: "add" }));
    } else if (arch === "hemi3") {
      obj = createWelvetHemispheres(JSON.stringify({ dim, n: 3, combine: "add" }));
    } else {
      obj = createWelvetParallel(JSON.stringify({ Dim: dim, OutFeat: dim, Branches: 2, Combine: "add" }));
    }
    if (isErr(obj)) {
      report.record("mega_cameral", name, classifyOpError(obj.error), obj.error);
      return;
    }
    const inp = fill(dim);
    const tgt = fill(dim, 0);
    tgt[0] = 1;
    if (obj.setChildModes) obj.setChildModes(JSON.stringify([mode, mode]));
    if (obj.setBranchModes) {
      const n = arch === "hemi3" ? 3 : 2;
      obj.setBranchModes(JSON.stringify(Array(n).fill(mode)));
    }
    const tr = obj.trainStackMSE
      ? obj.trainStackMSE(inp, tgt, mode, 0.05)
      : obj.trainMSE!(inp, tgt, mode, 0.05);
    if (isErr(tr)) {
      report.record("mega_cameral", name, classifyOpError(tr.error), tr.error);
    } else {
      report.record("mega_cameral", name, "OK", "");
    }
    obj.free?.();
  } catch (e) {
    report.record("mega_cameral", name, "FAIL", String(e));
  }
}

function runVolumetricDense(
  report: SuiteReport,
  dtypes: { id: number; name: string }[],
  formats: { id: number; name: string }[],
) {
  const sizes = [1, 2, 3];
  report.log(`\n## MEGA volumetric dense grids ${sizes.join("/")} × dtypes/formats × backends`);
  for (const n of sizes) {
    for (const be of BACKENDS) {
      for (const dt of dtypes) {
        const name = `vol/${n}³/dt=${dt.name}/none/${be.name}`;
        runVolCell(report, n, dt.id, 0, be.json, name);
      }
      for (const fmt of formats) {
        const name = `vol/${n}³/dt=float32/fmt=${fmt.name}/${be.name}`;
        runVolCell(report, n, 1, fmt.id, be.json, name);
      }
    }
  }
}

function runVolCell(
  report: SuiteReport,
  n: number,
  dtype: number,
  format: number,
  backend: string,
  name: string,
) {
  try {
    const dim = format === 20 ? 64 : 8;
    const g = createWelvetGrid(JSON.stringify({
      depth: n, rows: n, cols: n, layers_per_cell: 1, backend,
    }));
    for (let z = 0; z < n; z++) {
      for (let y = 0; y < n; y++) {
        for (let x = 0; x < n; x++) {
          const p = g.placeDense(JSON.stringify({
            z, y, x, l: 0, in: dim, out: dim, act: "relu", dtype, format,
          }));
          if (isErr(p)) {
            report.record("mega_vol", name, classifyOpError(p.error), `place ${z},${y},${x}: ${p.error}`);
            g.free?.();
            return;
          }
        }
      }
    }
    // chain along z for n>1 on the (0,0) column
    for (let z = 0; z < n - 1; z++) {
      g.setRemoteLink(z, 0, 0, 0, z + 1, 0, 0, 0);
    }
    const inp = fill(dim);
    const fwd = g.forward(inp);
    if (isErr(fwd)) {
      // remote hop may GAP on multi-cell — still counts
      report.record("mega_vol", name, classifyOpError(fwd.error), `fwd: ${fwd.error}`);
    } else {
      const tr = g.trainSGD(inp, fill(dim, 0), 0.01);
      if (isErr(tr)) report.record("mega_vol", name, classifyOpError(tr.error), `train: ${tr.error}`);
      else report.record("mega_vol", name, "OK", "");
    }
    g.free?.();
  } catch (e) {
    report.record("mega_vol", name, "FAIL", String(e));
  }
}
