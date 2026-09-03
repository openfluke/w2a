import { LAYER_DEFS, type LayerPlaceDef } from "./catalog.js";
import { SuiteReport, isErr, classifyOpError } from "./report.js";

export interface MatrixOpts {
  /** Restrict dtypes (default: all from listWelvetDTypes). */
  dtypes?: { id: number; name: string }[];
  /** Restrict formats (default: FormatNone only). */
  formats?: { id: number; name: string }[];
  /** Restrict layers. */
  layers?: LayerPlaceDef[];
  /** Backend name on grid JSON. */
  backend?: string;
  /** Also run entity roundtrip. */
  entity?: boolean;
  /** Also run forward. */
  forward?: boolean;
  /** Also run trainSGD when not skipTrain. */
  train?: boolean;
  report?: SuiteReport;
}

function fill(n: number, v = 0.1): Float32Array {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = v * ((i % 7) + 1);
  return a;
}

function placeBody(def: LayerPlaceDef, dtype: number, format: number): string {
  return JSON.stringify({ z: 0, y: 0, x: 0, l: 0, dtype, format, act: "relu", ...def.spec });
}

/**
 * Layer x dtype x format matrix — place (+ optional forward/train/entity).
 * Mirrors Go suites train_matrix + matrix_bench accessibility on WASM.
 */
export function runLayersMatrix(opts: MatrixOpts = {}): SuiteReport {
  const report = opts.report ?? new SuiteReport();
  const dtypes = opts.dtypes ?? (JSON.parse(listWelvetDTypes()) as { id: number; name: string }[]);
  const formats = opts.formats ?? [{ id: 0, name: "none" }];
  const layers = opts.layers ?? LAYER_DEFS;
  const backend = opts.backend ?? "cpu_tiled";
  const doFwd = opts.forward !== false;
  const doTrain = opts.train !== false;
  const doEntity = opts.entity !== false;

  report.log(`\n## layers matrix layers=${layers.length} dtypes=${dtypes.length} formats=${formats.length} backend=${backend}`);

  for (const layer of layers) {
    for (const dt of dtypes) {
      for (const fmt of formats) {
        const caseName = `${layer.id}/dt=${dt.name}/fmt=${fmt.name}/${backend}`;
        const t0 = performance.now();
        try {
          if (typeof welvetPermutationOK === "function" && !welvetPermutationOK(layer.id, dt.id, fmt.id, 0)) {
            report.record("layers", caseName, "SKIP", "PermutationOK=false", performance.now() - t0);
            continue;
          }
          const g = createWelvetGrid(JSON.stringify({
            depth: 1, rows: 1, cols: 1, layers_per_cell: 1, backend,
          }));
          const fn = (g as unknown as Record<string, (s: string) => unknown>)[layer.method];
          if (typeof fn !== "function") {
            report.record("layers", caseName, "FAIL", `missing ${layer.method}`, performance.now() - t0);
            continue;
          }
          const placed = fn.call(g, placeBody(layer, dt.id, fmt.id));
          if (isErr(placed)) {
            report.record("layers", caseName, classifyOpError(placed.error), `place: ${placed.error}`, performance.now() - t0);
            g.free?.();
            continue;
          }

          let trainOut = layer.outLen;
          if (doFwd) {
            const inp = fill(layer.inLen);
            const fwd = layer.shape
              ? g.forward(inp, JSON.stringify(layer.shape))
              : g.forward(inp);
            if (isErr(fwd)) {
              report.record("layers", caseName, classifyOpError(fwd.error), `fwd: ${fwd.error}`, performance.now() - t0);
              g.free?.();
              continue;
            }
            if (fwd.output && fwd.output.length > 0) trainOut = fwd.output.length;
          }

          if (doTrain && !layer.skipTrain) {
            const tr = layer.shape
              ? g.trainSGD(fill(layer.inLen), JSON.stringify(layer.shape), fill(trainOut, 0), 0.01)
              : g.trainSGD(fill(layer.inLen), fill(trainOut, 0), 0.01);
            if (isErr(tr)) {
              report.record("layers", caseName, classifyOpError(tr.error), `train: ${tr.error}`, performance.now() - t0);
              g.free?.();
              continue;
            }
          }

          if (doEntity) {
            const ent = g.serializeEntity();
            if (isErr(ent) || !(ent instanceof Uint8Array) || ent.length < 4) {
              const msg = isErr(ent) ? ent.error : "bad entity bytes";
              report.record("layers", caseName, classifyOpError(msg), `entity: ${msg}`, performance.now() - t0);
              g.free?.();
              continue;
            }
            const back = DeserializeGrid(ent);
            if (isErr(back)) {
              report.record("layers", caseName, classifyOpError(back.error), `deser: ${back.error}`, performance.now() - t0);
              g.free?.();
              continue;
            }
            back.free?.();
          }

          report.record("layers", caseName, "OK", "", performance.now() - t0);
          g.free?.();
        } catch (e) {
          report.record("layers", caseName, "FAIL", String(e), performance.now() - t0);
        }
      }
    }
  }
  return report;
}

/** Dense × all formats (Float32). Uses dim=64 so AffinePacked group size is OK. */
export function runDenseQuantMatrix(report?: SuiteReport): SuiteReport {
  const r = report ?? new SuiteReport();
  const formats = JSON.parse(listWelvetFormats()) as { id: number; name: string }[];
  const dense64: LayerPlaceDef = {
    id: "dense",
    method: "placeDense",
    spec: { in: 64, out: 64, act: "relu" },
    inLen: 64,
    outLen: 64,
  };
  return runLayersMatrix({
    report: r,
    layers: [dense64],
    dtypes: [{ id: 1, name: "float32" }],
    formats,
    backend: "cpu_tiled",
  });
}

/** FormatNone × all dtypes × all layers (place+fwd+train+entity). */
export function runFullDtypeLayerMatrix(report?: SuiteReport): SuiteReport {
  const r = report ?? new SuiteReport();
  const dtypes = JSON.parse(listWelvetDTypes()) as { id: number; name: string }[];
  return runLayersMatrix({
    report: r,
    dtypes,
    formats: [{ id: 0, name: "none" }],
    backend: "cpu_tiled",
  });
}
