import { SuiteReport, isErr, classifyOpError } from "./report.js";

export type TrainModesLayersOpts = {
  /** FormatNone × all dtypes (default true for mega/full). */
  dtypeSweep?: boolean;
  /** Packable quants @ float32 (default true for mega). */
  quantSweep?: boolean;
  /** Only these kinds (default: all from WASM). */
  kinds?: string[];
  /** Only these modes (default: all named). */
  modes?: string[];
  quietProgressEvery?: number;
};

/**
 * Native test49 / cameral_poly port:
 * every layer kind as dual hemispheres in Stack[Parallel] × every named TrainMode
 * × (FormatNone×dtypes + quants@f32).
 */
export function runTrainModesLayersMatrix(
  report?: SuiteReport,
  opts: TrainModesLayersOpts = {},
): SuiteReport {
  const r = report ?? new SuiteReport();
  if (typeof TrainCameralPoly !== "function" || typeof listWelvetCameralPolyKinds !== "function") {
    r.record("train_modes_layers", "api", "FAIL", "TrainCameralPoly / listWelvetCameralPolyKinds missing");
    return r;
  }

  const kinds = opts.kinds ?? (JSON.parse(listWelvetCameralPolyKinds()) as string[]);
  const modes = opts.modes ?? (JSON.parse(listWelvetNamedTrainModes()) as string[]);
  const dtypes = opts.dtypeSweep !== false
    ? (JSON.parse(listWelvetDTypes()) as { id: number; name: string }[])
    : [{ id: 1, name: "float32" }];
  const formats = opts.quantSweep
    ? (JSON.parse(listWelvetFormats()) as { id: number; name: string }[])
    : [{ id: 0, name: "none" }];

  const fnCells = kinds.length * modes.length * dtypes.length;
  const qCells = opts.quantSweep
    ? kinds.length * modes.length * formats.filter((f) => f.id !== 0).length
    : 0;
  r.log(`\n## train_modes × layers (cameral poly) kinds=${kinds.length} modes=${modes.length} → ~${fnCells + qCells} cells`);

  let n = 0;
  const every = opts.quietProgressEvery ?? 0;

  for (const kind of kinds) {
    for (const mode of modes) {
      for (const dt of dtypes) {
        const name = `${kind}/${mode}/dt=${dt.name}/none`;
        cell(r, name, () => TrainCameralPoly(kind, mode, dt.id, 0));
        n++;
        if (every > 0 && n % every === 0) {
          r.log(`  … train_modes_layers progress ${n}`);
        }
      }
    }
  }

  if (opts.quantSweep) {
    for (const kind of kinds) {
      for (const mode of modes) {
        for (const fmt of formats) {
          if (fmt.id === 0) continue;
          // AffinePacked needs cols%64 — poly uses dim=32; skip AffinePacked
          if (fmt.name.toLowerCase().includes("affine")) {
            r.record("train_modes_layers", `${kind}/${mode}/fmt=${fmt.name}`, "SKIP", "poly dim=32 not AffinePacked-aligned");
            continue;
          }
          // GDN v0 only packs FormatNone / BinaryPacked
          if (kind === "gdn" && fmt.name !== "BinaryPacked" && fmt.id !== 0) {
            r.record("train_modes_layers", `${kind}/${mode}/fmt=${fmt.name}`, "SKIP", "gdn Pack only None|BinaryPacked");
            continue;
          }
          const name = `${kind}/${mode}/dt=float32/fmt=${fmt.name}`;
          cell(r, name, () => TrainCameralPoly(kind, mode, 1, fmt.id));
          n++;
          if (every > 0 && n % every === 0) {
            r.log(`  … train_modes_layers progress ${n}`);
          }
        }
      }
    }
  }

  return r;
}

function cell(
  r: SuiteReport,
  name: string,
  fn: () => { loss?: number; note?: string; error?: string },
) {
  const t0 = performance.now();
  try {
    const out = fn();
    if (isErr(out)) {
      const status = /grid|mesh|unimplemented|not support|freeze|shadow/i.test(out.error)
        ? "GAP"
        : classifyOpError(out.error);
      r.record("train_modes_layers", name, status, out.error, performance.now() - t0);
      return;
    }
    r.record(
      "train_modes_layers",
      name,
      "OK",
      `loss=${out.loss}${out.note && out.note !== "ok" ? " " + out.note : ""}`,
      performance.now() - t0,
    );
  } catch (e) {
    r.record("train_modes_layers", name, "FAIL", String(e), performance.now() - t0);
  }
}
