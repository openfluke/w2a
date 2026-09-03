import { W2A_SUITE_CATALOG, LAYER_DEFS } from "./catalog.js";
import { SuiteReport } from "./report.js";
import { runFullDtypeLayerMatrix, runDenseQuantMatrix, runLayersMatrix } from "./layers_matrix.js";
import { runTrainModesMatrix, runCameralMatrix } from "./train_modes.js";
import { runSystemsMatrix, runSevenMatrix } from "./systems_matrix.js";
import { runMegaW2AMatrix } from "./mega_matrix.js";
import { runHonestyMatrix } from "./honesty.js";
import { runDnaHonestyMatrix } from "./dna_honesty.js";
import { runEvolutionHonestyMatrix } from "./evolution_honesty.js";
import { runActsMatrix } from "./acts.js";
import { runNativeOnlySkipCatalog } from "./native_only.js";
import { runStepTweenHonesty } from "./step_tween_honesty.js";
import { runWeightsMatrix, runStubSuitesMatrix } from "./stub_suites.js";
import { runTrainModesLayersMatrix } from "./train_modes_layers.js";

export { LAYER_DEFS, W2A_SUITE_CATALOG } from "./catalog.js";
export type { LayerPlaceDef, SuiteId } from "./catalog.js";
export { SuiteReport } from "./report.js";
export {
  runFullDtypeLayerMatrix,
  runDenseQuantMatrix,
  runLayersMatrix,
} from "./layers_matrix.js";
export { runTrainModesMatrix, runCameralMatrix } from "./train_modes.js";
export { runTrainModesLayersMatrix } from "./train_modes_layers.js";
export { runSystemsMatrix, runSevenMatrix } from "./systems_matrix.js";
export { runMegaW2AMatrix } from "./mega_matrix.js";
export { runHonestyMatrix } from "./honesty.js";
export { runDnaHonestyMatrix } from "./dna_honesty.js";
export { runEvolutionHonestyMatrix } from "./evolution_honesty.js";
export { runActsMatrix } from "./acts.js";
export { runNativeOnlySkipCatalog } from "./native_only.js";
export { runStepTweenHonesty } from "./step_tween_honesty.js";
export { runWeightsMatrix, runStubSuitesMatrix } from "./stub_suites.js";

export type RunAllOptions = {
  /** quick ≈ 100; full ≈ 900; mega ≈ 200k+ (Go w2a-scale product) */
  profile?: "quick" | "full" | "mega";
  log?: (s: string) => void;
  failOnFail?: boolean;
};

function runPortableHonesty(report: SuiteReport, profile: "quick" | "full" | "mega") {
  runNativeOnlySkipCatalog(report);
  runHonestyMatrix(report);
  runActsMatrix(report);
  runStepTweenHonesty(report);
  runDnaHonestyMatrix(report, { fullDtype: profile !== "quick" });
  runEvolutionHonestyMatrix(report, { fullLayers: profile === "mega" || profile === "full" });
}

export async function runAllW2ASuites(opts: RunAllOptions = {}): Promise<SuiteReport> {
  const profile = opts.profile ?? "full";
  const report = new SuiteReport(opts.log ?? console.log, {
    quiet: profile === "mega",
    progressEvery: profile === "mega" ? 2000 : 0,
  });

  report.log(`=== Welvet WASM W2A matrix (${profile}) engine=${welvetEngineVersion()} ===`);
  report.log(`catalog (${W2A_SUITE_CATALOG.length}): ${W2A_SUITE_CATALOG.join(", ")}`);

  if (typeof listWelvetSuiteCatalog === "function") {
    const fromWasm = JSON.parse(listWelvetSuiteCatalog()) as string[];
    const missing = W2A_SUITE_CATALOG.filter((s) => !fromWasm.includes(s));
    if (missing.length) {
      report.record("catalog", "wasm_parity", "FAIL", `missing in WASM: ${missing.join(",")}`);
    } else {
      report.record("catalog", "wasm_parity", "OK", `${fromWasm.length} suites`);
    }
  }

  runPortableHonesty(report, profile);
  runSystemsMatrix(report);
  runSevenMatrix(report);
  runWeightsMatrix(report);
  runStubSuitesMatrix(report);

  if (profile === "mega") {
    runCameralMatrix(report);
    runTrainModesMatrix(report, { dtypeSweep: true });
    runTrainModesLayersMatrix(report, {
      dtypeSweep: true,
      quantSweep: true,
      quietProgressEvery: 2000,
    });
    runMegaW2AMatrix({ report, progressEvery: 2000, splitOps: true });
  } else if (profile === "quick") {
    runCameralMatrix(report);
    runTrainModesMatrix(report);
    runTrainModesLayersMatrix(report, {
      dtypeSweep: false,
      quantSweep: false,
      kinds: ["dense", "swiglu", "mha"],
      modes: ["NormalBP", "Tween", "TweenSplit", "MeshBP"],
    });
    runLayersMatrix({
      report,
      layers: LAYER_DEFS,
      dtypes: [{ id: 1, name: "float32" }],
      formats: [{ id: 0, name: "none" }],
      forward: true,
      train: true,
      entity: true,
    });
  } else {
    runCameralMatrix(report);
    runTrainModesMatrix(report, { dtypeSweep: true });
    runTrainModesLayersMatrix(report, {
      dtypeSweep: true,
      quantSweep: false, // full profile: dtype×mode×kind; mega adds quants
    });
    runFullDtypeLayerMatrix(report);
    runDenseQuantMatrix(report);
    for (const backend of ["simd", "webgpu"]) {
      runLayersMatrix({
        report,
        layers: [LAYER_DEFS.find((l) => l.id === "dense")!],
        dtypes: [{ id: 1, name: "float32" }],
        formats: [{ id: 0, name: "none" }],
        backend,
        forward: true,
        train: true,
        entity: false,
      });
    }
  }

  const s = report.printSummary();
  if (opts.failOnFail !== false && s.fail > 0) {
    throw new Error(`W2A WASM matrix: ${s.fail} FAIL cells`);
  }
  return report;
}
