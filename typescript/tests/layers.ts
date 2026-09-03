/**
 * Place smoke for every Welvet layer kind (defaults; place must not error).
 */
import { init, createGrid, assertEngineVersion } from "../src/index.js";

const PLACES: { method: string; spec: Record<string, unknown> }[] = [
  { method: "placeDense", spec: { in: 8, out: 8, act: "relu", dtype: 1 } },
  { method: "placeMHA", spec: { dim: 8, DModel: 8, NumHeads: 2, SeqLen: 4 } },
  { method: "placeSwiGLU", spec: { dim: 8, InputDim: 8, IntermediateDim: 16 } },
  { method: "placeRMSNorm", spec: { dim: 8 } },
  { method: "placeLayerNorm", spec: { dim: 8 } },
  { method: "placeEmbedding", spec: { dim: 8, VocabSize: 32, EmbeddingDim: 8, SeqLen: 4 } },
  { method: "placeSoftmax", spec: { dim: 8 } },
  { method: "placeSequential", spec: { dim: 8, Depth: 2 } },
  { method: "placeResidual", spec: { dim: 8, Depth: 1 } },
  { method: "placeCNN1", spec: { dim: 16, InChannels: 1, Filters: 4, SeqLen: 16, Kernel: 3 } },
  { method: "placeCNN2", spec: { InChannels: 1, Filters: 4, Height: 8, Width: 8, Kernel: 3 } },
  { method: "placeCNN3", spec: { InChannels: 1, Filters: 2, Depth: 4, Height: 4, Width: 4, Kernel: 3 } },
  { method: "placeRNN", spec: { dim: 8, InputSize: 8, HiddenSize: 8, SeqLen: 4 } },
  { method: "placeLSTM", spec: { dim: 8, InputSize: 8, HiddenSize: 8, SeqLen: 4 } },
  { method: "placeConvT1", spec: { InChannels: 4, Filters: 2, SeqLen: 8, Kernel: 3 } },
  { method: "placeConvT2", spec: { InChannels: 4, Filters: 2, Height: 4, Width: 4, Kernel: 3 } },
  { method: "placeConvT3", spec: { InChannels: 2, Filters: 2, Depth: 4, Height: 4, Width: 4, Kernel: 3 } },
  { method: "placeParallel", spec: { dim: 8, OutFeat: 8, Branches: 2 } },
  { method: "placeStack", spec: { dim: 8, act: "relu" } },
	{ method: "placeKMeans", spec: { FeatureDim: 8, NumClusters: 4 } },
  { method: "placeMamba", spec: { DModel: 8, DState: 8, SeqLen: 4 } },
  { method: "placeMetacognition", spec: { Dim: 8 } },
  { method: "placeGDN", spec: { HiddenSize: 8, NumKeyHeads: 2, NumValueHeads: 2, KeyHeadDim: 4, ValueHeadDim: 4, ConvKernel: 3 } },
];

async function main() {
  console.log("=== Welvet WASM layers place matrix ===");
  await init();
  assertEngineVersion();

  let pass = 0;
  let fail = 0;
  for (const { method, spec } of PLACES) {
    const g = createGrid({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 });
    const body = JSON.stringify({ z: 0, y: 0, x: 0, l: 0, dtype: 1, format: 0, ...spec });
    const fn = (g as unknown as Record<string, (s: string) => { error?: string }>)[method];
    if (typeof fn !== "function") {
      console.error("  FAIL", method, "(missing method)");
      fail++;
      continue;
    }
    const r = fn.call(g, body);
    if (r && r.error) {
      console.error("  FAIL", method, r.error);
      fail++;
    } else {
      console.log("  PASS", method);
      pass++;
    }
    g.free();
  }
  console.log(`=== layers pass=${pass} fail=${fail} ===`);
  if (fail > 0) process.exit(1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
