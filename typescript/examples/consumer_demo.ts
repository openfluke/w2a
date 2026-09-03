/**
 * README / npm consumer smoke for @openfluke/welvet@1.1.1
 * Run: npx tsx examples/consumer_demo.ts
 */
import {
  init,
  createGrid,
  assertEngineVersion,
  WELVET_ENGINE_VERSION,
  DType,
  listNamedTrainModes,
  createBicameral,
  seedFrom,
} from "../src/index.js";

function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error(msg);
}

const HINT =
  "Rebuild: cd apps/w2a/typescript && npm run build:all";

export async function runConsumerDemo(): Promise<void> {
  assert(WELVET_ENGINE_VERSION === "1.1.1", `package version ${WELVET_ENGINE_VERSION}`);

  await init();
  assertEngineVersion();

  const wasmVer =
    typeof (globalThis as { welvetEngineVersion?: () => string }).welvetEngineVersion ===
    "function"
      ? (globalThis as { welvetEngineVersion: () => string }).welvetEngineVersion()
      : undefined;
  assert(
    wasmVer === WELVET_ENGINE_VERSION,
    wasmVer
      ? `stale main.wasm: WASM=${wasmVer} package=${WELVET_ENGINE_VERSION}. ${HINT}`
      : `missing welvetEngineVersion. ${HINT}`,
  );

  const g = createGrid({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 });
  const placed = g.placeDense(
    JSON.stringify({ in: 16, out: 8, act: "relu", dtype: DType.FLOAT32 }),
  );
  assert(!(placed as { error?: string }).error, `placeDense: ${(placed as { error?: string }).error}`);

  const input = new Float32Array(16);
  for (let i = 0; i < 16; i++) input[i] = 0.2 * Math.sin(i * 0.3);
  const fwd = g.forward(input);
  assert(!fwd.error, `forward: ${fwd.error}`);
  assert(fwd.output?.length === 8, `forward len=${fwd.output?.length}`);

  const target = new Float32Array(8);
  target[0] = 1;
  const tr = g.trainSGD(input, target, 0.05);
  assert(!tr.error, `trainSGD: ${tr.error}`);
  assert(typeof tr.loss === "number" && !Number.isNaN(tr.loss), `loss=${tr.loss}`);

  const entity = g.serializeEntity();
  assert(entity instanceof Uint8Array && entity.length > 8, "serializeEntity");

  const back = (
    globalThis as { DeserializeGrid?: (b: Uint8Array) => typeof g }
  ).DeserializeGrid?.(entity);
  assert(!!back && typeof back.forward === "function", "DeserializeGrid");
  const fwd2 = back!.forward(input);
  assert(!fwd2.error && fwd2.output?.length === 8, "roundtrip forward");

  const modes = listNamedTrainModes();
  assert(modes.length >= 10, `train modes ${modes.length}`);
  const stack = createBicameral({ in: 4, hidden: 4, out: 4 });
  const cam = stack.trainStackMSE(
    new Float32Array(4).fill(0.1),
    new Float32Array([1, 0, 0, 0]),
    modes[0],
    0.05,
  );
  assert(!cam.error, `cameral: ${cam.error}`);

  const seed = seedFrom("welvet", 42, true);
  assert(typeof seed === "string" && seed.length > 0, "seedFrom");

  back!.free?.();
  stack.free?.();
  g.free();

  console.log("=== consumer_demo OK ===");
  console.log(`  engine ${wasmVer}  forward=${fwd.output.length}  loss=${tr.loss}  entity=${entity.length}B  modes=${modes.length}`);
}

runConsumerDemo().catch((e) => {
  console.error(e);
  process.exit(1);
});
