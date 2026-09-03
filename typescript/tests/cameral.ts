/**
 * Cameral / TrainModes / CamSync / Tanhi smoke.
 */
import { init, createGrid, createBicameral, createHemispheres, listNamedTrainModes, assertEngineVersion } from "../src/index.js";

async function main() {
  console.log("=== Welvet WASM cameral ===");
  await init();
  assertEngineVersion();

  const modes = listNamedTrainModes();
  if (!Array.isArray(modes) || modes.length < 1) {
    console.error("FAIL no named train modes");
    process.exit(1);
  }
  console.log("  PASS named modes", modes.length, modes.slice(0, 5).join(","), "…");

  const st = createBicameral({ in: 4, hidden: 4, out: 4 });
  const input = new Float32Array([0.1, 0.2, 0.3, 0.4]);
  const target = new Float32Array([0.0, 1.0, 0.0, 0.0]);

  const mode = modes[0];
  st.setChildModes(JSON.stringify([mode, mode]));
  const r = st.trainStackMSE(input, target, mode, 0.05);
  if ((r as { error?: string }).error) {
    console.error("FAIL trainStackMSE", (r as { error: string }).error);
    process.exit(1);
  }
  console.log("  PASS trainStackMSE", mode, "loss=", r.loss);

	const hem = createHemispheres({ dim: 4, n: 2, combine: "add" });
  if ((hem as unknown as { error?: string }).error) {
    console.error("FAIL hemispheres", (hem as unknown as { error: string }).error);
    process.exit(1);
  }
  hem.setBranchModes(JSON.stringify([mode, mode]));
	hem.setCamSync(JSON.stringify({ Enabled: true, Alpha: 1.0 }));
  hem.setCamKit(JSON.stringify({ ShadowCoef: 1.0, DNAReg: 0, SurpriseThresh: 0 }));
  hem.setTanhi(JSON.stringify({ Enabled: false }));
  const pr = hem.trainMSE(input, target, mode, 0.05);
  if ((pr as { error?: string }).error) {
    console.error("FAIL trainMSE", (pr as { error: string }).error);
    process.exit(1);
  }
  console.log("  PASS Parallel.trainMSE + CamSync + Tanhi", "loss=", pr.loss);

  const g = createGrid();
  g.configureTanhi(JSON.stringify({ Enabled: false, Host: "127.0.0.1", Port: DefaultTanhiUDPPort() }));
  ConfigureTanhi(g._id, JSON.stringify({ Enabled: false }));
  EmitSweep("wasm-test");
  console.log("  PASS ConfigureTanhi / EmitSweep");

  // Spot-check a few more named modes on bicameral
  let checked = 0;
  for (const m of modes.slice(0, Math.min(8, modes.length))) {
    const s2 = createBicameral({ in: 4, hidden: 4, out: 4 });
    const tr = s2.trainStackMSE(input, target, m, 0.02);
    if ((tr as { error?: string }).error) {
      console.error("FAIL mode", m, (tr as { error: string }).error);
      process.exit(1);
    }
    checked++;
    s2.free();
  }
  console.log("  PASS train modes spot-check", checked);
  console.log("=== CAMERAL OK ===");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
