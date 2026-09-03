import { useEffect, useState } from "react";
import {
  initBrowser,
  createGrid,
  createBicameral,
  assertEngineVersion,
  engineVersion,
  listNamedTrainModes,
  DType,
  WELVET_ENGINE_VERSION,
} from "@openfluke/welvet";

type Status = "loading" | "ok" | "fail";

export function App() {
  const [status, setStatus] = useState<Status>("loading");
  const [lines, setLines] = useState<string[]>(["booting WASM…"]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const log: string[] = [];
      const push = (m: string) => {
        log.push(m);
        if (!cancelled) setLines([...log]);
      };
      try {
        if (typeof (globalThis as { Go?: unknown }).Go !== "function") {
          throw new Error("Go global missing — is /wasm_exec.js loaded?");
        }
        await initBrowser("/main.wasm");
        assertEngineVersion();
        push(`engine ${engineVersion()} (package ${WELVET_ENGINE_VERSION})`);

        const g = createGrid({ depth: 1, rows: 1, cols: 1, layers_per_cell: 1 });
        const placed = g.placeDense(
          JSON.stringify({ in: 8, out: 4, act: "relu", dtype: DType.FLOAT32 }),
        );
        if ((placed as { error?: string })?.error) {
          throw new Error(`placeDense: ${(placed as { error: string }).error}`);
        }
        const x = new Float32Array(8);
        for (let i = 0; i < 8; i++) x[i] = 0.1 * (i + 1);
        const fwd = g.forward(x);
        if (fwd.error) throw new Error(fwd.error);
        push(`dense forward len=${fwd.output.length}`);

        const tr = g.trainSGD(x, new Float32Array([1, 0, 0, 0]), 0.05);
        if (tr.error) throw new Error(tr.error);
        push(`trainSGD loss=${tr.loss}`);

        const modes = listNamedTrainModes();
        const stack = createBicameral({ in: 4, hidden: 4, out: 4 });
        const cam = stack.trainStackMSE(
          new Float32Array(4).fill(0.1),
          new Float32Array([1, 0, 0, 0]),
          modes[0],
          0.05,
        );
        if (cam.error) throw new Error(cam.error);
        push(`cameral ${modes[0]} loss=${cam.loss} modes=${modes.length}`);

        const ent = g.serializeEntity();
        if (!(ent instanceof Uint8Array) || ent.length < 8) {
          throw new Error("serializeEntity failed");
        }
        push(`entity ${ent.length}B`);

        stack.free?.();
        g.free();
        push("PASS");
        if (!cancelled) setStatus("ok");
      } catch (e) {
        push(`FAIL: ${e instanceof Error ? e.message : String(e)}`);
        if (!cancelled) setStatus("fail");
        console.error(e);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <main className="page">
      <h1>tdeploy / react</h1>
      <p className="meta">
        Vite + React loading <code>@openfluke/welvet@{WELVET_ENGINE_VERSION}</code> from npm
      </p>
      <div
        id="tdeploy-status"
        data-status={status}
        className={`badge ${status}`}
        role="status"
      >
        {status.toUpperCase()}
      </div>
      <pre id="tdeploy-log">{lines.join("\n")}</pre>
    </main>
  );
}
