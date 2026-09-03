/**
 * Welvet WASM loader — Node / Bun.
 * Loads assets from dist/ (postbuild) or assets/ (dev).
 */
export async function loadWelvetWASM(): Promise<void> {
  const fs = await import("fs");
  const url = await import("url");
  const path = await import("path");

  const __filename = url.fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);

  const candidates = [
    __dirname, // dist/
    path.join(__dirname, "..", "dist"),
    path.join(__dirname, "..", "assets"),
  ];

  let root = "";
  for (const c of candidates) {
    if (fs.existsSync(path.join(c, "main.wasm")) && fs.existsSync(path.join(c, "wasm_exec.js"))) {
      root = c;
      break;
    }
  }
  if (!root) {
    throw new Error("main.wasm / wasm_exec.js not found — run npm run build:wasm");
  }

  const wasmExecCode = fs.readFileSync(path.join(root, "wasm_exec.js"), "utf-8");
  // eslint-disable-next-line no-eval
  eval(wasmExecCode);

  const wasmBuffer = fs.readFileSync(path.join(root, "main.wasm"));
  // @ts-expect-error Go injected by wasm_exec.js
  const go = new Go();
  const { instance } = await WebAssembly.instantiate(wasmBuffer, go.importObject);
  go.run(instance);
  await new Promise<void>((r) => setTimeout(r, 50));
}

/** @deprecated Loom-compat */
export const loadLoomWASM = loadWelvetWASM;
