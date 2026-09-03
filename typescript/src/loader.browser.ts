/**
 * Welvet WASM browser loader.
 */
export async function loadWelvetWASMBrowser(wasmUrl?: string): Promise<void> {
  if (typeof (globalThis as Record<string, unknown>)["Go"] === "undefined") {
    const script = document.createElement("script");
    script.src = "/dist/wasm_exec.js";
    await new Promise<void>((resolve, reject) => {
      script.onload = () => resolve();
      script.onerror = () => reject(new Error("Failed to load wasm_exec.js"));
      document.head.appendChild(script);
    });
  }

  const response = await fetch(wasmUrl ?? "/dist/main.wasm");
  const wasmBuffer = await response.arrayBuffer();
  // @ts-expect-error Go injected by wasm_exec.js
  const go = new Go();
  const { instance } = await WebAssembly.instantiate(wasmBuffer, go.importObject);
  go.run(instance);
  await new Promise<void>((r) => setTimeout(r, 50));
}

/** @deprecated Loom-compat */
export const loadLoomWASMBrowser = loadWelvetWASMBrowser;
