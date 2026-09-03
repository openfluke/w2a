/** Browser stub — Node `loader.js` must not ship into the Vite bundle. */
export async function loadWelvetWASM(): Promise<void> {
  throw new Error("loadWelvetWASM is Node/Bun only — use initBrowser() in React");
}
export const loadLoomWASM = loadWelvetWASM;
