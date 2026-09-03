#!/usr/bin/env node
/** Copy main.wasm + wasm_exec.js from the published package into public/ */
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const reactRoot = path.resolve(here, "..");
const pkgDir = path.dirname(require.resolve("@openfluke/welvet/package.json"));
const dist = path.join(pkgDir, "dist");
const pub = path.join(reactRoot, "public");

fs.mkdirSync(pub, { recursive: true });
for (const f of ["main.wasm", "wasm_exec.js"]) {
  const src = path.join(dist, f);
  if (!fs.existsSync(src)) throw new Error(`missing ${src}`);
  fs.copyFileSync(src, path.join(pub, f));
  console.log(`synced ${f} → public/`);
}
