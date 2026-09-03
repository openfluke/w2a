#!/usr/bin/env node
/**
 * Headless Chrome smoke against Vite preview of the React app.
 * Expects #tdeploy-status[data-status=ok] after WASM boot.
 */
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const PORT = 4178;
const HOST = "127.0.0.1";
const URL = `http://${HOST}:${PORT}/`;

const chromePath =
  process.env.CHROME_PATH ||
  ["/usr/bin/google-chrome-stable", "/usr/bin/google-chrome", "/usr/bin/chromium"].find((p) =>
    fs.existsSync(p),
  );

if (!chromePath) {
  console.error("No Chrome/Chromium found. Set CHROME_PATH or install google-chrome.");
  process.exit(1);
}

function wait(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function waitForServer(timeoutMs = 90000) {
  const start = Date.now();
  let lastErr = "";
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(URL);
      if (res.ok || res.status === 404) return;
      lastErr = `HTTP ${res.status}`;
    } catch (e) {
      lastErr = String(e?.message || e);
    }
    await wait(250);
  }
  throw new Error(`preview not up at ${URL} (${lastErr})`);
}

function run(cmd, args, opts = {}) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { cwd: root, stdio: "inherit", ...opts });
    p.on("exit", (code) => (code === 0 ? resolve() : reject(new Error(`${cmd} exit ${code}`))));
  });
}

console.log("=== tdeploy/react browser smoke ===");
console.log("  building…");
await run("npm", ["run", "build"]);

const previewLog = [];
const preview = spawn(
  "npx",
  ["vite", "preview", "--host", HOST, "--port", String(PORT), "--strictPort"],
  { cwd: root, stdio: ["ignore", "pipe", "pipe"] },
);
preview.stdout.on("data", (d) => previewLog.push(String(d)));
preview.stderr.on("data", (d) => {
  previewLog.push(String(d));
  process.stderr.write(d);
});

try {
  await waitForServer();
  console.log(`  chrome: ${chromePath}`);
  const browser = await puppeteer.launch({
    executablePath: chromePath,
    headless: true,
    args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
  });
  const page = await browser.newPage();
  page.on("console", (msg) => console.log(`  [browser] ${msg.type()}: ${msg.text()}`));
  page.on("pageerror", (err) => console.error(`  [pageerror] ${err.message}`));

  await page.goto(URL, { waitUntil: "networkidle0", timeout: 120000 });
  await page.waitForFunction(
    () => {
      const el = document.querySelector("#tdeploy-status");
      const s = el?.getAttribute("data-status");
      return s === "ok" || s === "fail";
    },
    { timeout: 120000 },
  );

  const status = await page.$eval("#tdeploy-status", (el) => el.getAttribute("data-status"));
  const log = await page.$eval("#tdeploy-log", (el) => el.textContent || "");
  console.log(log);
  await browser.close();

  if (status !== "ok") {
    throw new Error(`status=${status}`);
  }
  console.log("=== tdeploy/react browser smoke OK ===");
} catch (e) {
  console.error("preview log:\n" + previewLog.join(""));
  throw e;
} finally {
  preview.kill("SIGTERM");
}
