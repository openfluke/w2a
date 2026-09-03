/**
 * @openfluke/welvet — Welvet 1.1.1 WASM/TypeScript bindings.
 *
 * Breaking rename from Loom 0.80: prefer createWelvetGrid / WELVET_ENGINE_VERSION.
 * Loom aliases (createLoomNetwork, LOOM_ENGINE_VERSION) remain for migration.
 */

export const WELVET_ENGINE_VERSION = "1.1.1";
/** @deprecated use WELVET_ENGINE_VERSION */
export const LOOM_ENGINE_VERSION = WELVET_ENGINE_VERSION;

import type { Grid, GridConfig, PlaceSpec, Stack, Parallel } from "./types.js";
import { DType, LayerType, Activation, Format, Backend, TrainMode, CombineMode } from "./types.js";
import { loadWelvetWASMBrowser } from "./loader.browser.js";

export * from "./types.js";
export { loadWelvetWASMBrowser, loadLoomWASMBrowser } from "./loader.browser.js";
export { loadWelvetWASM, loadLoomWASM } from "./loader.js";

function asJSON(v: object | string): string {
  return typeof v === "string" ? v : JSON.stringify(v);
}

function place(fn: (s: string) => unknown, spec: string | PlaceSpec): unknown {
  return fn(asJSON(spec as object));
}

export async function init(wasmUrl?: string): Promise<void> {
  if (typeof window !== "undefined" && typeof document !== "undefined") {
    return loadWelvetWASMBrowser(wasmUrl);
  }
  const mod = await import("./loader.js");
  return mod.loadWelvetWASM();
}

export async function initBrowser(wasmUrl?: string): Promise<void> {
  return loadWelvetWASMBrowser(wasmUrl);
}

export function createGrid(config: GridConfig | string = {}): Grid {
  const json = asJSON(typeof config === "string" ? config : {
    depth: 1, rows: 1, cols: 1, layers_per_cell: 1, ...config,
  });
  return createWelvetGrid(json);
}

/** Loom-compat alias */
export function createNetwork(config: GridConfig | string = {}): Grid {
  return createGrid(config);
}

export function engineVersion(): string {
  return welvetEngineVersion();
}

export function assertEngineVersion(): void {
  const v = welvetEngineVersion();
  if (v !== WELVET_ENGINE_VERSION) {
    throw new Error(`WASM version ${v} != package ${WELVET_ENGINE_VERSION}`);
  }
}

export function getInternalParity(): string[] {
  try {
    return JSON.parse(getWelvetInternalParity()) as string[];
  } catch {
    return [];
  }
}

export function createBicameral(cfg: object | string | number = {}): Stack {
  if (typeof cfg === "number") return createWelvetBicameral(cfg);
  return createWelvetBicameral(asJSON(cfg as object));
}

export function createHemispheres(cfg: object | string = {}): Parallel {
  return createWelvetHemispheres(asJSON(cfg as object));
}

export function createParallelLayer(cfg: object | string = {}): Parallel {
  return createWelvetParallel(asJSON(cfg as object));
}

export function listTrainModes(): string[] {
  return JSON.parse(listWelvetTrainModes()) as string[];
}

export function listNamedTrainModes(): string[] {
  return JSON.parse(listWelvetNamedTrainModes()) as string[];
}

export function listConcreteTrainModes(): string[] {
  return JSON.parse(listWelvetConcreteTrainModes()) as string[];
}

export function listLayerTypes(): string[] {
  return JSON.parse(listWelvetLayerTypes()) as string[];
}

export function listDTypes(): { id: number; name: string }[] {
  return JSON.parse(listWelvetDTypes()) as { id: number; name: string }[];
}

export function listFormats(): { id: number; name: string }[] {
  return JSON.parse(listWelvetFormats()) as { id: number; name: string }[];
}

export function seedFrom(...parts: unknown[]): string {
  return SeedFrom(JSON.stringify(parts)) as string;
}

export function createStore(
  rows: number,
  cols: number,
  dtype = 1,
  format = 0,
  data?: Float32Array,
): ReturnType<typeof createWelvetStore> {
  return createWelvetStore(rows, cols, dtype, format, data);
}

export function placeOn(grid: Grid, method: keyof Grid, spec: PlaceSpec): unknown {
  const fn = grid[method];
  if (typeof fn !== "function") throw new Error(`missing ${String(method)}`);
  return place(fn.bind(grid) as (s: string) => unknown, spec);
}

export { runAllW2ASuites, W2A_SUITE_CATALOG } from "./suites/index.js";
export type { RunAllOptions } from "./suites/index.js";

