export type CellStatus = "OK" | "GAP" | "FAIL" | "SKIP";

export interface CellResult {
  suite: string;
  case: string;
  status: CellStatus;
  note?: string;
  ms?: number;
}

export type SuiteReportOpts = {
  quiet?: boolean;
  progressEvery?: number;
  /** Keep every cell in memory (default: true unless quiet). */
  retainAll?: boolean;
};

export class SuiteReport {
  cells: CellResult[] = [];
  /** When quiet/mega, only FAIL cells are retained (counts still accurate). */
  retainAll: boolean;
  log: (line: string) => void;
  quiet: boolean;
  progressEvery: number;
  private counts = { ok: 0, gap: 0, fail: 0, skip: 0 };

  constructor(log: (line: string) => void = console.log, opts: SuiteReportOpts = {}) {
    this.log = log;
    this.quiet = !!opts.quiet;
    this.progressEvery = opts.progressEvery ?? 1000;
    this.retainAll = opts.retainAll ?? !this.quiet;
  }

  record(suite: string, caseName: string, status: CellStatus, note = "", ms?: number) {
    const cell: CellResult = { suite, case: caseName, status, note, ms };
    if (this.retainAll || status === "FAIL") {
      this.cells.push(cell);
    }
    if (status === "OK") this.counts.ok++;
    else if (status === "GAP") this.counts.gap++;
    else if (status === "FAIL") this.counts.fail++;
    else this.counts.skip++;

    const n = this.counts.ok + this.counts.gap + this.counts.fail + this.counts.skip;
    const shouldLog =
      !this.quiet ||
      status === "FAIL" ||
      (this.progressEvery > 0 && n % this.progressEvery === 0);

    if (shouldLog) {
      if (this.quiet && status !== "FAIL" && n % this.progressEvery === 0) {
        this.log(`  … progress ${n}  ok=${this.counts.ok} gap=${this.counts.gap} fail=${this.counts.fail} skip=${this.counts.skip}`);
      } else {
        const tag = status.padEnd(4);
        this.log(`  [${tag}] ${suite}/${caseName}${note ? " — " + note : ""}${ms != null ? ` (${ms.toFixed(1)}ms)` : ""}`);
      }
    }
  }

  summary() {
    const total = this.counts.ok + this.counts.gap + this.counts.fail + this.counts.skip;
    return { ...this.counts, total };
  }

  printSummary() {
    const s = this.summary();
    this.log(`\n=== SUMMARY ok=${s.ok} gap=${s.gap} fail=${s.fail} skip=${s.skip} total=${s.total} ===`);
    return s;
  }
}

export function isErr(r: unknown): r is { error: string } {
  return !!r && typeof r === "object" && "error" in r && typeof (r as { error: unknown }).error === "string";
}

/** Classify hard-errors that are expected gaps on WASM (SIMD/WebGPU unavailable). */
export function classifyOpError(err: string): CellStatus {
  const e = err.toLowerCase();
  if (
    e.includes("simd") ||
    e.includes("webgpu") ||
    e.includes("not available") ||
    e.includes("unimplemented") ||
    e.includes("stub") ||
    e.includes("unsupported op") ||
    e.includes("serialization:") ||
    e.includes("shape mismatch") ||
    e.includes("requires grid") ||
    e.includes("need [") ||
    e.includes("not multiple of") ||
    e.includes("multiple of group") ||
    e.includes("no recorded activation") ||
    e.includes("remote hop") ||
    e.includes("d_model") ||
    e.includes("width ") ||
    e.includes("mse shape") ||
    e.includes("training: mse") ||
    e.includes("pack only") ||
    e.includes("pack:") ||
    e.includes("setdtype")
  ) {
    return "GAP";
  }
  return "FAIL";
}
