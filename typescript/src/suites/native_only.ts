import { SuiteReport } from "./report.js";

type NativeCase = { suite: string; case: string; reason: string };

/** Explicit SKIP catalog for Go w2a cases that are host/SIMD/WebGPU-only. */
export function runNativeOnlySkipCatalog(report?: SuiteReport): SuiteReport {
  const r = report ?? new SuiteReport();
  r.log("\n## native-only (SKIP catalog)");

  let cases: NativeCase[] = [];
  if (typeof listWelvetNativeOnlyCases === "function") {
    try {
      cases = JSON.parse(listWelvetNativeOnlyCases()) as NativeCase[];
    } catch {
      cases = [];
    }
  }
  if (!cases.length) {
    r.record("native_only", "catalog", "FAIL", "listWelvetNativeOnlyCases empty/missing");
    return r;
  }

  for (const c of cases) {
    r.record("native_only", `${c.suite}/${c.case}`, "SKIP", c.reason);
  }
  return r;
}
