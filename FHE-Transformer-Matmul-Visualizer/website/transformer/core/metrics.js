// HE operation statistics for the transformer visualizer (spec §9.8).
//
// Counts are illustrative/educational unless a trace is produced by the Rotom
// compiler exporter (plannerSource === "rotom-compiler"). The page renders a
// "Illustrative counts" badge for educational traces.

import { emptyMetrics } from "./trace_schema.js";

export { emptyMetrics } from "./trace_schema.js";

const METRIC_KEYS = [
  "rotations",
  "additions",
  "ptCtMultiplications",
  "ctCtMultiplications",
  "masks",
  "relinearizations",
  "rescales",
  "multiplicativeDepth",
  "ciphertextCount",
];

/** Return a fresh zeroed metric object. */
export function newMetrics() {
  return emptyMetrics();
}

/** Copy a metrics object (so totals/deltas never alias). */
export function cloneMetrics(m) {
  const out = {};
  for (const key of METRIC_KEYS) out[key] = Number(m[key]) || 0;
  return out;
}

/** total = total + delta (element-wise, clamped at 0). */
export function addMetricsDelta(total, delta) {
  const out = {};
  for (const key of METRIC_KEYS) {
    const a = Number(total[key]) || 0;
    const b = Number(delta[key]) || 0;
    out[key] = Math.max(0, a + b);
  }
  return out;
}

/** True when every field of next is >= the same field of prev (monotone totals). */
export function isTotalGEQ(prev, next) {
  for (const key of METRIC_KEYS) {
    if (Number(next[key]) < Number(prev[key])) return false;
  }
  return true;
}

export function metricsKeys() {
  return METRIC_KEYS.slice();
}
