// Tests for metrics helpers (spec §9.8).

import { test } from "node:test";
import assert from "node:assert/strict";
import { emptyMetrics, cloneMetrics, addMetricsDelta, isTotalGEQ } from "../transformer/core/metrics.js";

test("emptyMetrics starts at zero", () => {
  const m = emptyMetrics();
  assert.equal(m.rotations, 0);
  assert.equal(m.ciphertextCount, 0);
});

test("cloneMetrics copies without aliasing", () => {
  const m = emptyMetrics();
  const c = cloneMetrics(m);
  c.additions = 5;
  assert.equal(m.additions, 0);
});

test("addMetricsDelta accumulates and clamps at zero", () => {
  const total = addMetricsDelta(emptyMetrics(), {
    ptCtMultiplications: 4,
    additions: 3,
    multiplicativeDepth: 1,
  });
  assert.equal(total.ptCtMultiplications, 4);
  assert.equal(total.additions, 3);
  assert.equal(total.multiplicativeDepth, 1);
  // Summing again with a negative (not legal) delta clamps to 0.
  const bad = addMetricsDelta(total, { additions: -10 });
  assert.equal(bad.additions, 0);
});

test("isTotalGEQ rejects decreasing totals", () => {
  const a = emptyMetrics();
  const b = { ...emptyMetrics(), rotations: 3 };
  assert.equal(isTotalGEQ(a, b), true);
  assert.equal(isTotalGEQ(b, a), false);
  assert.equal(isTotalGEQ(a, a), true);
});
