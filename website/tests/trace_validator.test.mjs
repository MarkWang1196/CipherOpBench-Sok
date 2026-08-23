// Trace validator tests (spec §19).

import { test } from "node:test";
import assert from "node:assert/strict";
import { validateTrace } from "../transformer/core/trace_validator.js";
import { buildMatmulScene } from "../transformer/core/scene_builder.js";
import { SCENES } from "../transformer/scenes/defaults.js";
import { matmul } from "../transformer/core/matrix.js";

function validTrace() {
  return buildMatmulScene({
    left: SCENES.xwq.left,
    right: SCENES.xwq.right,
    slotsPerCiphertext: 16,
    packingStrategy: "auto",
    options: {},
  });
}

test("accepts a valid generated trace", () => {
  const res = validateTrace(validTrace());
  assert.equal(res.ok, true, JSON.stringify(res.errors));
});

test("rejects a wrong schemaVersion", () => {
  const t = validTrace();
  t.schemaVersion = "9.9";
  assert.equal(validateTrace(t).ok, false);
});

test("rejects duplicate step ids and non-contiguous indices", () => {
  const t = validTrace();
  t.steps[1].id = t.steps[0].id;
  assert.ok(!validateTrace(t).ok);
  const t2 = validTrace();
  t2.steps[2].index = 99;
  assert.ok(!validateTrace(t2).ok);
});

test("rejects a ciphertext with the wrong slot count", () => {
  const t = validTrace();
  const step = t.steps.find((s) => s.ciphertexts.length);
  step.ciphertexts[0].slots = step.ciphertexts[0].slots.slice(0, 8);
  assert.ok(!validateTrace(t).ok);
});

test("rejects slot indices out of range", () => {
  const t = validTrace();
  const step = t.steps.find((s) => s.ciphertexts.length);
  step.ciphertexts[0].slots[0].slot = 999;
  assert.ok(!validateTrace(t).ok);
});

test("rejects a final tensor that does not match direct matmul", () => {
  const t = validTrace();
  t.finalTensor = matmul([[1]], [[1]]); // wrong shape + wrong values
  assert.ok(!validateTrace(t).ok);
});

test("rejects decreasing metrics totals", () => {
  const t = validTrace();
  const last = t.steps[t.steps.length - 1];
  last.metricsTotal = { ...last.metricsTotal, additions: -5 };
  assert.ok(!validateTrace(t).ok);
});

test("rejects missing matrixShape / slotsPerCiphertext", () => {
  const t = validTrace();
  t.parameters = { scheme: "CKKS" };
  assert.ok(!validateTrace(t).ok);
});
