// Input parser tests (spec §8.2, §20.3): validation, formats, determinism.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  parseAndValidateInputs,
  parseMatrixText,
  generateValues,
  mulberry32,
} from "../transformer/core/input_parser.js";

function baseState(overrides = {}) {
  return {
    m: "4",
    k: "4",
    n: "4",
    slots: "16",
    leftType: "ct",
    rightType: "pt",
    strategy: "auto",
    valueMode: "manual",
    seed: "",
    valuesTextA: "1 2 3 4\n5 6 7 8\n9 10 11 12\n13 14 15 16",
    valuesTextB: "1 0 2 1\n0 1 1 2\n2 1 0 1\n1 2 1 0",
    customLayouts: { left: "", right: "", output: "" },
    ...overrides,
  };
}

test("accepts valid inputs and produces clean inputs", () => {
  const r = parseAndValidateInputs(baseState());
  assert.equal(r.ok, true);
  assert.equal(r.inputs.m, 4);
  assert.equal(r.inputs.A.length, 4);
  assert.equal(r.inputs.B.length, 4);
});

test("rejects mismatched shapes with field-level errors", () => {
  const r = parseAndValidateInputs(baseState({ valuesTextB: "1 0\n0 1" })); // B is 2×2, expected 4×4
  assert.equal(r.ok, false);
  assert.ok(r.errors.values, "expected values error for wrong B shape");
});

test("rejects non-power-of-two slots", () => {
  const r = parseAndValidateInputs(baseState({ slots: "12" }));
  assert.equal(r.ok, false);
  assert.ok(r.errors.slots);
});

test("rejects non-finite values", () => {
  const r = parseAndValidateInputs(baseState({ valuesTextA: "1 2\nInfinity 4\n1 2\n3 4" }));
  assert.equal(r.ok, false);
  assert.ok(r.errors.values);
});

test("rejects dimensions over the product limit", () => {
  const r = parseAndValidateInputs(baseState({ m: "40", k: "40", n: "40" }));
  assert.equal(r.ok, false);
});

test("slots must fit the shared dim", () => {
  const r = parseAndValidateInputs(baseState({ slots: "8", k: "16" }));
  assert.equal(r.ok, false);
  assert.ok(r.errors.slots);
});

test("custom strategy requires parseable layouts", () => {
  const bad = parseAndValidateInputs(
    baseState({ strategy: "custom", customLayouts: { left: "roll(9,0) [0:4:1]", right: "[0:2:1]", output: "" } })
  );
  assert.equal(bad.ok, false);
  assert.ok(bad.errors.strategy);

  const missing = parseAndValidateInputs(
    baseState({ strategy: "custom", customLayouts: { left: "", right: "", output: "" } })
  );
  assert.equal(missing.ok, false);
});

test("sequential / random / zero-one generation is seeded and deterministic", () => {
  const a1 = generateValues(3, 3, "sequential", 1);
  const a2 = generateValues(3, 3, "sequential", 1);
  assert.deepEqual(a1, a2);
  assert.deepEqual(a1[0], [1, 2, 3]);

  const r1 = generateValues(3, 3, "random", 42);
  const r2 = generateValues(3, 3, "random", 42);
  assert.deepEqual(r1, r2);
  const r3 = generateValues(3, 3, "random", 43);
  assert.notDeepEqual(r1, r3);

  const z = generateValues(4, 4, "zero-one", 7);
  for (const row of z) for (const v of row) assert.ok(v === 0 || v === 1);
});

test("mulberry32 produces deterministic sequences", () => {
  const r = mulberry32(1234);
  const seq = [r(), r(), r()];
  const r2 = mulberry32(1234);
  assert.deepEqual(seq, [r2(), r2(), r2()]);
});

test("parseMatrixText accepts JSON, spaces, commas and semicolons", () => {
  assert.deepEqual(parseMatrixText("[[1,2],[3,4]]"), [[1, 2], [3, 4]]);
  assert.deepEqual(parseMatrixText("1 2\n3 4"), [[1, 2], [3, 4]]);
  assert.deepEqual(parseMatrixText("1,2;3,4"), [[1, 2], [3, 4]]);
  assert.throws(() => parseMatrixText("abc"), /not a finite number/);
  assert.throws(() => parseMatrixText(""), /No matrix values/);
});

test("auto-generated values flow into inputs", () => {
  const r = parseAndValidateInputs(baseState({ valueMode: "sequential", seed: "5" }));
  assert.equal(r.ok, true);
  assert.deepEqual(r.inputs.A[0], [1, 2, 3, 4]);
});
