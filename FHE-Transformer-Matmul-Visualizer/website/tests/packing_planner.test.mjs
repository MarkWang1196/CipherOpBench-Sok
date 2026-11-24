// Tests for the educational packing planner (spec §11, §20.3).

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  planMatmulPacking,
  nextPow2,
  padMatrixTo,
  cropMatrix,
} from "../transformer/core/packing_planner.js";

test("nextPow2 and padding helpers", () => {
  assert.equal(nextPow2(1), 1);
  assert.equal(nextPow2(3), 4);
  assert.equal(nextPow2(5), 8);
  assert.equal(nextPow2(16), 16);
  assert.throws(() => nextPow2(0), /positive/);
  const padded = padMatrixTo([[1, 2], [3, 4]], 4, 4);
  assert.deepEqual(padded[0], [1, 2, 0, 0]);
  assert.deepEqual(padded[3], [0, 0, 0, 0]);
  assert.deepEqual(cropMatrix(padded, 2, 2), [[1, 2], [3, 4]]);
});

test("2×3 · 3×4 has no fixed 4×4 / 16-slot assumptions", () => {
  const plan = planMatmulPacking({
    m: 2,
    k: 3,
    n: 4,
    slotsPerCiphertext: 16,
    leftSecret: true,
    rightSecret: false,
  });
  assert.deepEqual(plan.logicalShape, { m: 2, k: 3, n: 4 });
  assert.deepEqual(plan.paddedShape, { m: 2, k: 4, n: 4 });
  assert.equal(plan.displayMode, "full");
});

test("3×5 · 5×2 pads to 4×8·8×2 and never puts padding in the logical output", () => {
  const plan = planMatmulPacking({
    m: 3,
    k: 5,
    n: 2,
    slotsPerCiphertext: 16,
    leftSecret: true,
    rightSecret: false,
  });
  assert.deepEqual(plan.paddedShape, { m: 4, k: 8, n: 2 });
  assert.deepEqual(plan.padding, { rowsA: 1, shared: 3, colsB: 0, value: 0 });
  assert.deepEqual(plan.logicalShape, { m: 3, k: 5, n: 2 });
});

test("exceeds single ciphertext → multiple tiles", () => {
  const plan = planMatmulPacking({
    m: 5,
    k: 5,
    n: 6,
    slotsPerCiphertext: 16,
    leftSecret: true,
    rightSecret: false,
  });
  assert.ok(plan.tileGrid.n > 1, "expected column tiling for a 6-wide output");
  assert.ok(plan.tiling.tiles.length > 1);
});

test("Auto is deterministic for identical inputs", () => {
  const cfg = { m: 3, k: 5, n: 2, slotsPerCiphertext: 16, leftSecret: true, rightSecret: false };
  const a = planMatmulPacking(cfg);
  const b = planMatmulPacking(cfg);
  assert.deepEqual(a, b);
});

test("Row/Column strategies are recorded; Diagonal falls back with a warning", () => {
  const base = { m: 3, k: 5, n: 2, slotsPerCiphertext: 16, leftSecret: true, rightSecret: false };
  const row = planMatmulPacking({ ...base, strategy: "row" });
  assert.equal(row.packingStrategyRequested, "row");
  assert.equal(row.packingStrategyResolved, "row");

  const col = planMatmulPacking({ ...base, strategy: "column" });
  assert.equal(col.packingStrategyResolved, "column");

  const diag = planMatmulPacking({ ...base, strategy: "diagonal" });
  assert.equal(diag.packingStrategyRequested, "diagonal");
  assert.equal(diag.packingStrategyResolved, "auto");
  assert.ok(diag.warnings.length >= 1);
});

test("Custom layout parse failure blocks generation", () => {
  assert.throws(
    () =>
      planMatmulPacking({
        m: 3,
        k: 5,
        n: 2,
        slotsPerCiphertext: 16,
        leftSecret: true,
        rightSecret: false,
        strategy: "custom",
        customLayouts: { left: "roll(9,0) [0:4:1]", right: "[0:2:1]" },
      }),
    /Custom layout failed to parse/
  );
  assert.throws(
    () =>
      planMatmulPacking({
        m: 3,
        k: 5,
        n: 2,
        slotsPerCiphertext: 16,
        leftSecret: true,
        rightSecret: false,
        strategy: "custom",
        customLayouts: { left: "[0:4:1][1:4:1]" }, // missing right
      }),
    /requires layout strings/
  );
});

test("Custom strategy accepts parseable layouts and marks planner source user-custom", () => {
  const plan = planMatmulPacking({
    m: 3,
    k: 5,
    n: 2,
    slotsPerCiphertext: 16,
    leftSecret: true,
    rightSecret: false,
    strategy: "custom",
    customLayouts: { left: "[0:4:1][1:4:1]", right: "[1:2:1][0:4:1]" },
  });
  assert.equal(plan.plannerSource, "user-custom");
  assert.equal(plan.packingStrategyResolved, "custom");
});

test("different slot counts produce different plans", () => {
  const base = { m: 5, k: 5, n: 6, leftSecret: true, rightSecret: false };
  const p16 = planMatmulPacking({ ...base, slotsPerCiphertext: 16 });
  const p32 = planMatmulPacking({ ...base, slotsPerCiphertext: 32 });
  assert.notDeepEqual(p16.tileGrid, p32.tileGrid);
  assert.ok(p16.tileGrid.n > p32.tileGrid.n);
});

test("rejects non-power-of-two slots", () => {
  assert.throws(
    () =>
      planMatmulPacking({
        m: 2,
        k: 2,
        n: 2,
        slotsPerCiphertext: 12,
        leftSecret: true,
        rightSecret: false,
      }),
    /power of 2/
  );
});
