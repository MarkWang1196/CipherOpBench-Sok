// Tests for tiling (spec §11.4, §20.3).

import { test } from "node:test";
import assert from "node:assert/strict";
import { chooseTiles } from "../transformer/core/tiling.js";
import { nextPow2 } from "../transformer/core/packing_planner.js";

function pow2(x) {
  return nextPow2(x);
}

test("single tile when the padded output fits", () => {
  const t = chooseTiles(4, 4, 16);
  assert.deepEqual([t.R_t, t.C_t], [4, 4]);
  assert.deepEqual([t.nTileRows, t.nTileCols], [1, 1]);
  assert.equal(t.tiles.length, 1);
  assert.deepEqual(t.tiles[0].rowRange, [0, 4]);
});

test("tile dims are powers of two and R_t·C_t = n (or ≤ n)", () => {
  for (const [Mp, Np, n] of [[4, 4, 16], [16, 16, 32], [8, 8, 16], [16, 4, 32], [32, 32, 64]]) {
    const t = chooseTiles(Mp, Np, n);
    assert.ok(pow2(t.R_t) === t.R_t, "R_t power of two");
    assert.ok(pow2(t.C_t) === t.C_t, "C_t power of two");
    assert.ok(t.R_t * t.C_t <= n, `R_t*C_t (${t.R_t * t.C_t}) <= n (${n})`);
    for (const tile of t.tiles) {
      assert.equal(tile.slotGrid[0], t.R_t);
      assert.equal(tile.slotGrid[1], t.C_t);
      assert.equal(tile.rowRange[1] - tile.rowRange[0], t.R_t);
      assert.equal(tile.colRange[1] - tile.colRange[0], t.C_t);
    }
  }
});

test("oob tail columns when C_t > N′", () => {
  // Np=2, R_t=4, C_t=4 -> one column tile with 2 oob columns.
  const t = chooseTiles(4, 2, 16);
  assert.deepEqual([t.R_t, t.C_t], [4, 4]);
  assert.equal(t.tiles.length, 1);
  assert.equal(t.tiles[0].colsPerTile, 2);
  assert.deepEqual(t.tiles[0].colRange, [0, 4]);
});

test("rows tile when M′ > n", () => {
  const t = chooseTiles(32, 8, 16);
  assert.deepEqual([t.R_t, t.C_t], [16, 1]);
  assert.equal(t.nTileRows, 2);
  assert.equal(t.nTileCols, 8);
  assert.equal(t.tiles.length, 16);
  assert.deepEqual(t.tiles[0].rowRange, [0, 16]);
  assert.deepEqual(t.tiles[8].rowRange, [16, 32]);
});

test("boundary tile ranges are correct", () => {
  const t = chooseTiles(8, 16, 32);
  // R_t = 8, C_t = 4 -> 1 row tile, 4 col tiles.
  assert.deepEqual([t.nTileRows, t.nTileCols], [1, 4]);
  assert.deepEqual(t.tiles[3].colRange, [12, 16]);
});

test("rejects invalid inputs", () => {
  assert.throws(() => chooseTiles(0, 4, 16), /positive/);
  assert.throws(() => chooseTiles(4, 4, 0), /n > 0/);
});
