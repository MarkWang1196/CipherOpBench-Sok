// Tiling for the educational outer-product matmul planner.
//
// A tile is a rectangular block of the *output* grid that fits into a single
// ciphertext of `n` slots: the tile is `R_t × C_t` with `R_t·C_t ≤ n`, where
// `R_t = min(M′, n)` and `C_t = n / R_t` (both powers of two when the inputs
// are padded to powers of two).
//
// Because M′ and n are powers of two and R_t = min(M′, n), every row tile is
// full (M′ % R_t == 0). Column tiles are full whenever C_t ≤ N′; when
// C_t > N′ there is a single column tile whose trailing columns are
// *out-of-range* (oob), not padding.

import { nextPow2 } from "./packing_planner.js";

/**
 * Choose the tile grid for a padded output shape M′×N′ with `n` slots/CT.
 * @returns {{R_t, C_t, nTileRows, nTileCols, tiles: Array<{
 *   iTile, jTile, rowRange:[start,end), colRange:[start,end),
 *   rowsPerTile, colsPerTile, slotGrid:[R_t,C_t] }>}}
 */
export function chooseTiles(Mp, Np, n) {
  if (!(Number.isInteger(Mp) && Mp > 0) || !(Number.isInteger(Np) && Np > 0)) {
    throw new Error("chooseTiles requires positive integer padded dims");
  }
  if (!(Number.isInteger(n) && n > 0)) throw new Error("chooseTiles requires n > 0");

  const R_t = Math.min(Mp, n);
  if (n % R_t !== 0) {
    throw new Error(`tiling invariant broken: n=${n} not divisible by R_t=${R_t}`);
  }
  const C_t = n / R_t;
  const nTileRows = Math.ceil(Mp / R_t);
  const nTileCols = Math.ceil(Np / C_t);

  const tiles = [];
  for (let ti = 0; ti < nTileRows; ti++) {
    for (let tj = 0; tj < nTileCols; tj++) {
      const rowStart = ti * R_t;
      const colStart = tj * C_t;
      const rowsPerTile = Math.min(R_t, Mp - rowStart);
      const colsPerTile = Math.min(C_t, Np - colStart);
      tiles.push({
        iTile: ti,
        jTile: tj,
        rowRange: [rowStart, rowStart + R_t],
        colRange: [colStart, colStart + C_t],
        rowsPerTile,
        colsPerTile,
        slotGrid: [R_t, C_t],
      });
    }
  }
  return { R_t, C_t, nTileRows, nTileCols, tiles };
}
