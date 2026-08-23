// Progressive-display policy (spec §11.6): Full / Tile / Summary.
//
// The *trace* always contains the full per-tile slot data and the verified
// logical result; display mode only controls how the page renders it. The
// scene builder computes the mode from the input shape and tile/plane counts.

import { DISPLAY_MODES } from "./trace_schema.js";

/**
 * Choose a display mode for a matmul scene.
 *
 * @param {object} ctx
 * @param {number} ctx.m        logical rows of A
 * @param {number} ctx.k        shared dim
 * @param {number} ctx.n        logical cols of B
 * @param {number} ctx.planeCount total number of product planes (K′ × tiles)
 * @param {{m:number,n:number}} ctx.tileGrid number of tiles per axis
 */
export function chooseDisplayMode({ m, k, n, planeCount, tileGrid }) {
  const totalCells = m * k + k * n;
  const nTiles = tileGrid.m * tileGrid.n;

  if (totalCells <= 128 && planeCount <= 24 && nTiles <= 1) {
    return DISPLAY_MODES.FULL;
  }
  if (nTiles <= 9 && planeCount <= 64) {
    return DISPLAY_MODES.TILE;
  }
  return DISPLAY_MODES.SUMMARY;
}

/** Human label for a display mode. */
export function displayModeLabel(mode) {
  switch (mode) {
    case DISPLAY_MODES.FULL:
      return "Full";
    case DISPLAY_MODES.TILE:
      return "Tile";
    case DISPLAY_MODES.SUMMARY:
      return "Summary";
    default:
      return String(mode);
  }
}
