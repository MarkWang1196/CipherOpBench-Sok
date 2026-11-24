// Educational packing planner for arbitrary M×K · K×N matmuls (spec §11).
//
// The planner is deterministic: identical inputs always produce the identical
// plan. It guarantees a legal, explainable padding/tiling/packing, NOT that it
// matches the compiler-search structure (spec §11.2.8, §11.7). Results remain
// explicitly educational rather than being presented as optimized layouts.
//
// Core algorithm — Outer-Product Planes (OPP), verified against the real
// `layout_engine.js` engine:
//   * pad M→M′, K→K′, N→N′ (next powers of two)
//   * tile the output grid M′×N′ into R_t×C_t blocks with R_t·C_t ≤ n
//   * for each contraction index k, build product plane P_k: slot(i,j) =
//     A′[i,k]·B′[k,j] as repCol(A′[:,k]) ⊙ repRow(B′[k,:])
//   * reduce over k by cross-ciphertext addition; crop back to M×N
// Both repCol and repRow share the canonical slot mapping
//   (i = s mod R_t, j = s div R_t)
// which makes slot-wise multiply exact and provably correct.

import { Layout } from "../../layout_engine.js";
import { chooseTiles } from "./tiling.js";
import { chooseDisplayMode } from "./display_policy.js";
import { PLANNER_SOURCES, PACKING_STRATEGIES, REDUCTION_STRATEGIES } from "./trace_schema.js";

/** Next power of two >= x, for positive integers. */
export function nextPow2(x) {
  if (!Number.isInteger(x) || x <= 0) {
    throw new Error(`nextPow2 requires a positive integer, got ${x}`);
  }
  let p = 1;
  while (p < x) p *= 2;
  return p;
}

export function isPowerOfTwo(x) {
  const n = Number(x);
  return Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0;
}

/** Zero-pad (or crop via padMatrixTo with explicit size) a matrix to rows×cols. */
export function padMatrixTo(matrix, rows, cols, value = 0) {
  if (!Array.isArray(matrix)) throw new Error("padMatrixTo requires an array");
  const out = [];
  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) {
      const inRow = matrix[i];
      row.push(inRow && j < inRow.length ? Number(inRow[j]) : value);
    }
    out.push(row);
  }
  return out;
}

/** Crop a matrix to rows×cols (discard padding/oob). */
export function cropMatrix(matrix, rows, cols) {
  return matrix.slice(0, rows).map((r) => r.slice(0, cols));
}

/**
 * Pick the pack layout for the left operand. Row-major when it fits in one
 * ciphertext; otherwise a row-tiled pack (multiple ciphertexts).
 */
export function chooseLeftPackLayout(m, k, n, slotsPerCiphertext) {
  const Mp = nextPow2(m);
  const Kp = nextPow2(k);
  if (Mp * Kp <= slotsPerCiphertext) {
    return `[0:${Mp}:1][1:${Kp}:1]`;
  }
  // Tile rows: each ciphertext holds `chunkRows` full rows.
  const chunkRows = slotsPerCiphertext / Kp;
  if (chunkRows < 1 || !Number.isInteger(chunkRows)) {
    throw new Error(`shared dim K=${k} does not fit into n=${slotsPerCiphertext} slots`);
  }
  const rowCts = Mp / chunkRows;
  return `[0:${rowCts}:1][0:${chunkRows}:1][1:${Kp}:1]`;
}

/**
 * Build the complete physical plan for a matmul.
 *
 * @param {object} p
 * @param {number} p.m  rows of A (logical)
 * @param {number} p.k  shared/contraction dim
 * @param {number} p.n  cols of B (logical)
 * @param {number} p.slotsPerCiphertext
 * @param {boolean} p.leftSecret
 * @param {boolean} p.rightSecret
 * @param {string} p.strategy auto|row|column|diagonal|custom
 * @param {{left?:string,right?:string,output?:string}|null} p.customLayouts
 * @returns {object} the plan (see spec §9.6)
 */
export function planMatmulPacking({
  m,
  k,
  n,
  slotsPerCiphertext,
  leftSecret,
  rightSecret,
  strategy = PACKING_STRATEGIES.AUTO,
  customLayouts = null,
}) {
  const warnings = [];
  if (!isPowerOfTwo(slotsPerCiphertext)) {
    throw new Error(`Slots per ciphertext must be a power of 2 (got ${slotsPerCiphertext})`);
  }
  if (!(m > 0) || !(k > 0) || !(n > 0)) {
    throw new Error("M, K, N must be positive");
  }

  const requested = strategy;
  let resolved = requested;
  let leftLayout = null;
  let rightLayout = null;
  let outputLayout = null;
  let plannerSource = PLANNER_SOURCES.EDUCATIONAL_AUTO;

  if (requested === PACKING_STRATEGIES.CUSTOM) {
    if (!customLayouts || !customLayouts.left || !customLayouts.right) {
      throw new Error("Custom strategy requires layout strings for A and B");
    }
    try {
      Layout.fromString(customLayouts.left, slotsPerCiphertext, !!leftSecret);
      Layout.fromString(customLayouts.right, slotsPerCiphertext, !!rightSecret);
      if (customLayouts.output) {
        Layout.fromString(customLayouts.output, slotsPerCiphertext, false);
      }
    } catch (err) {
      throw new Error(`Custom layout failed to parse: ${err.message}`);
    }
    leftLayout = customLayouts.left;
    rightLayout = customLayouts.right;
    outputLayout = customLayouts.output || null;
    plannerSource = PLANNER_SOURCES.USER_CUSTOM;
  } else {
    if (requested === PACKING_STRATEGIES.DIAGONAL) {
      // The OPP planner broadcasts per-k planes; it does not use the classic
      // diagonal roll alignment (see the note in README / spec §12.5).
      warnings.push(
        "Diagonal roll-based alignment is not used by the educational outer-product planner; " +
          "falling back to auto (broadcast-per-k). Results remain correct."
      );
      resolved = PACKING_STRATEGIES.AUTO;
    }
    leftLayout = chooseLeftPackLayout(m, k, n, slotsPerCiphertext);
  }

  const Mp = nextPow2(m);
  const Kp = nextPow2(k);
  const Np = nextPow2(n);
  const tiling = chooseTiles(Mp, Np, slotsPerCiphertext);
  const planeCount = Kp * tiling.nTileRows * tiling.nTileCols;
  const displayMode = chooseDisplayMode({
    m,
    k,
    n,
    planeCount,
    tileGrid: { m: tiling.nTileRows, n: tiling.nTileCols },
  });

  const leftIsCt = !!leftSecret;
  const rightIsCt = !!rightSecret;
  const matmulType =
    leftIsCt && rightIsCt ? "ct-ct" : !leftIsCt && rightIsCt ? "pt-ct" : "ct-pt";

  return {
    logicalShape: { m, k, n },
    paddedShape: { m: Mp, k: Kp, n: Np },
    tileShape: { m: tiling.R_t, k: Kp, n: tiling.C_t },
    tileGrid: { m: tiling.nTileRows, n: tiling.nTileCols },
    padding: { rowsA: Mp - m, shared: Kp - k, colsB: Np - n, value: 0 },
    leftLayout,
    rightLayout,
    outputLayout,
    reductionStrategy: REDUCTION_STRATEGIES.CROSS_CIPHERTEXT_ADD,
    displayMode,
    matmulType,
    packingStrategyRequested: requested,
    packingStrategyResolved: resolved,
    plannerSource,
    warnings,
    tiling,
    planeCount,
    operandSecret: [leftIsCt, rightIsCt],
  };
}
