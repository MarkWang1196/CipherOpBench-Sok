// Pure matrix semantics for the transformer matmul visualizer.
//
// These functions are intentionally DOM-free and environment-free so they can
// be unit tested with `node --test` and reused by the scene builder, the
// trace validator and the formula renderer.
//
// IMPORTANT (spec §25.2): element identity is by tensor id + coordinates, never
// by numeric value. Two equal values from different coordinates are distinct.

function assertRectangular(matrix, name) {
  if (!Array.isArray(matrix) || matrix.length === 0) {
    throw new Error(`${name} must be a non-empty array`);
  }
  const w = matrix[0].length;
  if (!(w > 0)) throw new Error(`${name} must have at least one column`);
  for (const row of matrix) {
    if (!Array.isArray(row) || row.length !== w) {
      throw new Error(`${name} must be rectangular (every row the same length)`);
    }
  }
}

/**
 * C = A · B. A is m×k, B is k×n.
 * @throws on shape mismatch or non-finite entries.
 */
export function matmul(A, B) {
  assertRectangular(A, "A");
  assertRectangular(B, "B");
  const m = A.length;
  const k = A[0].length;
  const n = B[0].length;
  if (B.length !== k) {
    throw new Error(`matmul shape mismatch: A is ${m}×${k}, B is ${B.length}×${n}`);
  }
  const C = [];
  for (let i = 0; i < m; i++) {
    const row = [];
    for (let j = 0; j < n; j++) {
      let acc = 0;
      for (let p = 0; p < k; p++) {
        const a = Number(A[i][p]);
        const b = Number(B[p][j]);
        if (!Number.isFinite(a) || !Number.isFinite(b)) {
          throw new Error(`non-finite value in matmul at A[${i},${p}] / B[${p},${j}]`);
        }
        acc += a * b;
      }
      row.push(acc);
    }
    C.push(row);
  }
  return C;
}

/** Bᵀ[i][j] = B[j][i]. */
export function transpose(A) {
  assertRectangular(A, "A");
  const m = A.length;
  const n = A[0].length;
  const out = [];
  for (let j = 0; j < n; j++) {
    const col = [];
    for (let i = 0; i < m; i++) col.push(A[i][j]);
    out.push(col);
  }
  return out;
}

/**
 * All m·n·k scalar contributions of C = A·B.
 * Each contribution is the product of a single (left,right) operand pair.
 */
export function buildContributions(A, B) {
  assertRectangular(A, "A");
  assertRectangular(B, "B");
  const m = A.length;
  const k = A[0].length;
  const n = B[0].length;
  if (B.length !== k) {
    throw new Error(`matmul shape mismatch: A is ${m}×${k}, B is ${B.length}×${n}`);
  }
  const contributions = [];
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      for (let p = 0; p < k; p++) {
        const leftValue = Number(A[i][p]);
        const rightValue = Number(B[p][j]);
        contributions.push({
          id: `C:${i}:${j}:k:${p}`,
          left: { tensorId: "A", coords: [i, p], value: leftValue },
          right: { tensorId: "B", coords: [p, j], value: rightValue },
          outputCoords: [i, j],
          contractionIndex: p,
          value: leftValue * rightValue,
          expression: `A[${i},${p}] × B[${p},${j}]`,
        });
      }
    }
  }
  return contributions;
}

/**
 * Group contributions by output cell. Returns a Map keyed by `i:j`.
 * Each group holds exactly `k` contributions (one per contraction index).
 */
export function groupContributionsByOutput(contributions) {
  const map = new Map();
  for (const c of contributions) {
    const key = `${c.outputCoords[0]}:${c.outputCoords[1]}`;
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(c);
  }
  return map;
}

/**
 * Deep equality helper for the final-matrix verification (and tests).
 * Values are compared numerically (after Number() coercion).
 */
export function matricesEqual(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b)) return false;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i].length !== b[i].length) return false;
    for (let j = 0; j < a[i].length; j++) {
      // eslint-disable-next-line eqeqeq
      if (Number(a[i][j]) !== Number(b[i][j])) return false;
    }
  }
  return true;
}

export function matrixShape(matrix) {
  return [matrix.length, matrix[0].length];
}
