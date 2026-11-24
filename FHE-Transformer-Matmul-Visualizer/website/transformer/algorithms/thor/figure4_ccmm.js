// Paper-faithful THOR Figure 4 ciphertext-ciphertext matrix multiplication.
// Both Figure 4(a) Upper-Lower and Figure 4(b) Lower-Lower Basic are exposed.

import { matmul, matricesEqual } from "../../core/matrix.js";
import {
  addVectors,
  ctState,
  extendedUpperDiagonal,
  lowerDiagonal,
  makeSteps,
  matrixArtifact,
  mod,
  multiplyVectors,
  operator,
  rotateLeft,
  vectorArtifact,
} from "./paper_utils.js";

export const THOR_FIGURE4_PRESET = Object.freeze({
  A: [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
  ],
  B: [
    [1, 0, 2, 1],
    [0, 1, 1, 2],
    [2, 1, 0, 1],
    [1, 2, 1, 0],
  ],
});

export const THOR_CCMM_METHODS = Object.freeze({
  LOWER_LOWER: "lower-lower",
  UPPER_LOWER: "upper-lower",
});

const M = 2;
const N = 4;
const SLOTS = 8;
const C = 2;

function assertFigure4Shape(A, B, slots) {
  if (A.length !== M || A.some((row) => row.length !== N)) {
    throw new Error("THOR Figure 4 requires A to be 2x4 (edit values freely; keep the paper dimensions)");
  }
  if (B.length !== N || B.some((row) => row.length !== N)) {
    throw new Error("THOR Figure 4 requires B to be 4x4 (edit values freely; keep the paper dimensions)");
  }
  if (slots !== SLOTS) throw new Error("THOR Figure 4 uses exactly 8 slots: s=8, c=s/n=2");
}

function outputPack(matrix) {
  return [...lowerDiagonal(matrix, 0, N), ...lowerDiagonal(matrix, 1, N)];
}

function lowerLowerRows(A, B) {
  return Array.from({ length: N }, (_, ell) => {
    const aSegments = Array.from({ length: C }, (_, r) =>
      rotateLeft(lowerDiagonal(A, mod(r - ell, M), N), ell)
    );
    const bDiagonal = lowerDiagonal(B, ell, N);
    const aPacked = aSegments.flat();
    const bPacked = Array.from({ length: C }, () => bDiagonal).flat();
    return {
      ell,
      aPacked,
      bPacked,
      product: multiplyVectors(aPacked, bPacked),
      aLabel: `[rho^${ell}(L${mod(-ell, M)}(A)) | rho^${ell}(L${mod(1 - ell, M)}(A))]`,
      bLabel: `[L${ell}(B) | L${ell}(B)]`,
      tones: Array.from({ length: C }, () => Array.from({ length: N }, (_, t) => mod(ell + t, 4))).flat(),
    };
  });
}

function upperLowerRows(A, B) {
  const U = Array.from({ length: N }, (_, k) => extendedUpperDiagonal(A, k));
  const L = Array.from({ length: N }, (_, ell) => lowerDiagonal(B, ell, N));
  const specifications = [
    { a: [0, 0], b: [0, 1] },
    { a: [1, 3], b: [1, 0] },
    { a: [2, 2], b: [2, 3] },
    { a: [3, 1], b: [3, 2] },
  ];
  return specifications.map((specification, index) => {
    const aPacked = [U[specification.a[0]], rotateLeft(U[specification.a[1]], 1)].flat();
    const bPacked = [L[specification.b[0]], L[specification.b[1]]].flat();
    const tones = [
      ...Array.from({ length: N }, (_, t) => mod(specification.b[0] + t, 4)),
      ...Array.from({ length: N }, (_, t) => mod(specification.b[1] + t, 4)),
    ];
    return {
      ell: index,
      aPacked,
      bPacked,
      product: multiplyVectors(aPacked, bPacked),
      aLabel: `[U'${specification.a[0]}(A) | rho(U'${specification.a[1]}(A))]`,
      bLabel: `[L${specification.b[0]}(B) | L${specification.b[1]}(B)]`,
      tones,
    };
  });
}

function outputTones() {
  return Array.from({ length: C }, (_, r) =>
    Array.from({ length: N }, (_, t) => mod(r + t, M))
  ).flat();
}

function matrixToneA(_row, col) {
  return mod(col, 4);
}

function matrixToneB(row) {
  return mod(row, 4);
}

function matrixToneC(row) {
  return mod(row, 2);
}

export function buildThorFigure4CcmmTrace(options = {}) {
  const A = options.A || THOR_FIGURE4_PRESET.A;
  const B = options.B || THOR_FIGURE4_PRESET.B;
  const slots = Number(options.slotsPerCiphertext || SLOTS);
  const method = options.method || THOR_CCMM_METHODS.LOWER_LOWER;
  if (!Object.values(THOR_CCMM_METHODS).includes(method)) throw new Error(`Unknown THOR CC-MM method: ${method}`);
  assertFigure4Shape(A, B, slots);

  const result = matmul(A, B);
  const rows = method === THOR_CCMM_METHODS.UPPER_LOWER ? upperLowerRows(A, B) : lowerLowerRows(A, B);
  const packedResult = addVectors(rows.map((row) => row.product));
  const expectedPacked = outputPack(result);
  if (JSON.stringify(packedResult) !== JSON.stringify(expectedPacked)) {
    throw new Error(`THOR Figure 4 ${method} packed result failed verification`);
  }

  const methodLabel = method === THOR_CCMM_METHODS.UPPER_LOWER ? "Upper-Lower" : "Lower-Lower Basic";
  const inputA = matrixArtifact({ id: "A", label: "Ciphertext A", values: A, revealAt: 0, badge: "CT", meta: "2x4", toneForCell: matrixToneA });
  const inputB = matrixArtifact({ id: "B", label: "Ciphertext B", values: B, revealAt: 0, badge: "CT", meta: "4x4", toneForCell: matrixToneB });
  const outputC = matrixArtifact({ id: "C", label: "Decoded C", values: result, revealAt: 6, badge: "CT", meta: "2x4 - verified", toneForCell: matrixToneC });

  const encodedA = method === THOR_CCMM_METHODS.LOWER_LOWER
    ? vectorArtifact({
        id: "fig4-ct-a", label: "ct.A = [L0(A) | L1(A)]", values: outputPack(A), revealAt: 1,
        tones: [0, 1, 2, 3, 0, 1, 2, 3], meta: "multi-lower-diagonal input",
      })
    : vectorArtifact({
        id: "fig4-ct-a", label: "Encrypted extended upper diagonals U'k(A)", values: extendedUpperDiagonal(A, 0).concat(extendedUpperDiagonal(A, 1)), revealAt: 1,
        tones: [0, 1, 2, 3, 0, 1, 2, 3], meta: "Figure 4(a) input family",
      });

  const preparedRows = rows.map((row) => {
    const a = vectorArtifact({ id: `fig4-a-${row.ell}`, label: row.aLabel, values: row.aPacked, revealAt: 2, badge: "CT", tones: row.tones });
    const b = vectorArtifact({ id: `fig4-b-${row.ell}`, label: row.bLabel, values: row.bPacked, revealAt: 2, badge: "CT", tones: row.tones });
    return [a, operator(`fig4-mult-${row.ell}`, ".*", "CT x CT", 2), b];
  });
  const productArtifacts = rows.map((row) => vectorArtifact({
    id: `fig4-product-${row.ell}`, label: `P${row.ell}`, values: row.product, revealAt: 3,
    badge: "CT", kind: "product", tones: row.tones, meta: `${row.aLabel} .* ${row.bLabel}`,
  }));
  const outputArtifact = vectorArtifact({
    id: "fig4-ct-c", label: "ct.C = [L0(C) | L1(C)]", values: packedResult, revealAt: 5,
    badge: "CT", kind: "result", tones: outputTones(), meta: "multi-lower-diagonal output",
  });

  const graph = {
    id: `thor-figure4-${method}`,
    algorithmId: "thor-ccmm",
    title: `THOR Figure 4 - ${methodLabel} CC-MM`,
    subtitle: "Paper-faithful m=2, n=4, s=8, c=2. Both operands stay encrypted.",
    stages: [
      { id: "fig4-input", title: "Setup: encrypted input matrices", caption: "A is 2x4; B is 4x4; C is 2x4.", revealAt: 0, rows: [[inputA, operator("fig4-input-mult", "x", "CC-MM", 0), inputB]] },
      { id: "fig4-encode", title: method === THOR_CCMM_METHODS.UPPER_LOWER ? "Encode extended upper diagonals" : "Encode multi-lower diagonals", caption: method === THOR_CCMM_METHODS.UPPER_LOWER ? "Generate U'0...U'3 according to Equation (4)." : "Pack L0(A) and L1(A) into ct.A.", revealAt: 1, rows: [[encodedA]] },
      { id: "fig4-prepare", title: "Prepare rotated ciphertext pairs", caption: method === THOR_CCMM_METHODS.UPPER_LOWER ? "Arrange encrypted U'k(A) with the corresponding lower diagonals of B." : "Rotate A's lower diagonals and replicate each lower diagonal of B.", revealAt: 2, rows: preparedRows },
      { id: "fig4-products", title: "Evaluate parallel CT×CT products", caption: "Each row is one slot-wise tensor product; colors preserve the contraction lane.", revealAt: 3, rows: productArtifacts.map((artifact) => [artifact]) },
      { id: "fig4-sum", title: "Accumulate contraction lanes", caption: "Pairwise accumulation keeps the four products readable before the final sum.", revealAt: 4, rows: [
        [productArtifacts[0], operator("fig4-plus-01", "+", "pair 0", 4), productArtifacts[1]],
        [productArtifacts[2], operator("fig4-plus-23", "+", "pair 1", 4), productArtifacts[3]],
      ] },
      { id: "fig4-output", title: "Output multi-lower-diagonal encoding", caption: "The result remains encrypted and multi-lower-diagonal encoded.", revealAt: 5, rows: [[operator("fig4-total", "sum", "slot-wise", 5), outputArtifact]] },
      { id: "fig4-decode", title: "Visual check: decode and verify C", caption: "Decode only for visualization, then compare with direct A x B.", revealAt: 6, rows: [[outputArtifact, operator("fig4-decode-op", "->", "decode", 6), outputC]] },
    ],
  };

  const tensors = (includeOutput = false) => ({
    left: { id: "A", label: "Ciphertext A", shape: [2, 4], values: A },
    right: { id: "B", label: "Ciphertext B", shape: [4, 4], values: B },
    ...(includeOutput ? { output: { id: "C", label: "Result C", shape: [2, 4], values: result } } : {}),
  });
  const allProducts = rows.map((row) => ctState(`fig4-product-${row.ell}`, `P${row.ell}`, row.product, 8, true, "product"));
  const rotationCount = method === THOR_CCMM_METHODS.UPPER_LOWER ? 12 : 8;
  const plaintextMasks = method === THOR_CCMM_METHODS.UPPER_LOWER ? 8 : 16;
  const steps = makeSteps(`thor-fig4-${method}`, [
    { phase: "reference", title: "Setup: encrypted input matrices", description: "Both inputs are ciphertexts; the browser exposes values only for teaching.", formula: "C=A x B, A in R^(2x4), B in R^(4x4)", tensors: tensors() },
    { phase: "encode", title: method === THOR_CCMM_METHODS.UPPER_LOWER ? "Encode extended upper diagonals" : "Encode multi-lower diagonals", description: method === THOR_CCMM_METHODS.UPPER_LOWER ? "Use Equation (4) to extend A's encrypted upper diagonals." : "ct.A holds [L0(A)|L1(A)].", formula: method === THOR_CCMM_METHODS.UPPER_LOWER ? "U'k(A) from Eq. (4)" : "ct.A=[L0(A)|L1(A)]", tensors: tensors(), ciphertexts: [ctState("fig4-ct-a", "ct.A", method === THOR_CCMM_METHODS.LOWER_LOWER ? outputPack(A) : encodedA.cells.map((cell) => cell.value), 8, true)], delta: { ciphertextCount: 1 } },
    { phase: "rotate", title: "Prepare rotated ciphertext pairs", description: "Create the four A operands and the matching packed/replicated B operands.", formula: rows.map((row) => `${row.aLabel} .* ${row.bLabel}`).join("\n"), tensors: tensors(), ciphertexts: rows.flatMap((row) => [ctState(`fig4-a-${row.ell}`, row.aLabel, row.aPacked, 8, true), ctState(`fig4-b-${row.ell}`, row.bLabel, row.bPacked, 8, true)]), delta: { rotations: rotationCount, ptCtMultiplications: plaintextMasks, ciphertextCount: 7 } },
    { phase: "multiply", title: "Evaluate parallel CT×CT products", description: "Perform four CT x CT SIMD multiplications.", formula: "Pell=Aell .* Bell, ell=0..3", tensors: tensors(), ciphertexts: allProducts, delta: { ctCtMultiplications: 4, relinearizations: 1, rescales: 1, multiplicativeDepth: 1, ciphertextCount: 4 } },
    { phase: "reduce", title: "Accumulate contraction lanes", description: "Add the four product ciphertexts slot by slot.", formula: "ct.C=P0+P1+P2+P3", tensors: tensors(), ciphertexts: [ctState("fig4-partial", "ct.C partial", packedResult, 8, true, "partial-sum")], delta: { additions: 3, ciphertextCount: 1 } },
    { phase: "output", title: "Output multi-lower-diagonal encoding", description: "The encrypted result stores L0(C) and L1(C).", formula: `ct.C=[${packedResult.join(", ")}]`, tensors: tensors(), ciphertexts: [ctState("fig4-ct-c", "ct.C", packedResult, 8, true, "result")] },
    { phase: "output", title: "Visual check: decode and verify C", description: "Visualization-only decoding matches direct matrix multiplication.", formula: `Verified C=${JSON.stringify(result)}`, tensors: tensors(true), ciphertexts: [ctState("fig4-ct-c", "ct.C", packedResult, 8, true, "result")] },
  ]);

  const direct = matmul(A, B);
  if (!matricesEqual(result, direct)) throw new Error("THOR Figure 4 decoded result failed verification");

  return {
    schemaVersion: "1.0",
    id: `thor-ccmm-figure4-${method}`,
    title: `THOR Figure 4 ${methodLabel}`,
    parameters: { matrixShape: { m: 2, k: 4, n: 4 }, slotsPerCiphertext: 8, matmulType: "ct-ct", algorithmId: "thor-ccmm", ccmmMethod: method },
    operands: {
      left: { id: "A", label: "Ciphertext A", shape: [2, 4], values: A, secret: true },
      right: { id: "B", label: "Ciphertext B", shape: [4, 4], values: B, secret: true },
    },
    output: { id: "C", label: "Result C", shape: [2, 4] },
    plan: {
      algorithmId: "thor-ccmm", matmulType: "ct-ct",
      logicalShape: { m: 2, k: 4, n: 4 }, paddedShape: { m: 2, k: 4, n: 4 },
      tileGrid: { m: 1, n: 1 }, tileShape: { m: 2, n: 4 },
      packingStrategyRequested: method, packingStrategyResolved: method,
      plannerSource: "THOR Figure 4", reductionStrategy: "four-tensor-products-and-sum", displayMode: "paper-faithful",
      warnings: ["Paper mode fixes m=2, n=4, s=8, c=2; matrix values and CC-MM approach remain editable."],
    },
    graph,
    steps,
    finalTensor: result,
  };
}
