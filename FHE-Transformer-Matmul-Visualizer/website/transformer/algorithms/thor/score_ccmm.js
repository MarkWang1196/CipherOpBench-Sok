// Paper-faithful THOR Corollary 3.9 score (Q K^T) toy trace.
//
// THOR Section 6.2.2 computes the attention score in the transposed
// convention as S^T = K Q^T using Corollary 3.9:
//   L_r(A B) = sum_{ell in [m]} rho^ell( L'_{r-ell}(A) ) .* L_ell(B),  r in [n]
// where the extended lower diagonals are
//   L'_k(A) = L_k(A)              if k < m,
//   L'_k(A) = rho^{m floor(k/m)}( L_{k mod m}(A) )   if m <= k < n.
//
// The toy uses Q, K in R^{4x2} (n = 4 tokens, m = 2 dims), A = Q, B = K^T, so
// the encrypted diagonals directly reconstruct S = Q K^T in R^{4x4}. The two
// output ciphertexts hold ct.C0 = [L0(S) | L1(S)] and ct.C1 = [L2(S) | L3(S)]
// with c = s/n = 2 diagonals per ciphertext.

import { matmul, matricesEqual, transpose } from "../../core/matrix.js";
import {
  ctState,
  lowerDiagonal,
  makeSteps,
  matrixArtifact,
  mod,
  multiplyVectors,
  operator,
  rotateLeft,
  vectorArtifact,
} from "./paper_utils.js";

export const THOR_SCORE_PRESET = Object.freeze({
  Q: [
    [1, 2],
    [0, 1],
    [2, 0],
    [1, 1],
  ],
  K: [
    [1, 0],
    [1, 2],
    [0, 1],
    [2, 1],
  ],
});

const N = 4; // output rows (tokens)
const M = 2; // shared dimension
const SLOTS = 8;
const C = 2; // s / n

function assertScoreShape(Q, K, slots) {
  if (Q.length !== N || Q.some((row) => row.length !== M)) {
    throw new Error("THOR score requires Q to be 4x2 (edit values freely; keep the paper dimensions)");
  }
  if (K.length !== N || K.some((row) => row.length !== M)) {
    throw new Error("THOR score requires K to be 4x2 (edit values freely; keep the paper dimensions)");
  }
  if (slots !== SLOTS) throw new Error("THOR score uses exactly 8 slots: s=8, c=s/n=2");
}

export function buildThorScoreCcmmTrace(options = {}) {
  const Q = options.A || THOR_SCORE_PRESET.Q;
  const K = options.B || THOR_SCORE_PRESET.K;
  const slots = Number(options.slotsPerCiphertext || SLOTS);
  assertScoreShape(Q, K, slots);

  const KT = transpose(K);
  const S = matmul(Q, KT);
  const LQ = [lowerDiagonal(Q, 0, N), lowerDiagonal(Q, 1, N)];
  // Extended lower diagonals L'_k(Q), k in [4].
  const LQext = [LQ[0], LQ[1], rotateLeft(LQ[0], 2), rotateLeft(LQ[1], 2)];
  const LK = [lowerDiagonal(KT, 0, N), lowerDiagonal(KT, 1, N)];

  // For each output ciphertext j (diagonals {2j, 2j+1}) and contraction ell,
  // pack [rho^ell(L'_{2j-ell}) | rho^ell(L'_{2j+1-ell})] .* [L_ell(K^T) | L_ell(K^T)].
  const rows = [];
  for (let j = 0; j < 2; j += 1) {
    for (let ell = 0; ell < 2; ell += 1) {
      const aPacked = [0, 1].map((r) => rotateLeft(LQext[mod(2 * j + r - ell, 4)], ell)).flat();
      const bPacked = [...LK[ell], ...LK[ell]];
      rows.push({ j, ell, aPacked, bPacked, product: multiplyVectors(aPacked, bPacked) });
    }
  }

  const ctC0 = rows.filter((r) => r.j === 0).reduce((a, r) => a.map((v, i) => v + r.product[i]), new Array(8).fill(0));
  const ctC1 = rows.filter((r) => r.j === 1).reduce((a, r) => a.map((v, i) => v + r.product[i]), new Array(8).fill(0));
  const L_S = [0, 1, 2, 3].map((r) => lowerDiagonal(S, r, N));
  if (JSON.stringify(ctC0) !== JSON.stringify([...L_S[0], ...L_S[1]])) {
    throw new Error("THOR Corollary 3.9 ct.C0 failed verification");
  }
  if (JSON.stringify(ctC1) !== JSON.stringify([...L_S[2], ...L_S[3]])) {
    throw new Error("THOR Corollary 3.9 ct.C1 failed verification");
  }

  const inputQ = matrixArtifact({ id: "A", label: "Q", values: Q, revealAt: 0, badge: "CT", meta: "4x2", toneForCell: (_r, c) => mod(c, 4) });
  const inputK = matrixArtifact({ id: "B", label: "K", values: K, revealAt: 0, badge: "CT", meta: "4x2", toneForCell: (r, _c) => mod(r, 4) });
  const decoded = matrixArtifact({ id: "C", label: "Decoded S = QK^T", values: S, revealAt: 6, badge: "CT", meta: "4x4 - verified", toneForCell: (r, c) => mod(r + c, 4) });

  const ctA0 = vectorArtifact({ id: "score-ct-a0", label: "ct.A0 = [L'_0(Q) | L'_1(Q)]", values: [...LQext[0], ...LQext[1]], revealAt: 1, badge: "CT", tones: [0, 1, 2, 3, 0, 1, 2, 3] });
  const ctA1 = vectorArtifact({ id: "score-ct-a1", label: "ct.A1 = [L'_2(Q) | L'_3(Q)]", values: [...LQext[2], ...LQext[3]], revealAt: 1, badge: "CT", tones: [0, 1, 2, 3, 0, 1, 2, 3] });
  const ctB0 = vectorArtifact({ id: "score-ct-b0", label: "ct.B0 = [L_0(Kᵀ) | L_0(Kᵀ)]", values: [...LK[0], ...LK[0]], revealAt: 1, badge: "CT", tones: [0, 1, 2, 3, 0, 1, 2, 3] });
  const ctB1 = vectorArtifact({ id: "score-ct-b1", label: "ct.B1 = [L_1(Kᵀ) | L_1(Kᵀ)]", values: [...LK[1], ...LK[1]], revealAt: 1, badge: "CT", tones: [0, 1, 2, 3, 0, 1, 2, 3] });

  const prepared = rows.map((r) => {
    const a = vectorArtifact({ id: `score-a-${r.j}-${r.ell}`, label: `[ρ^${r.ell}(L'_{${mod(2 * r.j - r.ell, 4)}}) | ρ^${r.ell}(L'_{${mod(2 * r.j + 1 - r.ell, 4)}})]`, values: r.aPacked, revealAt: 2, badge: "CT", tones: [0, 1, 2, 3, 0, 1, 2, 3] });
    const b = vectorArtifact({ id: `score-b-${r.j}-${r.ell}`, label: `[L_${r.ell}(Kᵀ) | L_${r.ell}(Kᵀ)]`, values: r.bPacked, revealAt: 2, badge: "CT", tones: [0, 1, 2, 3, 0, 1, 2, 3] });
    return [a, operator(`score-mul-${r.j}-${r.ell}`, ".*", "CT x CT", 2), b];
  });
  const products = rows.map((r) => vectorArtifact({ id: `score-product-${r.j}-${r.ell}`, label: `P${r.j}${r.ell}`, values: r.product, revealAt: 3, badge: "CT", kind: "product", tones: [0, 1, 2, 3, 0, 1, 2, 3] }));
  const out0 = vectorArtifact({ id: "score-ct-c0", label: "ct.C0 = [L_0(S) | L_1(S)]", values: ctC0, revealAt: 5, badge: "CT", kind: "result", tones: [0, 1, 2, 3, 0, 1, 2, 3] });
  const out1 = vectorArtifact({ id: "score-ct-c1", label: "ct.C1 = [L_2(S) | L_3(S)]", values: ctC1, revealAt: 5, badge: "CT", kind: "result", tones: [0, 1, 2, 3, 0, 1, 2, 3] });

  const graph = {
    id: "thor-score-cor39",
    algorithmId: "thor-score",
    title: "THOR Corollary 3.9 - score QK^T",
    subtitle: "n=4, m=2, s=8, c=2. The encrypted diagonals reconstruct S = Q K^T in R^{4x4}.",
    stages: [
      { id: "score-input", title: "Setup: encrypted Q and K", caption: "THOR keeps both operands encrypted; the score is square 4x4.", revealAt: 0, rows: [[inputQ, operator("score-input-mult", "x", "CC-MM", 0), inputK]] },
      { id: "score-encode", title: "Encode extended lower diagonals", caption: "Pack L'_0..L'_3(Q) into two ciphertexts and the replicated lower diagonals of K^T.", revealAt: 1, rows: [[ctA0], [ctA1], [ctB0], [ctB1]] },
      { id: "score-prepare", title: "Prepare rotated ciphertext pairs", caption: "Corollary 3.9: rho^ell(L'_{r-ell}(A)) paired with L_ell(B).", revealAt: 2, rows: prepared },
      { id: "score-products", title: "Evaluate parallel CT×CT products", caption: "Each row is one slot-wise tensor product for the two output diagonals.", revealAt: 3, rows: products.map((p) => [p]) },
      { id: "score-sum", title: "Accumulate contraction lanes", caption: "Add the two products per output ciphertext.", revealAt: 4, rows: [
        [products[0], operator("score-plus0", "+", "j=0", 4), products[1]],
        [products[2], operator("score-plus1", "+", "j=1", 4), products[3]],
      ] },
      { id: "score-output", title: "Output multi-lower-diagonal encoding", caption: "Two ciphertexts hold L0|L1 and L2|L3 of the 4x4 score.", revealAt: 5, rows: [[out0], [out1]] },
      { id: "score-decode", title: "Visual check: decode and verify S", caption: "Decoding both ciphertexts reconstructs Q K^T.", revealAt: 6, rows: [[decoded]] },
    ],
  };
  const scoreStepCopy = [
    ["Input encrypted Q and K", "Q × Kᵀ"],
    ["Encode Q and Kᵀ lower diagonals", "Encode diagonals"],
    ["Prepare matching rotated diagonal pairs", "Pair diagonals"],
    ["Compute parallel ciphertext products", "CT products"],
    ["Sum products by output ciphertext", "Accumulate"],
    ["Keep multi-lower-diagonal score output", "Encoded scores"],
    ["Verify encrypted score matrix S", "Output"],
  ];
  graph.stages.forEach((stage, index) => {
    stage.stepTitle = scoreStepCopy[index][0];
    stage.shortTitle = scoreStepCopy[index][1];
  });

  const tensors = (includeOutput = false) => ({
    left: { id: "A", label: "Q", shape: [4, 2], values: Q },
    right: { id: "B", label: "Kᵀ", shape: [2, 4], values: KT },
    ...(includeOutput ? { output: { id: "C", label: "S", shape: [4, 4], values: S } } : {}),
  });
  const steps = makeSteps("thor-score", [
    { phase: "reference", title: "Setup: encrypted Q and K", description: "Both inputs are ciphertexts; the score S = Q K^T is 4x4.", formula: "S = Q K^T", tensors: tensors() },
    { phase: "encode", title: "Encode extended lower diagonals", description: "Corollary 3.9 needs L'_0..L'_3(Q) and L_0,L_1(K^T).", formula: "L'_k(Q), L_ell(K^T)", tensors: tensors(), ciphertexts: [
      ctState("score-ct-a0", "ct.A0", ctA0.cells.map((c) => c.value), 8, true), ctState("score-ct-a1", "ct.A1", ctA1.cells.map((c) => c.value), 8, true),
      ctState("score-ct-b0", "ct.B0", ctB0.cells.map((c) => c.value), 8, true), ctState("score-ct-b1", "ct.B1", ctB1.cells.map((c) => c.value), 8, true),
    ], delta: { ciphertextCount: 4 } },
    { phase: "rotate", title: "Prepare rotated ciphertext pairs", description: "Pair rho^ell(L'_{r-ell}(A)) with L_ell(B) for r in {2j, 2j+1}.", formula: rows.map((r) => `ρ^${r.ell}(L'_{${r.j * 2}-${r.ell}}) .* L_${r.ell}(Kᵀ)`).join("\n"), tensors: tensors(), delta: { rotations: 4, ciphertextCount: 4 } },
    { phase: "multiply", title: "Evaluate parallel CT×CT products", description: "Four SIMD ciphertext-ciphertext multiplications.", formula: "P_j,ell = ct.A .* ct.B", tensors: tensors(), delta: { ctCtMultiplications: 4, relinearizations: 1, rescales: 1, multiplicativeDepth: 1, ciphertextCount: 4 } },
    { phase: "reduce", title: "Accumulate contraction lanes", description: "ct.C0 = P00 + P01, ct.C1 = P10 + P11.", formula: "ct.Cj = Pj0 + Pj1", tensors: tensors(), delta: { additions: 2, ciphertextCount: 2 } },
    { phase: "output", title: "Output multi-lower-diagonal encoding", description: "ct.C0 and ct.C1 store the four lower diagonals of S.", formula: `ct.C0=[${ctC0.join(", ")}]; ct.C1=[${ctC1.join(", ")}]`, tensors: tensors(), ciphertexts: [ctState("score-ct-c0", "ct.C0", ctC0, 8, true, "result"), ctState("score-ct-c1", "ct.C1", ctC1, 8, true, "result")] },
    { phase: "output", title: "Visual check: decode and verify S", description: "Decode only for visualization; the protocol keeps the score encrypted.", formula: `Verified S=${JSON.stringify(S)}`, tensors: tensors(true) },
  ]);
  steps.forEach((step, index) => { step.title = graph.stages[index].stepTitle; });

  if (!matricesEqual(S, matmul(Q, KT))) throw new Error("THOR score decoded result failed verification");

  return {
    schemaVersion: "1.0",
    id: "thor-score-corollary39",
    title: "THOR Corollary 3.9 QK^T",
    parameters: { matrixShape: { m: 4, k: 2, n: 4 }, slotsPerCiphertext: 8, matmulType: "ct-ct", algorithmId: "thor-score" },
    operands: {
      left: { id: "A", label: "Q", shape: [4, 2], values: Q, secret: true },
      right: { id: "B", label: "Kᵀ", shape: [2, 4], values: KT, secret: true },
    },
    output: { id: "C", label: "S", shape: [4, 4] },
    plan: {
      algorithmId: "thor-score", matmulType: "ct-ct", visualStyle: "diagram-only",
      logicalShape: { m: 4, k: 2, n: 4 }, paddedShape: { m: 4, k: 2, n: 4 },
      tileGrid: { m: 1, n: 1 }, tileShape: { m: 4, n: 4 },
      packingStrategyRequested: "corollary-3.9", packingStrategyResolved: "corollary-3.9",
      plannerSource: "THOR Corollary 3.9 (Section 4.3)", reductionStrategy: "four-tensor-products-and-sum", displayMode: "paper-faithful",
      warnings: ["Paper mode fixes n=4, m=2, s=8, c=2; THOR encodes the score in the transposed form S^T = K Q^T, whose lower diagonals equal the upper diagonals of S."],
    },
    graph,
    steps,
    finalTensor: S,
  };
}
