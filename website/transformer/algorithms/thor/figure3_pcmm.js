// Paper-faithful THOR Figure 3 diagonal-block PC-MM trace.
// The numerical values are editable, but the paper preset keeps d=4, n=2,
// s=8, and c=2 so every slot can be checked against the published diagram.

import { matmul, matricesEqual } from "../../core/matrix.js";
import {
  PAPER_TONES,
  addMatrices,
  addVectors,
  block as matrixBlock,
  ctState,
  interleave,
  lowerDiagonal,
  makeSteps,
  matrixArtifact,
  multiplyVectors,
  operator,
  packLowerDiagonals,
  paperToneForPcmmA,
  paperToneForPcmmB,
  paperToneForPcmmC,
  ref,
  rotateLeft,
  upperDiagonal,
  vectorArtifact,
  verticalBlock,
} from "./paper_utils.js";

export const THOR_FIGURE3_PRESET = Object.freeze({
  A: [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [2, 1, 0, 1],
    [3, 2, 1, 2],
  ],
  B: [
    [1, 0],
    [0, 1],
    [2, 1],
    [1, 2],
  ],
});

const SLOTS = 8;
const BLOCK_SIZE = 2;

function assertFigure3Shape(A, B, slots) {
  if (A.length !== 4 || A.some((row) => row.length !== 4)) {
    throw new Error("THOR Figure 3 requires A to be 4x4 (edit values freely; keep the paper dimensions)");
  }
  if (B.length !== 4 || B.some((row) => row.length !== 2)) {
    throw new Error("THOR Figure 3 requires B to be 4x2 (edit values freely; keep the paper dimensions)");
  }
  if (slots !== SLOTS) throw new Error("THOR Figure 3 uses exactly 8 slots: s=8, c=s/d=2");
}

function expandUpperPair(first, second, diagonal) {
  const interlaced = interleave([
    upperDiagonal(first, diagonal, BLOCK_SIZE),
    upperDiagonal(second, diagonal, BLOCK_SIZE),
  ]);
  return [...interlaced, ...rotateLeft(interlaced, BLOCK_SIZE)];
}

function packedBTones() {
  return [0, 2, 0, 2, 1, 3, 1, 3];
}

function familyTones(diagonal) {
  const base = diagonal === 0 ? [0, 2, 0, 2] : [1, 3, 1, 3];
  return [...base, ...rotateLeft(base, 2)];
}

function packedBRefs() {
  const refs = [];
  for (let diagonal = 0; diagonal < 2; diagonal += 1) {
    for (let t = 0; t < 2; t += 1) {
      for (let blockRow = 0; blockRow < 2; blockRow += 1) {
        const row = blockRow * 2 + ((diagonal + t) % 2);
        refs.push([`B:${row}:${t}`]);
      }
    }
  }
  return refs;
}

function sourceRefsForPackedB() {
  return packedBRefs().map((keys) => keys.map((key) => {
    const [, row, col] = key.split(":").map(Number);
    return ref("B", row, col, 2);
  }));
}

function vectorRefsFromKeys(keys) {
  return keys.map((entries) => entries.map((key) => {
    const [tensorId, row, col] = key.split(":");
    const cols = tensorId === "A" ? 4 : 2;
    return ref(tensorId, Number(row), Number(col), cols);
  }));
}

function familyARefs(blockPairs, diagonal) {
  const short = [];
  for (let t = 0; t < 2; t += 1) {
    for (const [blockRow, blockCol] of blockPairs) {
      const row = blockRow * 2 + t;
      const col = blockCol * 2 + ((t + diagonal) % 2);
      short.push([`A:${row}:${col}`]);
    }
  }
  return [...short, ...rotateLeft(short, 2)];
}

export function buildThorFigure3PcmmTrace(options = {}) {
  const A = options.A || THOR_FIGURE3_PRESET.A;
  const B = options.B || THOR_FIGURE3_PRESET.B;
  const slots = Number(options.slotsPerCiphertext || SLOTS);
  assertFigure3Shape(A, B, slots);

  const A00 = matrixBlock(A, 0, 0, 2);
  const A01 = matrixBlock(A, 0, 1, 2);
  const A10 = matrixBlock(A, 1, 0, 2);
  const A11 = matrixBlock(A, 1, 1, 2);
  const B0 = verticalBlock(B, 0, 2);
  const B1 = verticalBlock(B, 1, 2);

  const alpha = matmul(A00, B0);
  const beta = matmul(A01, B1);
  const gamma = matmul(A10, B0);
  const delta = matmul(A11, B1);
  const C0 = addMatrices(alpha, beta);
  const C1 = addMatrices(gamma, delta);
  const C = [...C0, ...C1];
  const directC = matmul(A, B);
  if (!matricesEqual(C, directC)) throw new Error("THOR Figure 3 block decomposition failed verification");

  const ctB = packLowerDiagonals([B0, B1], 0, 2);
  const rho4B = rotateLeft(ctB, 4);
  const p00 = expandUpperPair(A00, A11, 0);
  const p10 = expandUpperPair(A10, A01, 0);
  const p01 = expandUpperPair(A00, A11, 1);
  const p11 = expandUpperPair(A10, A01, 1);

  const contribution00 = multiplyVectors(p00, ctB);
  const contribution10 = multiplyVectors(p10, ctB);
  const contribution01 = multiplyVectors(p01, rho4B);
  const contribution11 = multiplyVectors(p11, rho4B);
  const partial0 = addVectors([contribution00, contribution01]);
  const partial1 = addVectors([contribution10, contribution11]);
  const expectedPartial0 = packLowerDiagonals([alpha, delta], 0, 2);
  const expectedPartial1 = packLowerDiagonals([gamma, beta], 0, 2);
  if (JSON.stringify(partial0) !== JSON.stringify(expectedPartial0) || JSON.stringify(partial1) !== JSON.stringify(expectedPartial1)) {
    throw new Error("THOR Figure 3 parallel products do not match the paper packing");
  }

  const rotatedPartial1 = rotateLeft(partial1, 1);
  const betaMask = rotatedPartial1.map((value, index) => (index % 2 === 0 ? value : 0));
  const gammaMask = rotatedPartial1.map((value, index) => (index % 2 === 1 ? value : 0));
  const correctedGamma = rotateLeft(gammaMask, -2);
  const alignedPartial1 = addVectors([betaMask, correctedGamma]);
  const packedC = addVectors([partial0, alignedPartial1]);
  const expectedPackedC = packLowerDiagonals([C0, C1], 0, 2);
  if (JSON.stringify(packedC) !== JSON.stringify(expectedPackedC)) {
    throw new Error("THOR Figure 3 final multi-diagonal ciphertext failed verification");
  }

  const inputA = matrixArtifact({
    id: "A", label: "Plaintext A", values: A, revealAt: 0, badge: "PT",
    meta: "4x4 - four 2x2 blocks",
    toneForCell: (i, j) => paperToneForPcmmA(i, j, 2),
  });
  const inputB = matrixArtifact({
    id: "B", label: "Ciphertext B", values: B, revealAt: 0, badge: "CT",
    meta: "4x2 - B0 over B1",
    toneForCell: (i, j) => paperToneForPcmmB(i, j, 2),
  });
  const outputC = matrixArtifact({
    id: "C", label: "Decoded C", values: C, revealAt: 7, badge: "CT",
    meta: "C0 over C1 - verified",
    toneForCell: (i, j) => paperToneForPcmmC(i, j, 2),
  });

  const bTones = packedBTones();
  const diagonal0Tones = familyTones(0);
  const diagonal1Tones = familyTones(1);
  const partialTones = bTones;
  const rotatedPartialTones = rotateLeft(partialTones, 1);
  const betaMaskTones = rotatedPartialTones.map((tone, index) => index % 2 === 0 ? tone : PAPER_TONES.neutral);
  const gammaMaskTones = rotatedPartialTones.map((tone, index) => index % 2 === 1 ? tone : PAPER_TONES.neutral);
  const correctedGammaTones = rotateLeft(gammaMaskTones, -2);
  const alignedTones = [2, 0, 2, 0, 3, 1, 3, 1];
  const outputTones = [0, 2, 0, 2, 1, 3, 1, 3];

  const bProvenance = packedBRefs();
  const p00Prov = familyARefs([[0, 0], [1, 1]], 0);
  const p10Prov = familyARefs([[1, 0], [0, 1]], 0);
  const p01Prov = familyARefs([[0, 0], [1, 1]], 1);
  const p11Prov = familyARefs([[1, 0], [0, 1]], 1);
  const rho4BProv = rotateLeft(bProvenance, 4);

  const artifact = (cfg) => vectorArtifact(cfg);
  const ctBArtifact = artifact({ id: "fig3-ct-b", label: "ct.B = Lhat0(B0,B1)", values: ctB, revealAt: 1, badge: "CT", meta: "[barL0 | barL1]", tones: bTones, provenance: bProvenance });
  const rho4BArtifact = artifact({ id: "fig3-rho4-b", label: "rho^4(ct.B)", values: rho4B, revealAt: 2, badge: "CT", meta: "swap the two packed diagonals", tones: rotateLeft(bTones, 4), provenance: rho4BProv });
  const pArtifacts = [
    artifact({ id: "fig3-p00", label: "barU0(A00,A11) | rho^2", values: p00, revealAt: 2, badge: "PT", tones: diagonal0Tones, provenance: p00Prov }),
    artifact({ id: "fig3-p10", label: "barU0(A10,A01) | rho^2", values: p10, revealAt: 2, badge: "PT", tones: diagonal0Tones, provenance: p10Prov }),
    artifact({ id: "fig3-p01", label: "barU1(A00,A11) | rho^2", values: p01, revealAt: 2, badge: "PT", tones: diagonal1Tones, provenance: p01Prov }),
    artifact({ id: "fig3-p11", label: "barU1(A10,A01) | rho^2", values: p11, revealAt: 2, badge: "PT", tones: diagonal1Tones, provenance: p11Prov }),
  ];
  const partial0Artifact = artifact({ id: "fig3-partial-0", label: "Lhat0(A00B0,A11B1)", values: partial0, revealAt: 3, badge: "CT", kind: "product", tones: partialTones });
  const partial1Artifact = artifact({ id: "fig3-partial-1", label: "Lhat0(A10B0,A01B1)", values: partial1, revealAt: 3, badge: "CT", kind: "product", tones: partialTones });
  const rotatedArtifact = artifact({ id: "fig3-rho-partial", label: "rho(partial family 1)", values: rotatedPartial1, revealAt: 4, badge: "CT", tones: rotatedPartialTones });
  const betaArtifact = artifact({ id: "fig3-beta-mask", label: "beta slots", values: betaMask, revealAt: 5, badge: "CT", tones: betaMaskTones });
  const gammaArtifact = artifact({ id: "fig3-gamma-mask", label: "rho^-2(gamma slots)", values: correctedGamma, revealAt: 5, badge: "CT", tones: correctedGammaTones });
  const alignedArtifact = artifact({ id: "fig3-aligned", label: "Lhat0(A01B1,A10B0)", values: alignedPartial1, revealAt: 5, badge: "CT", tones: alignedTones });
  const resultArtifact = artifact({ id: "fig3-ct-c", label: "ct.C = Lhat0(C0,C1)", values: packedC, revealAt: 6, badge: "CT", kind: "result", tones: outputTones });

  const graph = {
    id: "thor-figure3-paper",
    algorithmId: "thor-pcmm",
    title: "THOR Figure 3 - diagonal-block PC-MM",
    subtitle: "Paper-faithful d=4, n=2, s=8 trace. Colors follow element identity through packing and rotations.",
    stages: [
      { id: "fig3-input", title: "Setup: input block matrices", caption: "A is plaintext; B is ciphertext. Colors match Figure 3.", revealAt: 0, rows: [[inputA, operator("fig3-times", "x", "block matmul", 0), inputB]] },
      { id: "fig3-pack", title: "Setup: multi-diagonal pack ct.B", caption: "Interleave the same lower diagonal of B0 and B1, then concatenate both diagonals.", revealAt: 1, rows: [[ctBArtifact]] },
      { id: "fig3-parallel-inputs", title: "Step 1a: prepare parallel operands", caption: "Two block families share ct.B and rho^4(ct.B).", revealAt: 2, rows: [
        [pArtifacts[0], operator("fig3-op00", ".*", "PT x CT", 2), ctBArtifact],
        [pArtifacts[1], operator("fig3-op10", ".*", "PT x CT", 2), ctBArtifact],
        [pArtifacts[2], operator("fig3-op01", ".*", "PT x CT", 2), rho4BArtifact],
        [pArtifacts[3], operator("fig3-op11", ".*", "PT x CT", 2), rho4BArtifact],
      ] },
      { id: "fig3-partials", title: "Step 1b: parallel submatrix products", caption: "alpha/delta and gamma/beta remain interlaced exactly as in the paper.", revealAt: 3, rows: [[partial0Artifact], [partial1Artifact]] },
      { id: "fig3-rotate", title: "Step 2a: rotate partial products", caption: "Rotate the second family left by one slot before separating beta and gamma.", revealAt: 4, rows: [[partial1Artifact, operator("fig3-rho", "rho", "left 1", 4), rotatedArtifact]] },
      { id: "fig3-mask", title: "Step 2b: mask and align wraparound", caption: "Keep beta in even slots; rotate gamma right by two slots; then add.", revealAt: 5, rows: [
        [betaArtifact, operator("fig3-plus-mask", "+", "combine", 5), gammaArtifact],
        [operator("fig3-to-align", "=", "aligned family", 5), alignedArtifact],
      ] },
      { id: "fig3-aggregate", title: "Step 3: aggregate aligned products", caption: "Add the two aligned partial block families.", revealAt: 6, rows: [
        [partial0Artifact, operator("fig3-plus", "+", "slot-wise", 6), alignedArtifact],
        [operator("fig3-eq", "=", "encoded output", 6), resultArtifact],
      ] },
      { id: "fig3-output", title: "Visual check: decode and verify C", caption: "ct.C remains multi-lower-diagonal encoded; decoding reconstructs C0 over C1.", revealAt: 7, rows: [[resultArtifact, operator("fig3-decode", "->", "decode", 7), outputC]] },
    ],
  };

  const tensors = (includeOutput = false) => ({
    left: { id: "A", label: "Plaintext A", shape: [4, 4], values: A },
    right: { id: "B", label: "Ciphertext B", shape: [4, 2], values: B },
    ...(includeOutput ? { output: { id: "C", label: "Result C", shape: [4, 2], values: C } } : {}),
  });
  const bSourceRefs = sourceRefsForPackedB();
  const steps = makeSteps("thor-fig3", [
    { phase: "reference", title: "Setup: input block matrices", description: "Split A into four 2x2 blocks and B into B0 over B1.", formula: "C0=A00B0+A01B1; C1=A10B0+A11B1", tensors: tensors() },
    { phase: "encode", title: "Setup: multi-diagonal pack ct.B", description: "Interleave lower diagonals across B0 and B1.", formula: "ct.B=[barL0(B0,B1)|barL1(B0,B1)]", tensors: tensors(), ciphertexts: [ctState("fig3-ct-b", "ct.B", ctB, 8, true, "tensor", bSourceRefs)], delta: { ciphertextCount: 1 } },
    { phase: "multiply", title: "Step 1a: prepare parallel operands", description: "Prepare upper-diagonal plaintext masks and rho^4(ct.B).", formula: "four PT vectors; ct.B and rho^4(ct.B)", tensors: tensors(), ciphertexts: [ctState("fig3-ct-b", "ct.B", ctB, 8, true, "tensor", bSourceRefs), ctState("fig3-rho4-b", "rho^4(ct.B)", rho4B, 8, true)], delta: { rotations: 1, ciphertextCount: 1 } },
    { phase: "multiply", title: "Step 1b: parallel submatrix products", description: "Four PT x CT products produce two block families.", formula: "T0=Lhat0(alpha,delta); T1=Lhat0(gamma,beta)", tensors: tensors(), ciphertexts: [ctState("fig3-partial-0", "T0", partial0, 8, true, "product"), ctState("fig3-partial-1", "T1", partial1, 8, true, "product")], delta: { ptCtMultiplications: 4, additions: 2, multiplicativeDepth: 1, ciphertextCount: 2 } },
    { phase: "rotate", title: "Step 2a: rotate partial products", description: "rho(T1) moves beta into even slots and gamma into odd slots.", formula: "rho(T1)", tensors: tensors(), ciphertexts: [ctState("fig3-rho-partial", "rho(T1)", rotatedPartial1, 8, true)], delta: { rotations: 1 } },
    { phase: "mask", title: "Step 2b: mask and align wraparound", description: "Separate beta/gamma, apply rho^-2 to gamma, then add.", formula: "T1_aligned=mask_beta(rho(T1))+rho^-2(mask_gamma(rho(T1)))", tensors: tensors(), ciphertexts: [ctState("fig3-aligned", "T1 aligned", alignedPartial1, 8, true)], delta: { masks: 2, rotations: 1, additions: 1 } },
    { phase: "reduce", title: "Step 3: aggregate aligned products", description: "T0 plus the aligned T1 yields the encoded result.", formula: "ct.C=T0+T1_aligned", tensors: tensors(), ciphertexts: [ctState("fig3-ct-c", "ct.C", packedC, 8, true, "result")], delta: { additions: 1, ciphertextCount: 1 } },
    { phase: "output", title: "Visual check: decode and verify C", description: "Decode both lower diagonals of C0 and C1 and compare with direct A x B.", formula: `Verified C=${JSON.stringify(C)}`, tensors: tensors(true), ciphertexts: [ctState("fig3-ct-c", "ct.C", packedC, 8, true, "result")] },
  ]);

  return {
    schemaVersion: "1.0",
    id: "thor-pcmm-figure3-paper",
    title: "THOR Figure 3 PC-MM",
    parameters: { matrixShape: { m: 4, k: 4, n: 2 }, slotsPerCiphertext: 8, matmulType: "pt-ct", algorithmId: "thor-pcmm" },
    operands: {
      left: { id: "A", label: "Plaintext A", shape: [4, 4], values: A, secret: false },
      right: { id: "B", label: "Ciphertext B", shape: [4, 2], values: B, secret: true },
    },
    output: { id: "C", label: "Result C", shape: [4, 2] },
    plan: {
      algorithmId: "thor-pcmm", matmulType: "pt-ct",
      logicalShape: { m: 4, k: 4, n: 2 }, paddedShape: { m: 4, k: 4, n: 2 },
      tileGrid: { m: 1, n: 1 }, tileShape: { m: 4, n: 2 },
      packingStrategyRequested: "thor-multi-diagonal", packingStrategyResolved: "thor-multi-diagonal",
      plannerSource: "THOR Figure 3", reductionStrategy: "internal-rotation-mask-aggregate", displayMode: "paper-faithful",
      warnings: ["Paper mode fixes d=4, n=2, s=8; matrix values remain fully editable."],
    },
    graph,
    steps,
    finalTensor: C,
  };
}
