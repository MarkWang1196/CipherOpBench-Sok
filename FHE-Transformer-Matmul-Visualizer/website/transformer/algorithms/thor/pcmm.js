// THOR-inspired plaintext-ciphertext matrix multiplication teaching trace.
//
// The preset follows the dataflow in THOR Figure 3 (diagonal extraction,
// multi-diagonal packing, encrypted rotations, SIMD products and aggregation)
// while keeping every slot numerically verifiable in the browser.  It uses the
// concrete 4x4 · 4x2 case from the figure's block geometry.  No HE library is
// invoked: CT/PT are semantic labels for the visualization.

import { matmul, transpose } from "../../core/matrix.js";
import { emptyMetrics } from "../../core/trace_schema.js";
import { nextPow2 } from "../../core/packing_planner.js";

export const THOR_PCMM_PRESET = Object.freeze({
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

export function cyclicDiagonal(matrix, offset) {
  const rows = matrix.length;
  const cols = matrix[0].length;
  if (rows !== cols) throw new Error("THOR teaching preset requires a square left matrix");
  return Array.from({ length: rows }, (_, row) => matrix[row][(row + offset) % cols]);
}

export function rotateLeft(values, amount) {
  if (!values.length) return [];
  const n = values.length;
  const shift = ((amount % n) + n) % n;
  return values.map((_, i) => values[(i + shift) % n]);
}

export function segmentedRotateLeft(values, amount, segmentSize) {
  if (!Number.isInteger(segmentSize) || segmentSize <= 0 || values.length % segmentSize !== 0) {
    throw new Error("segmentSize must evenly divide the packed vector");
  }
  const out = [];
  for (let start = 0; start < values.length; start += segmentSize) {
    out.push(...rotateLeft(values.slice(start, start + segmentSize), amount));
  }
  return out;
}

function addVectors(vectors) {
  return vectors[0].map((_, i) => vectors.reduce((sum, vector) => sum + vector[i], 0));
}

function multiplyVectors(left, right) {
  return left.map((value, i) => value * right[i]);
}

function matrixArtifact(id, label, values, revealAt, secret = false, meta = "") {
  return {
    id,
    type: "matrix",
    label,
    badge: secret ? "CT" : "PT",
    secret,
    meta: meta || `${values.length}×${values[0].length}`,
    revealAt,
    rows: values.map((row, i) =>
      row.map((value, j) => ({
        value,
        label: String(value),
        provenance: [`${id}:${i}:${j}`],
        tone: (i + j) % 4,
      }))
    ),
  };
}

function vectorArtifact({ id, label, values, revealAt, badge, meta, provenance, tone, kind = "vector" }) {
  return {
    id,
    type: kind,
    label,
    badge,
    meta,
    revealAt,
    cells: values.map((value, i) => ({
      value,
      label: String(value),
      provenance: provenance?.[i] || [`${id}:${i}`],
      tone: Array.isArray(tone) ? tone[i] : tone ?? i % 4,
    })),
  };
}

function operator(id, symbol, label, revealAt) {
  return { id, type: "operator", symbol, label, revealAt };
}

function makeSlots(id, values, kind, sourceRefs = []) {
  return values.map((value, slot) => ({
    id: `${id}:slot:${slot}`,
    logicalId: `${id}:${slot}`,
    slot,
    value,
    expression: `${id}[${slot}] = ${value}`,
    kind,
    sourceRefs: sourceRefs[slot] || [],
  }));
}

function ctState(id, label, values, secret, kind = "tensor", sourceRefs = []) {
  return {
    id,
    label,
    secret,
    illustrative: true,
    layout: "THOR teaching layout · 2 segments × 4 slots",
    slots: makeSlots(id, values, kind, sourceRefs),
  };
}

function tensorState(A, B, C, includeOutput = false) {
  return {
    left: { id: "A", label: "Plaintext A", shape: [4, 4], values: A },
    right: { id: "B", label: "Encrypted B", shape: [4, 2], values: B },
    ...(includeOutput
      ? { output: { id: "C", label: "Result C", shape: [4, 2], values: C } }
      : {}),
  };
}

function metricStep(previous, delta = {}) {
  const metricsDelta = { ...emptyMetrics(), ...delta };
  const metricsTotal = { ...previous };
  for (const key of Object.keys(metricsTotal)) metricsTotal[key] += metricsDelta[key] || 0;
  return { metricsDelta, metricsTotal };
}

function makeStep(index, cfg, previousMetrics) {
  const metrics = metricStep(previousMetrics, cfg.delta);
  return {
    id: `thor-pcmm-${index}`,
    index,
    phase: cfg.phase,
    title: cfg.title,
    description: cfg.description,
    formula: { plain: cfg.formula },
    notes: cfg.notes || [],
    operation: { type: cfg.operation },
    tensors: cfg.tensors,
    ciphertexts: cfg.ciphertexts || [],
    durationMs: cfg.durationMs || 1500,
    ...metrics,
  };
}

function cellRefsForPackedB(columns) {
  return columns.flatMap((column, j) =>
    column.map((value, i) => [{ tensorId: "B", coords: [i, j], linear: i * 2 + j, logicalId: `B:${i * 2 + j}` }])
  );
}

/** Build the fixed, fully verified THOR PC-MM story used by the paper view. */
export function buildThorPcmmFigure3Trace(options = {}) {
  const A = options.A || THOR_PCMM_PRESET.A;
  const B = options.B || THOR_PCMM_PRESET.B;
  if (A.length !== 4 || A.some((row) => row.length !== 4)) {
    throw new Error("THOR Figure 3 preset currently expects A to be 4×4");
  }
  if (B.length !== 4 || B.some((row) => row.length !== 2)) {
    throw new Error("THOR Figure 3 preset currently expects B to be 4×2");
  }

  const C = matmul(A, B);
  const columns = transpose(B);
  const packedB = columns.flat();
  const bRefs = cellRefsForPackedB(columns);
  const diagonals = Array.from({ length: 4 }, (_, k) => cyclicDiagonal(A, k));
  const packedDiagonals = diagonals.map((diagonal) => [...diagonal, ...diagonal]);
  const rotatedB = Array.from({ length: 4 }, (_, k) => segmentedRotateLeft(packedB, k, 4));
  const products = packedDiagonals.map((diagonal, k) => multiplyVectors(diagonal, rotatedB[k]));
  const packedC = addVectors(products);
  const expectedPackedC = columns.map((_, j) => C.map((row) => row[j])).flat();
  if (JSON.stringify(packedC) !== JSON.stringify(expectedPackedC)) {
    throw new Error("THOR PC-MM trace failed packed-result verification");
  }

  const diagRefs = diagonals.map((_, k) =>
    Array.from({ length: 8 }, (_, slot) => {
      const row = slot % 4;
      const col = (row + k) % 4;
      return [`A:${row}:${col}`];
    })
  );
  const rotatedBRefs = rotatedB.map((_, k) =>
    Array.from({ length: 8 }, (_, slot) => {
      const segment = Math.floor(slot / 4);
      const row = (slot % 4 + k) % 4;
      return [`B:${row}:${segment}`];
    })
  );
  const productRefs = products.map((_, k) =>
    Array.from({ length: 8 }, (_, slot) => [...diagRefs[k][slot], ...rotatedBRefs[k][slot]])
  );
  const resultRefs = Array.from({ length: 8 }, (_, slot) => {
    const column = Math.floor(slot / 4);
    const row = slot % 4;
    return [`C:${row}:${column}`, ...Array.from({ length: 4 }, (_, k) => `A:${row}:${(row + k) % 4}`), ...Array.from({ length: 4 }, (_, k) => `B:${(row + k) % 4}:${column}`)];
  });

  const inputA = matrixArtifact("A", "Plaintext matrix A", A, 0, false, "4×4 · split into 2×2 blocks");
  const inputB = matrixArtifact("B", "Encrypted matrix B", B, 0, true, "4×2 · two encrypted columns");
  const resultMatrix = matrixArtifact("C", "Decoded result C", C, 6, true, "4×2 · verified");
  resultMatrix.rows.forEach((row, i) => row.forEach((cell, j) => {
    cell.provenance = resultRefs[j * 4 + i];
    cell.tone = j;
  }));

  const diagonalArtifacts = diagonals.map((values, k) => vectorArtifact({
    id: `diag-${k}`,
    label: `D${k}(A)`,
    values,
    revealAt: 1,
    badge: "PT",
    meta: `A[i,(i+${k}) mod 4]`,
    provenance: diagRefs[k].slice(0, 4),
    tone: k,
  }));
  const packedDiagArtifacts = packedDiagonals.map((values, k) => vectorArtifact({
    id: `packed-diag-${k}`,
    label: `Û${k}(A)`,
    values,
    revealAt: 2,
    badge: "PT",
    meta: "diagonal duplicated across 2 column segments",
    provenance: diagRefs[k],
    tone: k,
  }));
  const packedBArtifact = vectorArtifact({
    id: "packed-b",
    label: "ct.B",
    values: packedB,
    revealAt: 2,
    badge: "CT",
    meta: "[B[:,0] | B[:,1]]",
    provenance: bRefs.map((refs) => refs.map((ref) => `B:${ref.coords[0]}:${ref.coords[1]}`)),
    tone: [0, 1, 2, 3, 0, 1, 2, 3],
  });
  const rotationArtifacts = rotatedB.map((values, k) => vectorArtifact({
    id: `rot-b-${k}`,
    label: k === 0 ? "ct.B" : `RotSeg${k}(ct.B)`,
    values,
    revealAt: 3,
    badge: "CT",
    meta: k === 0 ? "no rotation" : `ρ${k} + boundary mask`,
    provenance: rotatedBRefs[k],
    tone: k,
  }));
  const productArtifacts = products.map((values, k) => vectorArtifact({
    id: `product-${k}`,
    label: `P${k} = Û${k} ⊙ RotSeg${k}(ct.B)`,
    values,
    revealAt: 4,
    badge: "CT",
    meta: "slot-wise PT×CT",
    provenance: productRefs[k],
    tone: k,
    kind: "product",
  }));
  const packedCArtifact = vectorArtifact({
    id: "packed-c",
    label: "ct.C",
    values: packedC,
    revealAt: 5,
    badge: "CT",
    meta: "Σₖ Pk · columns remain separated",
    provenance: resultRefs,
    tone: [0, 0, 0, 0, 1, 1, 1, 1],
    kind: "result",
  });

  const graph = {
    id: "thor-pcmm-figure-3",
    algorithmId: "thor-pcmm",
    title: "THOR PC-MM · diagonal packing dataflow",
    subtitle: "A concrete 4×4 · 4×2 plaintext–ciphertext multiplication, organized like THOR Figure 3.",
    stages: [
      {
        id: "input",
        title: "Input matrices",
        caption: "A is plaintext; B is encrypted. The browser keeps values visible for teaching.",
        revealAt: 0,
        rows: [[inputA, operator("input-times", "×", "matrix multiplication", 0), inputB]],
      },
      {
        id: "diagonal",
        title: "1 · Extract cyclic diagonals",
        caption: "Each diagonal selects A[i,(i+k) mod 4], preserving the source-cell identity.",
        revealAt: 1,
        rows: diagonalArtifacts.map((artifact) => [artifact]),
      },
      {
        id: "encode",
        title: "2 · Multi-diagonal encoding",
        caption: "Duplicate each A diagonal for both encrypted B-column segments; pack B once.",
        revealAt: 2,
        rows: [...packedDiagArtifacts.map((artifact) => [artifact]), [packedBArtifact]],
      },
      {
        id: "rotate",
        title: "3 · Internal rotations",
        caption: "Segmented rotations are modeled as a global HE rotation plus a boundary mask/correction.",
        revealAt: 3,
        rows: rotationArtifacts.map((artifact) => [artifact]),
      },
      {
        id: "multiply",
        title: "4 · Parallel SIMD products",
        caption: "The four diagonal lanes multiply independently and can execute in parallel.",
        revealAt: 4,
        rows: productArtifacts.map((artifact) => [artifact]),
      },
      {
        id: "aggregate",
        title: "5 · Aggregation",
        caption: "Add the four partial ciphertexts slot by slot; no matrix values are revealed.",
        revealAt: 5,
        rows: [[operator("sum", "Σ", "add P0…P3", 5), packedCArtifact]],
      },
      {
        id: "output",
        title: "6 · Reconstruct output",
        caption: "Split the two 4-slot segments into the two columns of C.",
        revealAt: 6,
        rows: [[packedCArtifact, operator("decode", "→", "segment decode", 6), resultMatrix]],
      },
    ],
  };

  const diagSourceRefs = diagonals.map((_, k) =>
    Array.from({ length: 8 }, (_, slot) => {
      const row = slot % 4;
      const col = (row + k) % 4;
      return [{ tensorId: "A", coords: [row, col], linear: row * 4 + col, logicalId: `A:${row * 4 + col}` }];
    })
  );
  const rotationSourceRefs = rotatedB.map((_, k) =>
    Array.from({ length: 8 }, (_, slot) => {
      const column = Math.floor(slot / 4);
      const row = (slot % 4 + k) % 4;
      return [{ tensorId: "B", coords: [row, column], linear: row * 2 + column, logicalId: `B:${row * 2 + column}` }];
    })
  );
  const productSourceRefs = products.map((_, k) =>
    Array.from({ length: 8 }, (_, slot) => [...diagSourceRefs[k][slot], ...rotationSourceRefs[k][slot]])
  );
  const resultSourceRefs = Array.from({ length: 8 }, (_, slot) => {
    const column = Math.floor(slot / 4);
    const row = slot % 4;
    return Array.from({ length: 4 }, (_, k) => [
      { tensorId: "A", coords: [row, (row + k) % 4], linear: row * 4 + ((row + k) % 4), logicalId: `A:${row * 4 + ((row + k) % 4)}` },
      { tensorId: "B", coords: [(row + k) % 4, column], linear: ((row + k) % 4) * 2 + column, logicalId: `B:${((row + k) % 4) * 2 + column}` },
    ]).flat();
  });

  const tensors = tensorState(A, B, C, false);
  const total0 = emptyMetrics();
  const steps = [];
  steps.push(makeStep(0, {
    phase: "reference", operation: "reference", title: "Plaintext reference",
    description: "Start from A · B and preserve the 2×2 block geometry used in THOR Figure 3.",
    formula: "C = A · B,  A ∈ R⁴ˣ⁴ (PT),  B ∈ R⁴ˣ² (CT)", tensors,
  }, total0));
  steps.push(makeStep(1, {
    phase: "diagonal", operation: "extract-diagonal", title: "Extract A's cyclic diagonals",
    description: "Re-index A into four cyclic diagonals; every output row consumes one entry from each diagonal.",
    formula: "Dₖ(A)[i] = A[i,(i+k) mod 4]", tensors,
    ciphertexts: diagonals.map((v, k) => ctState(`diag-${k}`, `D${k}(A)`, [...v, ...v], false, "tensor", diagSourceRefs[k])),
  }, steps.at(-1).metricsTotal));
  steps.push(makeStep(2, {
    phase: "encode", operation: "multi-diagonal-pack", title: "Pack A diagonals and encrypted B",
    description: "A diagonals are duplicated for two column segments; B columns are concatenated into one ciphertext.",
    formula: "Ûₖ(A) = [Dₖ(A) | Dₖ(A)]\nct.B = [B[:,0] | B[:,1]]", tensors,
    ciphertexts: [
      ...packedDiagonals.map((v, k) => ctState(`packed-diag-${k}`, `Û${k}(A)`, v, false, "tensor", diagSourceRefs[k])),
      ctState("packed-b", "ct.B", packedB, true, "tensor", bRefs),
    ], delta: { ciphertextCount: 1 },
  }, steps.at(-1).metricsTotal));
  steps.push(makeStep(3, {
    phase: "rotate", operation: "internal-rotate", title: "Rotate inside each B-column segment",
    description: "The teaching model shows the logical segmented rotation produced by HE rotations and masks.",
    formula: "RotSegₖ(ct.B)[4j+i] = B[(i+k) mod 4,j]", tensors,
    ciphertexts: rotatedB.map((v, k) => ctState(`rot-b-${k}`, k ? `RotSeg${k}(ct.B)` : "ct.B", v, true, "tensor", rotationSourceRefs[k])),
    delta: { rotations: 6, masks: 6, ciphertextCount: 3 },
  }, steps.at(-1).metricsTotal));
  steps.push(makeStep(4, {
    phase: "multiply", operation: "ct-pt-multiply", title: "Multiply four diagonal lanes in parallel",
    description: "Each lane is a slot-wise PT×CT product; the contraction dimension is now distributed across lanes.",
    formula: "Pₖ = Ûₖ(A) ⊙ RotSegₖ(ct.B),  k = 0…3", tensors,
    ciphertexts: products.map((v, k) => ctState(`product-${k}`, `P${k}`, v, true, "product", productSourceRefs[k])),
    delta: { ptCtMultiplications: 4, multiplicativeDepth: 1, ciphertextCount: 4 },
  }, steps.at(-1).metricsTotal));
  steps.push(makeStep(5, {
    phase: "reduce", operation: "aggregate", title: "Aggregate partial ciphertexts",
    description: "Slot-wise additions combine all four diagonal contributions into the packed result.",
    formula: "ct.C = P₀ + P₁ + P₂ + P₃", tensors,
    ciphertexts: [ctState("packed-c", "ct.C", packedC, true, "partial-sum", resultSourceRefs)],
    delta: { additions: 3, ciphertextCount: 1 },
  }, steps.at(-1).metricsTotal));
  steps.push(makeStep(6, {
    phase: "output", operation: "decode", title: "Reconstruct and verify C",
    description: "Map the first four slots to C[:,0] and the next four slots to C[:,1].",
    formula: `ct.C = [${packedC.join(", ")}]\nC = ${JSON.stringify(C)}\nVerified: C equals direct A · B.`,
    tensors: tensorState(A, B, C, true),
    ciphertexts: [ctState("packed-c", "ct.C", packedC, true, "result", resultSourceRefs)],
  }, steps.at(-1).metricsTotal));

  return {
    schemaVersion: "1.0",
    id: "thor-pcmm-figure-3",
    title: "THOR PC-MM Figure 3 teaching trace",
    parameters: {
      matrixShape: { m: 4, k: 4, n: 2 },
      slotsPerCiphertext: 8,
      matmulType: "pt-ct",
      algorithmId: "thor-pcmm",
    },
    operands: {
      left: { id: "A", label: "Plaintext A", shape: [4, 4], values: A, secret: false },
      right: { id: "B", label: "Encrypted B", shape: [4, 2], values: B, secret: true },
    },
    output: { id: "C", label: "Result C", shape: [4, 2] },
    plan: {
      algorithmId: "thor-pcmm",
      matmulType: "pt-ct",
      logicalShape: { m: 4, k: 4, n: 2 },
      paddedShape: { m: 4, k: 4, n: 2 },
      tileGrid: { m: 1, n: 1 },
      tileShape: { m: 4, n: 2 },
      packingStrategyRequested: "thor-multi-diagonal",
      packingStrategyResolved: "thor-multi-diagonal",
      plannerSource: "paper-preset",
      reductionStrategy: "internal-rotation-and-add",
      displayMode: "full",
      warnings: ["Figure 3 teaching preset is fixed at 4×4 · 4×2; arbitrary-shape THOR planning is a separate adapter."],
    },
    graph,
    steps,
    finalTensor: C,
  };
}

// Generalized browser-side THOR PC-MM model. It applies the same diagonal SIMD
// identity to arbitrary M×K · K×N inputs by padding A to a D×D square, where
// D=nextPow2(max(M,K)), and packing as many B columns as the slot capacity
// allows. The story canvas shows the first output tile and up to eight lanes;
// all logical output cells are still computed and verified.
export function buildThorPcmmDynamicTrace(options = {}) {
  const A = options.A || THOR_PCMM_PRESET.A;
  const B = options.B || THOR_PCMM_PRESET.B;
  const M = A.length;
  const K = A[0]?.length || 0;
  const N = B[0]?.length || 0;
  if (!(M > 0 && K > 0 && N > 0) || A.some((row) => row.length !== K)) {
    throw new Error("THOR PC-MM requires a non-empty rectangular A matrix");
  }
  if (B.length !== K || B.some((row) => row.length !== N)) {
    throw new Error(`THOR PC-MM shape mismatch: A is ${M}×${K}, B is ${B.length}×${N}`);
  }

  const D = nextPow2(Math.max(M, K));
  const slots = Number(options.slotsPerCiphertext || nextPow2(D * Math.min(N, 2)));
  if (!Number.isInteger(slots) || slots < D || slots > 512 || (slots & (slots - 1)) !== 0) {
    throw new Error(`THOR slots must be a power of two between ${D} and 512`);
  }
  const segmentsPerCiphertext = Math.floor(slots / D);
  const columnsPerTile = Math.max(1, Math.min(N, segmentsPerCiphertext));
  const tileCount = Math.ceil(N / columnsPerTile);
  const tileStart = 0;
  const tileColumns = Math.min(columnsPerTile, N - tileStart);
  const C = matmul(A, B);

  const paddedA = Array.from({ length: D }, (_, i) =>
    Array.from({ length: D }, (_, j) => (i < M && j < K ? Number(A[i][j]) : 0))
  );
  const diagonals = Array.from({ length: D }, (_, k) => cyclicDiagonal(paddedA, k));
  const packedB = Array(slots).fill(0);
  for (let segment = 0; segment < tileColumns; segment++) {
    const column = tileStart + segment;
    for (let row = 0; row < K; row++) packedB[segment * D + row] = Number(B[row][column]);
  }
  const packedDiagonals = diagonals.map((diagonal) => {
    const vector = Array(slots).fill(0);
    for (let segment = 0; segment < tileColumns; segment++) {
      for (let row = 0; row < D; row++) vector[segment * D + row] = diagonal[row];
    }
    return vector;
  });
  const rotatedB = Array.from({ length: D }, (_, k) => segmentedRotateLeft(packedB, k, D));
  const products = packedDiagonals.map((diagonal, k) => multiplyVectors(diagonal, rotatedB[k]));
  const packedC = addVectors(products);
  const expectedPackedC = Array(slots).fill(0);
  for (let segment = 0; segment < tileColumns; segment++) {
    const column = tileStart + segment;
    for (let row = 0; row < M; row++) expectedPackedC[segment * D + row] = C[row][column];
  }
  if (JSON.stringify(packedC) !== JSON.stringify(expectedPackedC)) {
    throw new Error("THOR PC-MM dynamic trace failed packed-result verification");
  }

  const aRef = (slot, diagonalIndex) => {
    const row = slot % D;
    const col = (row + diagonalIndex) % D;
    if (row >= M || col >= K || Math.floor(slot / D) >= tileColumns) return [];
    return [{ tensorId: "A", coords: [row, col], linear: row * K + col, logicalId: `A:${row * K + col}` }];
  };
  const bRef = (slot, diagonalIndex) => {
    const segment = Math.floor(slot / D);
    const sourceRow = (slot % D + diagonalIndex) % D;
    const column = tileStart + segment;
    if (segment >= tileColumns || sourceRow >= K || column >= N) return [];
    return [{ tensorId: "B", coords: [sourceRow, column], linear: sourceRow * N + column, logicalId: `B:${sourceRow * N + column}` }];
  };
  const diagSourceRefs = diagonals.map((_, k) =>
    Array.from({ length: slots }, (_, slot) => aRef(slot, k))
  );
  const rotationSourceRefs = rotatedB.map((_, k) =>
    Array.from({ length: slots }, (_, slot) => bRef(slot, k))
  );
  const productSourceRefs = products.map((_, k) =>
    Array.from({ length: slots }, (_, slot) => [...diagSourceRefs[k][slot], ...rotationSourceRefs[k][slot]])
  );
  const resultSourceRefs = Array.from({ length: slots }, (_, slot) => {
    const segment = Math.floor(slot / D);
    const row = slot % D;
    const column = tileStart + segment;
    if (segment >= tileColumns || row >= M || column >= N) return [];
    return Array.from({ length: K }, (_, p) => [
      { tensorId: "A", coords: [row, p], linear: row * K + p, logicalId: `A:${row * K + p}` },
      { tensorId: "B", coords: [p, column], linear: p * N + column, logicalId: `B:${p * N + column}` },
    ]).flat();
  });

  const refsToKeys = (refs) => refs.map((ref) => `${ref.tensorId}:${ref.coords.join(":")}`);
  const graphVector = ({ id, label, values, revealAt, badge, meta, refs, tone, kind }) => {
    const visibleCount = Math.min(values.length, 32);
    const artifact = vectorArtifact({
      id,
      label,
      values: values.slice(0, visibleCount),
      revealAt,
      badge,
      meta: `${meta}${values.length > visibleCount ? ` · first ${visibleCount}/${values.length} slots` : ""}`,
      provenance: refs.slice(0, visibleCount).map(refsToKeys),
      tone,
      kind,
    });
    artifact.truncated = values.length > visibleCount;
    return artifact;
  };
  const graphMatrix = (id, label, values, revealAt, secret, meta) => {
    const shown = values.slice(0, 8).map((row) => row.slice(0, 8));
    const artifact = matrixArtifact(id, label, shown, revealAt, secret, meta);
    artifact.rows.forEach((row, i) => row.forEach((cell, j) => {
      cell.provenance = [`${id}:${i}:${j}`];
    }));
    return artifact;
  };
  const dynamicCt = (id, label, values, secret, kind, refs) => ({
    ...ctState(id, label, values, secret, kind, refs),
    layout: `THOR diagonal · ${segmentsPerCiphertext} segment(s) × ${D} slots`,
  });
  const tensors = (includeOutput = false) => ({
    left: { id: "A", label: "Plaintext A", shape: [M, K], values: A },
    right: { id: "B", label: "Encrypted B", shape: [K, N], values: B },
    ...(includeOutput ? { output: { id: "C", label: "Result C", shape: [M, N], values: C } } : {}),
  });

  const laneLimit = Math.min(D, 8);
  const laneIndices = Array.from({ length: laneLimit }, (_, i) => i);
  const laneNote = D > laneLimit
    ? [{ id: "lane-summary", type: "note", label: `${D - laneLimit} additional lanes`, text: `The canvas shows the first ${laneLimit} of ${D} padded diagonal lanes. All ${D} lanes are included in the verified computation.`, revealAt: 1 }]
    : [];
  const diagArtifacts = laneIndices.map((k) => graphVector({
    id: `diag-${k}`, label: `D${k}(A)`, values: diagonals[k], revealAt: 1, badge: "PT",
    meta: `A′[i,(i+${k}) mod ${D}]`, refs: diagSourceRefs[k].slice(0, D), tone: k,
  }));
  const packedDiagArtifacts = laneIndices.map((k) => graphVector({
    id: `packed-diag-${k}`, label: `Û${k}(A)`, values: packedDiagonals[k], revealAt: 2, badge: "PT",
    meta: `repeated across ${tileColumns} active column segment(s)`, refs: diagSourceRefs[k], tone: k,
  }));
  const packedBArtifact = graphVector({
    id: "packed-b", label: "ct.B tile 0", values: packedB, revealAt: 2, badge: "CT",
    meta: `columns 0…${tileColumns - 1}`, refs: rotationSourceRefs[0], tone: 0,
  });
  const rotationArtifacts = laneIndices.map((k) => graphVector({
    id: `rot-b-${k}`, label: k === 0 ? "ct.B" : `RotSeg${k}(ct.B)`, values: rotatedB[k], revealAt: 3,
    badge: "CT", meta: k === 0 ? "no rotation" : `ρ${k} + boundary mask`, refs: rotationSourceRefs[k], tone: k,
  }));
  const productArtifacts = laneIndices.map((k) => graphVector({
    id: `product-${k}`, label: `P${k} = Û${k} ⊙ RotSeg${k}(ct.B)`, values: products[k], revealAt: 4,
    badge: "CT", meta: "slot-wise PT×CT", refs: productSourceRefs[k], tone: k, kind: "product",
  }));
  const packedCArtifact = graphVector({
    id: "packed-c", label: "ct.C tile 0", values: packedC, revealAt: 5, badge: "CT",
    meta: `Σₖ Pk · ${tileColumns} output column segment(s)`, refs: resultSourceRefs, tone: 0, kind: "result",
  });
  const inputA = graphMatrix("A", "Plaintext matrix A", A, 0, false, `${M}×${K}`);
  const inputB = graphMatrix("B", "Encrypted matrix B", B, 0, true, `${K}×${N}`);
  const resultMatrix = graphMatrix("C", "Decoded result C", C, 6, true, `${M}×${N} · verified`);
  resultMatrix.rows.forEach((row, i) => row.forEach((cell, j) => {
    cell.provenance = [`C:${i}:${j}`, ...Array.from({ length: K }, (_, p) => [`A:${i}:${p}`, `B:${p}:${j}`]).flat()];
    cell.tone = j % 4;
  }));

  const graph = {
    id: `thor-pcmm-${M}x${K}x${N}-s${slots}`,
    algorithmId: "thor-pcmm",
    title: "THOR PC-MM · configurable diagonal packing",
    subtitle: `${M}×${K} · ${K}×${N}, padded diagonal width D=${D}, ${slots} slots/CT, ${tileCount} output tile(s).`,
    stages: [
      { id: "input", title: "Input matrices", caption: "A is plaintext and B is encrypted.", revealAt: 0, rows: [[inputA, operator("input-times", "×", "matrix multiplication", 0), inputB]] },
      { id: "diagonal", title: "1 · Extract cyclic diagonals", caption: `Pad A to ${D}×${D}, then extract Dₖ(A′)[i]=A′[i,(i+k) mod ${D}].`, revealAt: 1, rows: [...diagArtifacts.map((a) => [a]), ...laneNote.map((a) => [a])] },
      { id: "encode", title: "2 · Multi-diagonal encoding", caption: `Pack ${tileColumns} B column segment(s) in the representative ciphertext tile.`, revealAt: 2, rows: [...packedDiagArtifacts.map((a) => [a]), [packedBArtifact]] },
      { id: "rotate", title: "3 · Internal rotations", caption: `Rotate independently inside each ${D}-slot segment using HE rotations and masks.`, revealAt: 3, rows: rotationArtifacts.map((a) => [a]) },
      { id: "multiply", title: "4 · Parallel SIMD products", caption: `All ${D} diagonal lanes are multiplied; up to eight are expanded here.`, revealAt: 4, rows: productArtifacts.map((a) => [a]) },
      { id: "aggregate", title: "5 · Aggregation", caption: `Add ${D} partial ciphertexts for each of ${tileCount} output tile(s).`, revealAt: 5, rows: [[operator("sum", "Σ", `add P0…P${D - 1}`, 5), packedCArtifact]] },
      { id: "output", title: "6 · Reconstruct output", caption: `Crop padded rows and join ${tileCount} column tile(s) into C.`, revealAt: 6, rows: [[packedCArtifact, operator("decode", "→", "tile decode", 6), resultMatrix]] },
    ],
  };

  const visibleCtLanes = laneIndices;
  const total0 = emptyMetrics();
  const steps = [];
  steps.push(makeStep(0, {
    phase: "reference", operation: "reference", title: "Plaintext reference",
    description: `Start from configurable ${M}×${K} · ${K}×${N} PT×CT matrix multiplication.`,
    formula: `C = A · B,  A ∈ R${M}ˣ${K} (PT),  B ∈ R${K}ˣ${N} (CT)`, tensors: tensors(),
  }, total0));
  steps.push(makeStep(1, {
    phase: "diagonal", operation: "extract-diagonal", title: "Pad and extract cyclic diagonals",
    description: `A is zero-padded to ${D}×${D}; the canvas expands ${laneLimit} of ${D} lanes.`,
    formula: `Dₖ(A′)[i] = A′[i,(i+k) mod ${D}]`, tensors: tensors(),
    ciphertexts: visibleCtLanes.map((k) => dynamicCt(`diag-${k}`, `D${k}(A′)`, packedDiagonals[k], false, "tensor", diagSourceRefs[k])),
  }, steps.at(-1).metricsTotal));
  steps.push(makeStep(2, {
    phase: "encode", operation: "multi-diagonal-pack", title: "Pack diagonals and encrypted B tiles",
    description: `${columnsPerTile} B column(s) fit per ciphertext; ${tileCount} output tile(s) are required.`,
    formula: `columns/CT = floor(${slots}/${D}) = ${segmentsPerCiphertext}\noutput tiles = ceil(${N}/${columnsPerTile}) = ${tileCount}`, tensors: tensors(),
    ciphertexts: [...visibleCtLanes.map((k) => dynamicCt(`packed-diag-${k}`, `Û${k}(A′)`, packedDiagonals[k], false, "tensor", diagSourceRefs[k])), dynamicCt("packed-b", "ct.B tile 0", packedB, true, "tensor", rotationSourceRefs[0])],
    delta: { ciphertextCount: tileCount },
  }, steps.at(-1).metricsTotal));
  steps.push(makeStep(3, {
    phase: "rotate", operation: "internal-rotate", title: "Rotate inside packed B segments",
    description: "Logical segmented rotations are implemented by global rotations plus masks/corrections.",
    formula: `RotSegₖ(ct.B)[${D}j+i] = B[(i+k) mod ${D},j]`, tensors: tensors(),
    ciphertexts: visibleCtLanes.map((k) => dynamicCt(`rot-b-${k}`, k ? `RotSeg${k}(ct.B)` : "ct.B", rotatedB[k], true, "tensor", rotationSourceRefs[k])),
    delta: { rotations: (D - 1) * 2 * tileCount, masks: (D - 1) * 2 * tileCount, ciphertextCount: Math.max(0, D - 1) * tileCount },
  }, steps.at(-1).metricsTotal));
  steps.push(makeStep(4, {
    phase: "multiply", operation: "ct-pt-multiply", title: "Multiply diagonal lanes in parallel",
    description: `Each of ${D} padded diagonal lanes performs one slot-wise PT×CT multiplication per tile.`,
    formula: `Pₖ = Ûₖ(A′) ⊙ RotSegₖ(ct.B),  k = 0…${D - 1}`, tensors: tensors(),
    ciphertexts: visibleCtLanes.map((k) => dynamicCt(`product-${k}`, `P${k}`, products[k], true, "product", productSourceRefs[k])),
    delta: { ptCtMultiplications: D * tileCount, multiplicativeDepth: 1, ciphertextCount: D * tileCount },
  }, steps.at(-1).metricsTotal));
  steps.push(makeStep(5, {
    phase: "reduce", operation: "aggregate", title: "Aggregate partial ciphertexts",
    description: `All ${D} lane products are added for every output tile.`,
    formula: `ct.C[t] = Σₖ₌₀^${D - 1} Pₖ[t]`, tensors: tensors(),
    ciphertexts: [dynamicCt("packed-c", "ct.C tile 0", packedC, true, "partial-sum", resultSourceRefs)],
    delta: { additions: Math.max(0, D - 1) * tileCount, ciphertextCount: tileCount },
  }, steps.at(-1).metricsTotal));
  steps.push(makeStep(6, {
    phase: "output", operation: "decode", title: "Reconstruct and verify C",
    description: `Crop padded rows and concatenate ${tileCount} decoded output tile(s).`,
    formula: `C = ${JSON.stringify(C)}\nVerified: dynamic THOR trace equals direct A · B.`, tensors: tensors(true),
    ciphertexts: [dynamicCt("packed-c", "ct.C tile 0", packedC, true, "result", resultSourceRefs)],
  }, steps.at(-1).metricsTotal));

  const tiles = Array.from({ length: tileCount }, (_, jTile) => ({
    iTile: 0, jTile, rowRange: [0, M], colRange: [jTile * columnsPerTile, Math.min(N, (jTile + 1) * columnsPerTile)],
  }));
  return {
    schemaVersion: "1.0",
    id: `thor-pcmm-${M}x${K}x${N}-s${slots}`,
    title: "THOR PC-MM configurable teaching trace",
    parameters: { matrixShape: { m: M, k: K, n: N }, slotsPerCiphertext: slots, matmulType: "pt-ct", algorithmId: "thor-pcmm" },
    operands: {
      left: { id: "A", label: "Plaintext A", shape: [M, K], values: A, secret: false },
      right: { id: "B", label: "Encrypted B", shape: [K, N], values: B, secret: true },
    },
    output: { id: "C", label: "Result C", shape: [M, N] },
    plan: {
      algorithmId: "thor-pcmm", matmulType: "pt-ct",
      logicalShape: { m: M, k: K, n: N }, paddedShape: { m: D, k: D, n: nextPow2(N) },
      tileGrid: { m: 1, n: tileCount }, tileShape: { m: M, n: columnsPerTile },
      tiling: { nTileRows: 1, nTileCols: tileCount, tiles },
      packingStrategyRequested: options.packingStrategy || "auto",
      packingStrategyResolved: "thor-multi-diagonal",
      plannerSource: "thor-dynamic-browser",
      reductionStrategy: "internal-rotation-and-add",
      displayMode: D <= 8 && slots <= 32 ? "full" : tileCount <= 9 ? "tile" : "summary",
      warnings: D > laneLimit ? [`Story view expands ${laneLimit}/${D} lanes; the verified computation uses all lanes.`] : [],
    },
    graph, steps, finalTensor: C,
  };
}
