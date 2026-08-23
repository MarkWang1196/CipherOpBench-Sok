// MOAI matrix-multiplication toy traces.
//
// The three builders mirror the paper algorithms and the released C++:
//   Algorithm 2 / ct_pt_matrix_mul_wo_pre: Col -> Col, no rotations.
//   Algorithm 3/6 / ct_ct_matrix_mul_colpacking: Col x Col -> Diag.
//   Algorithm 4/7 / ct_ct_matrix_mul_diagpacking: Diag x Col -> Col (BSGS).

import { matmul, transpose } from "../../core/matrix.js";
import { parseAndValidateInputs, parseMatrixText } from "../../core/input_parser.js";
import {
  addVectors,
  ctState,
  makeSteps,
  matrixArtifact,
  mod,
  multiplyVectors,
  operator,
  rotateLeft,
  upperDiagonal,
  vectorArtifact,
} from "../thor/paper_utils.js";

export const MOAI_OPERATION_IDS = Object.freeze({
  QUERY: "moai-query",
  KEY: "moai-key",
  VALUE: "moai-value",
  SCORE: "moai-score",
  CONTEXT: "moai-context",
  OUTPUT: "moai-output",
  FC1: "moai-fc1",
  FC2: "moai-fc2",
  POOLER: "moai-pooler",
  CLASSIFIER: "moai-classifier",
});

const BATCHES = 2;
const SLOTS = 8;
const SUBSCRIPT = ["₀", "₁"];
const SUBSCRIPT_DIGITS = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"];

function subscript(value) {
  return String(value).split("").map((digit) => SUBSCRIPT_DIGITS[Number(digit)]).join("");
}

const X44 = Object.freeze([
  [1, 2, 0, 1],
  [0, 1, 3, 2],
  [2, 0, 1, 1],
  [1, 1, 2, 0],
]);

const W42 = Object.freeze([
  [1, 0],
  [0, 1],
  [2, 1],
  [1, 2],
]);

const Q42 = Object.freeze([
  [1, 2],
  [0, 1],
  [2, 0],
  [1, 1],
]);

const K42 = Object.freeze([
  [1, 0],
  [1, 2],
  [0, 1],
  [2, 1],
]);

const S44 = Object.freeze([
  [0.5, 0.25, 0.125, 0.125],
  [0.25, 0.5, 0.125, 0.125],
  [0.125, 0.125, 0.5, 0.25],
  [0.125, 0.125, 0.25, 0.5],
]);

const V42 = Object.freeze([
  [1, 0],
  [0, 1],
  [2, 1],
  [1, 2],
]);

function operation(id, config) {
  return Object.freeze({ id, method: "moai", slotsPerCiphertext: SLOTS, batchCount: BATCHES, ...config });
}

function linear(id, shortLabel, leftLabel, rightLabel, outputLabel, group) {
  return operation(id, {
    shortLabel,
    group,
    kind: "col-pt-col",
    formula: `${leftLabel} × ${rightLabel}`,
    leftLabel,
    rightLabel,
    outputLabel,
    leftSecret: true,
    rightSecret: false,
    A: X44,
    B: W42,
    strategy: "column",
  });
}

export const MOAI_OPERATIONS = Object.freeze({
  [MOAI_OPERATION_IDS.QUERY]: linear(MOAI_OPERATION_IDS.QUERY, "XWq", "X", "Wq", "Q", "attention"),
  [MOAI_OPERATION_IDS.KEY]: linear(MOAI_OPERATION_IDS.KEY, "XWk", "X", "Wk", "K", "attention"),
  [MOAI_OPERATION_IDS.VALUE]: linear(MOAI_OPERATION_IDS.VALUE, "XWv", "X", "Wv", "V", "attention"),
  [MOAI_OPERATION_IDS.SCORE]: operation(MOAI_OPERATION_IDS.SCORE, {
    shortLabel: "QKᵀ",
    group: "attention",
    kind: "col-col-diag",
    inputMode: "paired-columns",
    formula: "Q × Kᵀ",
    leftLabel: "Q",
    rightLabel: "K",
    outputLabel: "S",
    leftSecret: true,
    rightSecret: true,
    A: Q42,
    B: K42,
    strategy: "column",
  }),
  [MOAI_OPERATION_IDS.CONTEXT]: operation(MOAI_OPERATION_IDS.CONTEXT, {
    shortLabel: "SV",
    group: "attention",
    kind: "diag-col-col",
    formula: "S × V",
    leftLabel: "S",
    rightLabel: "V",
    outputLabel: "H",
    leftSecret: true,
    rightSecret: true,
    A: S44,
    B: V42,
    strategy: "column",
  }),
  [MOAI_OPERATION_IDS.OUTPUT]: linear(MOAI_OPERATION_IDS.OUTPUT, "HWo", "H", "Wo", "O", "attention"),
  [MOAI_OPERATION_IDS.FC1]: linear(MOAI_OPERATION_IDS.FC1, "FC1", "X", "W1", "F1", "ffn"),
  [MOAI_OPERATION_IDS.FC2]: linear(MOAI_OPERATION_IDS.FC2, "FC2", "F1", "W2", "F2", "ffn"),
  [MOAI_OPERATION_IDS.POOLER]: linear(MOAI_OPERATION_IDS.POOLER, "Pooler", "CLS", "Wp", "P", "output"),
  [MOAI_OPERATION_IDS.CLASSIFIER]: linear(MOAI_OPERATION_IDS.CLASSIFIER, "Classifier", "P", "Wc", "Y", "output"),
});

export function listMoaiOperations(group = null) {
  return Object.values(MOAI_OPERATIONS).filter((item) => !group || item.group === group);
}

export function getMoaiOperation(id) {
  return MOAI_OPERATIONS[id] || null;
}

function cloneMatrix(matrix) {
  return matrix.map((row) => [...row]);
}

function assertShape(matrix, rows, cols, label) {
  if (!Array.isArray(matrix) || matrix.length !== rows || matrix.some((row) => !Array.isArray(row) || row.length !== cols)) {
    const actual = Array.isArray(matrix) ? `${matrix.length}×${matrix[0]?.length || 0}` : "invalid";
    throw new Error(`${label} must be ${rows}×${cols} (got ${actual})`);
  }
}

function columns(matrix) {
  return Array.from({ length: matrix[0].length }, (_, col) => matrix.map((row) => Number(row[col])));
}

function interleaveBatch(values) {
  return values.flatMap((value) => Array.from({ length: BATCHES }, () => Number(value)));
}

function batchArtifact({ id, label, values, revealAt, tone, kind = "ciphertext" }) {
  const artifact = vectorArtifact({
    id,
    label,
    values: interleaveBatch(values),
    revealAt,
    badge: "CT",
    kind,
    tones: Array(SLOTS).fill(tone),
  });
  artifact.cells.forEach((cell, index) => {
    cell.label = `${cell.value}${SUBSCRIPT[index % BATCHES]}`;
  });
  return artifact;
}

function scalarArtifact(id, label, value, revealAt) {
  return vectorArtifact({
    id,
    label,
    values: [value],
    revealAt,
    badge: "PT",
    tones: [4],
  });
}

function columnPackedMatrix(id, label, values, revealAt, badge = "CT", toneOffset = 0, colorPlaintext = false) {
  return matrixArtifact({
    id,
    label,
    values,
    revealAt,
    badge,
    meta: "",
    toneForCell: (_row, col) => badge === "PT" && !colorPlaintext ? 4 : (col + toneOffset) % 4,
  });
}

function diagonalPackedMatrix(id, label, values, revealAt) {
  const size = values.length;
  return matrixArtifact({
    id,
    label,
    values,
    revealAt,
    badge: "CT",
    meta: "",
    toneForCell: (row, col) => mod(col - row, size) % 4,
  });
}

function artifactCiphertexts(stage) {
  const seen = new Set();
  const ciphertexts = [];
  for (const row of stage.rows || []) {
    for (const artifact of row) {
      if (artifact.badge !== "CT" || !artifact.cells || seen.has(artifact.id)) continue;
      seen.add(artifact.id);
      ciphertexts.push(ctState(
        artifact.id,
        artifact.label,
        artifact.cells.map((cell) => cell.value),
        SLOTS,
        true,
        artifact.type === "product" ? "product" : artifact.type === "result" ? "result" : "tensor"
      ));
    }
  }
  return ciphertexts;
}

function tensors(leftLabel, rightLabel, outputLabel, A, B, C, includeOutput = false) {
  return {
    left: { id: "A", label: leftLabel, shape: [A.length, A[0].length], values: A },
    right: { id: "B", label: rightLabel, shape: [B.length, B[0].length], values: B },
    ...(includeOutput ? { output: { id: "C", label: outputLabel, shape: [C.length, C[0].length], values: C } } : {}),
  };
}

function stepsFromGraph(descriptor, graph, A, validationB, C, deltas) {
  return makeSteps(descriptor.id, graph.stages.map((stage, index) => ({
    phase: index === 0 ? "reference" : index === graph.stages.length - 1 ? "output" : "multiply",
    title: stage.stepTitle || stage.title,
    description: stage.description || "",
    formula: "",
    tensors: tensors(descriptor.leftLabel, descriptor.kind === "col-col-diag" ? "Kᵀ" : descriptor.rightLabel, descriptor.outputLabel, A, validationB, C, index === graph.stages.length - 1),
    ciphertexts: artifactCiphertexts(stage),
    delta: deltas[index] || {},
  })));
}

function describeStages(graph, descriptions) {
  graph.stages.forEach((stage, index) => Object.assign(stage, descriptions[index] || {}));
}

function traceShell(descriptor, A, validationB, C, graph, steps, packingFlow) {
  const m = A.length;
  const k = A[0].length;
  const n = validationB[0].length;
  return {
    schemaVersion: "1.0",
    id: `${descriptor.id}-toy`,
    title: `ICLR'26 MOAI ${descriptor.shortLabel}`,
    parameters: {
      matrixShape: { m, k, n },
      slotsPerCiphertext: SLOTS,
      matmulType: descriptor.leftSecret && descriptor.rightSecret ? "ct-ct" : "ct-pt",
      algorithmId: descriptor.id,
      batchCount: BATCHES,
    },
    operands: {
      left: { id: "A", label: descriptor.leftLabel, shape: [m, k], values: A, secret: descriptor.leftSecret },
      right: {
        id: "B",
        label: descriptor.kind === "col-col-diag" ? "Kᵀ" : descriptor.rightLabel,
        shape: [validationB.length, validationB[0].length],
        values: validationB,
        secret: descriptor.rightSecret,
      },
    },
    output: { id: "C", label: descriptor.outputLabel, shape: [m, n] },
    plan: {
      algorithmId: descriptor.id,
      method: "moai",
      methodLabel: "ICLR'26 MOAI",
      visualStyle: "diagram-only",
      matmulType: descriptor.leftSecret && descriptor.rightSecret ? "ct-ct" : "ct-pt",
      logicalShape: { m, k, n },
      paddedShape: { m, k, n },
      tileGrid: { m: 1, n: 1 },
      tileShape: { m, n },
      packingStrategyRequested: packingFlow,
      packingStrategyResolved: packingFlow,
      plannerSource: "MOAI Algorithms 2-7",
      reductionStrategy: packingFlow,
      displayMode: "diagram-only",
      warnings: [],
    },
    graph,
    steps,
    finalTensor: C,
  };
}

function buildColPtCol(descriptor, A, W) {
  assertShape(A, 4, 4, descriptor.leftLabel);
  assertShape(W, 4, 2, descriptor.rightLabel);
  const C = matmul(A, W);
  const xCols = columns(A);
  const cCols = columns(C);
  const packedX = xCols.map((values, index) => batchArtifact({
    id: `${descriptor.id}-x-${index}`,
    label: `${descriptor.leftLabel}${subscript(index)}`,
    values,
    revealAt: 1,
    tone: index % 4,
  }));
  const productFamilies = Array.from({ length: 2 }, (_, outCol) =>
    xCols.map((values, inCol) => batchArtifact({
      id: `${descriptor.id}-p-${outCol}-${inCol}`,
      label: `p${outCol}${inCol}`,
      values: values.map((value) => Number(W[inCol][outCol]) * value),
      revealAt: 2 + outCol,
      // Each product is a scalar multiple of one encrypted X column. Keep
      // that ciphertext-column identity through every output projection.
      tone: inCol % 4,
      kind: "product",
    }))
  );
  const outputCols = cCols.map((values, index) => batchArtifact({
    id: `${descriptor.id}-out-${index}`,
    label: `${descriptor.outputLabel}${subscript(index)}`,
    values,
    revealAt: 4,
    tone: index % 4,
    kind: "result",
  }));

  const productRows = (outCol) => xCols.map((_values, inCol) => [
    scalarArtifact(`${descriptor.id}-w-${outCol}-${inCol}`, `w${inCol}${outCol}`, W[inCol][outCol], 2 + outCol),
    operator(`${descriptor.id}-mul-${outCol}-${inCol}`, "⊙", "", 2 + outCol),
    packedX[inCol],
    operator(`${descriptor.id}-eq-${outCol}-${inCol}`, "=", "", 2 + outCol),
    productFamilies[outCol][inCol],
  ]);

  const graph = {
    id: `${descriptor.id}-moai-alg2`,
    algorithmId: descriptor.id,
    title: descriptor.shortLabel,
    subtitle: "",
    stages: [
      { id: `${descriptor.id}-input`, title: `${descriptor.leftLabel} × ${descriptor.rightLabel}`, caption: "", revealAt: 0, rows: [[
        columnPackedMatrix("A", descriptor.leftLabel, A, 0),
        operator(`${descriptor.id}-times`, "×", "", 0),
        // Weight columns are plaintext, but retain their output-column identity
        // so the provenance remains visible across the matrix multiplication.
        columnPackedMatrix("B", descriptor.rightLabel, W, 0, "PT", 0, true),
      ]] },
      { id: `${descriptor.id}-col`, title: "Col", caption: "", revealAt: 1, rows: packedX.map((artifact) => [artifact]) },
      { id: `${descriptor.id}-y0`, title: `${descriptor.outputLabel}₀`, caption: "", revealAt: 2, rows: productRows(0) },
      { id: `${descriptor.id}-y1`, title: `${descriptor.outputLabel}₁`, caption: "", revealAt: 3, rows: productRows(1) },
      { id: `${descriptor.id}-sum`, title: "Σ", caption: "", revealAt: 4, rows: outputCols.map((artifact, outCol) => [
        ...productFamilies[outCol].flatMap((product, index) => index ? [operator(`${descriptor.id}-plus-${outCol}-${index}`, "+", "", 4), product] : [product]),
        operator(`${descriptor.id}-sum-eq-${outCol}`, "=", "", 4),
        artifact,
      ]) },
      { id: `${descriptor.id}-output`, title: descriptor.outputLabel, caption: "", revealAt: 5, rows: [
        [outputCols[0], operator(`${descriptor.id}-join`, "∥", "", 5), outputCols[1]],
        [operator(`${descriptor.id}-decode`, "↓", "", 5), columnPackedMatrix("C", descriptor.outputLabel, C, 5)],
      ] },
    ],
  };
  describeStages(graph, [
    { stepTitle: "Input encrypted X and plaintext W", shortTitle: "Input", description: `Start from encrypted ${descriptor.leftLabel} and plaintext ${descriptor.rightLabel}; compute ${descriptor.outputLabel} = ${descriptor.leftLabel}${descriptor.rightLabel}.` },
    { stepTitle: "Interleaved column-pack X", shortTitle: "Col pack", description: "Encrypt each input column separately, with the two toy batches interleaved across ciphertext slots." },
    { stepTitle: `Form ${descriptor.outputLabel} column 0`, shortTitle: `${descriptor.outputLabel}₀ products`, description: "Multiply every encrypted input column by the corresponding plaintext scalar from weight column 0." },
    { stepTitle: `Form ${descriptor.outputLabel} column 1`, shortTitle: `${descriptor.outputLabel}₁ products`, description: "Repeat the scalar ciphertext-plaintext products for weight column 1; no ciphertext rotation is needed." },
    { stepTitle: "Sum products by output column", shortTitle: "Column sums", description: "Add the weighted input columns to obtain one ciphertext for each output column, then add the plaintext bias in the full algorithm." },
    { stepTitle: "Return column-packed output", shortTitle: "Col output", description: `Keep ${descriptor.outputLabel} as interleaved column ciphertexts for the next MOAI module; decoding is shown only for verification.` },
  ]);
  const steps = stepsFromGraph(descriptor, graph, A, W, C, [
    {},
    { ciphertextCount: 4 },
    { ptCtMultiplications: 4, multiplicativeDepth: 1 },
    { ptCtMultiplications: 4 },
    { additions: 6, rescales: 2, ciphertextCount: 2 },
    {},
  ]);
  return traceShell(descriptor, A, W, C, graph, steps, "Col→Col");
}

function buildColColDiag(descriptor, Q, K) {
  assertShape(Q, 4, 2, "Q");
  assertShape(K, 4, 2, "K");
  const KT = transpose(K);
  const S = matmul(Q, KT);
  const qCols = columns(Q);
  const kCols = columns(K);
  const qPacked = qCols.map((values, index) => batchArtifact({ id: `${descriptor.id}-q-${index}`, label: `Q${SUBSCRIPT[index]}`, values, revealAt: 1, tone: index }));
  const kPacked = kCols.map((values, index) => batchArtifact({ id: `${descriptor.id}-k-${index}`, label: `K${SUBSCRIPT[index]}`, values, revealAt: 1, tone: index + 2 }));
  const rotatedK = Array.from({ length: 4 }, (_, diagonal) => kCols.map((values, index) => batchArtifact({
    id: `${descriptor.id}-rk-${diagonal}-${index}`,
    label: diagonal ? `ρ${diagonal * BATCHES}(K${SUBSCRIPT[index]})` : `K${SUBSCRIPT[index]}`,
    values: rotateLeft(values, diagonal),
    revealAt: 2,
    // Rotations move values inside K's column ciphertext; they do not change
    // which source column owns the color.
    tone: index + 2,
  })));
  const products = Array.from({ length: 4 }, (_, diagonal) => qCols.map((values, index) => batchArtifact({
    id: `${descriptor.id}-p-${diagonal}-${index}`,
    label: `P${diagonal}${index}`,
    values: values.map((value, token) => value * rotateLeft(kCols[index], diagonal)[token]),
    revealAt: 3,
    // A CT×CT product inherits the encrypted right-operand column identity.
    tone: index + 2,
    kind: "product",
  })));
  const diagonals = Array.from({ length: 4 }, (_, diagonal) => batchArtifact({
    id: `${descriptor.id}-diag-${diagonal}`,
    label: `D${diagonal}`,
    values: upperDiagonal(S, diagonal, 4),
    revealAt: 3,
    tone: diagonal,
    kind: "result",
  }));

  const graph = {
    id: `${descriptor.id}-moai-alg3-6`,
    algorithmId: descriptor.id,
    title: descriptor.shortLabel,
    subtitle: "",
    stages: [
      { id: `${descriptor.id}-input`, title: "Q × Kᵀ", caption: "", revealAt: 0, rows: [[
        columnPackedMatrix("Q", "Q", Q, 0),
        operator(`${descriptor.id}-times`, "×", "", 0),
        columnPackedMatrix("K", "K", K, 0, "CT", 2),
      ]] },
      { id: `${descriptor.id}-col`, title: "Col", caption: "", revealAt: 1, rows: [[qPacked[0], qPacked[1]], [kPacked[0], kPacked[1]]] },
      { id: `${descriptor.id}-rotate`, title: "ρ {0,2,4,6}", caption: "", revealAt: 2, rows: rotatedK.map((family) => family) },
      { id: `${descriptor.id}-dot`, title: "⊙ +", caption: "", revealAt: 3, rows: products.map((family, diagonal) => [
        family[0],
        operator(`${descriptor.id}-plus-${diagonal}`, "+", "", 3),
        family[1],
        operator(`${descriptor.id}-eq-${diagonal}`, "=", "", 3),
        diagonals[diagonal],
      ]) },
      { id: `${descriptor.id}-output`, title: "Diag", caption: "", revealAt: 4, rows: [
        [diagonals[0], operator(`${descriptor.id}-join-1`, "∥", "", 4), diagonals[1]],
        [diagonals[2], operator(`${descriptor.id}-join-3`, "∥", "", 4), diagonals[3]],
        [operator(`${descriptor.id}-decode`, "↓", "", 4), diagonalPackedMatrix("C", "S", S, 4)],
      ] },
    ],
  };
  describeStages(graph, [
    { stepTitle: "Input encrypted Q and K", shortTitle: "Q × Kᵀ", description: "Both matrices arrive in MOAI's interleaved column packing; K is not explicitly transposed." },
    { stepTitle: "Column-pack Q and K", shortTitle: "Col inputs", description: "Represent each feature column of Q and K with one interleaved ciphertext." },
    { stepTitle: "Rotate K for every score diagonal", shortTitle: "Rotate K", description: "Rotate each K column by the interleaved offsets for diagonals 0 through 3, preserving each batched sequence." },
    { stepTitle: "Multiply and sum feature columns", shortTitle: "Products + sum", description: "For each output diagonal, multiply Q columns with the matching rotated K columns and add across the feature dimension." },
    { stepTitle: "Return diagonal-packed scores", shortTitle: "Diag output", description: "The four ciphertexts encode the four diagonals of QKᵀ, directly matching MOAI's diagonal-packed Softmax input." },
  ]);
  const steps = stepsFromGraph(descriptor, graph, Q, KT, S, [
    {},
    { ciphertextCount: 4 },
    { rotations: 6 },
    { ctCtMultiplications: 8, additions: 4, relinearizations: 4, rescales: 4, multiplicativeDepth: 1, ciphertextCount: 4 },
    {},
  ]);
  return traceShell(descriptor, Q, KT, S, graph, steps, "Col×Col→Diag");
}

function buildDiagColCol(descriptor, S, V) {
  assertShape(S, 4, 4, "S");
  assertShape(V, 4, 2, "V");
  const H = matmul(S, V);
  const diagonals = Array.from({ length: 4 }, (_, diagonal) => upperDiagonal(S, diagonal, 4));
  const vCols = columns(V);
  const cPacked = diagonals.map((values, index) => batchArtifact({ id: `${descriptor.id}-c-${index}`, label: `D${index}`, values, revealAt: 1, tone: index }));
  const vPacked = vCols.map((values, index) => batchArtifact({ id: `${descriptor.id}-v-${index}`, label: `V${SUBSCRIPT[index]}`, values, revealAt: 1, tone: index + 2 }));
  const beta = vCols.map((values, col) => [0, 1].map((step) => batchArtifact({
    id: `${descriptor.id}-beta-${col}-${step}`,
    label: step ? `ρ${step * BATCHES}(V${SUBSCRIPT[col]})` : `V${SUBSCRIPT[col]}`,
    values: rotateLeft(values, step),
    revealAt: 2,
    // Baby-step rotations preserve V's source column identity.
    tone: col + 2,
  })));
  const alignedC = diagonals.map((values, index) => {
    const alpha = Math.floor(index / 2);
    const shift = alpha ? 2 : 0;
    return batchArtifact({
      id: `${descriptor.id}-aligned-${index}`,
      label: shift ? `ρ${shift * BATCHES}(D${index})` : `D${index}`,
      values: rotateLeft(values, shift),
      revealAt: 3,
      tone: index,
    });
  });
  const products = vCols.map((_values, col) => diagonals.map((_diag, index) => {
    const r = index % 2;
    const alpha = Math.floor(index / 2);
    const aligned = rotateLeft(diagonals[index], alpha ? 2 : 0);
    const betaValues = rotateLeft(vCols[col], r);
    return batchArtifact({
      id: `${descriptor.id}-p-${col}-${index}`,
      label: `P${col}${index}`,
      values: multiplyVectors(aligned, betaValues),
      revealAt: 4,
      // The product carries the encrypted V column through the diagonal lane.
      tone: col + 2,
      kind: "product",
    });
  }));
  const groups = products.map((family, col) => [0, 1].map((alpha) => {
    // The paper's giant step (Alg. 7) runs over all interleaved batches; sum
    // both batch-0 and batch-1 slots and keep the full 8-slot interleaved form.
    const raw = addVectors(family.slice(alpha * 2, alpha * 2 + 2).map((artifact) => artifact.cells.map((cell) => cell.value)));
    const values = alpha ? rotateLeft(raw, 2 * BATCHES) : raw;
    const artifact = vectorArtifact({
      id: `${descriptor.id}-g-${col}-${alpha}`,
      label: alpha ? `ρ${2 * BATCHES}(G${col}${alpha})` : `G${col}${alpha}`,
      values,
      revealAt: 5,
      badge: "CT",
      tones: Array(SLOTS).fill(col + 2),
      kind: "product",
    });
    artifact.cells.forEach((cell, index) => { cell.label = `${cell.value}${SUBSCRIPT[index % BATCHES]}`; });
    return artifact;
  }));
  const outCols = columns(H).map((values, col) => batchArtifact({
    id: `${descriptor.id}-out-${col}`,
    label: `H${SUBSCRIPT[col]}`,
    values,
    revealAt: 5,
    tone: col + 2,
    kind: "result",
  }));

  const graph = {
    id: `${descriptor.id}-moai-alg4-7`,
    algorithmId: descriptor.id,
    title: descriptor.shortLabel,
    subtitle: "",
    stages: [
      { id: `${descriptor.id}-input`, title: "S × V", caption: "", revealAt: 0, rows: [[
        diagonalPackedMatrix("A", "S", S, 0),
        operator(`${descriptor.id}-times`, "×", "", 0),
        columnPackedMatrix("B", "V", V, 0, "CT", 2),
      ]] },
      { id: `${descriptor.id}-pack`, title: "Diag ∥ Col", caption: "", revealAt: 1, rows: [
        [cPacked[0], cPacked[1]],
        [cPacked[2], cPacked[3]],
        [...vPacked],
      ] },
      { id: `${descriptor.id}-baby`, title: "Baby ρ {0,2}", caption: "", revealAt: 2, rows: beta.map((family) => family) },
      { id: `${descriptor.id}-align`, title: "ρ(D₂,D₃)·4", caption: "", revealAt: 3, rows: [
        [alignedC[0], alignedC[1]],
        [alignedC[2], alignedC[3]],
      ] },
      { id: `${descriptor.id}-products`, title: "⊙", caption: "", revealAt: 4, rows: products.flatMap((family) => [family.slice(0, 2), family.slice(2, 4)]) },
      { id: `${descriptor.id}-sum`, title: "Σ + Giant ρ·4", caption: "", revealAt: 5, rows: groups.map((family, col) => [
        family[0],
        operator(`${descriptor.id}-plus-${col}`, "+", "", 5),
        family[1],
        operator(`${descriptor.id}-eq-${col}`, "=", "", 5),
        outCols[col],
      ]) },
      { id: `${descriptor.id}-output`, title: "Col", caption: "", revealAt: 6, rows: [
        [outCols[0], operator(`${descriptor.id}-join`, "∥", "", 6), outCols[1]],
        [operator(`${descriptor.id}-decode`, "↓", "", 6), columnPackedMatrix("C", "H", H, 6, "CT", 2)],
      ] },
    ],
  };
  describeStages(graph, [
    { stepTitle: "Input diagonal S and column-packed V", shortTitle: "S × V", description: "Softmax scores S remain diagonally packed, while V arrives in interleaved column packing." },
    { stepTitle: "Keep Diag and Col operand layouts", shortTitle: "Diag + Col", description: "Use the score diagonals and V columns directly, without a packing-format conversion." },
    { stepTitle: "Baby-step rotations of V", shortTitle: "Baby rotate", description: "Rotate each V column by the baby-step offsets so it can pair with both diagonals inside each BSGS group." },
    { stepTitle: "Align score diagonals for giant steps", shortTitle: "Align diagonals", description: "Pre-rotate the second diagonal group so its later giant-step rotation lands in the correct output slots." },
    { stepTitle: "Multiply aligned S and V lanes", shortTitle: "Lane products", description: "Multiply every aligned score diagonal by its matching baby-step V ciphertext." },
    { stepTitle: "Sum baby groups and giant rotations", shortTitle: "BSGS sum", description: "Add within each baby-step group, apply the giant-step rotation to the second group, and combine both groups per output column." },
    { stepTitle: "Return column-packed context", shortTitle: "Col output", description: "The context matrix is already in interleaved column packing for the following projection; decoding is visualization-only." },
  ]);
  const steps = stepsFromGraph(descriptor, graph, S, V, H, [
    {},
    { ciphertextCount: 6 },
    { rotations: 2 },
    { rotations: 2 },
    { ctCtMultiplications: 8, additions: 4, relinearizations: 4, rescales: 4, multiplicativeDepth: 1 },
    { rotations: 2, additions: 2, ciphertextCount: 2 },
    {},
  ]);
  return traceShell(descriptor, S, V, H, graph, steps, "Diag×Col→Col");
}

export function buildMoaiOperationTrace(id, options = {}) {
  const descriptor = getMoaiOperation(id);
  if (!descriptor) throw new Error(`Unknown MOAI operation: ${id}`);
  if (options.leftSecret != null && Boolean(options.leftSecret) !== descriptor.leftSecret) {
    throw new Error(`${descriptor.shortLabel} requires ${descriptor.leftSecret ? "ciphertext" : "plaintext"} A`);
  }
  if (options.rightSecret != null && Boolean(options.rightSecret) !== descriptor.rightSecret) {
    throw new Error(`${descriptor.shortLabel} requires ${descriptor.rightSecret ? "ciphertext" : "plaintext"} B`);
  }
  if (options.slotsPerCiphertext != null && Number(options.slotsPerCiphertext) !== SLOTS) {
    throw new Error("MOAI toy batching uses exactly 8 slots (4 tokens × 2 interleaved inputs)");
  }
  const A = options.A || descriptor.A;
  const B = options.B || descriptor.B;
  if (descriptor.kind === "col-pt-col") return buildColPtCol(descriptor, A, B);
  if (descriptor.kind === "col-col-diag") return buildColColDiag(descriptor, A, B);
  return buildDiagColCol(descriptor, A, B);
}

export function moaiOperationDefaults(id) {
  const descriptor = getMoaiOperation(id);
  if (!descriptor) return null;
  const paired = descriptor.inputMode === "paired-columns";
  return {
    m: descriptor.A.length,
    k: descriptor.A[0].length,
    n: paired ? descriptor.A.length : descriptor.B[0].length,
    A: cloneMatrix(descriptor.A),
    B: cloneMatrix(descriptor.B),
    leftSecret: descriptor.leftSecret,
    rightSecret: descriptor.rightSecret,
    slotsPerCiphertext: SLOTS,
    strategy: descriptor.strategy,
  };
}

export function parseMoaiOperationInputs(raw, descriptor) {
  if (descriptor.inputMode !== "paired-columns") return parseAndValidateInputs(raw);
  const errors = {};
  let A;
  let B;
  try {
    A = parseMatrixText(raw.valuesTextA);
    B = parseMatrixText(raw.valuesTextB);
    assertShape(A, descriptor.A.length, descriptor.A[0].length, descriptor.leftLabel);
    assertShape(B, descriptor.B.length, descriptor.B[0].length, descriptor.rightLabel);
  } catch (error) {
    errors.values = error.message;
  }
  if (Number(raw.slots) !== SLOTS) errors.slots = "MOAI toy batching uses exactly 8 slots";
  if (Object.keys(errors).length) return { ok: false, errors, inputs: null };
  return {
    ok: true,
    errors: {},
    inputs: {
      m: descriptor.A.length,
      k: descriptor.A[0].length,
      n: descriptor.A.length,
      A,
      B,
      leftSecret: String(raw.leftType).toLowerCase() === "ct",
      rightSecret: String(raw.rightType).toLowerCase() === "ct",
      slotsPerCiphertext: SLOTS,
      strategy: "column",
      customLayouts: null,
      valueMode: "manual",
      seed: raw.seed,
    },
  };
}
