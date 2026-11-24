// Euston matrix-multiplication traces.
//
// The four primitive builders mirror Figure 2 and the released CPU/GPU code:
//   PCCDMM: plaintext column-packed U x encrypted diagonal D -> column-packed E
//   CCPUMM: ciphertext column-packed E x plaintext unpacked H -> column-packed R
//   CCCRMM: ciphertext column-packed Q x row-packed K^T -> diagonal-packed P
//   CDCCMM: ciphertext diagonal-packed P x column-packed V -> column-packed Y
// Batch slots are contiguous m-element components, as required by Algorithm 1.

import { matmul, transpose } from "../../core/matrix.js";
import { parseAndValidateInputs } from "../../core/input_parser.js";
import { addVectors, ctState, makeSteps, matrixArtifact, mod, multiplyVectors, operator, rotateLeft, vectorArtifact } from "../thor/paper_utils.js";

export const EUSTON_OPERATION_IDS = Object.freeze({
  PCCDMM: "euston-pccdmm",
  CCPUMM: "euston-ccpumm",
  CCCRMM: "euston-cccrmm",
  CDCCMM: "euston-cdccmm",
  QUERY: "euston-query",
  KEY: "euston-key",
  VALUE: "euston-value",
  SCORE: "euston-score",
  CONTEXT: "euston-context",
  OUTPUT: "euston-output",
  FC1: "euston-fc1",
  FC2: "euston-fc2",
});

const BATCHES = 2;
const ROWS = 2;
const ACTIVE_SLOTS = BATCHES * ROWS;
const SLOTS = 8;
const SUB = ["₀", "₁"];

const X22 = Object.freeze([[1, 3], [2, 0]]);
const W22 = Object.freeze([[2, 4], [3, 1]]);
const W24 = Object.freeze([[1, 0, 2, 3], [2, 1, 3, 0]]);
const X24 = Object.freeze([[1, 3, 0, 2], [4, 0, 2, 1]]);
const W42 = Object.freeze([[1, 0], [0, 2], [3, 1], [2, 1]]);
const Q22 = Object.freeze([[2, 1], [1, 0]]);
const K22 = Object.freeze([[1, 4], [3, 2]]);
const S22 = Object.freeze([[0.75, 0.25], [0.375, 0.625]]);
const V22 = Object.freeze([[2, 1], [1, 3]]);
const U22 = Object.freeze([[1, 2], [3, 0]]);
const D22 = Object.freeze([[2, 0], [0, 4]]);

function operation(id, config) {
  return Object.freeze({ id, method: "euston", slotsPerCiphertext: SLOTS, batchCount: BATCHES, ...config });
}

function iwmm(id, shortLabel, rightLabel, outputLabel) {
  return operation(id, {
    shortLabel, group: "attention", kind: "iwmm", leftLabel: "A", rightLabel,
    outputLabel, leftSecret: true, rightSecret: false, A: X22, B: W22, strategy: "column",
  });
}

function cpmm(id, shortLabel, leftLabel, rightLabel, outputLabel, group, A, B) {
  return operation(id, {
    shortLabel, group, kind: "ccpumm", leftLabel, rightLabel, outputLabel,
    leftSecret: true, rightSecret: false, A, B, strategy: "column",
  });
}

export const EUSTON_OPERATIONS = Object.freeze({
  [EUSTON_OPERATION_IDS.PCCDMM]: operation(EUSTON_OPERATION_IDS.PCCDMM, {
    shortLabel: "PCCDMM", group: "paper", kind: "pccdmm", leftLabel: "U", rightLabel: "D",
    outputLabel: "E", leftSecret: false, rightSecret: true, A: U22, B: D22, strategy: "column",
  }),
  [EUSTON_OPERATION_IDS.CCPUMM]: cpmm(EUSTON_OPERATION_IDS.CCPUMM, "CCPUMM", "E", "H", "R", "paper", X22, W22),
  [EUSTON_OPERATION_IDS.CCCRMM]: operation(EUSTON_OPERATION_IDS.CCCRMM, {
    shortLabel: "CCCRMM", group: "paper", kind: "cccrmm", leftLabel: "Q", rightLabel: "K",
    outputLabel: "P", leftSecret: true, rightSecret: true, A: Q22, B: K22, strategy: "column",
  }),
  [EUSTON_OPERATION_IDS.CDCCMM]: operation(EUSTON_OPERATION_IDS.CDCCMM, {
    shortLabel: "CDCCMM", group: "paper", kind: "cdccmm", leftLabel: "P", rightLabel: "V",
    outputLabel: "Y", leftSecret: true, rightSecret: true, A: S22, B: V22, strategy: "column",
  }),
  [EUSTON_OPERATION_IDS.QUERY]: iwmm(EUSTON_OPERATION_IDS.QUERY, "XWq", "Wq", "Q"),
  [EUSTON_OPERATION_IDS.KEY]: iwmm(EUSTON_OPERATION_IDS.KEY, "XWk", "Wk", "K"),
  [EUSTON_OPERATION_IDS.VALUE]: iwmm(EUSTON_OPERATION_IDS.VALUE, "XWv", "Wv", "V"),
  [EUSTON_OPERATION_IDS.SCORE]: operation(EUSTON_OPERATION_IDS.SCORE, {
    shortLabel: "QKᵀ", group: "attention", kind: "cccrmm", leftLabel: "Q", rightLabel: "K",
    outputLabel: "P", leftSecret: true, rightSecret: true, A: Q22, B: K22, strategy: "column",
  }),
  [EUSTON_OPERATION_IDS.CONTEXT]: operation(EUSTON_OPERATION_IDS.CONTEXT, {
    shortLabel: "Softmax·V", group: "attention", kind: "cdccmm", leftLabel: "Softmax(P)", rightLabel: "V",
    outputLabel: "Y", leftSecret: true, rightSecret: true, A: S22, B: V22, strategy: "column",
  }),
  [EUSTON_OPERATION_IDS.OUTPUT]: cpmm(EUSTON_OPERATION_IDS.OUTPUT, "YWo", "Y", "Wo", "O", "attention", X22, W22),
  [EUSTON_OPERATION_IDS.FC1]: cpmm(EUSTON_OPERATION_IDS.FC1, "FC1", "X", "W1", "F1", "ffn", X22, W24),
  [EUSTON_OPERATION_IDS.FC2]: cpmm(EUSTON_OPERATION_IDS.FC2, "FC2", "GELU(F1)", "W2", "F2", "ffn", X24, W42),
});

export function listEustonOperations(group = null) {
  return Object.values(EUSTON_OPERATIONS).filter((item) => !group || item.group === group);
}

export function getEustonOperation(id) { return EUSTON_OPERATIONS[id] || null; }
function clone(matrix) { return matrix.map((row) => [...row]); }
function assertRect(matrix, rows, cols, label) {
  if (!Array.isArray(matrix) || matrix.length !== rows || matrix.some((row) => !Array.isArray(row) || row.length !== cols)) {
    throw new Error(`${label} must be ${rows}×${cols}`);
  }
}
function cols(matrix) { return Array.from({ length: matrix[0].length }, (_, j) => matrix.map((row) => Number(row[j]))); }
function batch(values) { return [...values, ...values]; }
function strip({ id, label, values, revealAt, tone = 0, badge = "CT", kind = "ciphertext", batched = true }) {
  const active = batched ? batch(values) : values;
  const slots = Array.from({ length: SLOTS }, (_, i) => Number(active[i] ?? 0));
  // The block color denotes the packed column/diagonal identity, regardless
  // of whether the operand is plaintext or ciphertext.
  const artifact = vectorArtifact({ id, label, values: slots, revealAt, badge, kind, tones: Array(slots.length).fill(tone) });
  artifact.cells.forEach((cell, i) => {
    if (i >= active.length) cell.label = "·";
    else if (i < ACTIVE_SLOTS) cell.sampleLane = Math.floor(i / ROWS);
  });
  return artifact;
}
function matrix(id, label, values, revealAt, badge = "CT", layout = "column", toneOffset = 0) {
  return matrixArtifact({ id, label, values, revealAt, badge, meta: "",
    toneForCell: (row, col) => layout === "diagonal" ? mod(col - row, values.length)
      : mod((layout === "row" ? row : col) + toneOffset, 4) });
}
const valuesOf = (artifact) => artifact.cells.map((cell) => cell.value);

function componentRotations(input, prefix, revealAt) {
  const masks = [[1, 0], [0, 1]].map((mask, i) =>
    strip({ id: `${prefix}-mask-${i}`, label: i ? "Wrap mask" : "In-block mask", values: mask, badge: "PT", revealAt }));
  const rows = [];
  const shifted = input.map((column, i) => {
    const parts = [1, -1].map((shift, j) => {
      const rotatedCells = rotateLeft(column.cells, shift);
      const rotated = strip({ id: `${prefix}-rot-${i}-${j}`, label: `Rot${shift > 0 ? "L" : "R"}₁(${column.label})`,
        values: rotatedCells.map((cell) => cell.value),
        revealAt, tone: column.cells[0].tone, batched: false });
      rotated.cells.forEach((cell, slot) => {
        cell.label = rotatedCells[slot].label;
        cell.sampleLane = rotatedCells[slot].sampleLane;
      });
      const masked = strip({ id: `${prefix}-part-${i}-${j}`, label: `Part ${j + 1}`,
        values: multiplyVectors(valuesOf(rotated), valuesOf(masks[j])), revealAt, tone: column.cells[0].tone, batched: false });
      rows.push([column, operator(`${rotated.id}-op`, "→", "", revealAt), rotated,
        operator(`${masked.id}-op`, "⊙", "", revealAt), masks[j], operator(`${masked.id}-eq`, "=", "", revealAt), masked]);
      return masked;
    });
    const result = strip({ id: `${prefix}-result-${i}`, label: `${column.label}⇐1`,
      values: addVectors(parts.map(valuesOf)).slice(0, ACTIVE_SLOTS), revealAt, tone: column.cells[0].tone, batched: false });
    rows.push([parts[0], operator(`${result.id}-plus`, "+", "", revealAt), parts[1], operator(`${result.id}-eq`, "=", "", revealAt), result]);
    return result;
  });
  return { rows, shifted };
}
function artifactCiphertexts(stage) {
  const result = [], seen = new Set();
  for (const row of stage.rows || []) for (const a of row) {
    if (a.badge !== "CT" || !a.cells || a.coefficients || seen.has(a.id)) continue;
    seen.add(a.id);
    result.push(ctState(a.id, a.label, a.cells.map((cell) => cell.value), SLOTS, true,
      a.type === "product" ? "product" : a.type === "result" ? "result" : "tensor"));
  }
  return result;
}
function tensorState(d, A, validationB, C, output = false) {
  return {
    left: { id: "A", label: d.leftLabel, shape: [A.length, A[0].length], values: A },
    right: { id: "B", label: d.kind === "cccrmm" ? "Kᵀ" : d.rightLabel,
      shape: [validationB.length, validationB[0].length], values: validationB },
    ...(output ? { output: { id: "C", label: d.outputLabel, shape: [C.length, C[0].length], values: C } } : {}),
  };
}
function graphSteps(d, graph, A, validationB, C, deltas) {
  return makeSteps(d.id, graph.stages.map((stage, i) => ({
    phase: i === 0 ? "reference" : i === graph.stages.length - 1 ? "output" : "multiply",
    title: stage.stepTitle, description: stage.description, formula: "",
    tensors: tensorState(d, A, validationB, C, i === graph.stages.length - 1),
    ciphertexts: artifactCiphertexts(stage), delta: deltas[i] || {},
  })));
}
function shell(d, A, validationB, C, graph, steps, flow) {
  const m = A.length, k = A[0].length, n = validationB[0].length;
  const matmulType = d.rightSecret ? (d.leftSecret ? "ct-ct" : "pt-ct") : "ct-pt";
  return {
    schemaVersion: "1.0", id: `${d.id}-toy`, title: `S&P'26 Euston ${d.shortLabel}`,
    parameters: { matrixShape: { m, k, n }, slotsPerCiphertext: SLOTS,
      matmulType,
      algorithmId: d.id, batchCount: BATCHES },
    operands: {
      left: { id: "A", label: d.leftLabel, shape: [m, k], values: A, secret: d.leftSecret },
      right: { id: "B", label: d.kind === "cccrmm" ? "Kᵀ" : d.rightLabel,
        shape: [validationB.length, validationB[0].length], values: validationB, secret: d.rightSecret },
    },
    output: { id: "C", label: d.outputLabel, shape: [m, n] },
    plan: { algorithmId: d.id, method: "euston", methodLabel: "S&P'26 Euston", visualStyle: "diagram-only",
      matmulType, logicalShape: { m, k, n },
      paddedShape: { m, k, n }, tileGrid: { m: 1, n: 1 }, tileShape: { m, n },
      packingStrategyRequested: flow, packingStrategyResolved: flow, plannerSource: "Euston Sections 4.1-4.2 and Figure 2",
      reductionStrategy: flow, displayMode: "diagram-only", batchLabel: "samples", warnings: [] },
    graph, steps, finalTensor: C,
  };
}

function outputStage(d, C, packed, revealAt, layout = "column") {
  return { id: `${d.id}-output`, title: d.outputLabel, stepTitle: `Return ${layout}-packed ${d.outputLabel}`,
    shortTitle: "Output", description: `Read the ${layout}s from the first batch component to verify ${d.outputLabel}.`, revealAt,
    rows: [packed, [operator(`${d.id}-down`, "↓", "", revealAt), matrix("C", d.outputLabel, C, revealAt, "CT", layout)]] };
}

function packedDiagonal(diagonal, id, revealAt) {
  return { ...vectorArtifact({ id, label: "E(f_d(x))", values: batch(diagonal), revealAt, badge: "CT",
    meta: "Polynomial coefficients", tones: Array(ACTIVE_SLOTS).fill(0) }), coefficients: true };
}

function expandDiagonal(compressed, prefix, revealAt) {
  const masks = Array.from({ length: BATCHES }, (_, b) => strip({
    id: `${prefix}-sample-${b}`, label: `s${SUB[b]}`, revealAt, badge: "PT", batched: false,
    values: Array.from({ length: ACTIVE_SLOTS }, (_, slot) => Number(Math.floor(slot / ROWS) === b)),
  }));
  const rows = [];
  const diagonal = Array.from({ length: ROWS }, (_, col) => {
    const parts = masks.map((mask, b) => {
      const scalar = compressed.cells[b * ROWS + col].value;
      const replicated = strip({ id: `${prefix}-d-${b}-${col}`, label: `d${SUB[b]},${col + 1}`,
        values: Array(SLOTS).fill(scalar), revealAt, tone: col, batched: false });
      const masked = strip({ id: `${replicated.id}-masked`, label: `d${SUB[b]},${col + 1} ⊙ s${SUB[b]}`,
        values: multiplyVectors(valuesOf(replicated), valuesOf(mask)).slice(0, ACTIVE_SLOTS),
        revealAt, tone: col, batched: false });
      rows.push([replicated, operator(`${replicated.id}-mul`, "⊙", "", revealAt), mask,
        operator(`${replicated.id}-eq`, "=", "", revealAt), masked]);
      return masked;
    });
    const combined = strip({ id: `${prefix}-diagonal-${col}`, label: `d${col + 1}`,
      values: addVectors(parts.map(valuesOf)).slice(0, ACTIVE_SLOTS), revealAt, tone: col, batched: false });
    rows.push([parts[0], operator(`${combined.id}-sum`, "+", "", revealAt), parts[1],
      operator(`${combined.id}-eq`, "=", "", revealAt), combined]);
    return combined;
  });
  return { rows, diagonal };
}

function buildPccdmm(d, U, D) {
  assertRect(U, 2, 2, d.leftLabel); assertRect(D, 2, 2, d.rightLabel);
  if (Number(D[0][1]) !== 0 || Number(D[1][0]) !== 0) throw new Error("D must remain diagonal for PCCDMM");
  const C = matmul(U, D);
  const compressed = packedDiagonal([D[0][0], D[1][1]], `${d.id}-d`, 1);
  const { rows: expansionRows, diagonal } = expandDiagonal(compressed, d.id, 2);
  const uPts = cols(U).map((values, i) => strip({ id: `${d.id}-u-${i}`, label: `u${i + 1}`, values, revealAt: 3, tone: i, badge: "PT" }));
  const products = uPts.map((u, i) => strip({ id: `${d.id}-e-${i}`, label: `e${i + 1}`,
    values: multiplyVectors(valuesOf(u), valuesOf(diagonal[i])).slice(0, ACTIVE_SLOTS), revealAt: 3, tone: i, kind: "result", batched: false }));
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: `${d.leftLabel} × ${d.rightLabel}`, stepTitle: "Input plaintext U and diagonal D", shortTitle: "U × D",
      description: "Set up the first offline multiplication, where each U column is scaled by one encrypted diagonal entry.", revealAt: 0,
      rows: [[matrix("A", d.leftLabel, U, 0, "PT", "column"), operator(`${d.id}-times`, "×", "", 0), matrix("B", d.rightLabel, D, 0, "CT", "diagonal")]] },
    { id: `${d.id}-seccomp`, title: "SecComp", stepTitle: "Compress the two diagonal vectors", shortTitle: "Compress D",
      description: "Encode each sample’s diagonal entries as coefficients of one encrypted polynomial.", revealAt: 1,
      rows: [[matrix("B2", d.rightLabel, D, 1, "CT", "diagonal"), operator(`${d.id}-comp`, "→", "", 1), compressed]] },
    { id: `${d.id}-secdecomp`, title: "SecDecomp", stepTitle: "Expand and select each sample’s diagonal", shortTitle: "Expand D",
      description: "Decompress each coefficient to a repeated scalar, then mask and add to keep it in the matching sample’s slots.", revealAt: 2,
      rows: [[compressed], ...expansionRows] },
    { id: `${d.id}-multiply`, title: "PCCDMM", stepTitle: "Multiply U columns by the diagonal", shortTitle: "Scale U",
      description: "Multiply each packed plaintext column by the expanded encrypted entry for that column.", revealAt: 3,
      rows: products.map((p, i) => [uPts[i], operator(`${d.id}-mul-${i}`, "⊙", "", 3), diagonal[i], operator(`${d.id}-eq-${i}`, "=", "", 3), p]) },
    outputStage(d, transpose(products.map((p) => valuesOf(p).slice(0, ROWS))), products, 4),
  ] };
  const steps = graphSteps(d, graph, U, D, C, [{}, { ciphertextCount: 1 },
    { masks: 4, ptCtMultiplications: 4, additions: 2, multiplicativeDepth: 1 },
    { ptCtMultiplications: 2, rescales: 2, multiplicativeDepth: 1 }, {}]);
  return shell(d, U, D, C, graph, steps, "column-packed × encrypted diagonal");
}

function buildCcpumm(d, A, B) {
  assertRect(A, 2, d.A[0].length, d.leftLabel); assertRect(B, d.B.length, d.B[0].length, d.rightLabel);
  const C = matmul(A, B), aCols = cols(A);
  const aCts = aCols.map((values, i) => strip({ id: `${d.id}-a-${i}`, label: `${d.leftLabel}${i + 1}`, values, revealAt: 1, tone: i }));
  const productsByOutput = cols(B).map((_, j) => aCols.map((values, i) => strip({ id: `${d.id}-p-${j}-${i}`, label: `p${j + 1}${i + 1}`,
    values: values.map((value) => value * Number(B[i][j])), revealAt: 2, tone: i, kind: "product" })));
  const outputs = productsByOutput.map((products, j) => strip({ id: `${d.id}-c-${j}`, label: `${d.outputLabel}${j + 1}`,
    values: addVectors(products.map(valuesOf)).slice(0, ACTIVE_SLOTS), revealAt: 3, tone: j, kind: "result", batched: false }));
  const multiplyRows = [];
  for (let j = 0; j < outputs.length; j++) for (let i = 0; i < aCols.length; i++) {
    const scalar = strip({ id: `${d.id}-w-${j}-${i}`, label: `${d.rightLabel}${i + 1}${j + 1}`, values: [B[i][j], B[i][j]], revealAt: 2, badge: "PT" });
    multiplyRows.push([aCts[i], operator(`${d.id}-mul-${j}-${i}`, "⊙", "", 2), scalar,
      operator(`${d.id}-eq-${j}-${i}`, "=", "", 2), productsByOutput[j][i]]);
  }
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: `${d.leftLabel} × ${d.rightLabel}`, stepTitle: "Input encrypted columns and plaintext weights", shortTitle: "Input",
      description: `Compute ${d.outputLabel} by multiplying encrypted ${d.leftLabel} by plaintext ${d.rightLabel}.`, revealAt: 0,
      rows: [[matrix("A", d.leftLabel, A, 0, "CT", "column"), operator(`${d.id}-times`, "×", "", 0), matrix("B", d.rightLabel, B, 0, "PT", "column")]] },
    { id: `${d.id}-column-pack`, title: "Column Pack", stepTitle: "Pack one ciphertext per input column", shortTitle: "Pack columns",
      description: "Place the rows of each sample in a contiguous slot block and concatenate the blocks for each column.", revealAt: 1, rows: [aCts] },
    { id: `${d.id}-multiply`, title: "① Extend ⊙", stepTitle: "Broadcast weights and multiply columns", shortTitle: "Multiply",
      description: "Repeat each weight across its sample’s row slots, then multiply it by the matching encrypted input column.", revealAt: 2, rows: multiplyRows },
    { id: `${d.id}-sum`, title: "② Σ", stepTitle: "Accumulate products by output column", shortTitle: "Sum products",
      description: "Add the contributions along the shared dimension to produce one encrypted vector per output column.", revealAt: 3,
      rows: outputs.map((out, j) => [...productsByOutput[j], operator(`${d.id}-sum-${j}`, "Σ", "", 3), out]) },
    outputStage(d, transpose(outputs.map((out) => valuesOf(out).slice(0, ROWS))), outputs, 4),
  ] };
  const mults = A[0].length * B[0].length;
  const steps = graphSteps(d, graph, A, B, C, [{}, { ciphertextCount: aCts.length },
    { ptCtMultiplications: mults, rescales: mults, multiplicativeDepth: 1 }, { additions: (A[0].length - 1) * B[0].length }, {}]);
  return shell(d, A, B, C, graph, steps, "ciphertext column-packed × plaintext unpacked");
}

function buildCccrmm(d, Q, K) {
  assertRect(Q, 2, 2, d.leftLabel); assertRect(K, 2, 2, d.rightLabel);
  const KT = transpose(K), C = matmul(Q, KT), qCols = cols(Q), kCols = cols(K);
  const qCts = qCols.map((v, i) => strip({ id: `${d.id}-q-${i}`, label: `q${i + 1}`, values: v, revealAt: 1, tone: i }));
  const kCts = kCols.map((v, i) => strip({ id: `${d.id}-k-${i}`, label: `k${i + 1}`, values: v, revealAt: 1, tone: i + 2 }));
  const { rows: rotationRows, shifted } = componentRotations(kCts, d.id, 2);
  const rotations = [kCts, shifted];
  const diagonals = rotations.map((keys, i) => strip({ id: `${d.id}-diag-${i}`, label: `p${i + 1}`,
    values: addVectors(keys.map((key, j) => multiplyVectors(valuesOf(qCts[j]), valuesOf(key)))).slice(0, ACTIVE_SLOTS),
    revealAt: 3, tone: i, kind: "result", batched: false }));
  const decoded = C.map((row, i) => row.map((_, j) => diagonals[mod(j - i, ROWS)].cells[i].value));
  const rows = [];
  for (let shift = 0; shift < 2; shift++) rows.push([qCts[0], operator(`${d.id}-m-${shift}-0`, "⊙", "", 3), rotations[shift][0],
    operator(`${d.id}-plus-${shift}`, "+", "", 3), qCts[1], operator(`${d.id}-m-${shift}-1`, "⊙", "", 3), rotations[shift][1],
    operator(`${d.id}-eq-${shift}`, "=", "", 3), diagonals[shift]]);
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: `${d.leftLabel} × ${d.rightLabel}ᵀ`, stepTitle: "Input Q and transposed K", shortTitle: "Q × Kᵀ",
      description: "Multiply columns of the encrypted query matrix with the corresponding encrypted rows of Kᵀ.", revealAt: 0,
      rows: [[matrix("A", d.leftLabel, Q, 0, "CT", "column"), operator(`${d.id}-times`, "×", "", 0), matrix("B", "Kᵀ", KT, 0, "CT", "row", 2)]] },
    { id: `${d.id}-column-pack`, title: "Column Pack", stepTitle: "Pack Q columns and Kᵀ rows", shortTitle: "Pack Q and K",
      description: "Store each query column and matching key row in contiguous sample blocks.", revealAt: 1, rows: [qCts, kCts] },
    { id: `${d.id}-rotate`, title: "① Component ρ", stepTitle: "Rotate K within each sample", shortTitle: "Rotate K",
      description: "Mask a left rotation and a right rotation, then add the retained slots to keep each sample’s wraparound separate.", revealAt: 2, rows: rotationRows },
    { id: `${d.id}-multiply`, title: "② CCCRMM ⊙ Σ", stepTitle: "Compute one score diagonal per shift", shortTitle: "Score diagonals",
      description: "For each key shift, multiply matching Q and K vectors slotwise and sum across features.", revealAt: 3, rows },
    outputStage(d, decoded, diagonals, 4, "diagonal"),
  ] };
  const steps = graphSteps(d, graph, Q, KT, C, [{}, { ciphertextCount: 4 },
    { rotations: 4, masks: 4, ptCtMultiplications: 4, additions: 2, rescales: 4, multiplicativeDepth: 1 },
    { ctCtMultiplications: 4, additions: 2, relinearizations: 2, rescales: 2, multiplicativeDepth: 1 }, {}]);
  return shell(d, Q, KT, C, graph, steps, "column-packed × row-packed → diagonal-packed");
}

function buildCdccmm(d, P, V) {
  assertRect(P, 2, 2, d.leftLabel); assertRect(V, 2, 2, d.rightLabel);
  const C = matmul(P, V), vCols = cols(V);
  const diagonals = [[P[0][0], P[1][1]], [P[0][1], P[1][0]]].map((values, i) =>
    strip({ id: `${d.id}-diag-${i}`, label: `ℓp${i + 1}`, values, revealAt: 1, tone: i }));
  const vCts = vCols.map((values, i) => strip({ id: `${d.id}-v-${i}`, label: `v${i + 1}`, values, revealAt: 1, tone: i + 2 }));
  const { rows: rotationRows, shifted } = componentRotations(vCts, d.id, 2);
  const rotated = [vCts, shifted];
  const outputs = vCts.map((_, i) => strip({ id: `${d.id}-y-${i}`, label: `${d.outputLabel}${i + 1}`,
    values: addVectors(diagonals.map((a, j) => multiplyVectors(valuesOf(a), valuesOf(rotated[j][i])))).slice(0, ACTIVE_SLOTS),
    revealAt: 3, tone: i, kind: "result", batched: false }));
  const rows = outputs.map((out, i) => [diagonals[0], operator(`${d.id}-m-${i}-0`, "⊙", "", 3), rotated[0][i],
    operator(`${d.id}-plus-${i}`, "+", "", 3), diagonals[1], operator(`${d.id}-m-${i}-1`, "⊙", "", 3), rotated[1][i],
    operator(`${d.id}-eq-${i}`, "=", "", 3), out]);
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: `${d.leftLabel} × ${d.rightLabel}`, stepTitle: "Input diagonal attention and column-packed V", shortTitle: "Attention × V",
      description: "Apply the encrypted attention weights to the corresponding encrypted value rows.", revealAt: 0,
      rows: [[matrix("A", d.leftLabel, P, 0, "CT", "diagonal"), operator(`${d.id}-times`, "×", "", 0), matrix("B", d.rightLabel, V, 0, "CT", "column")]] },
    { id: `${d.id}-pack`, title: "Diag + Column Pack", stepTitle: "Pack attention diagonals and value columns", shortTitle: "Pack operands",
      description: "Use one ciphertext per cyclic attention diagonal and one per V column, with a row block for each sample.", revealAt: 1, rows: [diagonals, vCts] },
    { id: `${d.id}-rotate`, title: "① Component ρ", stepTitle: "Rotate V within each sample", shortTitle: "Rotate V",
      description: "Use two complementary masks to wrap each V column within its sample block.", revealAt: 2, rows: rotationRows },
    { id: `${d.id}-multiply`, title: "② CDCCMM ⊙ Σ", stepTitle: "Multiply diagonals and sum by V column", shortTitle: "Accumulate",
      description: "Multiply each attention diagonal by its matching V shift and add the products to form a context column.", revealAt: 3, rows },
    outputStage(d, transpose(outputs.map((out) => valuesOf(out).slice(0, ROWS))), outputs, 4),
  ] };
  const steps = graphSteps(d, graph, P, V, C, [{}, { ciphertextCount: 4 },
    { rotations: 4, masks: 4, ptCtMultiplications: 4, additions: 2, rescales: 4, multiplicativeDepth: 1 },
    { ctCtMultiplications: 4, additions: 2, relinearizations: 2, rescales: 2, multiplicativeDepth: 1 }, {}]);
  return shell(d, P, V, C, graph, steps, "diagonal-packed × column-packed → column-packed");
}

function subtract(A, B) { return A.map((row, i) => row.map((v, j) => Number(v) - Number(B[i][j]))); }
function buildIwmm(d, A, W) {
  assertRect(A, 2, 2, d.leftLabel); assertRect(W, 2, 2, d.rightLabel);
  const U = [[0, 1], [1, 0]], D = [[3, 0], [0, 2]], H = [[1, 0], [0, -1]];
  const E = matmul(U, D), R = matmul(E, H), X = subtract(A, R), C = matmul(A, W);
  const HW = matmul(H, W), XW = matmul(X, W);
  const dComp = packedDiagonal([D[0][0], D[1][1]], `${d.id}-d`, 1);
  const { rows: expansionRows, diagonal } = expandDiagonal(dComp, d.id, 2);
  const uPts = cols(U).map((values, i) => strip({ id: `${d.id}-u-col-${i}`, label: `u${i + 1}`, values, revealAt: 2, tone: i, badge: "PT" }));
  const eCols = uPts.map((out, i) => strip({ id: `${d.id}-e-${i}`, label: `e${i + 1}`,
    values: multiplyVectors(valuesOf(out), valuesOf(diagonal[i])).slice(0, ACTIVE_SLOTS), revealAt: 2, tone: i, batched: false }));
  const rwCols = cols(HW).map((weights, i) => strip({ id: `${d.id}-rw-${i}`, label: `RW${i + 1}`,
    values: addVectors(eCols.map((col, j) => valuesOf(col).map((value) => value * weights[j]))).slice(0, ACTIVE_SLOTS),
    revealAt: 3, tone: i, batched: false }));
  const xwPts = cols(XW).map((values, i) => strip({ id: `${d.id}-xw-${i}`, label: `XW${i + 1}`, values, revealAt: 4, tone: i, badge: "PT" }));
  const outCols = rwCols.map((out, i) => strip({ id: `${d.id}-out-${i}`, label: `${d.outputLabel}${i + 1}`,
    values: addVectors([valuesOf(out), valuesOf(xwPts[i])]).slice(0, ACTIVE_SLOTS), revealAt: 5, tone: i, kind: "result", batched: false }));
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: `${d.leftLabel} × ${d.rightLabel}`, stepTitle: "Set up the private input projection", shortTitle: "Input",
      description: `Split the computation of ${d.outputLabel} into a precomputed encrypted mask product and an online masked-input product.`, revealAt: 0,
      rows: [[matrix("A", "A (client)", A, 0, "PT", "column"), operator(`${d.id}-times`, "×", "", 0), matrix("B", d.rightLabel, W, 0, "PT", "column")]] },
    { id: `${d.id}-svd`, title: "R = U · D · H", stepTitle: "Factor the offline mask and encrypt D", shortTitle: "Factor mask",
      description: "Use a fixed SVD example for the mask R. U and H stay plaintext, while the diagonal of D is compressed and encrypted.", revealAt: 1,
      rows: [[matrix(`${d.id}-r`, "R (client)", R, 1, "PT"), operator(`${d.id}-svd-eq`, "=", "", 1),
        matrix(`${d.id}-u`, "U", U, 1, "PT"), operator(`${d.id}-ud`, "×", "", 1),
        matrix(`${d.id}-d-matrix`, "D (client)", D, 1, "PT", "diagonal"), operator(`${d.id}-dh`, "×", "", 1),
        matrix(`${d.id}-h`, "H", H, 1, "PT")], [dComp]] },
    { id: `${d.id}-pccd`, title: "① PCCDMM", stepTitle: "Build the encrypted columns of UD", shortTitle: "PCCDMM",
      description: "Expand D into batched scalar vectors and multiply each plaintext U column by its matching diagonal.", revealAt: 2,
      rows: [[dComp], ...expansionRows, ...eCols.map((col, i) => [uPts[i], operator(`${col.id}-mul`, "⊙", "", 2), diagonal[i], operator(`${col.id}-eq`, "=", "", 2), col])] },
    { id: `${d.id}-ccpu`, title: "② CCPUMM", stepTitle: "Precompute the encrypted mask product RW", shortTitle: "CCPUMM",
      description: "Compute HW in plaintext, then multiply the encrypted columns of E = UD by HW to obtain RW.", revealAt: 3,
      rows: [[...eCols, operator(`${d.id}-hw`, "×", "", 3), matrix(`${d.id}-hw-plain`, "HW", HW, 3, "PT"),
        operator(`${d.id}-rweq`, "=", "", 3), ...rwCols]] },
    { id: `${d.id}-online`, title: "③ X = A − R; compute XW", stepTitle: "Multiply the masked input online", shortTitle: "Compute XW",
      description: "At the client, mask A with R. The server receives X = A − R and multiplies it by W in plaintext.", revealAt: 4,
      rows: [[matrix(`${d.id}-x`, "X", X, 4, "PT", "column"), operator(`${d.id}-x-times`, "×", "", 4), matrix("B2", d.rightLabel, W, 4, "PT", "column"),
        operator(`${d.id}-x-eq`, "=", "", 4), ...xwPts]] },
    { id: `${d.id}-output`, title: "④ AW = RW + XW", stepTitle: "Restore the projection with the mask product", shortTitle: "Restore AW",
      description: "Add the plaintext XW columns to the encrypted RW columns to recover the encrypted projection.", revealAt: 5,
      rows: outCols.map((out, i) => [rwCols[i], operator(`${d.id}-add-${i}`, "+", "", 5), xwPts[i], operator(`${d.id}-eq-${i}`, "=", "", 5), out])
        .concat([[operator(`${d.id}-down`, "↓", "", 5), matrix("C", d.outputLabel, transpose(outCols.map((col) => valuesOf(col).slice(0, ROWS))), 5)]]) },
  ] };
  const steps = graphSteps(d, graph, A, W, C, [{}, { ciphertextCount: 1 },
    { ptCtMultiplications: 6, masks: 4, additions: 2, rescales: 2, multiplicativeDepth: 1 },
    { ptCtMultiplications: 4, additions: 2, rescales: 4, multiplicativeDepth: 1 }, {}, { additions: 2 }]);
  return shell(d, A, W, C, graph, steps, "SVD-secure IWMM → column-packed");
}

export function buildEustonOperationTrace(id, options = {}) {
  const d = getEustonOperation(id); if (!d) throw new Error(`Unknown Euston operation: ${id}`);
  if (options.leftSecret != null && Boolean(options.leftSecret) !== d.leftSecret) throw new Error(`${d.shortLabel} has a fixed left operand type`);
  if (options.rightSecret != null && Boolean(options.rightSecret) !== d.rightSecret) throw new Error(`${d.shortLabel} has a fixed right operand type`);
  if (options.slotsPerCiphertext != null && Number(options.slotsPerCiphertext) !== SLOTS) throw new Error("Euston Figure 2 toy uses 4 active values in an 8-slot ciphertext");
  const A = options.A || d.A, B = options.B || d.B;
  if (d.kind === "pccdmm") return buildPccdmm(d, A, B);
  if (d.kind === "cccrmm") return buildCccrmm(d, A, B);
  if (d.kind === "cdccmm") return buildCdccmm(d, A, B);
  if (d.kind === "iwmm") return buildIwmm(d, A, B);
  return buildCcpumm(d, A, B);
}

export function eustonOperationDefaults(id) {
  const d = getEustonOperation(id); if (!d) return null;
  return { m: d.A.length, k: d.A[0].length, n: d.B[0].length, A: clone(d.A), B: clone(d.B),
    leftSecret: d.leftSecret, rightSecret: d.rightSecret, slotsPerCiphertext: SLOTS, strategy: "column" };
}

export function parseEustonOperationInputs(raw, d) {
  const result = parseAndValidateInputs(raw);
  if (!result.ok) return result;
  if (d.kind === "pccdmm" && (Number(result.inputs.B[0]?.[1]) !== 0 || Number(result.inputs.B[1]?.[0]) !== 0)) {
    return { ok: false, errors: { values: "D must remain diagonal for PCCDMM" }, inputs: null };
  }
  return result;
}
