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
import { ctState, makeSteps, matrixArtifact, operator, vectorArtifact } from "../thor/paper_utils.js";

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

const X22 = Object.freeze([[1, 2], [0, 1]]);
const W22 = Object.freeze([[1, 0], [2, 1]]);
const W24 = Object.freeze([[1, 0, 1, 0], [0, 1, 1, 2]]);
const X24 = Object.freeze([[1, 2, 0, 1], [0, 1, 2, 1]]);
const W42 = Object.freeze([[1, 0], [0, 1], [1, 1], [0, 1]]);
const Q22 = Object.freeze([[1, 2], [0, 1]]);
const K22 = Object.freeze([[1, 0], [1, 2]]);
const S22 = Object.freeze([[0.75, 0.25], [0.2, 0.8]]);
const V22 = Object.freeze([[1, 0], [2, 1]]);
const U22 = Object.freeze([[1, 2], [0, 1]]);
const D22 = Object.freeze([[2, 0], [0, 3]]);

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
function componentRotate(values, shift) {
  const s = ((shift % ROWS) + ROWS) % ROWS;
  const one = values.slice(s).concat(values.slice(0, s));
  return batch(one);
}

function strip({ id, label, values, revealAt, tone = 0, badge = "CT", kind = "ciphertext", batched = true }) {
  const active = batched ? batch(values) : values;
  const slots = Array.from({ length: SLOTS }, (_, i) => Number(active[i] ?? 0));
  // The block color denotes the packed column/diagonal identity, regardless
  // of whether the operand is plaintext or ciphertext.
  const artifact = vectorArtifact({ id, label, values: slots, revealAt, badge, kind, tones: Array(slots.length).fill(tone) });
  artifact.cells.forEach((cell, i) => {
    if (i >= ACTIVE_SLOTS) cell.label = "·";
    else cell.label = `${cell.value}${SUB[Math.floor(i / ROWS)]}`;
  });
  return artifact;
}
function matrix(id, label, values, revealAt, badge = "CT", layout = "column") {
  return matrixArtifact({ id, label, values, revealAt, badge, meta: "",
    toneForCell: (row, col) => layout === "diagonal" ? ((col - row) % 4 + 4) % 4
      : layout === "row" ? row % 4 : col % 4 });
}
function artifactCiphertexts(stage) {
  const result = [], seen = new Set();
  for (const row of stage.rows || []) for (const a of row) {
    if (a.badge !== "CT" || !a.cells || seen.has(a.id)) continue;
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
    title: stage.title, description: stage.caption || "", formula: "",
    tensors: tensorState(d, A, validationB, C, i === graph.stages.length - 1),
    ciphertexts: artifactCiphertexts(stage), delta: deltas[i] || {},
  })));
}
function shell(d, A, validationB, C, graph, steps, flow) {
  const m = A.length, k = A[0].length, n = validationB[0].length;
  return {
    schemaVersion: "1.0", id: `${d.id}-toy`, title: `S&P'26 Euston ${d.shortLabel}`,
    parameters: { matrixShape: { m, k, n }, slotsPerCiphertext: SLOTS,
      matmulType: d.leftSecret && d.rightSecret ? "ct-ct" : d.leftSecret || d.rightSecret ? "ct-pt" : "pt-pt",
      algorithmId: d.id, batchCount: BATCHES },
    operands: {
      left: { id: "A", label: d.leftLabel, shape: [m, k], values: A, secret: d.leftSecret },
      right: { id: "B", label: d.kind === "cccrmm" ? "Kᵀ" : d.rightLabel,
        shape: [validationB.length, validationB[0].length], values: validationB, secret: d.rightSecret },
    },
    output: { id: "C", label: d.outputLabel, shape: [m, n] },
    plan: { algorithmId: d.id, method: "euston", methodLabel: "S&P'26 Euston", visualStyle: "diagram-only",
      matmulType: d.leftSecret && d.rightSecret ? "ct-ct" : "ct-pt", logicalShape: { m, k, n },
      paddedShape: { m, k, n }, tileGrid: { m: 1, n: 1 }, tileShape: { m, n },
      packingStrategyRequested: flow, packingStrategyResolved: flow, plannerSource: "Euston Sections 4.1-4.2 and Figure 2",
      reductionStrategy: flow, displayMode: "diagram-only", batchLabel: "samples", warnings: [] },
    graph, steps, finalTensor: C,
  };
}

function outputStage(d, C, columnArtifacts, revealAt) {
  return { id: `${d.id}-output`, title: d.outputLabel, caption: "", revealAt,
    rows: [[...columnArtifacts], [operator(`${d.id}-down`, "↓", "", revealAt), matrix("C", d.outputLabel, C, revealAt)]] };
}

function buildPccdmm(d, U, D) {
  assertRect(U, 2, 2, d.leftLabel); assertRect(D, 2, 2, d.rightLabel);
  if (Number(D[0][1]) !== 0 || Number(D[1][0]) !== 0) throw new Error("D must remain diagonal for PCCDMM");
  const C = matmul(U, D), u = cols(U), c = cols(C), diagonal = [D[0][0], D[1][1]];
  const compressed = strip({ id: `${d.id}-d`, label: "⟦d⟧", values: diagonal, revealAt: 1, tone: 0 });
  const expanded = diagonal.map((value, i) => strip({ id: `${d.id}-de-${i}`, label: `d${i + 1}`, values: [value, value], revealAt: 2, tone: i }));
  const uPts = u.map((values, i) => strip({ id: `${d.id}-u-${i}`, label: `u${i + 1}`, values, revealAt: 3, tone: i, badge: "PT" }));
  const products = c.map((values, i) => strip({ id: `${d.id}-e-${i}`, label: `e${i + 1}`, values, revealAt: 3, tone: i, kind: "result" }));
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: `${d.leftLabel} × ${d.rightLabel}`, caption: "", revealAt: 0,
      rows: [[matrix("A", d.leftLabel, U, 0, "PT", "column"), operator(`${d.id}-times`, "×", "", 0), matrix("B", d.rightLabel, D, 0, "CT", "diagonal")]] },
    { id: `${d.id}-seccomp`, title: "SecComp", caption: "", revealAt: 1, rows: [[matrix("B2", d.rightLabel, D, 1), operator(`${d.id}-comp`, "→", "", 1), compressed]] },
    { id: `${d.id}-secdecomp`, title: "SecDecomp", caption: "", revealAt: 2, rows: [[compressed, operator(`${d.id}-decomp`, "→", "", 2), ...expanded]] },
    { id: `${d.id}-multiply`, title: "PCCDMM", caption: "", revealAt: 3,
      rows: products.map((p, i) => [uPts[i], operator(`${d.id}-mul-${i}`, "⊙", "", 3), expanded[i], operator(`${d.id}-eq-${i}`, "=", "", 3), p]) },
    outputStage(d, C, products, 4),
  ] };
  const steps = graphSteps(d, graph, U, D, C, [{}, { ciphertextCount: 1 }, {},
    { ptCtMultiplications: 2, rescales: 2, multiplicativeDepth: 1 }, {}]);
  return shell(d, U, D, C, graph, steps, "column-packed × encrypted diagonal");
}

function buildCcpumm(d, A, B) {
  assertRect(A, 2, d.A[0].length, d.leftLabel); assertRect(B, d.B.length, d.B[0].length, d.rightLabel);
  const C = matmul(A, B), aCols = cols(A), cCols = cols(C);
  const aCts = aCols.map((values, i) => strip({ id: `${d.id}-a-${i}`, label: `${d.leftLabel}${i + 1}`, values, revealAt: 1, tone: i }));
  const productsByOutput = cCols.map((_, j) => aCols.map((values, i) => strip({ id: `${d.id}-p-${j}-${i}`, label: `p${j + 1}${i + 1}`,
    values: values.map((value) => value * Number(B[i][j])), revealAt: 2, tone: i, kind: "product" })));
  const outputs = cCols.map((values, j) => strip({ id: `${d.id}-c-${j}`, label: `${d.outputLabel}${j + 1}`, values, revealAt: 3, tone: j, kind: "result" }));
  const multiplyRows = [];
  for (let j = 0; j < cCols.length; j++) for (let i = 0; i < aCols.length; i++) {
    const scalar = strip({ id: `${d.id}-w-${j}-${i}`, label: `${d.rightLabel}${i + 1}${j + 1}`, values: [B[i][j], B[i][j]], revealAt: 2, badge: "PT" });
    multiplyRows.push([aCts[i], operator(`${d.id}-mul-${j}-${i}`, "⊙", "", 2), scalar,
      operator(`${d.id}-eq-${j}-${i}`, "=", "", 2), productsByOutput[j][i]]);
  }
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: `${d.leftLabel} × ${d.rightLabel}`, caption: "", revealAt: 0,
      rows: [[matrix("A", d.leftLabel, A, 0, "CT", "column"), operator(`${d.id}-times`, "×", "", 0), matrix("B", d.rightLabel, B, 0, "PT", "column")]] },
    { id: `${d.id}-column-pack`, title: "Column Pack", caption: "", revealAt: 1, rows: [aCts] },
    { id: `${d.id}-multiply`, title: "① Extend ⊙", caption: "", revealAt: 2, rows: multiplyRows },
    { id: `${d.id}-sum`, title: "② Σ", caption: "", revealAt: 3,
      rows: outputs.map((out, j) => [...productsByOutput[j], operator(`${d.id}-sum-${j}`, "Σ", "", 3), out]) },
    outputStage(d, C, outputs, 4),
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
  const rotations = [0, 1].map((shift) => kCols.map((v, i) => strip({ id: `${d.id}-kr-${shift}-${i}`, label: shift ? `k${i + 1}⇐1` : `k${i + 1}`,
    values: componentRotate(v, shift), revealAt: 2, tone: i + 2, batched: false })));
  const diagonals = [
    [C[0][0], C[1][1]],
    [C[0][1], C[1][0]],
  ].map((v, i) => strip({ id: `${d.id}-diag-${i}`, label: `ℓp${i + 1}`, values: v, revealAt: 3, tone: i, kind: "result" }));
  const rows = [];
  for (let shift = 0; shift < 2; shift++) rows.push([qCts[0], operator(`${d.id}-m-${shift}-0`, "⊙", "", 3), rotations[shift][0],
    operator(`${d.id}-plus-${shift}`, "+", "", 3), qCts[1], operator(`${d.id}-m-${shift}-1`, "⊙", "", 3), rotations[shift][1],
    operator(`${d.id}-eq-${shift}`, "=", "", 3), diagonals[shift]]);
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: `${d.leftLabel} × ${d.rightLabel}ᵀ`, caption: "", revealAt: 0,
      rows: [[matrix("A", d.leftLabel, Q, 0, "CT", "column"), operator(`${d.id}-times`, "×", "", 0), matrix("K", d.rightLabel, K, 0, "CT", "row")]] },
    { id: `${d.id}-column-pack`, title: "Column Pack", caption: "", revealAt: 1, rows: [qCts, kCts] },
    { id: `${d.id}-rotate`, title: "① Component ρ", caption: "", revealAt: 2, rows: rotations },
    { id: `${d.id}-multiply`, title: "② CCCRMM ⊙ Σ", caption: "", revealAt: 3, rows },
    outputStage(d, C, diagonals, 4),
  ] };
  const steps = graphSteps(d, graph, Q, KT, C, [{}, { ciphertextCount: 4 },
    { rotations: 4, ptCtMultiplications: 4, additions: 2, rescales: 4, multiplicativeDepth: 1 },
    { ctCtMultiplications: 4, additions: 2, relinearizations: 2, rescales: 2, multiplicativeDepth: 1 }, {}]);
  return shell(d, Q, KT, C, graph, steps, "column-packed × row-packed → diagonal-packed");
}

function buildCdccmm(d, P, V) {
  assertRect(P, 2, 2, d.leftLabel); assertRect(V, 2, 2, d.rightLabel);
  const C = matmul(P, V), vCols = cols(V);
  const diagonals = [[P[0][0], P[1][1]], [P[0][1], P[1][0]]].map((values, i) =>
    strip({ id: `${d.id}-diag-${i}`, label: `ℓp${i + 1}`, values, revealAt: 1, tone: i }));
  const vCts = vCols.map((values, i) => strip({ id: `${d.id}-v-${i}`, label: `v${i + 1}`, values, revealAt: 1, tone: i + 2 }));
  const rotated = [0, 1].map((shift) => vCols.map((values, i) => strip({ id: `${d.id}-vr-${shift}-${i}`,
    label: shift ? `v${i + 1}⇐1` : `v${i + 1}`, values: componentRotate(values, shift), revealAt: 2, tone: i + 2, batched: false })));
  const outputs = cols(C).map((values, i) => strip({ id: `${d.id}-y-${i}`, label: `${d.outputLabel}${i + 1}`, values, revealAt: 3, tone: i, kind: "result" }));
  const rows = outputs.map((out, i) => [diagonals[0], operator(`${d.id}-m-${i}-0`, "⊙", "", 3), rotated[0][i],
    operator(`${d.id}-plus-${i}`, "+", "", 3), diagonals[1], operator(`${d.id}-m-${i}-1`, "⊙", "", 3), rotated[1][i],
    operator(`${d.id}-eq-${i}`, "=", "", 3), out]);
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: `${d.leftLabel} × ${d.rightLabel}`, caption: "", revealAt: 0,
      rows: [[matrix("A", d.leftLabel, P, 0, "CT", "diagonal"), operator(`${d.id}-times`, "×", "", 0), matrix("B", d.rightLabel, V, 0, "CT", "column")]] },
    { id: `${d.id}-pack`, title: "Diag + Column Pack", caption: "", revealAt: 1, rows: [diagonals, vCts] },
    { id: `${d.id}-rotate`, title: "① Component ρ", caption: "", revealAt: 2, rows: rotated },
    { id: `${d.id}-multiply`, title: "② CDCCMM ⊙ Σ", caption: "", revealAt: 3, rows },
    outputStage(d, C, outputs, 4),
  ] };
  const steps = graphSteps(d, graph, P, V, C, [{}, { ciphertextCount: 4 },
    { rotations: 4, ptCtMultiplications: 4, additions: 2, rescales: 4, multiplicativeDepth: 1 },
    { ctCtMultiplications: 4, additions: 2, relinearizations: 2, rescales: 2, multiplicativeDepth: 1 }, {}]);
  return shell(d, P, V, C, graph, steps, "diagonal-packed × column-packed → column-packed");
}

function subtract(A, B) { return A.map((row, i) => row.map((v, j) => Number(v) - Number(B[i][j]))); }
function add(A, B) { return A.map((row, i) => row.map((v, j) => Number(v) + Number(B[i][j]))); }
function buildIwmm(d, A, W) {
  assertRect(A, 2, 2, d.leftLabel); assertRect(W, 2, 2, d.rightLabel);
  const I = [[1, 0], [0, 1]], R = I, X = subtract(A, R), C = matmul(A, W), RW = matmul(R, W), XW = matmul(X, W);
  const dComp = strip({ id: `${d.id}-d`, label: "⟦d⟧", values: [1, 1], revealAt: 1, tone: 0 });
  const eCols = cols(I).map((values, i) => strip({ id: `${d.id}-e-${i}`, label: `e${i + 1}`, values, revealAt: 2, tone: i }));
  const rwCols = cols(RW).map((values, i) => strip({ id: `${d.id}-rw-${i}`, label: `RW${i + 1}`, values, revealAt: 3, tone: i }));
  const xwPts = cols(XW).map((values, i) => strip({ id: `${d.id}-xw-${i}`, label: `XW${i + 1}`, values, revealAt: 4, tone: i, badge: "PT" }));
  const outCols = cols(C).map((values, i) => strip({ id: `${d.id}-out-${i}`, label: `${d.outputLabel}${i + 1}`, values, revealAt: 5, tone: i, kind: "result" }));
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: `${d.leftLabel} × ${d.rightLabel}`, caption: "", revealAt: 0,
      rows: [[matrix("A", d.leftLabel, A, 0, "CT", "column"), operator(`${d.id}-times`, "×", "", 0), matrix("B", d.rightLabel, W, 0, "PT", "column")]] },
    { id: `${d.id}-svd`, title: "R = U · D · H", caption: "", revealAt: 1,
      rows: [[matrix(`${d.id}-u`, "U", I, 1, "PT", "column"), operator(`${d.id}-ud`, "×", "", 1), dComp,
        operator(`${d.id}-dh`, "×", "", 1), matrix(`${d.id}-h`, "H", I, 1, "PT", "column")]] },
    { id: `${d.id}-pccd`, title: "① PCCDMM", caption: "", revealAt: 2, rows: [[dComp, operator(`${d.id}-to-e`, "→", "", 2), ...eCols]] },
    { id: `${d.id}-ccpu`, title: "② CCPUMM", caption: "", revealAt: 3,
      rows: [[...eCols, operator(`${d.id}-hw`, "× HW", "", 3), ...rwCols]] },
    { id: `${d.id}-online`, title: "③ X = A − R; compute XW", caption: "Online phase: mask A with R, then multiply the public masked input X by W.", revealAt: 4,
      rows: [[matrix(`${d.id}-x`, "X", X, 4, "PT", "column"), operator(`${d.id}-x-times`, "×", "", 4), matrix("B2", d.rightLabel, W, 4, "PT", "column")], xwPts] },
    { id: `${d.id}-output`, title: "④ AW = RW + XW", caption: "Combine the encrypted offline term RW with the online term XW.", revealAt: 5,
      rows: outCols.map((out, i) => [rwCols[i], operator(`${d.id}-add-${i}`, "+", "", 5), xwPts[i], operator(`${d.id}-eq-${i}`, "=", "", 5), out])
        .concat([[operator(`${d.id}-down`, "↓", "", 5), matrix("C", d.outputLabel, add(RW, XW), 5)]]) },
  ] };
  const steps = graphSteps(d, graph, A, W, C, [{}, { ciphertextCount: 1 },
    { ptCtMultiplications: 2, rescales: 2, multiplicativeDepth: 1 },
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
  if (d.kind === "pccdmm" && (Number(result.inputs.B[0][1]) !== 0 || Number(result.inputs.B[1][0]) !== 0)) {
    return { ok: false, errors: { values: "D must remain diagonal for PCCDMM" }, inputs: null };
  }
  return result;
}
