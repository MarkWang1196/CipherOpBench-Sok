// NEXUS matrix-multiplication traces.
// Paper/code correspondence:
//   Figure 4 / Algorithm 1 -> polynomial ciphertext decompression;
//   Figure 5 / MMEvaluator::matrix_mul -> offline-online batched CT-PT MM;
//   Appendix C / Algorithm 5 -> column-packed CT-CT QK^T;
//   Section III-C -> row-wise encrypted multiplication after attention.

import { matmul, transpose } from "../../core/matrix.js";
import { parseAndValidateInputs } from "../../core/input_parser.js";
import { ctState, makeSteps, matrixArtifact, operator, vectorArtifact } from "../thor/paper_utils.js";

export const NEXUS_OPERATION_IDS = Object.freeze({
  DECOMPRESS: "nexus-decompress",
  OFFLINE_ONLINE: "nexus-offline-online",
  COLUMN_CCMM: "nexus-column-ccmm",
  QUERY: "nexus-query",
  KEY: "nexus-key",
  VALUE: "nexus-value",
  SCORE: "nexus-score",
  CONTEXT: "nexus-context",
  OUTPUT: "nexus-output",
  FC1: "nexus-fc1",
  FC2: "nexus-fc2",
});

const SLOTS = 8;
const BATCHES = 2;
const SUB = ["₀", "₁"];
const A24 = Object.freeze([[1, 2, 0, 1], [0, 1, 2, 1]]);
const W42 = Object.freeze([[1, 0], [0, 1], [1, 1], [0, 1]]);
const A22 = Object.freeze([[1, 2], [0, 1]]);
const W22 = Object.freeze([[1, 0], [2, 1]]);
const W24 = Object.freeze([[1, 0, 1, 0], [0, 1, 1, 2]]);
const K22 = Object.freeze([[1, 0], [1, 2]]);
const S22 = Object.freeze([[0.75, 0.25], [0.2, 0.8]]);
const V22 = Object.freeze([[1, 0], [2, 1]]);

function operation(id, config) {
  return Object.freeze({ id, method: "nexus", slotsPerCiphertext: SLOTS, batchCount: BATCHES, ...config });
}
function linear(id, shortLabel, leftLabel, rightLabel, outputLabel, group, A, B, kind = "offline-online") {
  return operation(id, { shortLabel, group, kind, leftLabel, rightLabel, outputLabel,
    leftSecret: false, rightSecret: true, A, B, strategy: "column" });
}
export const NEXUS_OPERATIONS = Object.freeze({
  [NEXUS_OPERATION_IDS.DECOMPRESS]: linear(NEXUS_OPERATION_IDS.DECOMPRESS, "Compress / Decompress", "A", "W", "Q", "paper", A22, W22, "decompress"),
  [NEXUS_OPERATION_IDS.OFFLINE_ONLINE]: linear(NEXUS_OPERATION_IDS.OFFLINE_ONLINE, "Offline / Online MM", "A", "W", "Q", "paper", A24, W42),
  [NEXUS_OPERATION_IDS.COLUMN_CCMM]: operation(NEXUS_OPERATION_IDS.COLUMN_CCMM, { shortLabel: "Column CCMM", group: "paper", kind: "ccmm",
    leftLabel: "Q", rightLabel: "K", outputLabel: "P", leftSecret: true, rightSecret: true, A: A22, B: K22, strategy: "column" }),
  [NEXUS_OPERATION_IDS.QUERY]: linear(NEXUS_OPERATION_IDS.QUERY, "XWq", "X", "Wq", "Q", "attention", A24, W42),
  [NEXUS_OPERATION_IDS.KEY]: linear(NEXUS_OPERATION_IDS.KEY, "XWk", "X", "Wk", "K", "attention", A24, W42),
  [NEXUS_OPERATION_IDS.VALUE]: linear(NEXUS_OPERATION_IDS.VALUE, "XWv", "X", "Wv", "V", "attention", A24, W42),
  [NEXUS_OPERATION_IDS.SCORE]: operation(NEXUS_OPERATION_IDS.SCORE, { shortLabel: "QKᵀ", group: "attention", kind: "ccmm",
    leftLabel: "Q", rightLabel: "K", outputLabel: "P", leftSecret: true, rightSecret: true, A: A22, B: K22, strategy: "column" }),
  [NEXUS_OPERATION_IDS.CONTEXT]: operation(NEXUS_OPERATION_IDS.CONTEXT, { shortLabel: "Softmax·V", group: "attention", kind: "context",
    leftLabel: "Softmax(P)", rightLabel: "V", outputLabel: "H", leftSecret: true, rightSecret: true, A: S22, B: V22, strategy: "column" }),
  [NEXUS_OPERATION_IDS.OUTPUT]: linear(NEXUS_OPERATION_IDS.OUTPUT, "HWo", "H", "Wo", "O", "attention", A24, W42),
  [NEXUS_OPERATION_IDS.FC1]: linear(NEXUS_OPERATION_IDS.FC1, "FC1", "X", "W1", "F1", "ffn", A22, W24),
  [NEXUS_OPERATION_IDS.FC2]: linear(NEXUS_OPERATION_IDS.FC2, "FC2", "F1", "W2", "F2", "ffn", A24, W42),
});

export function listNexusOperations(group = null) { return Object.values(NEXUS_OPERATIONS).filter((x) => !group || x.group === group); }
export function getNexusOperation(id) { return NEXUS_OPERATIONS[id] || null; }
function clone(A) { return A.map((r) => [...r]); }
function cols(A) { return Array.from({ length: A[0].length }, (_, j) => A.map((r) => Number(r[j]))); }
function assertRect(A, rows, columns, label) {
  if (!Array.isArray(A) || A.length !== rows || A.some((r) => !Array.isArray(r) || r.length !== columns)) throw new Error(`${label} must be ${rows}×${columns}`);
}
function interleaved(values) { return values.flatMap((v) => [Number(v), Number(v)]); }
function rotatePairs(values, amount) {
  const pairs = Array.from({ length: values.length / 2 }, (_, i) => values.slice(2 * i, 2 * i + 2));
  const s = ((amount % pairs.length) + pairs.length) % pairs.length;
  return pairs.slice(s).concat(pairs.slice(0, s)).flat();
}
function strip({ id, label, values, revealAt, tone = 0, badge = "CT", kind = "ciphertext", labels = null }) {
  const padded = Array.from({ length: SLOTS }, (_, i) => Number(values[i] ?? 0));
  const a = vectorArtifact({ id, label, values: padded, revealAt, badge, kind, tones: Array(SLOTS).fill(badge === "CT" ? tone : 4) });
  a.cells.forEach((c, i) => { c.label = labels?.[i] ?? (i < values.length ? String(values[i]) : "·"); });
  return a;
}
function batchColumn(values, id, label, revealAt, tone = 0, kind = "ciphertext", badge = "CT") {
  const labels = values.flatMap((v) => [`${v}b${SUB[0]}`, `${v}b${SUB[1]}`]);
  return strip({ id, label, values: interleaved(values), labels, revealAt, tone, kind, badge });
}
function scalarCt(value, id, label, revealAt, tone = 0, kind = "ciphertext") {
  return strip({ id, label, values: Array(SLOTS).fill(Number(value)), revealAt, tone, kind });
}
function matrix(id, label, values, revealAt, badge = "CT", layout = "column", toneOffset = 0) {
  return matrixArtifact({ id, label, values, revealAt, badge, meta: "",
    toneForCell: (row, col) => layout === "row" ? (row + toneOffset) % 4 : (col + toneOffset) % 4 });
}
function artifactCiphertexts(stage) {
  const out = [], seen = new Set();
  for (const row of stage.rows || []) for (const a of row) {
    if (a.badge !== "CT" || !a.cells || seen.has(a.id)) continue;
    seen.add(a.id); out.push(ctState(a.id, a.label, a.cells.map((c) => c.value), SLOTS, true,
      a.type === "product" ? "product" : a.type === "result" ? "result" : "tensor"));
  }
  return out;
}
function graphSteps(d, graph, A, B, C, deltas) {
  return makeSteps(d.id, graph.stages.map((stage, i) => ({ phase: i === 0 ? "reference" : i === graph.stages.length - 1 ? "output" : "multiply",
    title: stage.title, description: "", formula: "", tensors: {
      left: { id: "A", label: d.leftLabel, shape: [A.length, A[0].length], values: A },
      right: { id: "B", label: d.kind === "ccmm" ? "Kᵀ" : d.rightLabel, shape: [B.length, B[0].length], values: B },
      ...(i === graph.stages.length - 1 ? { output: { id: "C", label: d.outputLabel, shape: [C.length, C[0].length], values: C } } : {}),
    }, ciphertexts: artifactCiphertexts(stage), delta: deltas[i] || {} }))) ;
}
function shell(d, A, B, C, graph, steps, flow) {
  return { schemaVersion: "1.0", id: `${d.id}-toy`, title: `NDSS'25 NEXUS ${d.shortLabel}`,
    parameters: { matrixShape: { m: A.length, k: A[0].length, n: B[0].length }, slotsPerCiphertext: SLOTS,
      matmulType: d.leftSecret && d.rightSecret ? "ct-ct" : "ct-pt", algorithmId: d.id, batchCount: BATCHES },
    operands: { left: { id: "A", label: d.leftLabel, shape: [A.length, A[0].length], values: A, secret: d.leftSecret },
      right: { id: "B", label: d.kind === "ccmm" ? "Kᵀ" : d.rightLabel, shape: [B.length, B[0].length], values: B, secret: d.rightSecret } },
    output: { id: "C", label: d.outputLabel, shape: [C.length, C[0].length] },
    plan: { algorithmId: d.id, method: "nexus", methodLabel: "NDSS'25 NEXUS", visualStyle: "diagram-only",
      matmulType: d.leftSecret && d.rightSecret ? "ct-ct" : "ct-pt", logicalShape: { m: A.length, k: A[0].length, n: B[0].length },
      paddedShape: { m: A.length, k: A[0].length, n: B[0].length }, tileGrid: { m: 1, n: 1 }, tileShape: { m: C.length, n: C[0].length },
      packingStrategyRequested: flow, packingStrategyResolved: flow, plannerSource: "NEXUS Figures 3-5, Algorithms 1/5, and MMEvaluator::matrix_mul",
      reductionStrategy: flow, displayMode: "diagram-only", batchLabel: "samples", warnings: [] }, graph, steps, finalTensor: C };
}
function outputStage(d, C, artifacts, revealAt) {
  return { id: `${d.id}-output`, title: d.outputLabel, caption: "", revealAt,
    rows: [[...artifacts], [operator(`${d.id}-down`, "↓", "", revealAt), matrix("C", d.outputLabel, C, revealAt)]] };
}
function packedPolynomial(W, d, revealAt) {
  const flat = transpose(W).flat();
  const labels = flat.map((v, i) => `${v}x${i}`).concat(Array(Math.max(0, SLOTS - flat.length)).fill("·"));
  return strip({ id: `${d.id}-poly`, label: "E(p(x))", values: flat, labels, revealAt, tone: 0 });
}
function decompressedWeights(W, d, revealAt) {
  return transpose(W).flat().map((v, i) => scalarCt(v, `${d.id}-w-${i}`, `w${Math.floor(i / W.length)}${i % W.length}`, revealAt, i % 4));
}

function buildOfflineOnline(d, A, W, decompressionOnly = false) {
  assertRect(A, d.A.length, d.A[0].length, d.leftLabel); assertRect(W, d.B.length, d.B[0].length, d.rightLabel);
  const C = matmul(A, W), poly = packedPolynomial(W, d, 1), weights = decompressedWeights(W, d, 3);
  const even = strip({ id: `${d.id}-even`, label: "p+Subs₉(p)", values: poly.cells.map((c, i) => i % 2 ? 0 : c.value * 2), revealAt: 2, tone: 0 });
  const odd = strip({ id: `${d.id}-odd`, label: "x⁻¹p+Subs₉", values: poly.cells.map((c, i) => i % 2 ? c.value * 2 : 0), revealAt: 2, tone: 0 });
  const outputs = cols(C).map((v, j) => batchColumn(v, `${d.id}-out-${j}`, `${d.outputLabel}${j}`, 5, j, "result"));
  const stages = [
    { id: `${d.id}-input`, title: decompressionOnly ? d.rightLabel : `${d.leftLabel} × ${d.rightLabel}`, caption: "", revealAt: 0,
      rows: [decompressionOnly
        ? [matrix("B", d.rightLabel, W, 0)]
        : [matrix("A", d.leftLabel, A, 0, "PT", "column"), operator(`${d.id}-times`, "×", "", 0), matrix("B", d.rightLabel, W, 0, "CT", "column")]] },
    { id: `${d.id}-compress`, title: "Compress", caption: "", revealAt: 1,
      rows: [[matrix(`${d.id}-wm`, d.rightLabel, W, 1), operator(`${d.id}-pack`, "→", "", 1), poly]] },
    { id: `${d.id}-subs`, title: "⊞ Subs · x⁻²ⁱ", caption: "", revealAt: 2,
      rows: [[poly, operator(`${d.id}-branch`, "↙ ↘", "", 2), even, odd]] },
    { id: `${d.id}-decompress`, title: "Decompress", caption: "", revealAt: 3,
      rows: [[even, odd, operator(`${d.id}-tree`, "→", "", 3), ...weights]] },
  ];
  if (decompressionOnly) {
    stages.push({ id: `${d.id}-simd`, title: "ENC([w,…,w])", caption: "", revealAt: 4, rows: [weights] });
  } else {
    const U = A.map((row, i) => row.map((_v, j) => (i + j) % 2));
    const masked = A.map((row, i) => row.map((v, j) => Number(v) - U[i][j]));
    const V = matmul(U, W), maskedProduct = matmul(masked, W);
    const corrections = cols(V).map((v, j) => batchColumn(v, `${d.id}-v-${j}`, `ENC(V${j})`, 4, j));
    const online = cols(maskedProduct).map((v, j) => batchColumn(v, `${d.id}-on-${j}`, `(A−U)W${j}`, 5, j, "product"));
    stages.push({ id: `${d.id}-offline`, title: "Offline U·W", caption: "", revealAt: 4,
      rows: [[matrix(`${d.id}-u`, "U", U, 4, "PT", "column"), operator(`${d.id}-uw`, "×", "", 4), ...weights, operator(`${d.id}-veq`, "=", "", 4), ...corrections]] });
    stages.push({ id: `${d.id}-online`, title: "Online (A−U)W ⊞ ENC(UW)", caption: "", revealAt: 5,
      rows: outputs.map((o, j) => [online[j], operator(`${d.id}-add-${j}`, "⊞", "", 5), corrections[j], operator(`${d.id}-eq-${j}`, "=", "", 5), o]) });
  }
  if (!decompressionOnly) stages.push(outputStage(d, C, outputs, 6));
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages };
  const steps = graphSteps(d, graph, A, W, C, stages.map((_s, i) => i === 2 ? { substitutions: 2, additions: 2 } : i === 3 ? { substitutions: 6, ptCtMultiplications: 8 } : i === 4 ? { ptCtMultiplications: A[0].length * W[0].length } : i === 5 && !decompressionOnly ? { ptCtMultiplications: A[0].length * W[0].length, additions: W[0].length } : {}));
  return shell(d, A, W, C, graph, steps, decompressionOnly ? "compressed polynomial → repeated SIMD weights" : "offline-online batched compressed-weight MM");
}

function diagonalVectors(C) { return [[C[0][0], C[1][1]], [C[0][1], C[1][0]]]; }
function buildCcmm(d, Q, K) {
  assertRect(Q, 2, 2, d.leftLabel); assertRect(K, 2, 2, d.rightLabel);
  const KT = transpose(K), C = matmul(Q, KT), q = cols(Q), k = cols(K);
  const qcts = q.map((v, i) => batchColumn(v, `${d.id}-q-${i}`, `q${i}`, 1, i));
  const kcts = k.map((v, i) => batchColumn(v, `${d.id}-k-${i}`, `k${i}`, 1, i + 2));
  const rotations = [0, 1].map((r) => kcts.map((a, j) => strip({ id: `${d.id}-kr-${r}-${j}`, label: r ? `RotL₂(k${j})` : `k${j}`,
    values: r ? rotatePairs(a.cells.map((c) => c.value), 1) : a.cells.map((c) => c.value), revealAt: 2, tone: j + 2 })));
  const diagonals = diagonalVectors(C).map((v, i) => batchColumn(v, `${d.id}-p-${i}`, `P${i}`, 3, i, "result"));
  const rows = diagonals.map((out, r) => [qcts[0], operator(`${d.id}-m-${r}-0`, "⊠", "", 3), rotations[r][0],
    operator(`${d.id}-plus-${r}`, "⊞", "", 3), qcts[1], operator(`${d.id}-m-${r}-1`, "⊠", "", 3), rotations[r][1], operator(`${d.id}-eq-${r}`, "=", "", 3), out]);
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: "Q × Kᵀ", caption: "", revealAt: 0, rows: [[matrix("A", "Q", Q, 0, "CT", "column"), operator(`${d.id}-times`, "×", "", 0), matrix("K", "K", K, 0, "CT", "column", 2)]] },
    { id: `${d.id}-col`, title: "Column pack", caption: "", revealAt: 1, rows: [qcts, kcts] },
    { id: `${d.id}-rotate`, title: "RotL₂(K)", caption: "", revealAt: 2, rows: rotations },
    { id: `${d.id}-multiply`, title: "Σ_j q_j ⊠ RotL(k_j)", caption: "", revealAt: 3, rows },
    outputStage(d, C, diagonals, 4),
  ] };
  const steps = graphSteps(d, graph, Q, KT, C, [{}, { ciphertextCount: 4 }, { rotations: 2 },
    { ctCtMultiplications: 4, additions: 2, relinearizations: 2, rescales: 2, multiplicativeDepth: 1 }, {}]);
  return shell(d, Q, KT, C, graph, steps, "column-packed Algorithm 5 → diagonal-packed");
}

function buildContext(d, S, V) {
  assertRect(S, 2, 2, d.leftLabel); assertRect(V, 2, 2, d.rightLabel);
  const packDot = (values) => [values[0], values[0], values[1], values[1], values[0], values[0], values[1], values[1]];
  const dotLabels = (values) => packDot(values).map((value, slot) => `${value}b${SUB[slot % 2]}`);
  const C = matmul(S, V), rows = S.map((v, i) => strip({ id: `${d.id}-s-${i}`, label: `S${i}`,
    values: packDot(v), labels: dotLabels(v), revealAt: 1, tone: i }));
  const vc = cols(V).map((v, j) => strip({ id: `${d.id}-v-${j}`, label: `V${j}`,
    values: packDot(v), labels: dotLabels(v), revealAt: 1, tone: j + 2 }));
  const products = [], sums = [], outputs = [];
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
    const vals = [S[i][0] * V[0][j], S[i][1] * V[1][j]];
    products.push(strip({ id: `${d.id}-p-${i}-${j}`, label: `p${i}${j}`, values: packDot(vals),
      labels: dotLabels(vals), revealAt: 2, tone: (i + j) % 4, kind: "product" }));
    sums.push(strip({ id: `${d.id}-h-${i}-${j}`, label: `${d.outputLabel}${i}${j}`,
      values: Array(SLOTS).fill(C[i][j]), labels: Array.from({ length: SLOTS }, (_x, slot) => `${C[i][j]}b${SUB[slot % 2]}`),
      revealAt: 3, tone: (i + j) % 4, kind: "result" }));
  }
  cols(C).forEach((v, j) => outputs.push(batchColumn(v, `${d.id}-out-${j}`, `${d.outputLabel}${j}`, 3, j, "result")));
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: `${d.leftLabel} × V`, caption: "", revealAt: 0, rows: [[matrix("A", d.leftLabel, S, 0, "CT", "row"), operator(`${d.id}-times`, "×", "", 0), matrix("B", "V", V, 0, "CT", "column", 2)]] },
    { id: `${d.id}-row`, title: "Row pack", caption: "", revealAt: 1, rows: [rows, vc] },
    { id: `${d.id}-multiply`, title: "⊠", caption: "", revealAt: 2,
      rows: products.map((p, z) => [rows[Math.floor(z / 2)], operator(`${d.id}-mul-${z}`, "⊠", "", 2), vc[z % 2], operator(`${d.id}-eq-${z}`, "=", "", 2), p]) },
    { id: `${d.id}-sum`, title: "QuickSum", caption: "", revealAt: 3,
      rows: products.map((p, z) => [p, operator(`${d.id}-sum-${z}`, "Σ RotL₂", "", 3), sums[z]]) },
    outputStage(d, C, outputs, 4),
  ] };
  const steps = graphSteps(d, graph, S, V, C, [{}, { ciphertextCount: 4 }, { ctCtMultiplications: 4, relinearizations: 4, rescales: 4 }, { rotations: 4, additions: 4 }, {}]);
  return shell(d, S, V, C, graph, steps, "row-packed ciphertext MM + QuickSum");
}

export function buildNexusOperationTrace(id, options = {}) {
  const d = getNexusOperation(id); if (!d) throw new Error(`Unknown NEXUS operation: ${id}`);
  if (options.leftSecret != null && Boolean(options.leftSecret) !== d.leftSecret) throw new Error(`${d.shortLabel} has a fixed left operand type`);
  if (options.rightSecret != null && Boolean(options.rightSecret) !== d.rightSecret) throw new Error(`${d.shortLabel} has a fixed right operand type`);
  if (options.slotsPerCiphertext != null && Number(options.slotsPerCiphertext) !== SLOTS) throw new Error("NEXUS toy uses 8 slots");
  const A = options.A || d.A, B = options.B || d.B;
  if (d.kind === "ccmm") return buildCcmm(d, A, B);
  if (d.kind === "context") return buildContext(d, A, B);
  return buildOfflineOnline(d, A, B, d.kind === "decompress");
}
export function nexusOperationDefaults(id) {
  const d = getNexusOperation(id); if (!d) return null;
  return { m: d.A.length, k: d.A[0].length, n: d.B[0].length, A: clone(d.A), B: clone(d.B), leftSecret: d.leftSecret,
    rightSecret: d.rightSecret, slotsPerCiphertext: SLOTS, strategy: "column" };
}
export function parseNexusOperationInputs(raw) { return parseAndValidateInputs(raw); }
