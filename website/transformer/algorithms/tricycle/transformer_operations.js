// Tricycle matrix-multiplication traces.
// The graphical flows mirror Sections 5-6 and Appendix H of the paper and
// the tricyclic-matmul / tricycle implementations: tricyclic encoding,
// broadcasted/batch/concatenated products, BSGS, lazy relinearization,
// the sole I'K/I'V packing change, and free attention complexification.

import { matmul } from "../../core/matrix.js";
import { parseAndValidateInputs } from "../../core/input_parser.js";
import { ctState, makeSteps, matrixArtifact, operator, rotateLeft, vectorArtifact } from "../thor/paper_utils.js";

export const TRICYCLE_OPERATION_IDS = Object.freeze({
  BROADCAST: "tricycle-broadcast",
  BATCH: "tricycle-batch",
  CONCAT: "tricycle-concat",
  BLOCK_BSGS: "tricycle-block-bsgs",
  QUERY: "tricycle-query",
  KEY: "tricycle-key",
  VALUE: "tricycle-value",
  KV_RESHAPE: "tricycle-kv-reshape",
  SCORE: "tricycle-score",
  CONTEXT: "tricycle-context",
  OUTPUT: "tricycle-output",
  FC1: "tricycle-fc1",
  FC2: "tricycle-fc2",
  POOLER: "tricycle-pooler",
  CLASSIFIER: "tricycle-classifier",
});

const SLOTS = 32;
const HEADS = 2;
const X35 = Object.freeze([[1, 0, 2, 1, 0], [0, 1, 1, 0, 2], [1, 1, 0, 2, 1]]);
const W54 = Object.freeze([[1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 0]]);
const Q32 = Object.freeze([[1, 0], [0, 1], [1, 1]]);
const KT25 = Object.freeze([[1, 0, 1, 0, 1], [0, 1, 1, 1, 0]]);
const S35 = Object.freeze([[0.5, 0.2, 0.1, 0.1, 0.1], [0.1, 0.5, 0.2, 0.1, 0.1], [0.2, 0.1, 0.5, 0.1, 0.1]]);
const V52 = Object.freeze([[1, 0], [0, 1], [1, 1], [2, 0], [0, 2]]);
const H34 = Object.freeze([[1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 1, 1]]);
const WO45 = Object.freeze([[1, 0, 0, 1, 0], [0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 1]]);
const I53 = Object.freeze([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]]);
const K32 = Object.freeze([[1, 0], [0, 1], [1, 1]]);
const W45 = Object.freeze([[1, 0, 0, 1, 0], [0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 1]]);
const P15 = Object.freeze([[1, 0, 2, 1, 0]]);
const PW53 = Object.freeze([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]]);
const C13 = Object.freeze([[1, 2, 1]]);
const CW32 = Object.freeze([[1, 0], [0, 1], [1, 1]]);
const A44 = Object.freeze([[1, 2, 0, 1], [0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1]]);
const B42 = Object.freeze([[1, 0], [0, 1], [1, 1], [0, 1]]);

function operation(id, config) {
  return Object.freeze({ id, method: "tricycle", slotsPerCiphertext: SLOTS, strategy: "tricyclic", batchCount: config.batchCount ?? 1, ...config });
}
function linear(id, shortLabel, leftLabel, rightLabel, outputLabel, group, A, B, complexMode = "none", batchCount = 1) {
  return operation(id, { shortLabel, group, kind: "linear", leftLabel, rightLabel, outputLabel,
    leftSecret: true, rightSecret: false, A, B, complexMode, batchCount });
}

export const TRICYCLE_OPERATIONS = Object.freeze({
  [TRICYCLE_OPERATION_IDS.BROADCAST]: linear(TRICYCLE_OPERATION_IDS.BROADCAST, "Broadcast MM", "A", "B", "Z", "paper", X35, W54, "none", HEADS),
  [TRICYCLE_OPERATION_IDS.BATCH]: operation(TRICYCLE_OPERATION_IDS.BATCH, { shortLabel: "Batch MM", group: "paper", kind: "batch",
    leftLabel: "A", rightLabel: "B", outputLabel: "Z", leftSecret: true, rightSecret: true, A: S35, B: V52, batchCount: HEADS }),
  [TRICYCLE_OPERATION_IDS.CONCAT]: operation(TRICYCLE_OPERATION_IDS.CONCAT, { shortLabel: "Concat MM", group: "paper", kind: "concat",
    leftLabel: "A", rightLabel: "B", outputLabel: "Z", leftSecret: true, rightSecret: false, A: H34, B: WO45, batchCount: HEADS }),
  [TRICYCLE_OPERATION_IDS.BLOCK_BSGS]: operation(TRICYCLE_OPERATION_IDS.BLOCK_BSGS, { shortLabel: "Block BSGS", group: "paper", kind: "block",
    leftLabel: "A", rightLabel: "B", outputLabel: "Z", leftSecret: true, rightSecret: false, A: A44, B: B42, batchCount: 2 }),
  [TRICYCLE_OPERATION_IDS.QUERY]: linear(TRICYCLE_OPERATION_IDS.QUERY, "XWq", "X", "Wq", "Q", "attention", X35, W54, "qkv", HEADS),
  [TRICYCLE_OPERATION_IDS.KEY]: linear(TRICYCLE_OPERATION_IDS.KEY, "XWk", "X", "Wk", "K", "attention", X35, W54, "qkv", HEADS),
  [TRICYCLE_OPERATION_IDS.VALUE]: linear(TRICYCLE_OPERATION_IDS.VALUE, "XWv", "X", "Wv", "V", "attention", X35, W54, "qkv", HEADS),
  [TRICYCLE_OPERATION_IDS.KV_RESHAPE]: operation(TRICYCLE_OPERATION_IDS.KV_RESHAPE, { shortLabel: "I′K / I′V", group: "attention", kind: "reshape",
    leftLabel: "I′", rightLabel: "K / V", outputLabel: "K′ / V′", leftSecret: false, rightSecret: true, A: I53, B: K32, batchCount: HEADS }),
  [TRICYCLE_OPERATION_IDS.SCORE]: operation(TRICYCLE_OPERATION_IDS.SCORE, { shortLabel: "QKᵀ", group: "attention", kind: "score",
    leftLabel: "Q", rightLabel: "K′ᵀ", outputLabel: "P", leftSecret: true, rightSecret: true, A: Q32, B: KT25, batchCount: HEADS }),
  [TRICYCLE_OPERATION_IDS.CONTEXT]: operation(TRICYCLE_OPERATION_IDS.CONTEXT, { shortLabel: "Softmax·V", group: "attention", kind: "context",
    leftLabel: "Softmax(P)", rightLabel: "V′", outputLabel: "H", leftSecret: true, rightSecret: true, A: S35, B: V52, batchCount: HEADS }),
  [TRICYCLE_OPERATION_IDS.OUTPUT]: operation(TRICYCLE_OPERATION_IDS.OUTPUT, { shortLabel: "HWo", group: "attention", kind: "concat",
    leftLabel: "H", rightLabel: "Wo", outputLabel: "O", leftSecret: true, rightSecret: false, A: H34, B: WO45, complexMode: "output", batchCount: HEADS }),
  [TRICYCLE_OPERATION_IDS.FC1]: linear(TRICYCLE_OPERATION_IDS.FC1, "FC1", "X", "Wff1", "F1", "ffn", X35, W54, "ff1"),
  [TRICYCLE_OPERATION_IDS.FC2]: linear(TRICYCLE_OPERATION_IDS.FC2, "FC2", "GELU(F1)", "Wff2", "F2", "ffn", H34, W45, "ff2"),
  [TRICYCLE_OPERATION_IDS.POOLER]: linear(TRICYCLE_OPERATION_IDS.POOLER, "Pooler", "X[CLS]", "Wp", "T", "output", P15, PW53),
  [TRICYCLE_OPERATION_IDS.CLASSIFIER]: linear(TRICYCLE_OPERATION_IDS.CLASSIFIER, "Classifier", "Tanh(T)", "Wc", "Logits", "output", C13, CW32),
});

export function listTricycleOperations(group = null) {
  return Object.values(TRICYCLE_OPERATIONS).filter((item) => !group || item.group === group);
}
export function getTricycleOperation(id) { return TRICYCLE_OPERATIONS[id] || null; }
function clone(A) { return A.map((row) => [...row]); }
function assertRect(A, rows, columns, label) {
  if (!Array.isArray(A) || A.length !== rows || A.some((row) => !Array.isArray(row) || row.length !== columns)) {
    throw new Error(`${label} must be ${rows}×${columns}`);
  }
}
function zlabel(real, imag) {
  const r = Number(real); const i = Number(imag);
  return `${r}${i < 0 ? "−" : "+"}${Math.abs(i)}i`;
}
function valueOf(item) { return typeof item === "object" ? Number(item.value) : Number(item); }
function labelOf(item, fallback) { return typeof item === "object" && item.label != null ? String(item.label) : fallback; }

function strip({ id, label, values, labels = null, revealAt, tone = 0, badge = "CT", kind = "ciphertext" }) {
  const padded = Array.from({ length: SLOTS }, (_, i) => Number(values[i] ?? 0));
  const artifact = vectorArtifact({ id, label, values: padded, revealAt, badge, kind,
    // Color follows the packed source identity, not the encryption type.
    tones: Array(SLOTS).fill(tone) });
  artifact.cells.forEach((cell, i) => { cell.label = labels?.[i] ?? (i < values.length ? String(values[i]) : "·"); });
  return artifact;
}
function triStrip(tensor, dims, id, label, revealAt, tone = 0, badge = "CT", kind = "ciphertext") {
  const [h, m, n] = dims; const count = h * m * n; const values = []; const labels = [];
  for (let slot = 0; slot < count; slot += 1) {
    const hi = slot % h; const ri = slot % m; const ci = slot % n;
    const item = tensor[hi][ri][ci]; const value = valueOf(item);
    values.push(value); labels.push(`${labelOf(item, value)}|h${hi}r${ri}c${ci}`);
  }
  return strip({ id, label, values, labels, revealAt, tone, badge, kind });
}
function matrix(id, label, values, revealAt, badge = "CT", layout = "column", toneOffset = 0) {
  return matrixArtifact({ id, label, values, revealAt, badge, meta: "",
    toneForCell: (row, col) => layout === "row" ? (row + toneOffset) % 4
      : layout === "diagonal" ? ((col - row) % 4 + 4) % 4 : (col + toneOffset) % 4 });
}
function plainDiag(B, id, label, revealAt, tone = 0) {
  const flat = B.flat().map(Number);
  return strip({ id, label, values: Array.from({ length: SLOTS }, (_, i) => flat[i % flat.length]), revealAt, tone, badge: "PT" });
}
function artifactCiphertexts(stage) {
  const out = []; const seen = new Set();
  for (const row of stage.rows || []) for (const artifact of row) {
    if (artifact.badge !== "CT" || !artifact.cells || seen.has(artifact.id)) continue;
    seen.add(artifact.id);
    out.push(ctState(artifact.id, artifact.label, artifact.cells.map((cell) => cell.value), SLOTS, true,
      artifact.type === "product" ? "product" : artifact.type === "result" ? "result" : "tensor"));
  }
  return out;
}
function graphSteps(d, graph, A, B, C, deltas = []) {
  return makeSteps(d.id, graph.stages.map((stage, i) => ({
    phase: i === 0 ? "reference" : i === graph.stages.length - 1 ? "output" : "multiply",
    title: stage.title, description: "", formula: "", tensors: {
      left: { id: "A", label: d.leftLabel, shape: [A.length, A[0].length], values: A },
      right: { id: "B", label: d.rightLabel, shape: [B.length, B[0].length], values: B },
      ...(i === graph.stages.length - 1 ? { output: { id: "C", label: d.outputLabel, shape: [C.length, C[0].length], values: C } } : {}),
    }, ciphertexts: artifactCiphertexts(stage), delta: deltas[i] || {},
  })));
}
function matmulType(d) {
  if (d.leftSecret && d.rightSecret) return "ct-ct";
  if (!d.leftSecret && d.rightSecret) return "pt-ct";
  return "ct-pt";
}
function shell(d, A, B, C, graph, steps, flow) {
  return { schemaVersion: "1.0", id: `${d.id}-toy`, title: `Cryptology ePrint'25 Tricycle ${d.shortLabel}`,
    parameters: { matrixShape: { m: A.length, k: A[0].length, n: B[0].length }, slotsPerCiphertext: SLOTS,
      matmulType: matmulType(d), algorithmId: d.id, batchCount: d.batchCount },
    operands: { left: { id: "A", label: d.leftLabel, shape: [A.length, A[0].length], values: A, secret: d.leftSecret },
      right: { id: "B", label: d.rightLabel, shape: [B.length, B[0].length], values: B, secret: d.rightSecret } },
    output: { id: "C", label: d.outputLabel, shape: [C.length, C[0].length] },
    plan: { algorithmId: d.id, method: "tricycle", methodLabel: "Cryptology ePrint'25 Tricycle", visualStyle: "diagram-only",
      matmulType: matmulType(d), logicalShape: { m: A.length, k: A[0].length, n: B[0].length },
      paddedShape: { m: A.length, k: A[0].length, n: B[0].length }, tileGrid: { m: 1, n: 1 },
      tileShape: { m: C.length, n: C[0].length }, packingStrategyRequested: "tricyclic",
      packingStrategyResolved: flow, plannerSource: "Tricycle Sections 5-6, Appendix H, and tricycle::transformer_layer_agnostic_preprocess",
      reductionStrategy: flow, displayMode: "diagram-only", batchLabel: d.batchCount > 1 ? "heads" : "tensor blocks", warnings: [] },
    graph, steps, finalTensor: C };
}
function outputStage(d, C, encoded, revealAt) {
  return { id: `${d.id}-output`, title: d.outputLabel, caption: "", revealAt,
    rows: [[encoded], [operator(`${d.id}-down`, "↓", "", revealAt), matrix("C", d.outputLabel, C, revealAt, "CT", "column")]] };
}
function realTensor(M) { return [M.map((row) => row.map(Number))]; }
function duplicateRealTensor(M, heads = HEADS) {
  return Array.from({ length: heads }, () => M.map((row) => row.map(Number)));
}
function complexHeadsFromColumns(M, heads = HEADS) {
  return Array.from({ length: heads }, (_, h) => M.map((row) => [{ value: Number(row[2 * h]), label: zlabel(row[2 * h], row[2 * h + 1]) }]));
}
function duplicateComplexColumns(M, heads = HEADS) {
  return Array.from({ length: heads }, () => M.map((row) => [{ value: Number(row[0]), label: zlabel(row[0], row[1]) }]));
}
function duplicateComplexRows(M, heads = HEADS) {
  return Array.from({ length: heads }, () => [Array.from({ length: M[0].length }, (_, j) => ({ value: Number(M[0][j]), label: zlabel(M[0][j], M[1][j]) }))]);
}
function outputComplexTensor(C, heads = HEADS) { return duplicateComplexColumns(C, heads); }
function concatWeightTensor(B) {
  return Array.from({ length: HEADS }, (_, h) => [B[0].map((_v, j) => ({ value: Number(B[2 * h][j]), label: zlabel(B[2 * h][j], -B[2 * h + 1][j]) }))]);
}

function buildLinear(d, A, B) {
  assertRect(A, d.A.length, d.A[0].length, d.leftLabel); assertRect(B, d.B.length, d.B[0].length, d.rightLabel);
  const C = matmul(A, B); const isQkv = d.complexMode === "qkv" || d.id === TRICYCLE_OPERATION_IDS.BROADCAST;
  const inputTensor = realTensor(A); const inputCt = triStrip(inputTensor, [1, A.length, A[0].length], `${d.id}-x`, "φ(X)", 1, 0);
  let outputTensor;
  if (B[0].length === 4) outputTensor = complexHeadsFromColumns(C);
  else if (C[0].length === 2) outputTensor = duplicateComplexColumns(C, 1);
  else outputTensor = realTensor(C);
  const outDims = B[0].length === 4 ? [HEADS, C.length, 1] : [1, C.length, Math.max(1, C[0].length / (C[0].length === 2 ? 2 : 1))];
  const weightTensor = B[0].length === 4 ? complexHeadsFromColumns(B) : realTensor(B);
  const weightDims = B[0].length === 4 ? [HEADS, B.length, 1] : [1, B.length, B[0].length];
  const weight = triStrip(weightTensor, weightDims, `${d.id}-w`, isQkv ? "φ(WL+iWR)" : "φ(W)", 1, 4, "PT");
  const baby0 = strip({ id: `${d.id}-b0`, label: "β₀", values: inputCt.cells.map((c) => c.value), revealAt: 2, tone: 0 });
  const baby1 = strip({ id: `${d.id}-b1`, label: "β₁", values: rotateLeft(inputCt.cells.map((c) => c.value), A.length), revealAt: 2, tone: 0 });
  const diag0 = plainDiag(B, `${d.id}-d0`, "D₀", 3, 0); const diag1 = plainDiag(B.map((row) => [...row].reverse()), `${d.id}-d1`, "D₁", 3, 1);
  const p0 = strip({ id: `${d.id}-p0`, label: "β₀⊙D₀", values: baby0.cells.map((c, i) => c.value * diag0.cells[i].value), revealAt: 3, tone: 1, kind: "product" });
  const p1 = strip({ id: `${d.id}-p1`, label: "β₁⊙D₁", values: baby1.cells.map((c, i) => c.value * diag1.cells[i].value), revealAt: 3, tone: 1, kind: "product" });
  const out = triStrip(outputTensor, outDims, `${d.id}-z`, d.outputLabel, 4, 2, "CT", "result");
  const stages = [
    { id: `${d.id}-input`, title: `${d.leftLabel} × ${d.rightLabel}`, caption: "", revealAt: 0,
      rows: [[matrix("A", d.leftLabel, A, 0, "CT", "column"), operator(`${d.id}-times`, "×", "", 0), matrix("B", d.rightLabel, B, 0, "PT", "column")]] },
    { id: `${d.id}-phi`, title: "φ(h,m,n)", caption: "", revealAt: 1,
      rows: [[inputCt], [weight]] },
    { id: `${d.id}-baby`, title: "βᵢ = ρ(φ(X), i·hm)", caption: "", revealAt: 2, rows: [[baby0], [baby1]] },
    { id: `${d.id}-diag`, title: "βᵢ ⊙ BatchDiag(W,i)", caption: "", revealAt: 3,
      rows: [[baby0, operator(`${d.id}-mul0`, "⊙", "", 3), diag0, operator(`${d.id}-eq0`, "=", "", 3), p0],
        [baby1, operator(`${d.id}-mul1`, "⊙", "", 3), diag1, operator(`${d.id}-eq1`, "=", "", 3), p1]] },
    { id: `${d.id}-bsgs`, title: "Σg ρg(Σi βᵢ⊙Dgᵢ)", caption: "", revealAt: 4,
      rows: [[p0, operator(`${d.id}-sum`, "⊞ ρg ⊞", "", 4), p1, operator(`${d.id}-eqz`, "=", "", 4), out]] },
  ];
  let encodedOutput = out; let outputAt = 5;
  if (d.complexMode === "ff1") {
    const left = strip({ id: `${d.id}-real`, label: "F₁L", values: out.cells.map((c) => c.value), revealAt: 5, tone: 2, kind: "result" });
    const right = strip({ id: `${d.id}-imag`, label: "F₁R", values: out.cells.map((c, i) => i % 2 ? c.value : 0), revealAt: 5, tone: 3, kind: "result" });
    stages.push({ id: `${d.id}-decomplex`, title: "C+Conj(C) · −iC+Conj(−iC)", caption: "", revealAt: 5,
      rows: [[out, operator(`${d.id}-conj`, "Conj", "", 5), left, right]] });
    encodedOutput = left; outputAt = 6;
  } else if (d.complexMode === "ff2") {
    const rec = strip({ id: `${d.id}-recomplex`, label: "GELU(FL)+i·GELU(FR)", values: inputCt.cells.map((c) => c.value), revealAt: 2, tone: 0 });
    stages.splice(2, 0, { id: `${d.id}-recomplex-stage`, title: "FL+iFR", caption: "", revealAt: 2, rows: [[rec]] });
    stages.forEach((stage, index) => { stage.revealAt = index; for (const row of stage.rows) for (const artifact of row) artifact.revealAt = index; });
    const real = strip({ id: `${d.id}-realout`, label: "F₂", values: out.cells.map((c) => c.value), revealAt: 6, tone: 2, kind: "result" });
    stages.push({ id: `${d.id}-conjugate`, title: "C + Conj(C)", caption: "", revealAt: 6,
      rows: [[out, operator(`${d.id}-real-op`, "+ Conj", "", 6), real]] });
    encodedOutput = real; outputAt = 7;
  }
  stages.push(outputStage(d, C, encodedOutput, outputAt));
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages };
  const deltas = stages.map((_stage, i) => i === 2 ? { rotations: 2 } : i === 3 ? { ptCtMultiplications: 2, rescales: 2, multiplicativeDepth: 1 } : i === 4 ? { rotations: 2, additions: 2 } : {});
  return shell(d, A, B, C, graph, graphSteps(d, graph, A, B, C, deltas), "broadcasted tricyclic CT×PT + BSGS");
}

function buildBatch(d, A, B, { score = false, masking = false } = {}) {
  assertRect(A, d.A.length, d.A[0].length, d.leftLabel); assertRect(B, d.B.length, d.B[0].length, d.rightLabel);
  const C = matmul(A, B);
  const leftTensor = score ? duplicateComplexColumns(A) : duplicateRealTensor(A);
  const rightTensor = score ? duplicateComplexRows(B) : duplicateComplexColumns(B);
  const leftDims = score ? [HEADS, A.length, 1] : [HEADS, A.length, A[0].length];
  const rightDims = score ? [HEADS, 1, B[0].length] : [HEADS, B.length, 1];
  const outTensor = score ? duplicateRealTensor(C) : outputComplexTensor(C);
  const outDims = score ? [HEADS, C.length, C[0].length] : [HEADS, C.length, 1];
  const left = triStrip(leftTensor, leftDims, `${d.id}-a`, score ? "φ(Q)" : "φ(S)", 1, 0);
  const right = triStrip(rightTensor, rightDims, `${d.id}-b`, score ? "φ(K′)" : "φ(V′)", 1, 1);
  const transposed = score ? strip({ id: `${d.id}-kt`, label: "φ(K′ᵀ)", values: right.cells.map((c) => c.value),
    labels: right.cells.map((c) => c.label), revealAt: 2, tone: 1 }) : null;
  const maskTarget = score ? left : right;
  const masked = strip({ id: `${d.id}-mask`, label: score ? "ρ(Q)⊙M/√dk" : "ρ(V′)⊙M", values: maskTarget.cells.map((c, i) => i < 30 ? c.value : 0),
    labels: maskTarget.cells.map((c, i) => i < 30 ? c.label : "0"), revealAt: score ? 3 : 2, tone: score ? 0 : 1 });
  const lhs = score ? masked : left; const rhs = score ? transposed : masked; const mulAt = score ? 4 : 3;
  const product = strip({ id: `${d.id}-prod`, label: "EvalMultNoRelin", values: lhs.cells.map((c, i) => c.value * rhs.cells[i].value),
    revealAt: mulAt, tone: 2, kind: "product" });
  const sum = triStrip(outTensor, outDims, `${d.id}-sum`, d.outputLabel, mulAt + 1, 2, "CT", "result");
  const stages = [
    { id: `${d.id}-input`, title: `${d.leftLabel} × ${d.rightLabel}`, caption: "", revealAt: 0,
      rows: [[matrix("A", d.leftLabel, A, 0, "CT", score ? "column" : "row"), operator(`${d.id}-times`, "×", "", 0), matrix("B", d.rightLabel, B, 0, "CT", "column", 2)]] },
    { id: `${d.id}-phi`, title: "φ(h,m,n)", caption: "", revealAt: 1, rows: [[left], [right]] },
  ];
  if (score) stages.push({ id: `${d.id}-transpose`, title: "φ(K′) = φ(K′ᵀ)", caption: "", revealAt: 2,
    rows: [[right, operator(`${d.id}-free`, "≡", "", 2), transposed]] });
  if (masking || score) stages.push({ id: `${d.id}-mask-stage`, title: score ? "ρ(Q) ⊙ M/√dk" : "ρ(V′) ⊙ M", caption: "", revealAt: score ? 3 : 2,
    rows: [[maskTarget, operator(`${d.id}-mask-op`, "⊙ M", "", score ? 3 : 2), masked]] });
  stages.push({ id: `${d.id}-bsgs`, title: score ? "Right-BSGS ⊠" : "Left-BSGS ⊠", caption: "", revealAt: mulAt,
    rows: [[lhs, operator(`${d.id}-mul`, "⊠", "", mulAt), rhs, operator(`${d.id}-eq`, "=", "", mulAt), product]] });
  stages.push({ id: `${d.id}-lazy`, title: "Σbaby → Relinearize → ρgiant", caption: "", revealAt: mulAt + 1,
    rows: [[product, operator(`${d.id}-lazy-op`, "Σ → Relin → ρ", "", mulAt + 1), sum]] });
  let encodedOutput = sum; let outputAt = mulAt + 2;
  if (score) {
    const real = strip({ id: `${d.id}-real`, label: "P", values: sum.cells.map((c) => c.value), labels: sum.cells.map((c) => c.label), revealAt: mulAt + 2, tone: 2, kind: "result" });
    stages.push({ id: `${d.id}-conj`, title: "½(C + Conj(C))", caption: "", revealAt: mulAt + 2,
      rows: [[sum, operator(`${d.id}-conj-op`, "+ Conj", "", mulAt + 2), real]] });
    encodedOutput = real; outputAt += 1;
  }
  stages.push(outputStage(d, C, encodedOutput, outputAt));
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages };
  const deltas = stages.map((stage) => stage.id.endsWith("-bsgs") ? { rotations: 3, ctCtMultiplications: 3, rescales: 1, multiplicativeDepth: 1 }
    : stage.id.endsWith("-lazy") ? { additions: 2, relinearizations: 1, rotations: 1 } : {});
  return shell(d, A, B, C, graph, graphSteps(d, graph, A, B, C, deltas), "tricyclic CT×CT batch MM + BSGS + lazy relinearization");
}

function buildConcat(d, A, B) {
  assertRect(A, d.A.length, d.A[0].length, d.leftLabel); assertRect(B, d.B.length, d.B[0].length, d.rightLabel);
  const C = matmul(A, B); const aTensor = complexHeadsFromColumns(A); const bTensor = concatWeightTensor(B);
  const left = triStrip(aTensor, [HEADS, A.length, 1], `${d.id}-a`, "φ(HL+iHR)", 1, 0);
  const weight = triStrip(bTensor, [HEADS, 1, B[0].length], `${d.id}-w`, "φ(WoL−iWoR)", 1, 4, "PT");
  const baby0 = strip({ id: `${d.id}-b0`, label: "β₀", values: left.cells.map((c) => c.value), labels: left.cells.map((c) => c.label), revealAt: 2, tone: 0 });
  const baby1 = strip({ id: `${d.id}-b1`, label: "β₁", values: rotateLeft(left.cells.map((c) => c.value), A.length), revealAt: 2, tone: 0 });
  const product = strip({ id: `${d.id}-prod`, label: "Σc,d β⊙ConcatDiag", values: baby0.cells.map((c, i) => c.value * weight.cells[i].value + baby1.cells[i].value), revealAt: 3, tone: 2, kind: "product" });
  const complexOut = triStrip(realTensor(C), [1, C.length, C[0].length], `${d.id}-complex`, "O+i·junk", 4, 2, "CT", "result");
  const realOut = strip({ id: `${d.id}-real`, label: d.outputLabel, values: complexOut.cells.map((c) => c.value), labels: complexOut.cells.map((c) => c.label), revealAt: 5, tone: 2, kind: "result" });
  const stages = [
    { id: `${d.id}-input`, title: `${d.leftLabel} × ${d.rightLabel}`, caption: "", revealAt: 0,
      rows: [[matrix("A", d.leftLabel, A, 0, "CT", "column"), operator(`${d.id}-times`, "×", "", 0), matrix("B", d.rightLabel, B, 0, "PT", "column")]] },
    { id: `${d.id}-phi`, title: "φ(h,m,n)", caption: "", revealAt: 1, rows: [[left], [weight]] },
    { id: `${d.id}-baby`, title: "βᵢ = ρ(φ(H), i·hm)", caption: "", revealAt: 2, rows: [[baby0], [baby1]] },
    { id: `${d.id}-concat`, title: "Σhead Σc βᵢ ⊙ ConcatDiag", caption: "", revealAt: 3,
      rows: [[baby0, operator(`${d.id}-mul`, "⊙", "", 3), weight, operator(`${d.id}-sum`, "Σh,c", "", 3), product]] },
    { id: `${d.id}-giant`, title: "Σg ρg(Σi ·)", caption: "", revealAt: 4,
      rows: [[product, operator(`${d.id}-rho`, "Σ ρg", "", 4), complexOut]] },
    { id: `${d.id}-conj`, title: "C + Conj(C)", caption: "", revealAt: 5,
      rows: [[complexOut, operator(`${d.id}-conj-op`, "+ Conj", "", 5), realOut]] },
    outputStage(d, C, realOut, 6),
  ];
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages };
  return shell(d, A, B, C, graph, graphSteps(d, graph, A, B, C, [{}, {}, { rotations: 2 },
    { ptCtMultiplications: 4, additions: 3, rescales: 1, multiplicativeDepth: 1 }, { rotations: 2, additions: 2 }, { additions: 1 }, {}]),
    "concatenated tricyclic CT×PT across heads");
}

function buildReshape(d, I, K) {
  assertRect(I, 5, 3, d.leftLabel); assertRect(K, 3, 2, d.rightLabel);
  const C = matmul(I, K); const V = K.map((row) => [row[1], row[0]]); const VC = matmul(I, V);
  const ipt = triStrip(realTensor(I), [1, 5, 3], `${d.id}-i`, "φ(I′)", 1, 4, "PT");
  const kct = triStrip(duplicateComplexColumns(K), [HEADS, 3, 1], `${d.id}-k`, "φ(K)", 1, 0);
  const vct = triStrip(duplicateComplexColumns(V), [HEADS, 3, 1], `${d.id}-v`, "φ(V)", 1, 1);
  const kout = triStrip(duplicateComplexColumns(C), [HEADS, 5, 1], `${d.id}-ko`, "φ(K′)", 3, 0, "CT", "result");
  const vout = triStrip(duplicateComplexColumns(VC), [HEADS, 5, 1], `${d.id}-vo`, "φ(V′)", 3, 1, "CT", "result");
  const stages = [
    { id: `${d.id}-input`, title: "I′ × K / V", caption: "", revealAt: 0,
      rows: [[matrix("A", "I′", I, 0, "PT", "column"), operator(`${d.id}-times`, "×", "", 0), matrix("B", "K / V", K, 0, "CT", "column", 2)]] },
    { id: `${d.id}-phi`, title: "φ(1,L+r,L) · φ(h,L,dk/2)", caption: "", revealAt: 1, rows: [[ipt], [kct], [vct]] },
    { id: `${d.id}-ptct`, title: "ρ(φ(K/V)) ⊙ BatchDiag(I′)", caption: "", revealAt: 2,
      rows: [[kct, operator(`${d.id}-kmul`, "⊙ D(I′)", "", 2)], [vct, operator(`${d.id}-vmul`, "⊙ D(I′)", "", 2)]] },
    { id: `${d.id}-reshape`, title: "Σρ · +r zero rows", caption: "", revealAt: 3,
      rows: [[kout], [vout]] },
    outputStage(d, C, kout, 4),
  ];
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages };
  return shell(d, I, K, C, graph, graphSteps(d, graph, I, K, C, [{}, {}, { rotations: 2, ptCtMultiplications: 2, rescales: 2, multiplicativeDepth: 1 }, { additions: 2 }, {}]),
    "broadcasted PT×CT tall-identity reshape");
}

function blockValues(M, br, bc, rows = 2, cols = 2) {
  const out = []; for (let i = 0; i < rows; i += 1) for (let j = 0; j < cols; j += 1) out.push(Number(M[br * rows + i]?.[bc * cols + j] ?? 0));
  return out;
}
function blockStrip(values, id, label, revealAt, tone, badge = "CT", kind = "ciphertext") {
  return strip({ id, label, values, labels: values.map((v, i) => `${v}|${i >> 1}${i & 1}`), revealAt, tone, badge, kind });
}
function buildBlock(d, A, B) {
  assertRect(A, 4, 4, d.leftLabel); assertRect(B, 4, 2, d.rightLabel); const C = matmul(A, B);
  const x11 = blockStrip(blockValues(A, 0, 0), `${d.id}-x11`, "X₁₁", 1, 0); const x12 = blockStrip(blockValues(A, 0, 1), `${d.id}-x12`, "X₁₂", 1, 1);
  const x21 = blockStrip(blockValues(A, 1, 0), `${d.id}-x21`, "X₂₁", 1, 2); const x22 = blockStrip(blockValues(A, 1, 1), `${d.id}-x22`, "X₂₂", 1, 3);
  const y11 = blockStrip(blockValues(B, 0, 0), `${d.id}-y11`, "Y₁₁", 1, 4, "PT"); const y21 = blockStrip(blockValues(B, 1, 0), `${d.id}-y21`, "Y₂₁", 1, 4, "PT");
  const babies = [x11, x12, x21, x22].map((x, i) => strip({ id: `${d.id}-beta-${i}`, label: `β(X${(i >> 1) + 1}${(i & 1) + 1})`,
    values: rotateLeft(x.cells.map((c) => c.value), i % 2), revealAt: 2, tone: i }));
  const p1 = strip({ id: `${d.id}-p1`, label: "GS(X₁₁,Y₁₁)+GS(X₁₂,Y₂₁)", values: babies[0].cells.map((c, i) => c.value * y11.cells[i].value + babies[1].cells[i].value * y21.cells[i].value), revealAt: 3, tone: 1, kind: "product" });
  const p2 = strip({ id: `${d.id}-p2`, label: "GS(X₂₁,Y₁₁)+GS(X₂₂,Y₂₁)", values: babies[2].cells.map((c, i) => c.value * y11.cells[i].value + babies[3].cells[i].value * y21.cells[i].value), revealAt: 3, tone: 2, kind: "product" });
  const cout = triStrip(realTensor(C), [1, 4, 2], `${d.id}-out`, "Z", 4, 2, "CT", "result");
  const stages = [
    { id: `${d.id}-input`, title: "4×4 · 4×2", caption: "", revealAt: 0,
      rows: [[matrix("A", "X", A, 0, "CT", "column"), operator(`${d.id}-times`, "×", "", 0), matrix("B", "Y", B, 0, "PT", "column")]] },
    { id: `${d.id}-blocks`, title: "2×2 blocks", caption: "", revealAt: 1, rows: [[x11], [x12], [x21], [x22], [y11], [y21]] },
    { id: `${d.id}-babies`, title: "Shared baby rotations", caption: "", revealAt: 2, rows: babies.map((b) => [b]) },
    { id: `${d.id}-aggregate`, title: "Aggregate equal giant steps", caption: "", revealAt: 3, rows: [[p1], [p2]] },
    { id: `${d.id}-lazy`, title: "ρgiant → lazy Relin", caption: "", revealAt: 4,
      rows: [[p1, p2, operator(`${d.id}-lazy-op`, "ρ → Relin", "", 4), cout]] },
    outputStage(d, C, cout, 5),
  ];
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages };
  return shell(d, A, B, C, graph, graphSteps(d, graph, A, B, C, [{}, {}, { rotations: 4 }, { ptCtMultiplications: 4, additions: 2, rescales: 2, multiplicativeDepth: 1 }, { rotations: 2, relinearizations: 2 }, {}]),
    "optimized block BSGS with shared rotations");
}

export function buildTricycleOperationTrace(id, options = {}) {
  const d = getTricycleOperation(id); if (!d) throw new Error(`Unknown Tricycle operation: ${id}`);
  if (options.leftSecret != null && Boolean(options.leftSecret) !== d.leftSecret) throw new Error(`${d.shortLabel} has a fixed left operand type`);
  if (options.rightSecret != null && Boolean(options.rightSecret) !== d.rightSecret) throw new Error(`${d.shortLabel} has a fixed right operand type`);
  if (options.slotsPerCiphertext != null && Number(options.slotsPerCiphertext) !== SLOTS) throw new Error("Tricycle toy uses 32 CKKS slots");
  const A = options.A || d.A; const B = options.B || d.B;
  if (d.kind === "reshape") return buildReshape(d, A, B);
  if (d.kind === "score") return buildBatch(d, A, B, { score: true, masking: true });
  if (d.kind === "context") return buildBatch(d, A, B, { masking: true });
  if (d.kind === "batch") return buildBatch(d, A, B);
  if (d.kind === "concat") return buildConcat(d, A, B);
  if (d.kind === "block") return buildBlock(d, A, B);
  return buildLinear(d, A, B);
}

export function tricycleOperationDefaults(id) {
  const d = getTricycleOperation(id); if (!d) return null;
  return { m: d.A.length, k: d.A[0].length, n: d.B[0].length, A: clone(d.A), B: clone(d.B),
    leftSecret: d.leftSecret, rightSecret: d.rightSecret, slotsPerCiphertext: SLOTS, strategy: "tricyclic" };
}
export function parseTricycleOperationInputs(raw) { return parseAndValidateInputs(raw); }
