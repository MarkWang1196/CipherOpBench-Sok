// STIP compact-packing matrix-multiplication traces.
//
// The stages follow the paper algorithms and the released STIP_func code:
// AMCP group rotations (Algorithm 3), DHP complex head pairing (Algorithm 2),
// RIHP score/L2 packing (Algorithms 1/4/5), and RIHP+BSGS attention output
// (Algorithm 6).  The toy keeps m=t=k=2 and S=8 so every physical slot is
// visible while preserving STIP's sequence-major interleaving.

import { matmul, squaredL2Distances, transpose } from "../../core/matrix.js";
import { parseAndValidateInputs } from "../../core/input_parser.js";
import { addVectors, ctState, makeSteps, matrixArtifact, mod, multiplyVectors, operator, rotateLeft, vectorArtifact } from "../thor/paper_utils.js";
import { complexLabel, multiplyComplex, sumComplexVectors } from "../complex_slots.js";

export const STIP_OPERATION_IDS = Object.freeze({
  RIHP: "stip-rihp",
  L2: "stip-l2",
  DHP: "stip-dhp",
  AMCP: "stip-amcp",
  BSGS: "stip-bsgs",
  QUERY: "stip-query",
  KEY: "stip-key",
  VALUE: "stip-value",
  SCORE: "stip-score",
  DISTANCE: "stip-distance",
  CONTEXT: "stip-context",
  OUTPUT: "stip-output",
  FC1: "stip-fc1",
  FC2: "stip-fc2",
});

const SLOTS = 8;
const BATCHES = 2;
const M = 2;
const STRIDE = SLOTS / M;
const SUB = ["₀", "₁"];
const X24 = Object.freeze([[1, 3, 0, 2], [4, 0, 2, 1]]);
const W42 = Object.freeze([[1, 0], [0, 2], [3, 1], [2, 1]]);
const W24 = Object.freeze([[1, 0, 2, 3], [2, 1, 3, 0]]);
const Q22 = Object.freeze([[2, 1], [1, 0]]);
const K22 = Object.freeze([[1, 4], [3, 2]]);
const G22 = Object.freeze([[0.5, 0.875], [0.25, 0.75]]);
const V22 = Object.freeze([[2, 1], [1, 3]]);

function operation(id, config) {
  return Object.freeze({ id, method: "stip", slotsPerCiphertext: SLOTS, batchCount: BATCHES, ...config });
}
function projection(id, shortLabel, rightLabel, outputLabel, group = "attention", kind = "projection") {
  return operation(id, { shortLabel, group, kind, leftLabel: kind === "output" ? "H" : "X", rightLabel, outputLabel,
    leftSecret: true, rightSecret: false, A: X24, B: W42, strategy: "column" });
}
export const STIP_OPERATIONS = Object.freeze({
  [STIP_OPERATION_IDS.RIHP]: operation(STIP_OPERATION_IDS.RIHP, { shortLabel: "RIHP QKᵀ", group: "paper", kind: "rihp",
    leftLabel: "Q", rightLabel: "K", outputLabel: "P", leftSecret: true, rightSecret: true, A: Q22, B: K22, strategy: "column" }),
  [STIP_OPERATION_IDS.L2]: operation(STIP_OPERATION_IDS.L2, { shortLabel: "RIHP L2", group: "paper", kind: "l2",
    leftLabel: "Q", rightLabel: "K", outputLabel: "D²", leftSecret: true, rightSecret: true, A: Q22, B: K22, strategy: "column" }),
  [STIP_OPERATION_IDS.DHP]: projection(STIP_OPERATION_IDS.DHP, "DHP", "W⁰/W¹", "Y⁰/Y¹", "paper", "dhp"),
  [STIP_OPERATION_IDS.AMCP]: projection(STIP_OPERATION_IDS.AMCP, "AMCP", "W", "Y", "paper", "amcp"),
  [STIP_OPERATION_IDS.BSGS]: operation(STIP_OPERATION_IDS.BSGS, { shortLabel: "RIHP + BSGS", group: "paper", kind: "context",
    leftLabel: "G", rightLabel: "V", outputLabel: "H", leftSecret: true, rightSecret: true, A: G22, B: V22, strategy: "column" }),
  [STIP_OPERATION_IDS.QUERY]: projection(STIP_OPERATION_IDS.QUERY, "XWq", "Wq", "Q"),
  [STIP_OPERATION_IDS.KEY]: projection(STIP_OPERATION_IDS.KEY, "XWk", "Wk", "K"),
  [STIP_OPERATION_IDS.VALUE]: projection(STIP_OPERATION_IDS.VALUE, "XWv", "Wv", "V"),
  [STIP_OPERATION_IDS.SCORE]: operation(STIP_OPERATION_IDS.SCORE, { shortLabel: "QKᵀ", group: "attention", kind: "rihp",
    leftLabel: "Q", rightLabel: "K", outputLabel: "P", leftSecret: true, rightSecret: true, A: Q22, B: K22, strategy: "column" }),
  [STIP_OPERATION_IDS.DISTANCE]: operation(STIP_OPERATION_IDS.DISTANCE, { shortLabel: "L2(Q,K)", group: "attention", kind: "l2",
    leftLabel: "Q", rightLabel: "K", outputLabel: "D²", leftSecret: true, rightSecret: true, A: Q22, B: K22, strategy: "column" }),
  [STIP_OPERATION_IDS.CONTEXT]: operation(STIP_OPERATION_IDS.CONTEXT, { shortLabel: "GK·V", group: "attention", kind: "context",
    leftLabel: "GK(D²)", rightLabel: "V", outputLabel: "H", leftSecret: true, rightSecret: true, A: G22, B: V22, strategy: "column" }),
  [STIP_OPERATION_IDS.OUTPUT]: projection(STIP_OPERATION_IDS.OUTPUT, "HWo", "Wo", "O", "attention", "output"),
  [STIP_OPERATION_IDS.FC1]: operation(STIP_OPERATION_IDS.FC1, { shortLabel: "FC1", group: "ffn", kind: "pcmm",
    leftLabel: "X", rightLabel: "W1", outputLabel: "F1", leftSecret: true, rightSecret: false, A: Q22, B: W24, strategy: "column" }),
  [STIP_OPERATION_IDS.FC2]: operation(STIP_OPERATION_IDS.FC2, { shortLabel: "FC2", group: "ffn", kind: "pcmm",
    leftLabel: "ReLU(F1)", rightLabel: "W2", outputLabel: "F2", leftSecret: true, rightSecret: false, A: X24, B: W42, strategy: "column" }),
});

export function listStipOperations(group = null) {
  return Object.values(STIP_OPERATIONS).filter((item) => !group || item.group === group);
}
export function getStipOperation(id) { return STIP_OPERATIONS[id] || null; }
function clone(A) { return A.map((row) => [...row]); }
function cols(A) { return Array.from({ length: A[0].length }, (_, j) => A.map((row) => Number(row[j]))); }
function assertRect(A, rows, columns, label) {
  if (!Array.isArray(A) || A.length !== rows || A.some((row) => !Array.isArray(row) || row.length !== columns)) {
    throw new Error(`${label} must be ${rows}×${columns}`);
  }
}
function strip({ id, label, values, revealAt, tone = 0, badge = "CT", kind = "ciphertext", sampleLanes = false }) {
  const padded = Array.from({ length: SLOTS }, (_, i) => values[i] ?? 0);
  const a = vectorArtifact({ id, label, values: padded.map((v) => Array.isArray(v) ? v[0] : Number(v)), revealAt, badge, kind,
    tones: Array(SLOTS).fill(badge === "CT" ? tone : 4) });
  a.cells.forEach((cell, i) => {
    const value = padded[i];
    if (Array.isArray(value)) cell.imaginary = value[1];
    const text = Array.isArray(value) ? complexLabel(value) : String(value);
    cell.label = values[i] == null ? "·" : text;
    if (sampleLanes && values[i] != null) cell.sampleLane = i % BATCHES;
  });
  return a;
}
const valuesOf = (artifact) => artifact.cells.map((cell) => cell.value);
const complexValuesOf = (artifact) => artifact.cells.map((cell) => [cell.value, cell.imaginary ?? 0]);

function columnSlots(values) {
  return values.flatMap((value) => [value, value, null, null]);
}
function batchColumn(values, id, label, revealAt, tone = 0, kind = "ciphertext") {
  return strip({ id, label, values: columnSlots(values), sampleLanes: true, revealAt, tone, kind });
}
function complexColumn(pairs, id, label, revealAt, tone = 0, kind = "ciphertext") {
  return strip({ id, label, values: columnSlots(pairs), sampleLanes: true, revealAt, tone, kind });
}
function readColumn(artifact) {
  return Array.from({ length: M }, (_, row) => artifact.cells[row * STRIDE].value);
}
function readDiagonals(diagonals) {
  return Array.from({ length: M }, (_, i) =>
    Array.from({ length: M }, (_, j) => diagonals[mod(j - i, M)].cells[i * STRIDE].value));
}
function matrix(id, label, values, revealAt, badge = "CT", layout = "column", toneOffset = 0) {
  return matrixArtifact({ id, label, values, revealAt, badge, meta: "",
    toneForCell: (row, col) => {
      if (layout === "diagonal") return mod(col - row, values.length);
      if (layout === "row") return (row + toneOffset) % 4;
      return (col + toneOffset) % 4;
    } });
}
function artifactCiphertexts(stage) {
  const out = [], seen = new Set();
  for (const row of stage.rows || []) for (const a of row) {
    if (a.badge !== "CT" || !a.cells || seen.has(a.id)) continue;
    seen.add(a.id);
    out.push(ctState(a.id, a.label, a.cells.map((c) => c.value), SLOTS, true,
      a.type === "product" ? "product" : a.type === "result" ? "result" : "tensor"));
  }
  return out;
}
function graphSteps(d, graph, left, right, C, deltas) {
  return makeSteps(d.id, graph.stages.map((stage, i) => ({
    phase: i === 0 ? "reference" : i === graph.stages.length - 1 ? "output" : "multiply",
    title: stage.stepTitle, description: stage.description, formula: "", tensors: {
      left: { id: "A", label: d.leftLabel, shape: [left.length, left[0].length], values: left },
      right: { id: "B", label: d.kind === "rihp" ? "Kᵀ" : d.rightLabel, shape: [right.length, right[0].length], values: right },
      ...(i === graph.stages.length - 1 ? { output: { id: "C", label: d.outputLabel, shape: [C.length, C[0].length], values: C } } : {}),
    }, ciphertexts: artifactCiphertexts(stage), delta: deltas[i] || {},
  })));
}
function shell(d, A, B, C, graph, steps, flow) {
  const shape = { m: A.length, k: A[0].length, n: C[0].length };
  const distance = d.kind === "l2";
  return { schemaVersion: "1.0", id: `${d.id}-toy`, title: `ePrint'26 STIP ${d.shortLabel}`,
    parameters: { matrixShape: shape, slotsPerCiphertext: SLOTS, matmulType: d.rightSecret ? "ct-ct" : "ct-pt",
      algorithmId: d.id, batchCount: BATCHES, ...(distance ? { resultOperation: "squared-l2" } : {}) },
    operands: { left: { id: "A", label: d.leftLabel, shape: [A.length, A[0].length], values: A, secret: d.leftSecret },
      right: { id: "B", label: d.kind === "rihp" ? "Kᵀ" : d.rightLabel, shape: [B.length, B[0].length], values: B, secret: d.rightSecret } },
    output: { id: "C", label: d.outputLabel, shape: [C.length, C[0].length] },
    plan: { algorithmId: d.id, method: "stip", methodLabel: "ePrint'26 STIP", visualStyle: "diagram-only",
      matmulType: d.rightSecret ? "ct-ct" : "ct-pt", logicalShape: shape, paddedShape: shape, tileGrid: { m: 1, n: 1 },
      ...(distance ? { formula: "Squared L2 distances between Q and K" } : {}),
      tileShape: { m: C.length, n: C[0].length }, packingStrategyRequested: flow, packingStrategyResolved: flow,
      plannerSource: "STIP Algorithms 1-9 and STIP_func/main.cpp", reductionStrategy: flow,
      displayMode: "diagram-only", batchLabel: "samples", warnings: [] }, graph, steps, finalTensor: C };
}
function outputStage(d, decoded, artifacts, revealAt, layout = "column") {
  return { id: `${d.id}-output`, title: d.outputLabel, stepTitle: `Verify the packed ${d.outputLabel} output`,
    shortTitle: "Output", description: `Read the first sample’s ${layout === "diagonal" ? "cyclic diagonals" : "output columns"} back into matrix form.`, revealAt,
    rows: [artifacts, [operator(`${d.id}-down`, "↓", "", revealAt), matrix("C", d.outputLabel, decoded, revealAt, "CT", layout)]] };
}

function amcpPacked(A, groups, id) {
  return groups.map((indices, g) => {
    const values = A.flatMap((row) => indices.flatMap((j) => [row[j], row[j]]));
    const artifact = strip({ id: `${id}-${g}`, label: `C${g}`, values, sampleLanes: true, revealAt: 1, tone: g });
    artifact.cells.forEach((cell, slot) => {
      cell.featureIndex = indices[Math.floor(slot / BATCHES) % indices.length];
    });
    return artifact;
  });
}

function amcpAlign(packed, id) {
  const u = [1, 1, 0, 0, 1, 1, 0, 0];
  const v = u.map((x) => 1 - x);
  const rows = [];
  const rotated = packed.map((group, i) => {
    const left = strip({ id: `${id}-rot-left-${i}`, label: `RotL₂(C${i}) ⊙ u`, revealAt: 2, tone: i, sampleLanes: true,
      values: multiplyVectors(rotateLeft(valuesOf(group), BATCHES), u) });
    const right = strip({ id: `${id}-rot-right-${i}`, label: `RotR₂(C${i}) ⊙ v`, revealAt: 2, tone: i, sampleLanes: true,
      values: multiplyVectors(rotateLeft(valuesOf(group), -BATCHES), v) });
    const aligned = strip({ id: `${id}-rot-${i}`, label: `C${i} shifted`, revealAt: 2, tone: i, sampleLanes: true,
      values: addVectors([valuesOf(left), valuesOf(right)]) });
    rows.push([group, operator(`${aligned.id}-align`, "→", "", 2), left,
      operator(`${aligned.id}-sum`, "+", "", 2), right, operator(`${aligned.id}-eq`, "=", "", 2), aligned]);
    return aligned;
  });
  return { rotated, rows };
}

function projectionInput(d, A, W) {
  return { id: `${d.id}-input`, title: `${d.leftLabel} × ${d.rightLabel}`, stepTitle: "Input activation and projection weights", shortTitle: "Input",
    description: `Compute ${d.outputLabel} from encrypted ${d.leftLabel} and the plaintext weights ${d.rightLabel}.`, revealAt: 0,
    rows: [[matrix("A", d.leftLabel, A, 0, "CT", "column"), operator(`${d.id}-times`, "×", "", 0), matrix("B", d.rightLabel, W, 0, "PT", "column")]] };
}

function complexWeightProducts(inputs, weightAt, id, revealAt) {
  const products = inputs.map((input, i) => {
    const weights = Array.from({ length: SLOTS }, (_, slot) => weightAt(i, slot));
    const plain = strip({ id: `${id}-w-${i}`, label: `W̃${i}`, values: weights, revealAt, badge: "PT" });
    const product = strip({ id: `${id}-product-${i}`, label: `Term ${i + 1}`, kind: "product", revealAt, tone: i % 4,
      values: valuesOf(input).map((value, slot) => multiplyComplex([value, 0], weights[slot])) });
    return { values: complexValuesOf(product),
      row: [input, operator(`${product.id}-mul`, "⊙", "", revealAt), plain, operator(`${product.id}-eq`, "=", "", revealAt), product] };
  });
  return { values: sumComplexVectors(products.map((p) => p.values)), rows: products.map((p) => p.row) };
}

function unpackHeads(d, hybrid, revealAt) {
  const slots = complexValuesOf(hybrid);
  const outputs = [0, 1].map((part) => batchColumn(
    Array.from({ length: M }, (_, row) => slots[row * STRIDE][part]),
    `${d.id}-out-${part}`, `${d.outputLabel}${SUB[part]}`, revealAt, part, "result"));
  const stage = { id: `${d.id}-unpack`, title: "Conj", stepTitle: "Separate the two projected heads", shortTitle: "Split heads",
    description: "Use the sum and difference with the conjugate to recover the real and imaginary head values in separate columns.", revealAt,
    rows: [[hybrid, operator(`${d.id}-conj-op`, "Re, Im", "", revealAt), ...outputs]] };
  return { outputs, stage };
}

function buildProjection(d, A, W) {
  assertRect(A, 2, 4, d.leftLabel); assertRect(W, 4, 2, d.rightLabel);
  const C = matmul(A, W);
  const useAmcp = d.kind !== "dhp";
  const groups = [[0, 1], [2, 3]];
  const packed = useAmcp ? amcpPacked(A, groups, `${d.id}-amcp`)
    : cols(A).map((col, i) => batchColumn(col, `${d.id}-column-${i}`, `x${i}`, 1, i));
  const stages = [projectionInput(d, A, W), {
    id: `${d.id}-pack`, title: useAmcp ? "AMCP" : "Column pack",
    stepTitle: useAmcp ? "Interleave features within each token" : "Pack one ciphertext per activation column",
    shortTitle: useAmcp ? "AMCP pack" : "Pack columns",
    description: useAmcp ? "Place both sample lanes of each feature together inside each token’s four-slot group."
      : "Keep corresponding rows and sample lanes in the same slots across all activation columns.", revealAt: 1, rows: [packed],
  }];
  const deltas = [{}, { ciphertextCount: packed.length }];
  let inputs = packed;
  if (useAmcp) {
    const { rotated, rows } = amcpAlign(packed, d.id);
    inputs = packed.flatMap((group, i) => [group, rotated[i]]);
    stages.push({ id: `${d.id}-rotate-stage`, title: "RotL₂ · u + RotR₂ · v", stepTitle: "Align the two feature lanes", shortTitle: "Align features",
      description: "Rotate left and right by two slots, then select the complementary lanes to swap features within each token.", revealAt: 2, rows });
    deltas.push({ rotations: 4, masks: 4, ptCtMultiplications: 4, additions: 2 });
  }
  const multiplyAt = stages.length;
  const pureAmcp = d.kind === "amcp";
  const products = complexWeightProducts(inputs, (i, slot) => {
    const lane = Math.floor((slot % STRIDE) / BATCHES);
    const feature = useAmcp ? groups[Math.floor(i / 2)][mod(lane + i, 2)] : i;
    if (pureAmcp) return [Number(W[feature][lane]), 0];
    return W[feature].map(Number);
  }, d.id, multiplyAt);
  const hybrid = strip({ id: `${d.id}-hybrid`, label: pureAmcp ? "Packed head lanes" : "Y⁰+iY¹",
    values: pureAmcp ? products.values.map(([v]) => v) : products.values, revealAt: multiplyAt, tone: 1, kind: "product" });
  stages.push({ id: `${d.id}-project`, title: pureAmcp ? "AMCP ⊙ Σ" : "DHP ⊙ Σ",
    stepTitle: pureAmcp ? "Apply lane-specific projection weights" : "Project two heads using complex weights",
    shortTitle: pureAmcp ? "AMCP products" : "DHP products",
    description: pureAmcp ? "Use the aligned plaintext weights for each head lane, then sum all feature products in place."
      : "Multiply by W⁰+iW¹ and sum the aligned feature products. The two head projections occupy the real and imaginary parts.", revealAt: multiplyAt,
    rows: [...products.rows, [operator(`${d.id}-sum`, "Σ", "", multiplyAt), hybrid]] });
  deltas.push({ ptCtMultiplications: inputs.length, additions: inputs.length - 1, rescales: 1, multiplicativeDepth: 1 });
  if (pureAmcp) {
    const decoded = A.map((row, i) => W[0].map((_, j) => hybrid.cells[i * STRIDE + j * BATCHES].value));
    stages.push({ ...outputStage(d, decoded, [hybrid], stages.length),
      description: "Read each head from its separate slot lane within the token group." });
  } else {
    const { outputs, stage } = unpackHeads(d, hybrid, stages.length);
    stages.push(stage, outputStage(d, transpose(outputs.map(readColumn)), outputs, stages.length + 1));
    deltas.push({ additions: 2, ptCtMultiplications: 2 });
  }
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages };
  return shell(d, A, W, C, graph, graphSteps(d, graph, A, W, C, deltas),
    pureAmcp ? "AMCP packed head lanes" : useAmcp ? "AMCP input alignment + DHP" : "DHP column packing");
}

function buildOutputProjection(d, A, W) {
  assertRect(A, 2, 4, d.leftLabel); assertRect(W, 4, 2, d.rightLabel);
  const C = matmul(A, W);
  const groups = [[0, 2], [1, 3]];
  const packed = amcpPacked(A, groups, `${d.id}-heads`);
  const headParts = [], productRows = [];
  for (let j = 0; j < W[0].length; j++) {
    const products = packed.map((input, i) => {
      const weights = Array.from({ length: SLOTS }, (_, slot) => W[groups[i][Math.floor((slot % STRIDE) / BATCHES)]][j]);
      const plain = strip({ id: `${d.id}-w-${i}-${j}`, label: `Wo${i},${j}`, values: weights, revealAt: 2, badge: "PT" });
      const product = strip({ id: `${d.id}-product-${i}-${j}`, label: `Term ${i + 1}`, kind: "product", revealAt: 2, tone: j,
        values: multiplyVectors(valuesOf(input), weights) });
      productRows.push([input, operator(`${product.id}-mul`, "⊙", "", 2), plain, operator(`${product.id}-eq`, "=", "", 2), product]);
      return product;
    });
    const part = strip({ id: `${d.id}-parts-${j}`, label: `Head contributions to O${j}`,
      values: addVectors(products.map(valuesOf)), revealAt: 2, tone: j });
    productRows.push([...products, operator(`${part.id}-sum`, "Σ", "", 2), part]);
    headParts.push(part);
  }
  const mask = strip({ id: `${d.id}-mask`, label: "Keep first head lane", badge: "PT", revealAt: 3, values: [1, 1, 0, 0, 1, 1, 0, 0] });
  const outputs = headParts.map((part, j) => {
    const values = valuesOf(part);
    const reduced = multiplyVectors(addVectors([values, rotateLeft(values, BATCHES)]), valuesOf(mask));
    return batchColumn([reduced[0], reduced[STRIDE]], `${d.id}-out-${j}`, `O${j}`, 3, j, "result");
  });
  const stages = [projectionInput(d, A, W),
    { id: `${d.id}-pack`, title: "AMCP", stepTitle: "Pack matching features from both heads", shortTitle: "Pack head lanes",
      description: "Within each token and feature group, place the two heads in different slot lanes.", revealAt: 1, rows: [packed] },
    { id: `${d.id}-project`, title: "AMCP ⊙ Σ", stepTitle: "Apply the output weights per head", shortTitle: "Project heads",
      description: "Multiply by each head’s weights and add across features while keeping the head contributions in separate lanes.", revealAt: 2, rows: productRows },
    { id: `${d.id}-aggregate`, title: "Σ ρ₂ AMCP", stepTitle: "Rotate, add, and keep the combined head lane", shortTitle: "Merge heads",
      description: "A two-slot rotation aligns the second head with the first. Add the contributions and mask all other lanes.", revealAt: 3,
      rows: outputs.map((out, j) => [headParts[j], operator(`${out.id}-add`, "⊞ RotL₂", "", 3), operator(`${out.id}-mask`, "⊙", "", 3),
        mask, operator(`${out.id}-eq`, "=", "", 3), out]) },
    outputStage(d, transpose(outputs.map(readColumn)), outputs, 4)];
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages };
  return shell(d, A, W, C, graph, graphSteps(d, graph, A, W, C, [{}, { ciphertextCount: 2 },
    { ptCtMultiplications: 4, additions: 2, rescales: 2, multiplicativeDepth: 1 },
    { rotations: 2, additions: 2, masks: 2, ptCtMultiplications: 2 }, {}]), "AMCP output projection and head aggregation");
}

function scoreDiagonals(C) { return [[C[0][0], C[1][1]], [C[0][1], C[1][0]]]; }
function buildRihp(d, Q, K) {
  assertRect(Q, 2, 2, d.leftLabel); assertRect(K, 2, 2, d.rightLabel);
  const KT = transpose(K), C = matmul(Q, KT), q = cols(Q), k = cols(K);
  const qct = q.map((v, i) => batchColumn(v, `${d.id}-q-${i}`, `q${i}`, 1, i));
  const kct = k.map((v, i) => batchColumn(v, `${d.id}-k-${i}`, `k${i}`, 1, i + 2));
  const kh = kct.map((key, i) => {
    const shifted = rotateLeft(valuesOf(key), STRIDE);
    return complexColumn(k[i].map((value, row) => [value, shifted[row * STRIDE]]), `${d.id}-kh-${i}`, `k̃${i}`, 2, i + 2);
  });
  const terms = kh.map((key, j) => complexValuesOf(key).map((slot, i) => multiplyComplex([qct[j].cells[i].value, 0], slot)));
  const sums = sumComplexVectors(terms);
  const hybrid = complexColumn([sums[0], sums[STRIDE]], `${d.id}-hybrid`, "C₀=P₀+iP₁", 3, 1, "product");
  const unpacked = [0, 1].map((part) => batchColumn([sums[0][part], sums[STRIDE][part]], `${d.id}-p-${part}`, `P${part}`, 4, part, "result"));
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: "Q × Kᵀ", stepTitle: "Input Q and K for the dot-product scores", shortTitle: "Q × Kᵀ",
      description: "Compute the two cyclic score diagonals together using the first RIHP strategy.", revealAt: 0,
      rows: [[matrix("A", "Q", Q, 0, "CT", "column"), operator(`${d.id}-times`, "×", "", 0), matrix("B", "Kᵀ", KT, 0, "CT", "row", 2)]] },
    { id: `${d.id}-column`, title: "Col", stepTitle: "Pack Q and K by feature column", shortTitle: "Pack columns",
      description: "Keep two sample lanes per token. Consecutive tokens are four physical slots apart in this demo.", revealAt: 1, rows: [qct, kct] },
    { id: `${d.id}-hybrid-k`, title: "K + i·RotL₁(K)", stepTitle: "Put the shifted keys in the imaginary part", shortTitle: "Hybrid K",
      description: "Pair each key column with its one-token rotation, which corresponds to a four-slot left rotation here.", revealAt: 2,
      rows: kh.map((key, j) => [kct[j], operator(`${key.id}-op`, "+ i·RotL₄", "", 2), key]) },
    { id: `${d.id}-rihp`, title: "RIHP ⊙ Σ", stepTitle: "Compute both score diagonals in one ciphertext", shortTitle: "RIHP product",
      description: "Multiply the real Q columns by the hybrid K columns and sum over features. Each complex part holds one score diagonal.", revealAt: 3,
      rows: [[...qct, operator(`${d.id}-mul`, "⊙", "", 3), ...kh, operator(`${d.id}-sum`, "Σ", "", 3), hybrid]] },
    { id: `${d.id}-conj`, title: "½(C+Conj C) · −½i(C−Conj C)", stepTitle: "Unpack the real score diagonals", shortTitle: "Split diagonals",
      description: "Use the conjugate sum and difference to separate the two diagonals into real ciphertexts.", revealAt: 4,
      rows: [[hybrid, operator(`${d.id}-split`, "Conj", "", 4), ...unpacked]] },
    outputStage(d, readDiagonals(unpacked), unpacked, 5, "diagonal"),
  ] };
  const steps = graphSteps(d, graph, Q, KT, C, [{}, { ciphertextCount: 4 }, { rotations: 2, ptCtMultiplications: 2 },
    { ctCtMultiplications: 2, additions: 1, relinearizations: 1, rescales: 1, multiplicativeDepth: 1 }, { additions: 2 }, {}]);
  return shell(d, Q, KT, C, graph, steps, "RIHP column × column → diagonal");
}

function buildL2(d, Q, K) {
  assertRect(Q, 2, 2, d.leftLabel); assertRect(K, 2, 2, d.rightLabel);
  const C = squaredL2Distances(Q, K), q = cols(Q), k = cols(K);
  const qh = complexColumn(q[0].map((v, i) => [v, q[1][i]]), `${d.id}-qh`, "q₀+iq₁", 1, 0);
  const kh = complexColumn(k[0].map((v, i) => [v, -k[1][i]]), `${d.id}-kh`, "k₀−ik₁", 1, 2);
  const norm = (a) => complexValuesOf(a).filter((_, slot) => slot % STRIDE === 0).map(([real, imag]) => real ** 2 + imag ** 2);
  const sq = batchColumn(norm(qh), `${d.id}-sq`, "S_Q", 2, 0, "product");
  const sk = batchColumn(norm(kh), `${d.id}-sk`, "S_K", 2, 2, "product");
  const crossRows = [], combineRows = [];
  const diags = Array.from({ length: M }, (_, shift) => {
    const offset = shift * STRIDE;
    const rotated = rotateLeft(complexValuesOf(kh), offset);
    const keys = shift ? complexColumn([rotated[0], rotated[STRIDE]], `${d.id}-k-shift`, "RotL₄(k̂*)", 3, 2) : kh;
    const raw = complexValuesOf(qh).map((slot, i) => multiplyComplex(slot, rotated[i]));
    const product = complexColumn([raw[0], raw[STRIDE]], `${d.id}-c-${shift}`, `C${shift}`, 3, shift + 1, "product");
    crossRows.push([qh, operator(`${product.id}-mul`, "⊙", "", 3), keys, operator(`${product.id}-eq`, "=", "", 3), product]);
    const rotatedNorm = rotateLeft(valuesOf(sk), offset);
    const keysNorm = shift ? batchColumn([rotatedNorm[0], rotatedNorm[STRIDE]], `${d.id}-sk-rot`, "RotL₄(S_K)", 4, 2) : sk;
    const distance = [0, STRIDE].map((slot) => [sq.cells[slot].value + keysNorm.cells[slot].value - 2 * raw[slot][0], -2 * raw[slot][1]]);
    const beforeReal = complexColumn(distance, `${d.id}-t-${shift}`, `T${shift}`, 4, shift);
    const result = batchColumn(distance.map(([real]) => real), `${d.id}-d-${shift}`, `D²${shift}`, 4, shift, "result");
    combineRows.push([sq, operator(`${result.id}-plus`, "+", "", 4), keysNorm, operator(`${result.id}-minus`, "−2·", "", 4),
      product, operator(`${result.id}-eq`, "=", "", 4), beforeReal, operator(`${result.id}-real`, "Re", "", 4), result]);
    return result;
  });
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: "Squared distance between Q and K", stepTitle: "Input queries and keys for Gaussian attention", shortTitle: "Q and K",
      description: "Compute squared L2 distances between query and key rows before applying the Gaussian kernel.", revealAt: 0,
      rows: [[matrix("A", "Q", Q, 0, "CT", "column"), matrix("B", "K", K, 0, "CT", "column", 2)]] },
    { id: `${d.id}-pair`, title: "RIHP", stepTitle: "Pair feature columns in complex slots", shortTitle: "Pair features",
      description: "Place the two query features in the real and imaginary parts, and use the conjugate pairing for the keys.", revealAt: 1, rows: [[qh, kh]] },
    { id: `${d.id}-norm`, title: "z ⊙ Conj(z)", stepTitle: "Compute query and key squared norms", shortTitle: "Squared norms",
      description: "Multiply each complex feature pair by its conjugate to sum the squares of both features.", revealAt: 2,
      rows: [[qh, operator(`${d.id}-qnorm`, "⊙Conj", "", 2), sq], [kh, operator(`${d.id}-knorm`, "⊙Conj", "", 2), sk]] },
    { id: `${d.id}-cross`, title: "Σ q̂ ⊙ Rot(k̂*)", stepTitle: "Compute the rotated query–key cross terms", shortTitle: "Cross terms",
      description: "Multiply the complex queries by the aligned keys. Each product’s real part contains the dot product for that shift.", revealAt: 3, rows: crossRows },
    { id: `${d.id}-combine`, title: "Re(S_Q + Rot(S_K) − 2C)", stepTitle: "Combine norms and extract the distance", shortTitle: "Distance diagonals",
      description: "Align the key norm to each cross term, subtract twice the product, and take the real part to get the squared distance.", revealAt: 4, rows: combineRows },
    outputStage(d, readDiagonals(diags), diags, 5, "diagonal"),
  ] };
  const steps = graphSteps(d, graph, Q, K, C, [{}, { ciphertextCount: 2 },
    { ctCtMultiplications: 2, relinearizations: 2, rescales: 2, multiplicativeDepth: 1 },
    { rotations: 1, ctCtMultiplications: 2, relinearizations: 2, rescales: 2 }, { rotations: 1, additions: 6, ptCtMultiplications: 2 }, {}]);
  return shell(d, Q, K, C, graph, steps, "RIHP squared L2 → diagonal");
}

function buildContext(d, G, V) {
  assertRect(G, 2, 2, d.leftLabel); assertRect(V, 2, 2, d.rightLabel);
  const C = matmul(G, V), diagonals = scoreDiagonals(G).map((v, i) => batchColumn(v, `${d.id}-g-${i}`, `G${i}`, 1, i));
  const vc = cols(V), vpair = complexColumn(vc[0].map((v, i) => [v, vc[1][i]]), `${d.id}-vpair`, "v₀+iv₁", 1, 2);
  const shifted = rotateLeft(complexValuesOf(vpair), STRIDE);
  const baby = complexColumn([shifted[0], shifted[STRIDE]], `${d.id}-baby`, "RotL₄(v̂)", 2, 2);
  const babies = [vpair, baby];
  const products = babies.map((ct, shift) => {
    const values = complexValuesOf(ct).map((slot, i) => multiplyComplex(slot, [diagonals[shift].cells[i].value, 0]));
    return complexColumn([values[0], values[STRIDE]], `${d.id}-gv-${shift}`, `G${shift}⊙β${shift}`, 3, shift, "product");
  });
  const sum = sumComplexVectors(products.map(complexValuesOf));
  const hybrid = complexColumn([sum[0], sum[STRIDE]], `${d.id}-hybrid`, "H₀+iH₁", 3, 1, "product");
  const outputs = [0, 1].map((part) => batchColumn([sum[0][part], sum[STRIDE][part]], `${d.id}-out-${part}`,
    `${d.outputLabel}${SUB[part]}`, 4, part, "result"));
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: `${d.leftLabel} × V`, stepTitle: "Input Gaussian weights and encrypted values", shortTitle: "Weights × V",
      description: "Multiply the diagonal-packed attention weights by V to obtain the encrypted context.", revealAt: 0,
      rows: [[matrix("A", d.leftLabel, G, 0, "CT", "diagonal"), operator(`${d.id}-times`, "×", "", 0), matrix("B", "V", V, 0, "CT", "column", 2)]] },
    { id: `${d.id}-pack`, title: "Diag + RIHP", stepTitle: "Pair the two value columns", shortTitle: "RIHP pack",
      description: "Combine the value columns as v₀+iv₁ and keep one ciphertext for each real attention diagonal.", revealAt: 1, rows: [[...diagonals, vpair]] },
    { id: `${d.id}-baby`, title: "BSGS β", stepTitle: "Prepare the value baby steps", shortTitle: "Baby rotations",
      description: "Keep the original value pair and rotate it by one token to align it with the second diagonal.", revealAt: 2,
      rows: [[vpair, operator(`${d.id}-baby-rot`, "RotL₄", "", 2), baby]] },
    { id: `${d.id}-bsgs`, title: "Σ_g Rot(Σ_b G ⊙ β)", stepTitle: "Accumulate the complex context columns", shortTitle: "BSGS products",
      description: "Multiply each attention diagonal by its matching baby step and add. With two tokens, both terms fit one giant group at offset zero.", revealAt: 3,
      rows: [...products.map((p, i) => [diagonals[i], operator(`${p.id}-mul`, "⊙", "", 3), babies[i], operator(`${p.id}-eq`, "=", "", 3), p]),
        [...products, operator(`${d.id}-bsgs-op`, "Σ", "", 3), hybrid]] },
    { id: `${d.id}-conj`, title: "Conj", stepTitle: "Unpack the real context columns", shortTitle: "Split columns",
      description: "Use conjugate extraction to recover the two context columns from the real and imaginary channels.", revealAt: 4,
      rows: [[hybrid, operator(`${d.id}-conj-op`, "Re, Im", "", 4), ...outputs]] },
    outputStage(d, transpose(outputs.map(readColumn)), outputs, 5),
  ] };
  const steps = graphSteps(d, graph, G, V, C, [{}, { ciphertextCount: 3 }, { rotations: 1 },
    { ctCtMultiplications: 2, additions: 1, relinearizations: 1, rescales: 1, multiplicativeDepth: 1 },
    { additions: 2, ptCtMultiplications: 2 }, {}]);
  return shell(d, G, V, C, graph, steps, "diagonal × RIHP columns with BSGS");
}

function buildPcmm(d, A, W) {
  assertRect(A, d.A.length, d.A[0].length, d.leftLabel); assertRect(W, d.B.length, d.B[0].length, d.rightLabel);
  const C = matmul(A, W), ac = cols(A).map((v, i) => batchColumn(v, `${d.id}-a-${i}`, `x${i}`, 1, i % 4));
  const products = cols(W).map((_, j) => ac.map((a, i) => batchColumn(readColumn(a).map((value) => value * W[i][j]),
    `${d.id}-p-${j}-${i}`, `p${j}${i}`, 2, i % 4, "product")));
  const outputs = products.map((set, j) => batchColumn(addVectors(set.map(readColumn)), `${d.id}-out-${j}`, `${d.outputLabel}${j}`, 3, j % 4, "result"));
  const rows = [];
  products.forEach((set, j) => set.forEach((p, i) => rows.push([ac[i], operator(`${d.id}-mul-${j}-${i}`, `⊙${W[i][j]}`, "", 2), p])));
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    projectionInput(d, A, W),
    { id: `${d.id}-col`, title: "Col", stepTitle: "Pack the feed-forward input by column", shortTitle: "Pack columns",
      description: "Store each encrypted input feature in its own ciphertext with rows and sample lanes aligned.", revealAt: 1, rows: [ac] },
    { id: `${d.id}-multiply`, title: "⊙ PT", stepTitle: "Scale encrypted columns by their weights", shortTitle: "Weight products",
      description: "Apply the matching plaintext weight to each encrypted input column for each output feature.", revealAt: 2, rows },
    { id: `${d.id}-sum`, title: "Σ", stepTitle: "Sum the feed-forward column products", shortTitle: "Sum products",
      description: "Add the scaled columns along the shared dimension to obtain one encrypted vector per output feature.", revealAt: 3,
      rows: outputs.map((o, j) => [...products[j], operator(`${d.id}-sum-${j}`, "Σ", "", 3), o]) },
    outputStage(d, transpose(outputs.map(readColumn)), outputs, 4),
  ] };
  const mults = A[0].length * W[0].length;
  const steps = graphSteps(d, graph, A, W, C, [{}, { ciphertextCount: ac.length },
    { ptCtMultiplications: mults, rescales: mults, multiplicativeDepth: 1 }, { additions: (A[0].length - 1) * W[0].length }, {}]);
  return shell(d, A, W, C, graph, steps, "column-packed CPMM");
}

export function buildStipOperationTrace(id, options = {}) {
  const d = getStipOperation(id); if (!d) throw new Error(`Unknown STIP operation: ${id}`);
  if (options.leftSecret != null && Boolean(options.leftSecret) !== d.leftSecret) throw new Error(`${d.shortLabel} has a fixed left operand type`);
  if (options.rightSecret != null && Boolean(options.rightSecret) !== d.rightSecret) throw new Error(`${d.shortLabel} has a fixed right operand type`);
  if (options.slotsPerCiphertext != null && Number(options.slotsPerCiphertext) !== SLOTS) throw new Error("STIP toy uses 8 CKKS slots");
  const A = options.A || d.A, B = options.B || d.B;
  if (d.kind === "rihp") return buildRihp(d, A, B);
  if (d.kind === "l2") return buildL2(d, A, B);
  if (d.kind === "context") return buildContext(d, A, B);
  if (d.kind === "pcmm") return buildPcmm(d, A, B);
  if (d.kind === "output") return buildOutputProjection(d, A, B);
  return buildProjection(d, A, B);
}
export function stipOperationDefaults(id) {
  const d = getStipOperation(id); if (!d) return null;
  return { m: d.A.length, k: d.A[0].length, n: d.B[0].length, A: clone(d.A), B: clone(d.B),
    leftSecret: d.leftSecret, rightSecret: d.rightSecret, slotsPerCiphertext: SLOTS, strategy: "column" };
}
export function parseStipOperationInputs(raw) { return parseAndValidateInputs(raw); }
