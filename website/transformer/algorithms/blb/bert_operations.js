// BLB matrix-multiplication toy traces.
//
// The visual flows mirror the paper and the released SCI implementation:
//   * BSGS ciphertext-plaintext projection with spatial-first packing.
//   * Multi-head packed ciphertext-ciphertext MatMul (spatial x reduce -> diagonal).
//   * MPC <-> CKKS boundaries around Softmax, followed by padded S x V.
//   * Modified diagonal ciphertext-plaintext MatMul for the output projection.

import { matmul, transpose } from "../../core/matrix.js";
import { parseAndValidateInputs, parseMatrixText } from "../../core/input_parser.js";
import {
  ctState,
  makeSteps,
  matrixArtifact,
  operator,
  rotateLeft,
  upperDiagonal,
  vectorArtifact,
} from "../thor/paper_utils.js";

export const BLB_OPERATION_IDS = Object.freeze({
  QUERY: "blb-query",
  KEY: "blb-key",
  VALUE: "blb-value",
  SCORE: "blb-score",
  CONTEXT: "blb-context",
  OUTPUT: "blb-output",
  FC1: "blb-fc1",
  FC2: "blb-fc2",
  POOLER: "blb-pooler",
  CLASSIFIER: "blb-classifier",
});

const SLOTS = 16;
const HEADS = 2;
const SUB = ["₀", "₁", "₂", "₃"];

const X44 = Object.freeze([
  [1, 2, 0, 1],
  [0, 1, 3, 2],
  [2, 0, 1, 1],
  [1, 1, 2, 0],
]);
const W42 = Object.freeze([[1, 0], [0, 1], [2, 1], [1, 2]]);
const Q42 = Object.freeze([[1, 2], [0, 1], [2, 0], [1, 1]]);
const K42 = Object.freeze([[1, 0], [1, 2], [0, 1], [2, 1]]);
const S44 = Object.freeze([
  [0.5, 0.25, 0.125, 0.125],
  [0.25, 0.5, 0.125, 0.125],
  [0.125, 0.125, 0.5, 0.25],
  [0.125, 0.125, 0.25, 0.5],
]);
const V42 = Object.freeze([[1, 0], [0, 1], [2, 1], [1, 2]]);
const CLS14 = Object.freeze([[1, 2, 0, 1]]);
const W44 = Object.freeze([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]]);

function operation(id, config) {
  return Object.freeze({ id, method: "blb", slotsPerCiphertext: SLOTS, ...config });
}

function linear(id, shortLabel, leftLabel, rightLabel, outputLabel, group) {
  return operation(id, {
    shortLabel, group, kind: "spatial-pt", leftLabel, rightLabel, outputLabel,
    leftSecret: true, rightSecret: false, A: X44, B: W42, strategy: "column",
  });
}

export const BLB_OPERATIONS = Object.freeze({
  [BLB_OPERATION_IDS.QUERY]: linear(BLB_OPERATION_IDS.QUERY, "XWq", "X", "Wq", "Q", "attention"),
  [BLB_OPERATION_IDS.KEY]: linear(BLB_OPERATION_IDS.KEY, "XWk", "X", "Wk", "K", "attention"),
  [BLB_OPERATION_IDS.VALUE]: linear(BLB_OPERATION_IDS.VALUE, "XWv", "X", "Wv", "V", "attention"),
  [BLB_OPERATION_IDS.SCORE]: operation(BLB_OPERATION_IDS.SCORE, {
    shortLabel: "QKᵀ", group: "attention", kind: "mhp-ctct", inputMode: "paired-columns",
    leftLabel: "Q", rightLabel: "K", outputLabel: "S", leftSecret: true, rightSecret: true,
    A: Q42, B: K42, strategy: "column",
  }),
  [BLB_OPERATION_IDS.CONTEXT]: operation(BLB_OPERATION_IDS.CONTEXT, {
    shortLabel: "SV", group: "attention", kind: "mpc-ctct", leftLabel: "S", rightLabel: "V",
    outputLabel: "H", leftSecret: true, rightSecret: true, A: S44, B: V42, strategy: "column",
  }),
  [BLB_OPERATION_IDS.OUTPUT]: operation(BLB_OPERATION_IDS.OUTPUT, {
    shortLabel: "HWo", group: "attention", kind: "diag-pt", leftLabel: "H", rightLabel: "Wo",
    outputLabel: "O", leftSecret: true, rightSecret: false, A: X44, B: W42, strategy: "column",
  }),
  [BLB_OPERATION_IDS.FC1]: linear(BLB_OPERATION_IDS.FC1, "FC1", "X", "W1", "F1", "ffn"),
  [BLB_OPERATION_IDS.FC2]: linear(BLB_OPERATION_IDS.FC2, "FC2", "F1", "W2", "F2", "ffn"),
  [BLB_OPERATION_IDS.POOLER]: operation(BLB_OPERATION_IDS.POOLER, {
    shortLabel: "Pooler", group: "output", kind: "mpc-matmul", leftLabel: "CLS", rightLabel: "Wp",
    outputLabel: "P", leftSecret: true, rightSecret: true, A: CLS14, B: W44, strategy: "column",
  }),
  [BLB_OPERATION_IDS.CLASSIFIER]: operation(BLB_OPERATION_IDS.CLASSIFIER, {
    shortLabel: "Classifier", group: "output", kind: "mpc-matmul", leftLabel: "P", rightLabel: "Wc",
    outputLabel: "Y", leftSecret: true, rightSecret: true, A: CLS14, B: W42, strategy: "column",
  }),
});

export function listBlbOperations(group = null) {
  return Object.values(BLB_OPERATIONS).filter((item) => !group || item.group === group);
}

export function getBlbOperation(id) {
  return BLB_OPERATIONS[id] || null;
}

function cloneMatrix(matrix) { return matrix.map((row) => [...row]); }
function columns(matrix) { return Array.from({ length: matrix[0].length }, (_, j) => matrix.map((row) => Number(row[j]))); }
function flattenSpatial(matrix) { return columns(matrix).flat(); }
function flattenReduce(matrix) { return matrix.flat().map(Number); }
function pad16(values) { return Array.from({ length: SLOTS }, (_, i) => Number(values[i] ?? 0)); }
function headPack(values) { return pad16([...values, ...values]); }
function assertShape(matrix, rows, cols, label) {
  if (!Array.isArray(matrix) || matrix.length !== rows || matrix.some((row) => !Array.isArray(row) || row.length !== cols)) {
    throw new Error(`${label} must be ${rows}×${cols}`);
  }
}

function packedArtifact({ id, label, values, revealAt, tone = 0, badge = "CT", kind = "ciphertext", heads = false }) {
  const packed = heads ? headPack(values) : pad16(values);
  const artifact = vectorArtifact({ id, label, values: packed, revealAt, badge, kind, tones: Array(SLOTS).fill(tone) });
  if (heads) artifact.cells.forEach((cell, index) => { cell.label = `${cell.value}h${index < values.length ? "₀" : "₁"}`; });
  return artifact;
}

function mpcMatrix(id, label, values, revealAt) {
  return matrixArtifact({ id, label, values, revealAt, badge: "MPC", meta: "", toneForCell: () => 4 });
}

function ptMatrix(id, label, values, revealAt, toneOffset = 0) {
  return matrixArtifact({ id, label, values, revealAt, badge: "PT", meta: "",
    // Plaintext weights are reordered to match BLB's packed output columns.
    toneForCell: (_row, col) => (col + toneOffset) % 4 });
}

function spatialMatrix(id, label, values, revealAt, badge = "CT", toneOffset = 0) {
  return matrixArtifact({ id, label, values, revealAt, badge, meta: "",
    toneForCell: (_row, col) => badge === "CT" ? (col + toneOffset) % 4 : 4 });
}

function diagonalMatrix(id, label, values, revealAt) {
  return matrixArtifact({
    id, label, values, revealAt, badge: "CT", meta: "",
    toneForCell: (row, col) => ((col - row) % values.length + values.length) % values.length,
  });
}

function artifactCiphertexts(stage) {
  const seen = new Set();
  const out = [];
  for (const row of stage.rows || []) for (const artifact of row) {
    if (artifact.badge !== "CT" || !artifact.cells || seen.has(artifact.id)) continue;
    seen.add(artifact.id);
    out.push(ctState(artifact.id, artifact.label, artifact.cells.map((cell) => cell.value), SLOTS, true,
      artifact.type === "product" ? "product" : artifact.type === "result" ? "result" : "tensor"));
  }
  return out;
}

function tensors(descriptor, A, validationB, C, includeOutput = false) {
  return {
    left: { id: "A", label: descriptor.leftLabel, shape: [A.length, A[0].length], values: A },
    right: { id: "B", label: descriptor.kind === "mhp-ctct" ? "Kᵀ" : descriptor.rightLabel,
      shape: [validationB.length, validationB[0].length], values: validationB },
    ...(includeOutput ? { output: { id: "C", label: descriptor.outputLabel, shape: [C.length, C[0].length], values: C } } : {}),
  };
}

function stepsFromGraph(descriptor, graph, A, validationB, C, deltas) {
  return makeSteps(descriptor.id, graph.stages.map((stage, index) => ({
    phase: index === 0 ? "reference" : index === graph.stages.length - 1 ? "output" : "multiply",
    title: stage.stepTitle || stage.title, description: stage.description || "", formula: "",
    tensors: tensors(descriptor, A, validationB, C, index === graph.stages.length - 1),
    ciphertexts: artifactCiphertexts(stage), delta: deltas[index] || {},
  })));
}

function describeStages(graph, descriptions) {
  graph.stages.forEach((stage, index) => Object.assign(stage, descriptions[index] || {}));
}

function traceShell(descriptor, A, validationB, C, graph, steps, packingFlow, { batchCount = 1, matmulType = null } = {}) {
  const m = A.length, k = A[0].length, n = validationB[0].length;
  const type = matmulType || (descriptor.leftSecret && descriptor.rightSecret ? "ct-ct" : "ct-pt");
  return {
    schemaVersion: "1.0", id: `${descriptor.id}-toy`, title: `USENIX Security'25 BLB ${descriptor.shortLabel}`,
    parameters: { matrixShape: { m, k, n }, slotsPerCiphertext: SLOTS, matmulType: type,
      algorithmId: descriptor.id, batchCount },
    operands: {
      left: { id: "A", label: descriptor.leftLabel, shape: [m, k], values: A, secret: descriptor.leftSecret },
      right: { id: "B", label: descriptor.kind === "mhp-ctct" ? "Kᵀ" : descriptor.rightLabel,
        shape: [validationB.length, validationB[0].length], values: validationB, secret: descriptor.rightSecret },
    },
    output: { id: "C", label: descriptor.outputLabel, shape: [m, n] },
    plan: {
      algorithmId: descriptor.id, method: "blb", methodLabel: "USENIX Security'25 BLB",
      visualStyle: "diagram-only", matmulType: type, logicalShape: { m, k, n }, paddedShape: { m, k, n },
      tileGrid: { m: 1, n: 1 }, tileShape: { m, n }, packingStrategyRequested: packingFlow,
      packingStrategyResolved: packingFlow, plannerSource: "BLB Sections 4-5 and Appendix C",
      reductionStrategy: packingFlow, displayMode: "diagram-only", batchLabel: batchCount > 1 ? "heads" : "",
      warnings: [],
    },
    graph, steps, finalTensor: C,
  };
}

function buildSpatialPt(descriptor, A, W) {
  assertShape(A, 4, 4, descriptor.leftLabel); assertShape(W, 4, 2, descriptor.rightLabel);
  const C = matmul(A, W);
  const x = packedArtifact({ id: `${descriptor.id}-x`, label: descriptor.leftLabel, values: flattenSpatial(A), revealAt: 1, tone: 0 });
  const baby = [0, 1].map((shift) => packedArtifact({ id: `${descriptor.id}-baby-${shift}`, label: `ρ${shift * 4}(${descriptor.leftLabel})`, values: rotateLeft(flattenSpatial(A), shift * 4), revealAt: 2, tone: shift }));
  const partials = Array.from({ length: 2 }, (_, out) => Array.from({ length: 4 }, (_, inner) => {
    const values = A.map((row) => row[inner] * W[inner][out]);
    return packedArtifact({ id: `${descriptor.id}-p-${out}-${inner}`, label: `p${out}${inner}`, values, revealAt: 3,
      // Every partial sum still comes from the encrypted A column k.
      tone: inner % 4, kind: "product" });
  }));
  const outCols = columns(C).map((values, col) => packedArtifact({ id: `${descriptor.id}-out-${col}`,
    label: `${descriptor.outputLabel}${SUB[col]}`, values, revealAt: 4, tone: col, kind: "result" }));
  const graph = { id: `${descriptor.id}-blb-bsgs`, algorithmId: descriptor.id, title: descriptor.shortLabel, subtitle: "", stages: [
    { id: `${descriptor.id}-input`, title: `${descriptor.leftLabel} × ${descriptor.rightLabel}`, caption: "", revealAt: 0, rows: [[
      spatialMatrix("A", descriptor.leftLabel, A, 0), operator(`${descriptor.id}-times`, "×", "", 0), ptMatrix("B", descriptor.rightLabel, W, 0),
    ]] },
    { id: `${descriptor.id}-spatial`, title: "Spatial-first", caption: "", revealAt: 1, rows: [[x]] },
    { id: `${descriptor.id}-baby`, title: "Baby ρ", caption: "", revealAt: 2, rows: [baby] },
    { id: `${descriptor.id}-products`, title: "⊙ PT", caption: "", revealAt: 3, rows: partials.flatMap((family) => [family]) },
    { id: `${descriptor.id}-giant`, title: "Giant ρ + Σ", caption: "", revealAt: 4, rows: outCols.map((out, col) => [
      ...partials[col].flatMap((p, i) => i ? [operator(`${descriptor.id}-plus-${col}-${i}`, "+", "", 4), p] : [p]),
      operator(`${descriptor.id}-eq-${col}`, "=", "", 4), out,
    ]) },
    { id: `${descriptor.id}-output`, title: descriptor.outputLabel, caption: "", revealAt: 5, rows: [[
      outCols[0], operator(`${descriptor.id}-join`, "∥", "", 5), outCols[1]],
      [operator(`${descriptor.id}-decode`, "↓", "", 5), spatialMatrix("C", descriptor.outputLabel, C, 5)],
    ] },
  ] };
  describeStages(graph, [
    { stepTitle: "Input encrypted activation and plaintext weight", shortTitle: "Input", description: `Compute ${descriptor.outputLabel} = ${descriptor.leftLabel}${descriptor.rightLabel} with the activation encrypted and the model weight plaintext.` },
    { stepTitle: "Pack activation in spatial-first order", shortTitle: "Spatial pack", description: "Store consecutive spatial positions before moving along the reduction dimension, matching BLB's consistent linear-layer packing." },
    { stepTitle: "Generate BSGS baby rotations", shortTitle: "Baby rotate", description: "Rotate the packed activation for the baby-step offsets shared by the plaintext-weight products." },
    { stepTitle: "Multiply plaintext weight lanes", shortTitle: "PT products", description: "Apply the plaintext weight masks to the matching rotated activation lanes in parallel." },
    { stepTitle: "Apply giant rotations and sum", shortTitle: "Giant sum", description: "Rotate the baby-step partials by their giant offsets and add across the reduction dimension for each output column." },
    { stepTitle: "Return spatial-first output", shortTitle: "Spatial output", description: `Keep ${descriptor.outputLabel} in BLB's spatial-first layout for the following fused operators; decoding is visualization-only.` },
  ]);
  const steps = stepsFromGraph(descriptor, graph, A, W, C, [{}, { ciphertextCount: 1 }, { rotations: 1 },
    { ptCtMultiplications: 8, multiplicativeDepth: 1 }, { rotations: 2, additions: 6, rescales: 2, ciphertextCount: 2 }, {}]);
  return traceShell(descriptor, A, W, C, graph, steps, "Spatial-first BSGS", { batchCount: HEADS });
}

function buildMhpCtCt(descriptor, Q, K) {
  assertShape(Q, 4, 2, "Q"); assertShape(K, 4, 2, "K");
  const KT = transpose(K), S = matmul(Q, KT);
  const q = packedArtifact({ id: `${descriptor.id}-q`, label: "Q(h₀∥h₁)", values: flattenSpatial(Q), revealAt: 1, tone: 0, heads: true });
  const kr = packedArtifact({ id: `${descriptor.id}-k`, label: "Kᵀ(h₀∥h₁)", values: flattenReduce(KT), revealAt: 1, tone: 1, heads: true });
  const rotated = Array.from({ length: 4 }, (_, t) => packedArtifact({ id: `${descriptor.id}-r-${t}`, label: `ρ${t}(Kᵀ)`,
    values: rotateLeft(flattenReduce(KT), t), revealAt: 2, tone: t, heads: true }));
  // Per-slot products q .* rho_t(K^T); the paper's inner-rotation + diagonal
  // alignment steps then accumulate these into the diagonal ciphertexts.
  const products = rotated.map((r, t) => packedArtifact({ id: `${descriptor.id}-pr-${t}`, label: `P${t}`,
    values: q.cells.map((c, s) => c.value * r.cells[s].value), revealAt: 3, tone: t, kind: "product" }));
  const diagonals = Array.from({ length: 4 }, (_, t) => packedArtifact({ id: `${descriptor.id}-d-${t}`, label: `D${t}`,
    values: upperDiagonal(S, t, 4), revealAt: 4, tone: t, heads: true, kind: "result" }));
  const graph = { id: `${descriptor.id}-blb-ctct`, algorithmId: descriptor.id, title: descriptor.shortLabel, subtitle: "", stages: [
    { id: `${descriptor.id}-input`, title: "Q × Kᵀ", caption: "", revealAt: 0, rows: [[
      spatialMatrix("A", "Q", Q, 0), operator(`${descriptor.id}-times`, "×", "", 0), spatialMatrix("K", "K", K, 0, "CT", 2),
    ]] },
    { id: `${descriptor.id}-mhp`, title: "MHP", caption: "", revealAt: 1, rows: [[q], [kr]] },
    { id: `${descriptor.id}-inner`, title: "Inner ρ + mask", caption: "", revealAt: 2, rows: rotated.map((r) => [r]) },
    { id: `${descriptor.id}-row`, title: "⊙", caption: "", revealAt: 3, rows: products.map((p, t) => [q,
      operator(`${descriptor.id}-mul-${t}`, "⊙", "", 3), rotated[t], operator(`${descriptor.id}-eq-${t}`, "=", "", 3), p]) },
    { id: `${descriptor.id}-diag`, title: "Inner ρ + Σ → Diag", caption: "", revealAt: 4, rows: [diagonals] },
    { id: `${descriptor.id}-output`, title: "Diag", caption: "", revealAt: 5, rows: [[
      operator(`${descriptor.id}-decode`, "↓", "", 5), diagonalMatrix("C", "S", S, 5),
    ]] },
  ] };
  describeStages(graph, [
    { stepTitle: "Input encrypted Q and K", shortTitle: "Q × Kᵀ", description: "Form the attention-score product from encrypted Q and K; the visual matrix transpose only states the logical operation." },
    { stepTitle: "Pack multiple heads in parallel", shortTitle: "MHP pack", description: "Use BLB multi-head packing so neighboring heads share ciphertext rotations and multiplications." },
    { stepTitle: "Prepare K rows with inner rotations", shortTitle: "Inner rotate", description: "Use masked inner rotations to align each reduce-first K row while preserving independent packed heads." },
    { stepTitle: "Multiply aligned Q and K lanes", shortTitle: "CT products", description: "Multiply Q with each aligned K lane, producing ciphertext partial products for the score diagonals." },
    { stepTitle: "Align and sum into score diagonals", shortTitle: "Diagonal sum", description: "Apply the remaining inner/diagonal rotations and add partial products into a diagonal packing of QKᵀ." },
    { stepTitle: "Return diagonal-packed scores", shortTitle: "Diag output", description: "Keep the encrypted attention scores diagonal-packed for the following attention computation; decoding is only a check." },
  ]);
  const steps = stepsFromGraph(descriptor, graph, Q, KT, S, [{}, { ciphertextCount: 2 },
    { rotations: 8, masks: 4, ptCtMultiplications: 4 }, { rotations: 3, ctCtMultiplications: 4, additions: 4, relinearizations: 4, rescales: 4, multiplicativeDepth: 1 },
    { rotations: 4, masks: 8, ptCtMultiplications: 8, additions: 4 }, {}]);
  return traceShell(descriptor, Q, KT, S, graph, steps, "MHP spatial×reduce→diagonal", { batchCount: HEADS });
}

function buildMpcCtCt(descriptor, S, V) {
  assertShape(S, 4, 4, "S"); assertShape(V, 4, 2, "V");
  const H = matmul(S, V), Vpad = V.map((row) => [...row, 0, 0]), Hpad = H.map((row) => [...row, 0, 0]);
  const sct = packedArtifact({ id: `${descriptor.id}-s`, label: "S(h₀∥h₁)", values: flattenSpatial(S), revealAt: 1, tone: 0 });
  const vct = packedArtifact({ id: `${descriptor.id}-v`, label: "V∥0(h₀∥h₁)", values: flattenReduce(Vpad), revealAt: 1, tone: 1 });
  const rotated = Array.from({ length: 4 }, (_, t) => packedArtifact({ id: `${descriptor.id}-rv-${t}`, label: `ρ${t}(V∥0)`,
    values: rotateLeft(flattenReduce(Vpad), t), revealAt: 2, tone: t }));
  // Per-slot products s .* rho_t(V||0); the inner-rotation + diagonal alignment
  // steps accumulate these into the padded diagonal ciphertexts.
  const products = rotated.map((r, t) => packedArtifact({ id: `${descriptor.id}-pr-${t}`, label: `P${t}`,
    values: sct.cells.map((c, s) => c.value * r.cells[s].value), revealAt: 3, tone: t, kind: "product" }));
  const diagonals = Array.from({ length: 4 }, (_, t) => packedArtifact({ id: `${descriptor.id}-dh-${t}`, label: `D${t}`,
    values: upperDiagonal(Hpad, t, 4), revealAt: 4, tone: t, kind: "result" }));
  const graph = { id: `${descriptor.id}-blb-softmax-v`, algorithmId: descriptor.id, title: descriptor.shortLabel, subtitle: "", stages: [
    { id: `${descriptor.id}-input`, title: "S × V", caption: "", revealAt: 0, rows: [[
      mpcMatrix("A", "S", S, 0), operator(`${descriptor.id}-times`, "×", "", 0), mpcMatrix("B", "V", V, 0),
    ]] },
    { id: `${descriptor.id}-convert`, title: "MPC → CKKS", caption: "", revealAt: 1, rows: [[sct], [vct]] },
    { id: `${descriptor.id}-inner`, title: "Inner ρ + mask", caption: "", revealAt: 2, rows: rotated.map((r) => [r]) },
    { id: `${descriptor.id}-row`, title: "⊙", caption: "", revealAt: 3, rows: products.map((p, t) => [sct,
      operator(`${descriptor.id}-mul-${t}`, "⊙", "", 3), rotated[t], operator(`${descriptor.id}-eq-${t}`, "=", "", 3), p]) },
    { id: `${descriptor.id}-diag`, title: "Inner ρ + Σ → Diag", caption: "", revealAt: 4, rows: [diagonals] },
    { id: `${descriptor.id}-output`, title: "Diag → H", caption: "", revealAt: 5, rows: [[
      diagonalMatrix(`${descriptor.id}-padded`, "H∥0", Hpad, 5), operator(`${descriptor.id}-crop`, "↘", "", 5), spatialMatrix("C", "H", H, 5),
    ]] },
  ] };
  describeStages(graph, [
    { stepTitle: "Input MPC-shared S and V", shortTitle: "S × V", description: "The Softmax output and V are secret-shared at the nonlinear boundary before the encrypted context multiplication." },
    { stepTitle: "Convert shares from MPC to CKKS", shortTitle: "MPC → CKKS", description: "Securely convert both operands to CKKS and arrange them in the spatial/reduce layouts required by BLB's ct-ct MatMul." },
    { stepTitle: "Prepare V rows with inner rotations", shortTitle: "Inner rotate", description: "Use masked inner rotations to align each packed V row with the corresponding score lanes." },
    { stepTitle: "Multiply aligned S and V lanes", shortTitle: "CT products", description: "Evaluate ciphertext-ciphertext products for the aligned score and value lanes." },
    { stepTitle: "Align and sum context diagonals", shortTitle: "Diagonal sum", description: "Rotate and add the partial products to obtain a padded diagonal packing of the context result." },
    { stepTitle: "Crop padding and return H", shortTitle: "Context output", description: "Remove the visualization's padded columns and expose the logical context matrix H; the encrypted pipeline retains the compatible packed form." },
  ]);
  const steps = stepsFromGraph(descriptor, graph, S, V, H, [{}, { ciphertextCount: 2 },
    { rotations: 8, masks: 4, ptCtMultiplications: 4 }, { rotations: 3, ctCtMultiplications: 4, additions: 4, relinearizations: 4, rescales: 4, multiplicativeDepth: 1 },
    { rotations: 4, masks: 8, ptCtMultiplications: 8, additions: 4 }, {}]);
  return traceShell(descriptor, S, V, H, graph, steps, "MPC→MHP spatial×reduce→diagonal", { batchCount: HEADS });
}

function buildDiagPt(descriptor, H, W) {
  assertShape(H, 4, 4, "H"); assertShape(W, 4, 2, "Wo");
  const O = matmul(H, W);
  const n = H.length, outCols = W[0].length;
  // BLB Appendix C.2 modified matmulcp: E[i,j] = sum_t D_t[i] * W[(i+t) mod L, j].
  // D_t is replicated across the L column-blocks of the spatial-first output
  // layout (slot j*L + i = column j, row i), and the plaintext mask M_t holds
  // W[(i+t) mod L, j], so sum_t (D_t .* M_t) reproduces E exactly.
  const replicatedDiagonal = (t) => pad16(Array.from({ length: outCols }, () => upperDiagonal(H, t, n)).flat());
  const weightMask = (t) => {
    const mask = new Array(SLOTS).fill(0);
    for (let j = 0; j < outCols; j += 1) for (let i = 0; i < n; i += 1) {
      mask[j * n + i] = Number(W[(i + t) % n][j]);
    }
    return mask;
  };
  const diagonals = Array.from({ length: 4 }, (_, t) => packedArtifact({ id: `${descriptor.id}-d-${t}`, label: `D${t}`,
    values: replicatedDiagonal(t), revealAt: 1, tone: t }));
  const weightMasks = Array.from({ length: 4 }, (_, t) => packedArtifact({ id: `${descriptor.id}-w-${t}`, label: `W${t}`,
    values: weightMask(t), revealAt: 2, tone: 4, badge: "PT" }));
  const products = Array.from({ length: 4 }, (_, t) => packedArtifact({ id: `${descriptor.id}-p-${t}`, label: `P${t}`,
    values: replicatedDiagonal(t).map((v, s) => Number(v) * weightMask(t)[s]), revealAt: 3, tone: t, kind: "product" }));
  const out = packedArtifact({ id: `${descriptor.id}-out`, label: "O", values: flattenSpatial(O), revealAt: 4, tone: 0, kind: "result" });
  const expectedO = pad16(flattenSpatial(O));
  const accumulated = products.reduce((a, p) => a.map((v, s) => v + p.cells[s].value), new Array(SLOTS).fill(0));
  if (JSON.stringify(accumulated) !== JSON.stringify(expectedO)) {
    throw new Error("BLB output projection diagonal products failed verification");
  }
  const graph = { id: `${descriptor.id}-blb-diag-pt`, algorithmId: descriptor.id, title: descriptor.shortLabel, subtitle: "", stages: [
    { id: `${descriptor.id}-input`, title: "H × Wo", caption: "", revealAt: 0, rows: [[
      diagonalMatrix("A", "H", H, 0), operator(`${descriptor.id}-times`, "×", "", 0), ptMatrix("B", "Wo", W, 0),
    ]] },
    { id: `${descriptor.id}-diag`, title: "Diag", caption: "", revealAt: 1, rows: [diagonals] },
    { id: `${descriptor.id}-weights`, title: "Wo ⧉", caption: "", revealAt: 2, rows: [weightMasks] },
    { id: `${descriptor.id}-products`, title: "ρ(H) ⊙ Wo", caption: "", revealAt: 3, rows: products.map((p, t) => [diagonals[t],
      operator(`${descriptor.id}-mul-${t}`, "⊙", "", 3), weightMasks[t], operator(`${descriptor.id}-eq-${t}`, "=", "", 3), p]) },
    { id: `${descriptor.id}-sum`, title: "Σ", caption: "", revealAt: 4, rows: [[...products.flatMap((p, i) => i ? [operator(`${descriptor.id}-plus-${i}`, "+", "", 4), p] : [p]),
      operator(`${descriptor.id}-eq`, "=", "", 4), out]] },
    { id: `${descriptor.id}-output`, title: "Spatial-first", caption: "", revealAt: 5, rows: [[out],
      [operator(`${descriptor.id}-decode`, "↓", "", 5), spatialMatrix("C", "O", O, 5)]] },
  ] };
  describeStages(graph, [
    { stepTitle: "Input diagonal H and plaintext Wo", shortTitle: "H × Wo", description: "Consume the diagonal-packed multi-head context directly and multiply by the plaintext output-projection weight." },
    { stepTitle: "Replicate context diagonals", shortTitle: "H diagonals", description: "Expose each cyclic diagonal of H across the output-column blocks used by the modified ct-pt MatMul." },
    { stepTitle: "Encode reordered Wo masks", shortTitle: "Weight masks", description: "Reorder and duplicate plaintext weight entries so each mask matches one context diagonal and its target output column." },
    { stepTitle: "Multiply diagonal and weight lanes", shortTitle: "PT products", description: "Multiply every replicated H diagonal with its corresponding plaintext Wo mask." },
    { stepTitle: "Sum diagonal products", shortTitle: "Product sum", description: "Add all diagonal contributions; their slot layout now equals the projected output in spatial-first order." },
    { stepTitle: "Return spatial-first projection", shortTitle: "Spatial output", description: "The output projection is already in BLB's spatial-first packing, avoiding a separate repacking operation." },
  ]);
  const steps = stepsFromGraph(descriptor, graph, H, W, O, [{}, { ciphertextCount: 4 }, {},
    { rotations: 4, ptCtMultiplications: 4, multiplicativeDepth: 1 }, { additions: 3, ciphertextCount: 1 }, {}]);
  return traceShell(descriptor, H, W, O, graph, steps, "Diagonal×PT→spatial-first", { batchCount: HEADS });
}

function buildMpcMatmul(descriptor, A, B) {
  const C = matmul(A, B);
  const a0 = mpcMatrix(`${descriptor.id}-a0`, `[${descriptor.leftLabel}]₀`, A, 1);
  const a1 = mpcMatrix(`${descriptor.id}-a1`, `[${descriptor.leftLabel}]₁`, A.map((r) => r.map(() => 0)), 1);
  const b0 = mpcMatrix(`${descriptor.id}-b0`, `[${descriptor.rightLabel}]₀`, B, 1);
  const b1 = mpcMatrix(`${descriptor.id}-b1`, `[${descriptor.rightLabel}]₁`, B.map((r) => r.map(() => 0)), 1);
  const p00 = mpcMatrix(`${descriptor.id}-p00`, "P₀₀", C, 2);
  const zero = C.map((r) => r.map(() => 0));
  const p01 = mpcMatrix(`${descriptor.id}-p01`, "P₀₁", zero, 2);
  const p10 = mpcMatrix(`${descriptor.id}-p10`, "P₁₀", zero, 2);
  const p11 = mpcMatrix(`${descriptor.id}-p11`, "P₁₁", zero, 2);
  const graph = { id: `${descriptor.id}-blb-mpc`, algorithmId: descriptor.id, title: descriptor.shortLabel, subtitle: "", stages: [
    { id: `${descriptor.id}-input`, title: `${descriptor.leftLabel} × ${descriptor.rightLabel}`, caption: "", revealAt: 0, rows: [[
      mpcMatrix("A", descriptor.leftLabel, A, 0), operator(`${descriptor.id}-times`, "×", "", 0), mpcMatrix("B", descriptor.rightLabel, B, 0),
    ]] },
    { id: `${descriptor.id}-shares`, title: "⟦ · ⟧₀ ∥ ⟦ · ⟧₁", caption: "", revealAt: 1, rows: [[a0, a1], [b0, b1]] },
    { id: `${descriptor.id}-products`, title: "⊙ MPC", caption: "", revealAt: 2, rows: [[p00, p01], [p10, p11]] },
    { id: `${descriptor.id}-sum`, title: "Σ mod 2ˡ", caption: "", revealAt: 3, rows: [[p00,
      operator(`${descriptor.id}-plus1`, "+", "", 3), p01, operator(`${descriptor.id}-plus2`, "+", "", 3), p10,
      operator(`${descriptor.id}-plus3`, "+", "", 3), p11]] },
    { id: `${descriptor.id}-output`, title: descriptor.outputLabel, caption: "", revealAt: 4, rows: [[mpcMatrix("C", descriptor.outputLabel, C, 4)]] },
  ] };
  describeStages(graph, [
    { stepTitle: "Input secret-shared matrices", shortTitle: "Input", description: `Compute ${descriptor.outputLabel} = ${descriptor.leftLabel}${descriptor.rightLabel} inside BLB's MPC domain.` },
    { stepTitle: "Split both operands into additive shares", shortTitle: "Share inputs", description: "Represent each operand as the sum of two additive shares over the MPC ring." },
    { stepTitle: "Multiply all share pairs", shortTitle: "Share products", description: "Evaluate the four pairwise products needed when multiplying two additively shared matrices." },
    { stepTitle: "Add products modulo 2ˡ", shortTitle: "Modular sum", description: "Combine all share-product terms modulo the MPC ring to reconstruct shares of the matrix product." },
    { stepTitle: "Return secret-shared output", shortTitle: "MPC output", description: `Keep ${descriptor.outputLabel} secret-shared for the surrounding hybrid protocol.` },
  ]);
  const steps = stepsFromGraph(descriptor, graph, A, B, C, [{}, {}, {}, { additions: 3 }, {}]);
  return traceShell(descriptor, A, B, C, graph, steps, "MPC secret-share MatMul", { matmulType: "mpc-mpc" });
}

export function buildBlbOperationTrace(id, options = {}) {
  const descriptor = getBlbOperation(id);
  if (!descriptor) throw new Error(`Unknown BLB operation: ${id}`);
  if (options.leftSecret != null && Boolean(options.leftSecret) !== descriptor.leftSecret) throw new Error(`${descriptor.shortLabel} has a fixed left operand type`);
  if (options.rightSecret != null && Boolean(options.rightSecret) !== descriptor.rightSecret) throw new Error(`${descriptor.shortLabel} has a fixed right operand type`);
  if (options.slotsPerCiphertext != null && Number(options.slotsPerCiphertext) !== SLOTS) throw new Error("BLB toy MHP uses exactly 16 slots");
  const A = options.A || descriptor.A, B = options.B || descriptor.B;
  if (descriptor.kind === "spatial-pt") return buildSpatialPt(descriptor, A, B);
  if (descriptor.kind === "mhp-ctct") return buildMhpCtCt(descriptor, A, B);
  if (descriptor.kind === "mpc-ctct") return buildMpcCtCt(descriptor, A, B);
  if (descriptor.kind === "diag-pt") return buildDiagPt(descriptor, A, B);
  return buildMpcMatmul(descriptor, A, B);
}

export function blbOperationDefaults(id) {
  const descriptor = getBlbOperation(id);
  if (!descriptor) return null;
  return { m: descriptor.A.length, k: descriptor.A[0].length,
    n: descriptor.inputMode === "paired-columns" ? descriptor.A.length : descriptor.B[0].length,
    A: cloneMatrix(descriptor.A), B: cloneMatrix(descriptor.B), leftSecret: descriptor.leftSecret,
    rightSecret: descriptor.rightSecret, slotsPerCiphertext: SLOTS, strategy: descriptor.strategy };
}

export function parseBlbOperationInputs(raw, descriptor) {
  if (descriptor.inputMode !== "paired-columns") return parseAndValidateInputs(raw);
  const errors = {}; let A; let B;
  try {
    A = parseMatrixText(raw.valuesTextA); B = parseMatrixText(raw.valuesTextB);
    assertShape(A, 4, 2, descriptor.leftLabel); assertShape(B, 4, 2, descriptor.rightLabel);
  } catch (error) { errors.values = error.message; }
  if (Number(raw.slots) !== SLOTS) errors.slots = "BLB toy MHP uses exactly 16 slots";
  if (Object.keys(errors).length) return { ok: false, errors, inputs: null };
  return { ok: true, errors: {}, inputs: { m: 4, k: 2, n: 4, A, B,
    leftSecret: String(raw.leftType).toLowerCase() === "ct", rightSecret: String(raw.rightType).toLowerCase() === "ct",
    slotsPerCiphertext: SLOTS, strategy: "column", customLayouts: null, valueMode: "manual", seed: raw.seed } };
}
