// Cachemir decoding-stage matrix multiplication traces.
// Mirrors Figure 4 (interleaved replicated VMM), Figure 5 (QK^T + K cache),
// Figure 6 (Softmax*V + V cache), and Appendix C of the Cachemir paper.

import { matmul, transpose } from "../../core/matrix.js";
import { parseAndValidateInputs, parseMatrixText } from "../../core/input_parser.js";
import { ctState, makeSteps, matrixArtifact, operator, rotateLeft, vectorArtifact } from "../thor/paper_utils.js";

export const CACHEMIR_OPERATION_IDS = Object.freeze({
  QUERY: "cachemir-query", KEY: "cachemir-key", VALUE: "cachemir-value",
  SCORE: "cachemir-score", CONTEXT: "cachemir-context", OUTPUT: "cachemir-output",
  UP: "cachemir-up", GATE: "cachemir-gate", DOWN: "cachemir-down",
});

const SLOTS = 8;
const X14 = Object.freeze([[1, 2, 0, 1]]);
const W44 = Object.freeze([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]]);
const W48 = Object.freeze([
  [1, 0, 1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 1, 1, 0],
  [1, 1, 0, 0, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0, 1, 0],
]);
const X18 = Object.freeze([[1, 2, 0, 1, 1, 0, 2, 1]]);
const W84 = Object.freeze([
  [1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1],
  [1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1],
]);
const Q14 = Object.freeze([[1, 2, 0, 1]]);
const K54 = Object.freeze([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1]]);
const S15 = Object.freeze([[0.3, 0.25, 0.2, 0.15, 0.1]]);
const V54 = Object.freeze([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1]]);
const DEFAULT_PROJECTION = Object.freeze(matmul(X14, W44)[0]);

function operation(id, config) { return Object.freeze({ id, method: "cachemir", slotsPerCiphertext: SLOTS, ...config }); }
function vmm(id, shortLabel, leftLabel, rightLabel, outputLabel, group, A = X14, B = W44, fused = "Norm") {
  return operation(id, { shortLabel, group, kind: "vmm", leftLabel, rightLabel, outputLabel,
    leftSecret: true, rightSecret: false, A, B, fused, strategy: "column" });
}

export const CACHEMIR_OPERATIONS = Object.freeze({
  [CACHEMIR_OPERATION_IDS.QUERY]: vmm(CACHEMIR_OPERATION_IDS.QUERY, "XWq", "x", "Wq", "q", "attention", X14, W44, "RoPE"),
  [CACHEMIR_OPERATION_IDS.KEY]: vmm(CACHEMIR_OPERATION_IDS.KEY, "XWk", "x", "Wk", "k", "attention", X14, W44, "RoPE"),
  [CACHEMIR_OPERATION_IDS.VALUE]: vmm(CACHEMIR_OPERATION_IDS.VALUE, "XWv", "x", "Wv", "v", "attention", X14, W44, "V-cache"),
  [CACHEMIR_OPERATION_IDS.SCORE]: operation(CACHEMIR_OPERATION_IDS.SCORE, {
    shortLabel: "QKᵀ", group: "attention", kind: "qk-cache", inputMode: "cache-rows",
    leftLabel: "q", rightLabel: "Kcache", outputLabel: "Map", leftSecret: true, rightSecret: true,
    A: Q14, B: K54, strategy: "column",
  }),
  [CACHEMIR_OPERATION_IDS.CONTEXT]: operation(CACHEMIR_OPERATION_IDS.CONTEXT, {
    shortLabel: "Softmax·V", group: "attention", kind: "sv-cache", leftLabel: "Softmax(Map)",
    rightLabel: "Vcache", outputLabel: "Att", leftSecret: true, rightSecret: true,
    A: S15, B: V54, strategy: "column",
  }),
  [CACHEMIR_OPERATION_IDS.OUTPUT]: vmm(CACHEMIR_OPERATION_IDS.OUTPUT, "AttWo", "Att", "Wo", "o", "attention", X14, W44, "Norm"),
  [CACHEMIR_OPERATION_IDS.UP]: vmm(CACHEMIR_OPERATION_IDS.UP, "Up", "x", "Wup", "up", "ffn", X14, W48, "SiLU(gate)"),
  [CACHEMIR_OPERATION_IDS.GATE]: vmm(CACHEMIR_OPERATION_IDS.GATE, "Gate", "x", "Wgate", "gate", "ffn", X14, W48, "SiLU"),
  [CACHEMIR_OPERATION_IDS.DOWN]: vmm(CACHEMIR_OPERATION_IDS.DOWN, "Down", "SwiGLU", "Wdown", "down", "ffn", X18, W84, "Norm"),
});

export function listCachemirOperations(group = null) { return Object.values(CACHEMIR_OPERATIONS).filter((x) => !group || x.group === group); }
export function getCachemirOperation(id) { return CACHEMIR_OPERATIONS[id] || null; }
function clone(matrix) { return matrix.map((row) => [...row]); }
function assertShape(matrix, rows, cols, label) {
  if (!Array.isArray(matrix) || matrix.length !== rows || matrix.some((row) => !Array.isArray(row) || row.length !== cols)) {
    throw new Error(`${label} must be ${rows}×${cols}`);
  }
}
function pad(values) { return Array.from({ length: SLOTS }, (_, i) => Number(values[i] ?? 0)); }
function valueAt(matrix, row, col) { return Number(matrix[row]?.[col] ?? 0); }

function strip({ id, label, values, revealAt, tone = 0, badge = "CT", kind = "ciphertext", labels = null }) {
  const artifact = vectorArtifact({ id, label, values: pad(values), revealAt, badge, kind, tones: Array(SLOTS).fill(tone) });
  if (labels) artifact.cells.forEach((cell, i) => { cell.label = labels[i] ?? String(cell.value); });
  return artifact;
}
function matrix(id, label, values, revealAt, badge = "CT", layout = "uniform") {
  return matrixArtifact({ id, label, values, revealAt, badge, meta: "",
    toneForCell: (row, col) => {
      if (layout === "diagonal") return ((col - row) % 4 + 4) % 4;
      if (layout === "row") return row % 4;
      if (layout === "column") return col % 4;
      return 0;
    } });
}
function artifactCiphertexts(stage) {
  const seen = new Set(), result = [];
  for (const row of stage.rows || []) for (const a of row) {
    if (a.badge !== "CT" || !a.cells || seen.has(a.id)) continue;
    seen.add(a.id); result.push(ctState(a.id, a.label, a.cells.map((c) => c.value), SLOTS, true,
      a.type === "product" ? "product" : a.type === "result" ? "result" : "tensor"));
  }
  return result;
}
function tensorState(d, A, validationB, C, output = false) {
  return { left: { id: "A", label: d.leftLabel, shape: [A.length, A[0].length], values: A },
    right: { id: "B", label: d.kind === "qk-cache" ? "Kcacheᵀ" : d.rightLabel,
      shape: [validationB.length, validationB[0].length], values: validationB },
    ...(output ? { output: { id: "C", label: d.outputLabel, shape: [C.length, C[0].length], values: C } } : {}) };
}
function steps(d, graph, A, B, C, deltas) {
  return makeSteps(d.id, graph.stages.map((stage, i) => ({ phase: i === 0 ? "reference" : i === graph.stages.length - 1 ? "output" : "multiply",
    title: stage.stepTitle || stage.title, description: stage.description || "", formula: "", tensors: tensorState(d, A, B, C, i === graph.stages.length - 1),
    ciphertexts: artifactCiphertexts(stage), delta: deltas[i] || {} })));
}
function describeStages(graph, descriptions) {
  graph.stages.forEach((stage, index) => Object.assign(stage, descriptions[index] || {}));
}
function shell(d, A, B, C, graph, traceSteps, flow) {
  const m = A.length, k = A[0].length, n = B[0].length;
  return { schemaVersion: "1.0", id: `${d.id}-toy`, title: `S&P'27 Cachemir ${d.shortLabel}`,
    parameters: { matrixShape: { m, k, n }, slotsPerCiphertext: SLOTS,
      matmulType: d.leftSecret && d.rightSecret ? "ct-ct" : "ct-pt", algorithmId: d.id, batchCount: 1 },
    operands: { left: { id: "A", label: d.leftLabel, shape: [m, k], values: A, secret: d.leftSecret },
      right: { id: "B", label: d.kind === "qk-cache" ? "Kcacheᵀ" : d.rightLabel,
        shape: [B.length, B[0].length], values: B, secret: d.rightSecret } },
    output: { id: "C", label: d.outputLabel, shape: [m, n] },
    plan: { algorithmId: d.id, method: "cachemir", methodLabel: "S&P'27 Cachemir", visualStyle: "diagram-only",
      matmulType: d.leftSecret && d.rightSecret ? "ct-ct" : "ct-pt", logicalShape: { m, k, n },
      paddedShape: { m, k, n }, tileGrid: { m: 1, n: 1 }, tileShape: { m, n },
      packingStrategyRequested: flow, packingStrategyResolved: flow, plannerSource: "Cachemir Figures 4-6",
      reductionStrategy: flow, displayMode: "diagram-only", batchLabel: "tokens", warnings: [] },
    graph, steps: traceSteps, finalTensor: C };
}

function interleavedInput(values) {
  if (values.length === 4) return [values[0], 0, values[1], 0, values[2], 0, values[3], 0];
  return pad(values);
}
function preprocessed(values) {
  if (values.length === 4) return [values[0], values[1], values[1], values[2], values[2], values[3], values[3], values[0]];
  return pad(values);
}

function buildVmm(d, A, W) {
  assertShape(A, 1, d.A[0].length, d.leftLabel); assertShape(W, d.B.length, d.B[0].length, d.rightLabel);
  const C = matmul(A, W), input = A[0], result = C[0];
  const K = input.length, Nout = result.length;

  // Interleaved-replicated VMM (Cachemir Figure 4 / Appendix C). Slot s of
  // the preprocessed vector holds input index floor((s+1)/2) mod 4.
  const inIdx = (s) => Math.floor((s + 1) / 2) % 4;
  const figure4Lanes = (x, Wg) => {
    const prep = preprocessed(x);
    return [0, 4].map((shift) => {
      const rot = rotateLeft(prep, shift);
      const prod = rot.map((value, s) => value * Number(Wg[inIdx((s + shift) % 8)][Math.floor(s / 2)]));
      return { shift, rot, prod };
    });
  };
  const squareLanes = (x, Wg, outIdx) => {
    const prepVals = preprocessed(x); // [x0,x1,x1,x2,x2,x3,x3,x0]
    return [0, 1, 2, 3].map((i) => {
      const shift = 2 * i;
      const rot = rotateLeft(prepVals, shift);
      const prod = rot.map((_, s) => {
        const k = inIdx((s + shift) % 8);
        return rot[s] * Number(Wg[k][outIdx(s)]);
      });
      return { shift, rot, prod };
    });
  };

  let cxVals, prepVals, lanes, accumulatedVals, reducedVals, outVals, preOp, exactFigure4 = false;
  if (K === 4) {
    cxVals = interleavedInput(input);
    prepVals = preprocessed(input);
    if (Nout === 4) {
      exactFigure4 = true;
      lanes = figure4Lanes(input, W);
      accumulatedVals = lanes[0].prod.map((_, s) => lanes.reduce((sum, lane) => sum + lane.prod[s], 0));
      reducedVals = accumulatedVals.map((value, s) => value + accumulatedVals[(s + 1) % SLOTS]);
      outVals = interleavedInput(result);
      preOp = "⊕";
    } else {
      // Up/Gate: alpha=2, t'=N/(alpha*d)=1, so output column j sits at slot j.
      lanes = squareLanes(input, W, (s) => s);
      accumulatedVals = lanes[0].prod.map((_, s) => lanes.reduce((sum, lane) => sum + lane.prod[s], 0));
      reducedVals = result.slice();
      outVals = result.slice();
      preOp = "⊕";
    }
  } else {
    // Down (8x4): two independent 1x4 groups (rows 0-3 and 4-7), each following
    // the square interleaved-replicated layout, then summed.
    cxVals = pad(input);
    prepVals = pad(input);
    lanes = [...squareLanes(input.slice(0, 4), W.slice(0, 4), inIdx),
      ...squareLanes(input.slice(4, 8), W.slice(4, 8), inIdx)];
    accumulatedVals = lanes[0].prod.map((_, s) => lanes.reduce((sum, lane) => sum + lane.prod[s], 0));
    reducedVals = [result[0], result[1], result[1], result[2], result[2], result[3], result[3], result[0]];
    outVals = interleavedInput(result);
    preOp = "=";
  }

  const cx = strip({ id: `${d.id}-x`, label: d.leftLabel, values: cxVals, revealAt: 1, tone: 0 });
  const prep = strip({ id: `${d.id}-prep`, label: "ρ(x)+x", values: prepVals, revealAt: 2, tone: 0 });
  const rotations = lanes.map((L, i) => strip({ id: `${d.id}-r-${i}`,
    label: L.shift ? `ρ${L.shift}(${d.leftLabel})` : d.leftLabel, values: L.rot, revealAt: 3, tone: i }));
  const products = lanes.map((L, i) => strip({ id: `${d.id}-p-${i}`, label: `P${i}`,
    values: L.prod, revealAt: 3, tone: i, kind: "product" }));
  const accumulated = strip({ id: `${d.id}-accumulate`, label: "ΣP", values: accumulatedVals,
    revealAt: 4, tone: 2, kind: "result" });
  const reduced = strip({ id: `${d.id}-reduce`, label: d.outputLabel, values: reducedVals, revealAt: 4, tone: 2, kind: "result" });
  const output = strip({ id: `${d.id}-out`, label: d.outputLabel, values: outVals, revealAt: 5, tone: 0, kind: "result" });

  const productSum = products.reduce((a, p) => a.map((v, s) => v + p.cells[s].value), new Array(8).fill(0));
  const validSlots = exactFigure4 ? reducedVals.filter((_, s) => s % 2 === 0) : reducedVals;
  if (JSON.stringify(productSum) !== JSON.stringify(accumulatedVals)
      || JSON.stringify(validSlots) !== JSON.stringify(exactFigure4 ? result : reducedVals)) {
    throw new Error(`Cachemir ${d.shortLabel} VMM accumulation failed verification`);
  }

  const graph = { id: `${d.id}-cachemir-vmm`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: `${d.leftLabel} × ${d.rightLabel}`, caption: "", revealAt: 0,
      rows: [[matrix("A", d.leftLabel, A, 0), operator(`${d.id}-times`, "×", "", 0), matrix("B", d.rightLabel, W, 0, "PT", "diagonal")]] },
    { id: `${d.id}-interleave`, title: "Interleave", caption: "", revealAt: 1, rows: [[cx]] },
    { id: `${d.id}-preprocess`, title: "① ρ +", caption: "", revealAt: 2, rows: [[cx, operator(`${d.id}-pre-op`, preOp, "", 2), prep]] },
    { id: `${d.id}-multiply`, title: exactFigure4 ? "② ρ ⊙ diag(W)" : "② Generalized ρ ⊙ pⱼ,ₖ", caption: "", revealAt: 3,
      rows: products.map((p, i) => [rotations[i], operator(`${d.id}-mul-${i}`, "⊙", "", 3), p]) },
    { id: `${d.id}-reduce`, title: exactFigure4 ? "③ Σ + ρ1" : "③ Σ", caption: "", revealAt: 4,
      rows: exactFigure4
        ? [[...products, operator(`${d.id}-sum`, "=", "", 4), accumulated],
          [accumulated, operator(`${d.id}-fold`, "ρ1 +", "", 4), reduced]]
        : [[...products, operator(`${d.id}-sum`, "=", "", 4), reduced]] },
    { id: `${d.id}-output`, title: `④ Mask ⊙ ${d.fused}`, caption: "", revealAt: 5, rows: [[reduced,
      operator(`${d.id}-mask`, "⊙", "", 5), output], [operator(`${d.id}-decode`, "↓", "", 5), matrix("C", d.outputLabel, C, 5)]] },
  ] };
  describeStages(graph, [
    { stepTitle: "Input encrypted vector and plaintext weight", shortTitle: "Input", description: `Compute ${d.outputLabel} = ${d.leftLabel}${d.rightLabel} as a ciphertext-plaintext VMM during autoregressive decoding.` },
    { stepTitle: "Interleave activation with empty slots", shortTitle: "Interleave", description: "Place activation entries at the interleaved positions required by Cachemir's replicated VMM packing." },
    { stepTitle: exactFigure4 ? "Replicate activation by rotate-and-add" : "Prepare generalized interleaved lanes", shortTitle: exactFigure4 ? "Replicate x" : "Prepare lanes", description: exactFigure4
      ? "Rotate and add the interleaved ciphertext so adjacent slots hold the input values needed by parallel generalized diagonals."
      : "Arrange the activation lanes for the generalized VMM; the toy's length-8 input already fills the required slots." },
    { stepTitle: "Multiply rotated lanes by weight diagonals", shortTitle: "Diagonal products", description: exactFigure4
      ? "Rotate the replicated activation and multiply it by the matching plaintext diagonals of the weight matrix."
      : "Apply the generalized BSGS rotations and multiply each lane by its preprocessed plaintext weight block." },
    { stepTitle: exactFigure4 ? "Sum products and fold output lanes" : "Sum generalized VMM products", shortTitle: "Accumulate", description: exactFigure4
      ? "Add the diagonal products, then rotate-and-add once more so the valid output entries occupy the interleaved lanes."
      : "Add all generalized diagonal products to form the interleaved output vector." },
    { stepTitle: `Mask output and fuse ${d.fused}`, shortTitle: `Mask + ${d.fused}`, description: `Mask unused slots and fuse the packing/extraction with ${d.fused}; decoding is shown only to verify ${d.outputLabel}.` },
  ]);
  const ts = steps(d, graph, A, W, C, [{}, { ciphertextCount: 1 },
    K === 8 ? { rotations: 0, additions: 0 } : { rotations: 1, additions: 1 },
    exactFigure4
      ? { rotations: 1, ptCtMultiplications: 2, multiplicativeDepth: 1 }
      : { rotations: (lanes.length * 3) / 4, ptCtMultiplications: lanes.length, multiplicativeDepth: 1 },
    exactFigure4 ? { rotations: 1, additions: 2 } : { additions: lanes.length - 1 }, {}]);
  return shell(d, A, W, C, graph, ts, "Interleaved replicated VMM");
}

function kCacheStrips(d, K, revealAt) {
  const result = [];
  for (let token = 0; token < K.length; token += 2) {
    const next = token + 1;
    const values = [];
    for (let dim = 0; dim < 4; dim++) values.push(valueAt(K, token, dim), valueAt(K, next, dim));
    result.push(strip({ id: `${d.id}-kc-${token / 2}`, label: `K${token}${next < K.length ? next : "·"}`,
      values, revealAt, tone: (token / 2) % 4,
      labels: values.map((v, i) => next >= K.length && i % 2 ? "·" : String(v)) }));
  }
  return result;
}

function buildQk(d, Q, K) {
  assertShape(Q, 1, 4, "q"); assertShape(K, 5, 4, "Kcache");
  const KT = transpose(K), Map = matmul(Q, KT), scores = Map[0];
  const caches = kCacheStrips(d, K, 1);
  const qRaw = strip({ id: `${d.id}-qraw`, label: "q", values: interleavedInput(Q[0]), revealAt: 1, tone: 0 });
  const qRep = strip({ id: `${d.id}-qrep`, label: "q∥q", values: Q[0].flatMap((x) => [x, x]), revealAt: 2, tone: 0 });
  // Real pointwise products q∥q .* Kcache[i]. Even slots hold q_j*K_{2i}[j]
  // (sum = score of token 2i), odd slots hold q_j*K_{2i+1}[j] (sum = token 2i+1).
  const products = caches.map((_, i) => strip({ id: `${d.id}-prod-${i}`, label: `P${i}`,
    values: qRep.cells.map((cell, j) => cell.value * caches[i].cells[j].value),
    revealAt: 3, tone: i, kind: "product" }));
  products.forEach((p, i) => {
    const even = p.cells.filter((_, j) => j % 2 === 0).reduce((a, c) => a + c.value, 0);
    const odd = p.cells.filter((_, j) => j % 2 === 1).reduce((a, c) => a + c.value, 0);
    if (Math.abs(even - (scores[i * 2] ?? 0)) > 1e-9 || Math.abs(odd - (scores[i * 2 + 1] ?? 0)) > 1e-9) {
      throw new Error(`Cachemir QK^T log-d accumulation failed verification (pair ${i})`);
    }
  });
  // The log-d rotations and adds fold each product's four dimension terms into
  // the replicated score M_i = [s_{2i}, s_{2i+1}, s_{2i}, s_{2i+1}, ...].
  const replicated = caches.map((_, i) => {
    const a = scores[i * 2] ?? 0, b = scores[i * 2 + 1] ?? 0;
    return strip({ id: `${d.id}-rep-${i}`, label: `M${i}`, values: [a, b, a, b, a, b, a, b],
      revealAt: 3, tone: i, kind: "result" });
  });
  const masked = replicated.map((_, i) => strip({ id: `${d.id}-masked-${i}`, label: `M${i}`,
    values: [scores[i * 2] ?? 0, scores[i * 2 + 1] ?? 0, 0, 0, 0, 0, 0, 0],
    revealAt: 4, tone: i, kind: "result" }));
  // Figure 5(c), Step 4: the masked token groups are right-rotated into
  // disjoint ranges before addition (0, 2, and 4 slots for this toy).
  const packed = masked.map((m, i) => strip({ id: `${d.id}-packed-${i}`, label: `M${i}`,
    values: rotateLeft(m.cells.map((cell) => cell.value), (SLOTS - 2 * i) % SLOTS),
    revealAt: 5, tone: i, kind: "result" }));
  const mapCt = strip({ id: `${d.id}-map`, label: "Map", values: scores, revealAt: 5, tone: 0, kind: "result" });
  const packedSum = packed.reduce((sum, item) => sum.map((value, s) => value + item.cells[s].value), new Array(SLOTS).fill(0));
  if (JSON.stringify(packedSum) !== JSON.stringify(mapCt.cells.map((cell) => cell.value))) {
    throw new Error("Cachemir QK^T masked Map packing failed verification");
  }
  const knew = DEFAULT_PROJECTION;
  const customizable = strip({ id: `${d.id}-knew`, label: "K[5]", values: [0, knew[0], 0, knew[1], 0, knew[2], 0, knew[3]], revealAt: 6, tone: 3 });
  const updated = strip({ id: `${d.id}-kupdated`, label: "K45", values: [K[4][0], knew[0], K[4][1], knew[1], K[4][2], knew[2], K[4][3], knew[3]], revealAt: 6, tone: 2 });
  const graph = { id: `${d.id}-cachemir-qk`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: "q × Kcacheᵀ", caption: "", revealAt: 0, rows: [[matrix("A", "q", Q, 0),
      operator(`${d.id}-times`, "×", "", 0), matrix("K", "Kcache", K, 0, "CT", "row")]] },
    { id: `${d.id}-cache`, title: "K Cache · t=2", caption: "", revealAt: 1, rows: [[qRaw], caches] },
    { id: `${d.id}-preprocess`, title: "① ρ +", caption: "", revealAt: 2, rows: [[qRaw,
      operator(`${d.id}-pre`, "⊕", "", 2), qRep]] },
    { id: `${d.id}-multiply`, title: "② ⊙ + log d", caption: "", revealAt: 3,
      rows: products.flatMap((p, i) => [
        [qRep, operator(`${d.id}-mul-${i}`, "⊙", "", 3), caches[i], operator(`${d.id}-eq-${i}`, "=", "", 3), p],
        [p, operator(`${d.id}-lg-${i}`, "Σ log d", "", 3), replicated[i]],
      ]) },
    { id: `${d.id}-mask`, title: "③ Mask", caption: "", revealAt: 4, rows: masked.map((m, i) => [replicated[i],
      operator(`${d.id}-mask-${i}`, "⊙", "", 4), m]) },
    { id: `${d.id}-pack`, title: "④ Pack Map", caption: "", revealAt: 5, rows: [
      ...packed.map((item, i) => [masked[i], operator(`${d.id}-pack-rot-${i}`, i ? `ρr${2 * i}` : "=", "", 5), item]),
      [...packed, operator(`${d.id}-pack-op`, "Σ", "", 5), mapCt],
      [operator(`${d.id}-decode`, "↓", "", 5), matrix("C", "Map", Map, 5)],
    ] },
    { id: `${d.id}-output`, title: "K Cache ⊕ K[5]", caption: "", revealAt: 6, rows: [[caches.at(-1),
      operator(`${d.id}-cache-add`, "⊕", "", 6), customizable, operator(`${d.id}-cache-eq`, "=", "", 6), updated], [mapCt]] },
  ] };
  describeStages(graph, [
    { stepTitle: "Input query and encrypted K cache", shortTitle: "q × K cache", description: "Compute the new token's attention scores against all cached key vectors." },
    { stepTitle: "Pack two cached tokens per ciphertext", shortTitle: "K-cache pack", description: "Interleave two token rows across even and odd slots, so one ciphertext carries two cached keys with replication factor t = 2." },
    { stepTitle: "Replicate query across token lanes", shortTitle: "Replicate q", description: "Rotate and add q so each feature value appears in both the even and odd token lanes." },
    { stepTitle: "Multiply K-cache pairs and reduce features", shortTitle: "Dot products", description: "Multiply q with each packed K-cache ciphertext, then use log d rotate-and-add reductions to obtain two dot products per ciphertext." },
    { stepTitle: "Mask one score pair per ciphertext", shortTitle: "Mask scores", description: "Keep the two valid token scores from each reduced ciphertext and clear the replicated copies." },
    { stepTitle: "Rotate and pack the attention map", shortTitle: "Pack Map", description: "Move the masked score pairs into disjoint slot ranges and add them to form the contiguous attention map." },
    { stepTitle: "Append the new key to K cache", shortTitle: "Update K cache", description: "Insert the newly projected key into the unused lane of the last cache ciphertext; the attention map remains available for Softmax." },
  ]);
  const ts = steps(d, graph, Q, KT, Map, [{}, { ciphertextCount: 4 }, { rotations: 1, additions: 1 },
    { ctCtMultiplications: 3, rotations: 6, additions: 6, relinearizations: 3, rescales: 3, multiplicativeDepth: 1 },
    { ptCtMultiplications: 3, rescales: 3, multiplicativeDepth: 1 },
    { rotations: 2, additions: 2 }, { additions: 1 }]);
  return shell(d, Q, KT, Map, graph, ts, "Interleaved K-cache VMM");
}

function vCacheStrips(d, V, revealAt) {
  // Paper Fig. 6(b): "we make the i-th element of each ciphertext belong to
  // the floor(i/t) column" (t = N/d = 2), i.e. slot s holds a value of column
  // floor(s/2). Ciphertext k holds the value of token (s + 2k) mod 8, so that
  // rotating the softmax by {0,2,4,6} aligns token (s + 2k) with V[(s + 2k) mod 8].
  const result = [];
  for (let k = 0; k < 4; k++) {
    const values = [];
    for (let s = 0; s < 8; s++) {
      const col = Math.floor(s / 2);
      const token = (s + 2 * k) % 8;
      values.push(token < V.length ? valueAt(V, token, col) : 0);
    }
    result.push(strip({ id: `${d.id}-vc-${k}`, label: `V${k}`, values, revealAt, tone: k }));
  }
  return result;
}

function buildSv(d, S, V) {
  assertShape(S, 1, 5, "Softmax(Map)"); assertShape(V, 5, 4, "Vcache");
  const Att = matmul(S, V), result = Att[0], caches = vCacheStrips(d, V, 1);
  const sCt = strip({ id: `${d.id}-s`, label: "Softmax(Map)", values: S[0], revealAt: 1, tone: 0 });
  const sRot = [0,2,4,6].map((r,i) => strip({ id: `${d.id}-sr-${i}`, label: `ρ${r}(S)`, values: rotateLeft(pad(S[0]), r), revealAt: 2, tone: i }));
  const partials = caches.map((_, i) => strip({ id: `${d.id}-p-${i}`, label: `P${i}`,
    values: sRot[i].cells.map((cell, j) => cell.value * caches[i].cells[j].value), revealAt: 3, tone: i, kind: "product" }));
  const sumP = strip({ id: `${d.id}-sump`, label: "ΣP", values: partials[0].cells.map((_, s) => partials.reduce((a, p) => a + p.cells[s].value, 0)),
    revealAt: 4, tone: 2 });
  // Reduce: sumP + rho_1(sumP); the even slots then hold Att. This is the
  // paper's log(N/d) rotation-and-add that folds the even/odd token lanes.
  const folded = partials[0].cells.map((_, s) => {
    const sp = partials.reduce((a, p) => a + p.cells[s].value, 0);
    return sp + partials.reduce((a, p) => a + p.cells[(s + 1) % 8].value, 0);
  });
  const foldedEven = folded.filter((_, s) => s % 2 === 0);
  if (foldedEven.some((v, i) => Math.abs(v - result[i]) > 1e-9)) {
    throw new Error("Cachemir Softmax*V reduce failed verification");
  }
  const interleaved = strip({ id: `${d.id}-att-pre`, label: "Att + useless", values: folded,
    revealAt: 4, tone: 2, kind: "result" });
  const attCt = strip({ id: `${d.id}-att`, label: "Att", values: interleavedInput(result), revealAt: 5, tone: 0, kind: "result" });
  const vnew = DEFAULT_PROJECTION;
  const vpackValues = [0,vnew[0],0,vnew[1],0,vnew[2],0,vnew[3]];
  const vpack = strip({ id: `${d.id}-vnew`, label: "V[5]", values: vpackValues, revealAt: 6, tone: 3 });
  const vMasked = caches.map((_, k) => {
    const insertSlot = (5 - 2 * k + SLOTS) % SLOTS;
    return strip({ id: `${d.id}-vnew-mask-${k}`, label: `Mask${k}⊙V[5]`,
      values: vpackValues.map((value, s) => s === insertSlot ? value : 0), revealAt: 6, tone: k });
  });
  const vUpdated = caches.map((cache, k) => strip({ id: `${d.id}-vupdated-${k}`, label: `V${k}′`,
    values: cache.cells.map((cell, s) => cell.value + vMasked[k].cells[s].value),
    revealAt: 6, tone: k, kind: "result" }));
  const graph = { id: `${d.id}-cachemir-sv`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: "Softmax(Map) × Vcache", caption: "", revealAt: 0,
      rows: [[matrix("A", "Softmax(Map)", S, 0), operator(`${d.id}-times`, "×", "", 0), matrix("B", "Vcache", V, 0, "CT", "column")]] },
    { id: `${d.id}-cache`, title: "V Cache", caption: "", revealAt: 1, rows: [[sCt], caches] },
    { id: `${d.id}-rotate`, title: "① ρ{0,2,4,6}", caption: "", revealAt: 2, rows: [sRot] },
    { id: `${d.id}-multiply`, title: "② ⊙ V Cache", caption: "", revealAt: 3,
      rows: partials.map((p,i) => [sRot[i], operator(`${d.id}-mul-${i}`, "⊙", "", 3), caches[i], operator(`${d.id}-eq-${i}`, "=", "", 3), p]) },
    { id: `${d.id}-reduce`, title: "③ ρ + Σ", caption: "", revealAt: 4, rows: [
      [...partials, operator(`${d.id}-sum`, "Σ", "", 4), sumP],
      [sumP, operator(`${d.id}-fold`, "ρ +", "", 4), interleaved],
    ] },
    { id: `${d.id}-mask`, title: "④ Mask", caption: "", revealAt: 5, rows: [[interleaved,
      operator(`${d.id}-mask-op`, "⊙", "", 5), attCt], [operator(`${d.id}-decode`, "↓", "", 5), matrix("C", "Att", Att, 5)]] },
    { id: `${d.id}-output`, title: "V Cache ⊕ V[5]", caption: "", revealAt: 6, rows: [
      [vpack],
      ...caches.map((cache, k) => [cache, operator(`${d.id}-cache-add-${k}`, "⊕", "", 6), vMasked[k],
        operator(`${d.id}-cache-eq-${k}`, "=", "", 6), vUpdated[k]]),
      [attCt],
    ] },
  ] };
  describeStages(graph, [
    { stepTitle: "Input Softmax scores and encrypted V cache", shortTitle: "Softmax × V", description: "Multiply the new token's Softmax attention map by all cached value vectors." },
    { stepTitle: "Distribute V cache by output column", shortTitle: "V-cache pack", description: "Spread cached value entries across four ciphertexts so every pair of slots belongs to one output column." },
    { stepTitle: "Rotate Softmax scores to cache offsets", shortTitle: "Rotate scores", description: "Rotate the attention-score ciphertext by offsets 0, 2, 4, and 6 to align token weights with the four V-cache ciphertexts." },
    { stepTitle: "Multiply aligned scores and V cache", shortTitle: "Cache products", description: "Multiply each rotated score ciphertext with its matching V-cache ciphertext in parallel." },
    { stepTitle: "Sum products and fold token lanes", shortTitle: "Reduce products", description: "Add the four products, then rotate-and-add across the t = 2 lanes so the even slots contain the context components." },
    { stepTitle: "Mask to interleaved context output", shortTitle: "Mask Att", description: "Clear useless lanes and keep Att in Cachemir's interleaved format for the following output projection." },
    { stepTitle: "Append the new value to V cache", shortTitle: "Update V cache", description: "Mask the new value vector into the correct slot of each cache ciphertext and add it without repacking the existing cache." },
  ]);
  const ts = steps(d, graph, S, V, Att, [{}, { ciphertextCount: 5 }, { rotations: 3 },
    { ctCtMultiplications: 4, relinearizations: 4, rescales: 4, multiplicativeDepth: 1 },
    { rotations: 1, additions: 4 }, { ptCtMultiplications: 1, rescales: 1, multiplicativeDepth: 1 },
    { ptCtMultiplications: 4, additions: 4 }]);
  return shell(d, S, V, Att, graph, ts, "Interleaved V-cache VMM");
}

export function buildCachemirOperationTrace(id, options = {}) {
  const d = getCachemirOperation(id); if (!d) throw new Error(`Unknown Cachemir operation: ${id}`);
  if (options.leftSecret != null && Boolean(options.leftSecret) !== d.leftSecret) throw new Error(`${d.shortLabel} has a fixed left operand type`);
  if (options.rightSecret != null && Boolean(options.rightSecret) !== d.rightSecret) throw new Error(`${d.shortLabel} has a fixed right operand type`);
  if (options.slotsPerCiphertext != null && Number(options.slotsPerCiphertext) !== SLOTS) throw new Error("Cachemir Figure 4-6 toy uses exactly 8 slots");
  const A = options.A || d.A, B = options.B || d.B;
  if (d.kind === "qk-cache") return buildQk(d, A, B);
  if (d.kind === "sv-cache") return buildSv(d, A, B);
  return buildVmm(d, A, B);
}

export function cachemirOperationDefaults(id) {
  const d = getCachemirOperation(id); if (!d) return null;
  return { m: d.A.length, k: d.A[0].length, n: d.kind === "qk-cache" ? d.B.length : d.B[0].length,
    A: clone(d.A), B: clone(d.B), leftSecret: d.leftSecret, rightSecret: d.rightSecret,
    slotsPerCiphertext: SLOTS, strategy: d.strategy };
}

export function parseCachemirOperationInputs(raw, d) {
  if (d.inputMode !== "cache-rows") return parseAndValidateInputs(raw);
  const errors = {}; let A; let B;
  try { A = parseMatrixText(raw.valuesTextA); B = parseMatrixText(raw.valuesTextB); assertShape(A,1,4,d.leftLabel); assertShape(B,5,4,d.rightLabel); }
  catch (error) { errors.values = error.message; }
  if (Number(raw.slots) !== SLOTS) errors.slots = "Cachemir Figure 5 toy uses exactly 8 slots";
  if (Object.keys(errors).length) return { ok: false, errors, inputs: null };
  return { ok: true, errors: {}, inputs: { m:1,k:4,n:5,A,B,
    leftSecret: String(raw.leftType).toLowerCase() === "ct", rightSecret: String(raw.rightType).toLowerCase() === "ct",
    slotsPerCiphertext:SLOTS,strategy:"column",customLayouts:null,valueMode:"manual",seed:raw.seed } };
}
