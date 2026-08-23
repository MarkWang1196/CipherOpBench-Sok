// STIP compact-packing matrix-multiplication traces.
//
// The stages follow the paper algorithms and the released STIP_func code:
// AMCP group rotations (Algorithm 3), DHP complex head pairing (Algorithm 2),
// RIHP score/L2 packing (Algorithms 1/4/5), and RIHP+BSGS attention output
// (Algorithm 6).  The toy keeps m=t=k=2 and S=8 so every physical slot is
// visible while preserving STIP's sequence-major interleaving.

import { matmul, transpose } from "../../core/matrix.js";
import { parseAndValidateInputs } from "../../core/input_parser.js";
import { ctState, makeSteps, matrixArtifact, operator, vectorArtifact } from "../thor/paper_utils.js";

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
const SUB = ["₀", "₁"];
const X24 = Object.freeze([[1, 2, 0, 1], [0, 1, 2, 1]]);
const W42 = Object.freeze([[1, 0], [0, 1], [1, 1], [0, 1]]);
const W24 = Object.freeze([[1, 0, 1, 0], [0, 1, 1, 2]]);
const Q22 = Object.freeze([[1, 2], [0, 1]]);
const K22 = Object.freeze([[1, 0], [1, 2]]);
const G22 = Object.freeze([[0.88, 0.32], [0.61, 0.78]]);
const V22 = Object.freeze([[1, 0], [2, 1]]);

function operation(id, config) {
  return Object.freeze({ id, method: "stip", slotsPerCiphertext: SLOTS, batchCount: BATCHES, ...config });
}
function projection(id, shortLabel, rightLabel, outputLabel, group = "attention", kind = "projection") {
  return operation(id, { shortLabel, group, kind, leftLabel: "X", rightLabel, outputLabel,
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
function interleaved(values) {
  return values.flatMap((value) => [Number(value), Number(value)]);
}
function rotateStride(values, groups = 1) {
  const width = BATCHES * groups;
  const out = [...values];
  for (let s = 0; s < M; s += 1) {
    const base = s * width;
    const block = out.slice(base, base + width);
    out.splice(base, width, ...block.slice(BATCHES), ...block.slice(0, BATCHES));
  }
  return out;
}
function strip({ id, label, values, revealAt, tone = 0, badge = "CT", kind = "ciphertext", labels = null }) {
  const padded = Array.from({ length: SLOTS }, (_, i) => Number(values[i] ?? 0));
  const a = vectorArtifact({ id, label, values: padded, revealAt, badge, kind,
    tones: Array(SLOTS).fill(badge === "CT" ? tone : 4) });
  a.cells.forEach((cell, i) => { cell.label = labels?.[i] ?? (i < values.length ? String(values[i]) : "·"); });
  return a;
}
function batchColumn(values, id, label, revealAt, tone = 0, kind = "ciphertext") {
  const labels = values.flatMap((v) => [`${v}b${SUB[0]}`, `${v}b${SUB[1]}`]);
  return strip({ id, label, values: interleaved(values), labels, revealAt, tone, kind });
}
function complexColumn(real, imag, id, label, revealAt, tone = 0, kind = "ciphertext") {
  const values = interleaved(real);
  const labels = real.flatMap((v, i) => {
    const im = Number(imag[i] || 0); const z = `${v}${im < 0 ? "−" : "+"}${Math.abs(im)}i`;
    return [`${z}b${SUB[0]}`, `${z}b${SUB[1]}`];
  });
  return strip({ id, label, values, labels, revealAt, tone, kind });
}
function matrix(id, label, values, revealAt, badge = "CT", layout = "column", toneOffset = 0) {
  return matrixArtifact({ id, label, values, revealAt, badge, meta: "",
    toneForCell: (row, col) => {
      if (layout === "diagonal") return ((col - row) % 4 + 4) % 4;
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
    title: stage.title, description: "", formula: "", tensors: {
      left: { id: "A", label: d.leftLabel, shape: [left.length, left[0].length], values: left },
      right: { id: "B", label: d.rightLabel, shape: [right.length, right[0].length], values: right },
      ...(i === graph.stages.length - 1 ? { output: { id: "C", label: d.outputLabel, shape: [C.length, C[0].length], values: C } } : {}),
    }, ciphertexts: artifactCiphertexts(stage), delta: deltas[i] || {},
  })));
}
function shell(d, logicalA, logicalB, validationA, validationB, C, graph, steps, flow) {
  return { schemaVersion: "1.0", id: `${d.id}-toy`, title: `ePrint'26 STIP ${d.shortLabel}`,
    parameters: { matrixShape: { m: validationA.length, k: validationA[0].length, n: validationB[0].length },
      slotsPerCiphertext: SLOTS, matmulType: d.rightSecret ? "ct-ct" : "ct-pt", algorithmId: d.id, batchCount: BATCHES },
    operands: { left: { id: "A", label: d.leftLabel, shape: [validationA.length, validationA[0].length], values: validationA, secret: d.leftSecret },
      right: { id: "B", label: d.rightLabel, shape: [validationB.length, validationB[0].length], values: validationB, secret: d.rightSecret } },
    output: { id: "C", label: d.outputLabel, shape: [C.length, C[0].length] },
    plan: { algorithmId: d.id, method: "stip", methodLabel: "ePrint'26 STIP", visualStyle: "diagram-only",
      matmulType: d.rightSecret ? "ct-ct" : "ct-pt", logicalShape: { m: logicalA.length, k: logicalA[0].length, n: logicalB[0].length },
      paddedShape: { m: validationA.length, k: validationA[0].length, n: validationB[0].length }, tileGrid: { m: 1, n: 1 },
      tileShape: { m: C.length, n: C[0].length }, packingStrategyRequested: flow, packingStrategyResolved: flow,
      plannerSource: "STIP Algorithms 1-9 and STIP_func/main.cpp", reductionStrategy: flow,
      displayMode: "diagram-only", batchLabel: "samples", warnings: [] }, graph, steps, finalTensor: C };
}
function outputStage(d, C, artifacts, revealAt) {
  return { id: `${d.id}-output`, title: d.outputLabel, caption: "", revealAt,
    rows: [[...artifacts], [operator(`${d.id}-down`, "↓", "", revealAt), matrix("C", d.outputLabel, C, revealAt)]] };
}

function amcpPacked(A, id, revealAt) {
  const c = cols(A), groups = [];
  for (let g = 0; g < c.length; g += 2) {
    const values = [];
    for (let s = 0; s < M; s += 1) for (let feature = 0; feature < 2; feature += 1) {
      const v = Number(c[g + feature]?.[s] ?? 0); values.push(v, v);
    }
    const labels = [];
    for (let s = 0; s < M; s += 1) for (let feature = 0; feature < 2; feature += 1) {
      const v = Number(c[g + feature]?.[s] ?? 0); labels.push(`${v}b₀x${g + feature}`, `${v}b₁x${g + feature}`);
    }
    groups.push(strip({ id: `${id}-${g / 2}`, label: `C${g / 2}`, values, labels, revealAt, tone: g / 2 }));
  }
  return groups;
}
function buildProjection(d, A, W) {
  assertRect(A, 2, 4, d.leftLabel); assertRect(W, 4, 2, d.rightLabel);
  const C = matmul(A, W), packed = amcpPacked(A, `${d.id}-amcp`, 1);
  const rotations = packed.map((a, i) => strip({ id: `${d.id}-rot-${i}`, label: `ρ₂(C${i})`,
    values: rotateStride(a.cells.map((c) => c.value), 2), revealAt: 2, tone: i }));
  const hybrid = complexColumn(cols(C)[0], cols(C)[1], `${d.id}-hybrid`, "Y⁰+iY¹", 3, 1, "product");
  const outputs = cols(C).map((v, i) => batchColumn(v, `${d.id}-out-${i}`, `${d.outputLabel}${SUB[i]}`, 4, i, "result"));
  const aggregate = d.kind === "output" ? { id: `${d.id}-aggregate`, title: "Σ ρ₂ AMCP", caption: "", revealAt: 4,
    rows: [[hybrid, operator(`${d.id}-agg-op`, "Σρ₂", "", 4), ...outputs]] } :
    { id: `${d.id}-unpack`, title: "Conj", caption: "", revealAt: 4,
      rows: [[hybrid, operator(`${d.id}-conj-op`, "Conj", "", 4), ...outputs]] };
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: `${d.leftLabel} × ${d.rightLabel}`, caption: "", revealAt: 0,
      rows: [[matrix("A", d.leftLabel, A, 0, "CT", "column"), operator(`${d.id}-times`, "×", "", 0), matrix("B", d.rightLabel, W, 0, "PT", "column")]] },
    { id: `${d.id}-amcp-stage`, title: "AMCP", caption: "", revealAt: 1, rows: [packed] },
    { id: `${d.id}-rotate-stage`, title: "RotL₂ · u + RotR₂ · v", caption: "", revealAt: 2, rows: [rotations] },
    { id: `${d.id}-dhp-stage`, title: "DHP ⊙ Σ", caption: "", revealAt: 3,
      rows: [[...rotations, operator(`${d.id}-weights`, "⊙ W⁰+iW¹", "", 3), hybrid]] },
    aggregate, outputStage(d, C, outputs, 5),
  ] };
  const steps = graphSteps(d, graph, A, W, C, [{}, { ciphertextCount: packed.length }, { rotations: packed.length * 2, ptCtMultiplications: packed.length * 2 },
    { ptCtMultiplications: 8, additions: 6, rescales: 2, multiplicativeDepth: 1 }, { rotations: d.kind === "output" ? 1 : 0 }, {}]);
  return shell(d, A, W, A, W, C, graph, steps, "AMCP + DHP column packing");
}

function scoreDiagonals(C) { return [[C[0][0], C[1][1]], [C[0][1], C[1][0]]]; }
function buildRihp(d, Q, K) {
  assertRect(Q, 2, 2, d.leftLabel); assertRect(K, 2, 2, d.rightLabel);
  const KT = transpose(K), C = matmul(Q, KT), q = cols(Q), k = cols(K), diags = scoreDiagonals(C);
  const qct = q.map((v, i) => batchColumn(v, `${d.id}-q-${i}`, `q${i}`, 1, i));
  const kct = k.map((v, i) => batchColumn(v, `${d.id}-k-${i}`, `k${i}`, 1, i + 2));
  const kh = k.map((v, i) => complexColumn(v, [...v].reverse(), `${d.id}-kh-${i}`, `k̃${i}`, 2, i + 2));
  const hybrid = complexColumn(diags[0], diags[1], `${d.id}-hybrid`, "C₀=P₀+iP₁", 3, 1, "product");
  const unpacked = diags.map((v, i) => batchColumn(v, `${d.id}-p-${i}`, `P${i}`, 4, i, "result"));
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: "Q × Kᵀ", caption: "", revealAt: 0,
      rows: [[matrix("A", "Q", Q, 0, "CT", "column"), operator(`${d.id}-times`, "×", "", 0), matrix("K", "K", K, 0, "CT", "column", 2)]] },
    { id: `${d.id}-column`, title: "Col", caption: "", revealAt: 1, rows: [qct, kct] },
    { id: `${d.id}-hybrid-k`, title: "K + i·RotL₁(K)", caption: "", revealAt: 2, rows: [[...kct, operator(`${d.id}-kto`, "→", "", 2), ...kh]] },
    { id: `${d.id}-rihp`, title: "RIHP ⊙ Σ", caption: "", revealAt: 3,
      rows: [[...qct, operator(`${d.id}-mul`, "⊙", "", 3), ...kh, operator(`${d.id}-sum`, "Σ", "", 3), hybrid]] },
    { id: `${d.id}-conj`, title: "½(C+Conj C) · −½i(C−Conj C)", caption: "", revealAt: 4,
      rows: [[hybrid, operator(`${d.id}-split`, "Conj", "", 4), ...unpacked]] },
    outputStage(d, C, unpacked, 5),
  ] };
  const steps = graphSteps(d, graph, Q, KT, C, [{}, { ciphertextCount: 4 }, { rotations: 2, ptCtMultiplications: 2 },
    { ctCtMultiplications: 2, additions: 1, relinearizations: 1, rescales: 1, multiplicativeDepth: 1 }, { additions: 2 }, {}]);
  return shell(d, Q, K, Q, KT, C, graph, steps, "RIHP column × column → diagonal");
}

function squaredDistances(Q, K) {
  return Q.map((q) => K.map((k) => q.reduce((s, v, j) => s + (Number(v) - Number(k[j])) ** 2, 0)));
}
function l2Validation(Q, K) {
  const left = Q.map((q) => [q.reduce((s, v) => s + Number(v) ** 2, 0), ...q.map((v) => -2 * Number(v)), 1]);
  const right = [K.map(() => 1), ...Array.from({ length: K[0].length }, (_, j) => K.map((k) => Number(k[j]))),
    K.map((k) => k.reduce((s, v) => s + Number(v) ** 2, 0))];
  return { left, right };
}
function buildL2(d, Q, K) {
  assertRect(Q, 2, 2, d.leftLabel); assertRect(K, 2, 2, d.rightLabel);
  const C = squaredDistances(Q, K), validation = l2Validation(Q, K), q = cols(Q), k = cols(K);
  const qh = complexColumn(q[0], q[1], `${d.id}-qh`, "q₀+iq₁", 1, 0);
  const kh = complexColumn(k[0], k[1].map((v) => -v), `${d.id}-kh`, "k₀−ik₁", 1, 2);
  const sq = batchColumn(Q.map((row) => row.reduce((s, v) => s + v * v, 0)), `${d.id}-sq`, "S_Q", 2, 0, "product");
  const sk = batchColumn(K.map((row) => row.reduce((s, v) => s + v * v, 0)), `${d.id}-sk`, "S_K", 2, 2, "product");
  const diags = scoreDiagonals(C).map((v, i) => batchColumn(v, `${d.id}-d-${i}`, `D²${i}`, 4, i, "result"));
  const cross = scoreDiagonals(matmul(Q, transpose(K))).map((v, i) => batchColumn(v, `${d.id}-c-${i}`, `C${i}`, 3, i + 1, "product"));
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: "Q · K", caption: "", revealAt: 0,
      rows: [[matrix("A", "Q", Q, 0, "CT", "column"), operator(`${d.id}-dot`, "·", "", 0), matrix("K", "K", K, 0, "CT", "column", 2)]] },
    { id: `${d.id}-pair`, title: "RIHP", caption: "", revealAt: 1, rows: [[qh, kh]] },
    { id: `${d.id}-norm`, title: "z ⊙ Conj(z)", caption: "", revealAt: 2, rows: [[qh, operator(`${d.id}-qnorm`, "⊙Conj", "", 2), sq], [kh, operator(`${d.id}-knorm`, "⊙Conj", "", 2), sk]] },
    { id: `${d.id}-cross`, title: "Σ q̂ ⊙ Rot(k̂)", caption: "", revealAt: 3,
      rows: cross.map((c, i) => [qh, operator(`${d.id}-mul-${i}`, i ? "⊙RotL₂" : "⊙", "", 3), kh, operator(`${d.id}-eq-${i}`, "=", "", 3), c]) },
    { id: `${d.id}-combine`, title: "S_Q + Rot(S_K) − 2C", caption: "", revealAt: 4,
      rows: diags.map((out, i) => [sq, operator(`${d.id}-plus-${i}`, "+", "", 4), sk, operator(`${d.id}-minus-${i}`, "−2", "", 4), cross[i], operator(`${d.id}-eqd-${i}`, "=", "", 4), out]) },
    outputStage(d, C, diags, 5),
  ] };
  const steps = graphSteps(d, graph, validation.left, validation.right, C, [{}, { ciphertextCount: 2 },
    { ctCtMultiplications: 2, relinearizations: 2, rescales: 2, multiplicativeDepth: 1 },
    { rotations: 1, ctCtMultiplications: 2, additions: 1, relinearizations: 1, rescales: 1 }, { additions: 6 }, {}]);
  return shell(d, Q, K, validation.left, validation.right, C, graph, steps, "RIHP squared L2 → diagonal");
}

function buildContext(d, G, V) {
  assertRect(G, 2, 2, d.leftLabel); assertRect(V, 2, 2, d.rightLabel);
  const C = matmul(G, V), diagonals = scoreDiagonals(G).map((v, i) => batchColumn(v, `${d.id}-g-${i}`, `G${i}`, 1, i));
  const vc = cols(V), vpair = complexColumn(vc[0], vc[1], `${d.id}-vpair`, "v₀+iv₁", 1, 2);
  const babies = [vpair, complexColumn([...vc[0]].reverse(), [...vc[1]].reverse(), `${d.id}-baby`, "RotL₂(v̂)", 2, 2)];
  const giants = cols(C).map((v, i) => batchColumn(v, `${d.id}-out-${i}`, `${d.outputLabel}${SUB[i]}`, 4, i, "result"));
  const hybrid = complexColumn(cols(C)[0], cols(C)[1], `${d.id}-hybrid`, "H₀+iH₁", 3, 1, "product");
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: `${d.leftLabel} × V`, caption: "", revealAt: 0,
      rows: [[matrix("A", d.leftLabel, G, 0, "CT", "diagonal"), operator(`${d.id}-times`, "×", "", 0), matrix("B", "V", V, 0, "CT", "column", 2)]] },
    { id: `${d.id}-pack`, title: "Diag + RIHP", caption: "", revealAt: 1, rows: [[...diagonals, vpair]] },
    { id: `${d.id}-baby`, title: "BSGS β", caption: "", revealAt: 2, rows: [babies] },
    { id: `${d.id}-bsgs`, title: "Σ_g Rot(Σ_b G ⊙ β)", caption: "", revealAt: 3,
      rows: [[...diagonals, operator(`${d.id}-bsgs-op`, "⊙Σρ", "", 3), ...babies, operator(`${d.id}-bsgs-eq`, "=", "", 3), hybrid]] },
    { id: `${d.id}-conj`, title: "Conj", caption: "", revealAt: 4,
      rows: [[hybrid, operator(`${d.id}-conj-op`, "Conj", "", 4), ...giants]] },
    outputStage(d, C, giants, 5),
  ] };
  const steps = graphSteps(d, graph, G, V, C, [{}, { ciphertextCount: 3 }, { rotations: 1 },
    { ctCtMultiplications: 4, additions: 3, rotations: 1, relinearizations: 2, rescales: 2, multiplicativeDepth: 1 }, { additions: 2 }, {}]);
  return shell(d, G, V, G, V, C, graph, steps, "diagonal × RIHP columns with BSGS");
}

function buildPcmm(d, A, W) {
  assertRect(A, d.A.length, d.A[0].length, d.leftLabel); assertRect(W, d.B.length, d.B[0].length, d.rightLabel);
  const C = matmul(A, W), ac = cols(A).map((v, i) => batchColumn(v, `${d.id}-a-${i}`, `x${i}`, 1, i % 4));
  const products = cols(C).map((_, j) => ac.map((a, i) => batchColumn(a.cells.filter((_, x) => x < 4).map((c) => c.value * W[i][j]).filter((_, x) => x % 2 === 0),
    `${d.id}-p-${j}-${i}`, `p${j}${i}`, 2, i % 4, "product")));
  const outputs = cols(C).map((v, j) => batchColumn(v, `${d.id}-out-${j}`, `${d.outputLabel}${j}`, 3, j % 4, "result"));
  const rows = [];
  products.forEach((set, j) => set.forEach((p, i) => rows.push([ac[i], operator(`${d.id}-mul-${j}-${i}`, `⊙${W[i][j]}`, "", 2), p])));
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: `${d.leftLabel} × ${d.rightLabel}`, caption: "", revealAt: 0,
      rows: [[matrix("A", d.leftLabel, A, 0, "CT", "column"), operator(`${d.id}-times`, "×", "", 0), matrix("B", d.rightLabel, W, 0, "PT", "column")]] },
    { id: `${d.id}-col`, title: "Col", caption: "", revealAt: 1, rows: [ac] },
    { id: `${d.id}-multiply`, title: "⊙ PT", caption: "", revealAt: 2, rows },
    { id: `${d.id}-sum`, title: "Σ", caption: "", revealAt: 3, rows: outputs.map((o, j) => [...products[j], operator(`${d.id}-sum-${j}`, "Σ", "", 3), o]) },
    outputStage(d, C, outputs, 4),
  ] };
  const mults = A[0].length * W[0].length;
  const steps = graphSteps(d, graph, A, W, C, [{}, { ciphertextCount: ac.length },
    { ptCtMultiplications: mults, rescales: mults, multiplicativeDepth: 1 }, { additions: (A[0].length - 1) * W[0].length }, {}]);
  return shell(d, A, W, A, W, C, graph, steps, "column-packed CPMM");
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
  return buildProjection(d, A, B);
}
export function stipOperationDefaults(id) {
  const d = getStipOperation(id); if (!d) return null;
  return { m: d.A.length, k: d.A[0].length, n: d.B[0].length, A: clone(d.A), B: clone(d.B),
    leftSecret: d.leftSecret, rightSecret: d.rightSecret, slotsPerCiphertext: SLOTS, strategy: "column" };
}
export function parseStipOperationInputs(raw) { return parseAndValidateInputs(raw); }
