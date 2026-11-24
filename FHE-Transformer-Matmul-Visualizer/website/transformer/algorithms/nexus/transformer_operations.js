// NEXUS matrix-multiplication traces.
// Paper/code correspondence:
//   Figure 4 / Algorithm 1 -> polynomial ciphertext decompression;
//   Figure 5 / MMEvaluator::matrix_mul -> offline-online batched CT-PT MM;
//   Appendix C / Algorithm 5 -> column-packed CT-CT QK^T;
//   Section III-C -> row-wise encrypted multiplication after attention.

import { matmul, transpose } from "../../core/matrix.js";
import { parseAndValidateInputs } from "../../core/input_parser.js";
import { addVectors, ctState, makeSteps, matrixArtifact, mod, multiplyVectors, operator, rotateLeft, vectorArtifact } from "../thor/paper_utils.js";

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
const DEGREE = 2 * SLOTS;
const BATCHES = 2;
const A24 = Object.freeze([[1, 3, 0, 2], [4, 0, 2, 1]]);
const W42 = Object.freeze([[1, 0], [0, 2], [3, 1], [2, 1]]);
const A22 = Object.freeze([[2, 1], [1, 0]]);
const W22 = Object.freeze([[2, 4], [3, 1]]);
const W24 = Object.freeze([[1, 0, 2, 3], [2, 1, 3, 0]]);
const K22 = Object.freeze([[1, 4], [3, 2]]);
const S22 = Object.freeze([[0.75, 0.25], [0.375, 0.625]]);
const V22 = Object.freeze([[2, 1], [1, 3]]);

function operation(id, config) {
  return Object.freeze({ id, method: "nexus", slotsPerCiphertext: SLOTS, batchCount: BATCHES, ...config });
}
function linear(id, shortLabel, leftLabel, rightLabel, outputLabel, group, A, B, kind = "offline-online") {
  return operation(id, { shortLabel, group, kind, leftLabel, rightLabel, outputLabel,
    leftSecret: kind === "hidden-linear", rightSecret: kind !== "hidden-linear", A, B, strategy: "column" });
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
  [NEXUS_OPERATION_IDS.OUTPUT]: linear(NEXUS_OPERATION_IDS.OUTPUT, "HWo", "H", "Wo", "O", "attention", A24, W42, "hidden-linear"),
  [NEXUS_OPERATION_IDS.FC1]: linear(NEXUS_OPERATION_IDS.FC1, "FC1", "X", "W1", "F1", "ffn", A22, W24, "hidden-linear"),
  [NEXUS_OPERATION_IDS.FC2]: linear(NEXUS_OPERATION_IDS.FC2, "FC2", "GELU(F1)", "W2", "F2", "ffn", A24, W42, "hidden-linear"),
});

export function listNexusOperations(group = null) { return Object.values(NEXUS_OPERATIONS).filter((x) => !group || x.group === group); }
export function getNexusOperation(id) { return NEXUS_OPERATIONS[id] || null; }
function clone(A) { return A.map((r) => [...r]); }
function cols(A) { return Array.from({ length: A[0].length }, (_, j) => A.map((r) => Number(r[j]))); }
function assertRect(A, rows, columns, label) {
  if (!Array.isArray(A) || A.length !== rows || A.some((r) => !Array.isArray(r) || r.length !== columns)) throw new Error(`${label} must be ${rows}×${columns}`);
}
function strip({ id, label, values, revealAt, tone = 0, badge = "CT", kind = "ciphertext", slotsPerSampleBlock = 0 }) {
  const padded = Array.from({ length: SLOTS }, (_, i) => Number(values[i] ?? 0));
  const a = vectorArtifact({ id, label, values: padded, revealAt, badge, kind, tones: Array(SLOTS).fill(badge === "CT" ? tone : 4) });
  a.cells.forEach((cell, i) => {
    if (i >= values.length) cell.label = "·";
    else if (slotsPerSampleBlock) cell.sampleLane = Math.floor(i / slotsPerSampleBlock) % BATCHES;
  });
  return a;
}
const valuesOf = (artifact) => artifact.cells.map((cell) => cell.value);

// Section III-C concatenates sample blocks vertically in each packed column.
function batchColumn(values, id, label, revealAt, tone = 0, kind = "ciphertext", badge = "CT") {
  return strip({ id, label, values: [...values, ...values], slotsPerSampleBlock: values.length, revealAt, tone, kind, badge });
}

// In the score demo, repeating [row0 batch0, row0 batch1, row1 batch0,
// row1 batch1] across every slot implements a cyclic key shift per sample.
function periodicColumn(values, id, label, revealAt, tone = 0, kind = "ciphertext") {
  const slots = Array.from({ length: SLOTS }, (_, i) => values[Math.floor(i / BATCHES) % values.length]);
  return strip({ id, label, values: slots, slotsPerSampleBlock: 1, revealAt, tone, kind });
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
    if (a.badge !== "CT" || !a.cells || a.coefficients || seen.has(a.id)) continue;
    seen.add(a.id); out.push(ctState(a.id, a.label, a.cells.map((c) => c.value), SLOTS, true,
      a.type === "product" ? "product" : a.type === "result" ? "result" : "tensor"));
  }
  return out;
}
function graphSteps(d, graph, A, B, C, deltas) {
  return makeSteps(d.id, graph.stages.map((stage, i) => ({ phase: i === 0 ? "reference" : i === graph.stages.length - 1 ? "output" : "multiply",
    title: stage.stepTitle, description: stage.description, formula: "", tensors: {
      left: { id: "A", label: d.leftLabel, shape: [A.length, A[0].length], values: A },
      right: { id: "B", label: d.kind === "ccmm" ? "Kᵀ" : d.rightLabel, shape: [B.length, B[0].length], values: B },
      ...(i === graph.stages.length - 1 ? { output: { id: "C", label: d.outputLabel, shape: [C.length, C[0].length], values: C } } : {}),
    }, ciphertexts: artifactCiphertexts(stage), delta: deltas[i] || {} }))) ;
}
function shell(d, A, B, C, graph, steps, flow) {
  const matmulType = d.rightSecret ? (d.leftSecret ? "ct-ct" : "pt-ct") : "ct-pt";
  return { schemaVersion: "1.0", id: `${d.id}-toy`, title: `NDSS'25 NEXUS ${d.shortLabel}`,
    parameters: { matrixShape: { m: A.length, k: A[0].length, n: B[0].length }, slotsPerCiphertext: SLOTS,
      matmulType, algorithmId: d.id, batchCount: BATCHES },
    operands: { left: { id: "A", label: d.leftLabel, shape: [A.length, A[0].length], values: A, secret: d.leftSecret },
      right: { id: "B", label: d.kind === "ccmm" ? "Kᵀ" : d.rightLabel, shape: [B.length, B[0].length], values: B, secret: d.rightSecret } },
    output: { id: "C", label: d.outputLabel, shape: [C.length, C[0].length] },
    plan: { algorithmId: d.id, method: "nexus", methodLabel: "NDSS'25 NEXUS", visualStyle: "diagram-only",
      matmulType, logicalShape: { m: A.length, k: A[0].length, n: B[0].length },
      ...(d.kind === "decompress" ? { formula: "SecComp → SecDecomp(W)", inputShapes: [B.length + "×" + B[0].length], operandType: "Coefficient ↔ SIMD" } : {}),
      paddedShape: { m: A.length, k: A[0].length, n: B[0].length }, tileGrid: { m: 1, n: 1 }, tileShape: { m: C.length, n: C[0].length },
      packingStrategyRequested: flow, packingStrategyResolved: flow, plannerSource: "NEXUS Figures 3-5, Algorithms 1/5, and MMEvaluator::matrix_mul",
      reductionStrategy: flow, displayMode: "diagram-only", batchLabel: "samples", warnings: [] }, graph, steps, finalTensor: C };
}
function outputStage(d, decoded, artifacts, revealAt, description) {
  return { id: `${d.id}-output`, title: d.outputLabel, stepTitle: `Check the encrypted ${d.outputLabel} output`,
    shortTitle: "Output", description, revealAt,
    rows: [artifacts, [operator(`${d.id}-down`, "↓", "", revealAt), matrix("C", d.outputLabel, decoded, revealAt)]] };
}

function polynomialArtifact(coefficients, id, label, width, revealAt) {
  const values = coefficients.slice(0, width);
  const artifact = vectorArtifact({ id, label, values, revealAt, badge: "CT",
    meta: `Coefficients mod x^${DEGREE} + 1`, tones: values.map(() => 0) });
  artifact.cells.forEach((cell, i) => { cell.label = `${cell.value}x^${i}`; });
  return { ...artifact, coefficients: true };
}

// Monomials wrap with a sign in the CKKS polynomial ring x^N' = -1.
function remapExponents(coefficients, exponent) {
  const values = Array(DEGREE).fill(0);
  coefficients.forEach((coefficient, i) => {
    const power = mod(exponent(i), 2 * DEGREE);
    values[power % DEGREE] += power < DEGREE ? coefficient : -coefficient;
  });
  return values;
}

function splitPolynomial(coefficients, offset) {
  const shifted = remapExponents(coefficients, (i) => i - offset);
  return [coefficients, shifted].map((branch) =>
    addVectors([branch, remapExponents(branch, (i) => i * (DEGREE / offset + 1))]));
}

function decompress(coefficients) {
  let branches = [coefficients];
  for (let offset = 1; offset < DEGREE; offset *= 2) {
    const splits = branches.map((branch) => splitPolynomial(branch, offset));
    branches = splits.map(([even]) => even).concat(splits.map(([, odd]) => odd));
  }
  return branches.map((branch) => branch[0] / DEGREE);
}

function buildOfflineOnline(d, A, W, decompressionOnly = false) {
  assertRect(A, d.A.length, d.A[0].length, d.leftLabel); assertRect(W, d.B.length, d.B[0].length, d.rightLabel);
  const C = matmul(A, W);
  const flat = transpose(W).flat().map(Number);
  const coefficients = Array.from({ length: DEGREE }, (_, i) => flat[i] ?? 0);
  const [evenCoefficients, oddCoefficients] = splitPolynomial(coefficients, 1);
  const poly = polynomialArtifact(coefficients, `${d.id}-poly`, "Eₛ(p(x))", flat.length, 1);
  const even = polynomialArtifact(evenCoefficients, `${d.id}-even`, "p + Subs₁₇(p)", flat.length, 2);
  const odd = polynomialArtifact(oddCoefficients, `${d.id}-odd`, "p′ + Subs₁₇(p′), p′ = x⁻¹p", flat.length, 2);
  const decoded = decompress(coefficients);
  const weights = decoded.slice(0, flat.length).map((value, i) => scalarCt(value, `${d.id}-w-${i}`,
    `w${i % W.length},${Math.floor(i / W.length)}`, 3, Math.floor(i / W.length) % 4));
  const stages = [
    { id: `${d.id}-input`, title: decompressionOnly ? d.rightLabel : `${d.leftLabel} × ${d.rightLabel}`,
      stepTitle: decompressionOnly ? "Select the weights to compress" : "Set up the offline–online projection", shortTitle: "Input",
      description: decompressionOnly ? "Start with a weight matrix whose entries will be sent in a single encrypted polynomial."
        : "The client holds the input and the server holds the weights. The offline phase prepares an encrypted correction for the online product.", revealAt: 0,
      rows: [decompressionOnly
        ? [matrix("B", d.rightLabel, W, 0)]
        : [matrix("A", d.leftLabel, A, 0, "PT", "column"), operator(`${d.id}-times`, "×", "", 0), matrix("B", d.rightLabel, W, 0, "CT", "column")]] },
    { id: `${d.id}-compress`, title: "Compress", stepTitle: "Encode weights as polynomial coefficients", shortTitle: "Compress",
      description: "Flatten W column by column, encode its entries as coefficients, and encrypt the polynomial with the server’s key.", revealAt: 1,
      rows: [[matrix(`${d.id}-wm`, d.rightLabel, W, 1), operator(`${d.id}-pack`, "→", "", 1), poly]] },
    { id: `${d.id}-subs`, title: "⊞ Subs · x⁻²ⁱ", stepTitle: "Split the even and odd coefficients", shortTitle: "Split coefficients",
      description: "Add the substitution to keep the even powers of p. First multiply by x⁻¹ on the second branch to expose the odd coefficients.", revealAt: 2,
      rows: [[poly, operator(`${d.id}-branch`, "↙ ↘", "", 2), even, odd]] },
    { id: `${d.id}-decompress`, title: "Decompress", stepTitle: "Extract repeated SIMD weights", shortTitle: "Decompress",
      description: "Repeat the split and divide by the polynomial degree. Each resulting constant encrypts a SIMD vector with one weight repeated.", revealAt: 3,
      rows: [[even, odd, operator(`${d.id}-tree`, "→", "", 3), ...weights]] },
  ];
  if (decompressionOnly) {
    stages.push({ id: `${d.id}-simd`, title: "ENC([w,…,w])", stepTitle: "Return the broadcast weight ciphertexts", shortTitle: "SIMD weights",
      description: "Keep the non-padding coefficients in weight order. Each broadcast is ready for slotwise multiplication with an input column.", revealAt: 4, rows: [weights] });
  } else {
    const U = A.map((row, i) => row.map((_v, j) => (i + j) % 2));
    const masked = A.map((row, i) => row.map((v, j) => Number(v) - U[i][j]));
    const extractedW = W.map((row, i) => row.map((_, j) => weights[j * W.length + i].cells[0].value));
    const V = matmul(U, extractedW), maskedProduct = matmul(masked, W);
    const corrections = cols(V).map((v, j) => batchColumn(v, `${d.id}-v-${j}`, `ENC꜀(UW${j})`, 4, j));
    const online = cols(maskedProduct).map((v, j) => batchColumn(v, `${d.id}-on-${j}`, `(A−U)W${j}`, 5, j, "product", "PT"));
    const outputs = corrections.map((correction, j) => batchColumn(
      addVectors([valuesOf(correction), valuesOf(online[j])]).slice(0, A.length),
      `${d.id}-out-${j}`, `${d.outputLabel}${j}`, 5, j, "result"));
    stages.push({ id: `${d.id}-offline`, title: "Offline U·W", stepTitle: "Precompute the encrypted correction UW", shortTitle: "Offline UW",
      description: "The client multiplies a toy mask U by the expanded weights. Commutable encryption lets the server obtain UW under the client’s key.", revealAt: 4,
      rows: [[matrix(`${d.id}-u`, "U", U, 4, "PT", "column"), operator(`${d.id}-uw`, "×", "", 4), ...weights, operator(`${d.id}-veq`, "=", "", 4), ...corrections]] });
    stages.push({ id: `${d.id}-online`, title: "Online (A−U)W ⊞ ENC(UW)", stepTitle: "Add the online masked product", shortTitle: "Online correction",
      description: "The server computes (A−U)W in plaintext and adds the encrypted correction to recover encrypted AW.", revealAt: 5,
      rows: [[matrix(`${d.id}-masked`, "A−U", masked, 5, "PT"), operator(`${d.id}-m`, "×", "", 5), matrix("W-plain", d.rightLabel, W, 5, "PT")],
        ...outputs.map((o, j) => [online[j], operator(`${d.id}-add-${j}`, "⊞", "", 5), corrections[j], operator(`${d.id}-eq-${j}`, "=", "", 5), o])] });
    stages.push(outputStage(d, transpose(outputs.map((o) => valuesOf(o).slice(0, A.length))), outputs, 6,
      "Read the first sample block of each encrypted output column and compare it with the matrix product."));
  }
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages };
  const steps = graphSteps(d, graph, A, W, C, [{}, { ciphertextCount: 1 }, { additions: 2 },
    { additions: 2 * DEGREE - 4, ptCtMultiplications: flat.length }, decompressionOnly ? {} : {
      ptCtMultiplications: flat.length, additions: (A[0].length - 1) * W[0].length, multiplicativeDepth: 1,
    }, { additions: W[0].length }, {}]);
  return shell(d, A, W, C, graph, steps, decompressionOnly ? "compressed polynomial → repeated SIMD weights" : "offline-online batched compressed-weight MM");
}

function buildCcmm(d, Q, K) {
  assertRect(Q, 2, 2, d.leftLabel); assertRect(K, 2, 2, d.rightLabel);
  const KT = transpose(K), C = matmul(Q, KT), q = cols(Q), k = cols(K);
  const qcts = q.map((v, i) => periodicColumn(v, `${d.id}-q-${i}`, `q${i}`, 1, i));
  const kcts = k.map((v, i) => periodicColumn(v, `${d.id}-k-${i}`, `k${i}`, 1, i + 2));
  const shifted = kcts.map((a, j) => strip({ id: `${d.id}-kr-${j}`, label: `RotL₂(k${j})`,
    values: rotateLeft(valuesOf(a), BATCHES), slotsPerSampleBlock: 1,
    revealAt: 2, tone: j + 2 }));
  const rotations = [kcts, shifted];
  const diagonals = rotations.map((keys, i) => {
    const values = addVectors(keys.map((key, j) => multiplyVectors(valuesOf(qcts[j]), valuesOf(key))));
    return strip({ id: `${d.id}-p-${i}`, label: `P${i}`, values,
      slotsPerSampleBlock: 1, revealAt: 3, tone: i, kind: "result" });
  });
  const decoded = C.map((row, i) => row.map((_, j) => diagonals[mod(j - i, Q.length)].cells[i * BATCHES].value));
  const rows = diagonals.map((out, r) => [qcts[0], operator(`${d.id}-m-${r}-0`, "⊠", "", 3), rotations[r][0],
    operator(`${d.id}-plus-${r}`, "⊞", "", 3), qcts[1], operator(`${d.id}-m-${r}-1`, "⊠", "", 3), rotations[r][1], operator(`${d.id}-eq-${r}`, "=", "", 3), out]);
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: "Q × Kᵀ", stepTitle: "Input encrypted queries and keys", shortTitle: "Q × Kᵀ",
      description: "Compute QKᵀ from the corresponding feature columns of Q and K.", revealAt: 0,
      rows: [[matrix("A", "Q", Q, 0, "CT", "column"), operator(`${d.id}-times`, "×", "", 0), matrix("B", "Kᵀ", KT, 0, "CT", "row", 2)]] },
    { id: `${d.id}-col`, title: "Column pack", stepTitle: "Pack matching Q and K columns", shortTitle: "Pack columns",
      description: "In this batched demo, interleave the samples by row and repeat the pattern so key rotations wrap correctly.", revealAt: 1, rows: [qcts, kcts] },
    { id: `${d.id}-rotate`, title: "RotL₂(K)", stepTitle: "Rotate each K column by one row", shortTitle: "Rotate K",
      description: "A two-slot rotation advances the key row in both sample lanes while leaving the feature index unchanged.", revealAt: 2,
      rows: shifted.map((col, j) => [kcts[j], operator(`${col.id}-op`, "RotL₂", "", 2), col]) },
    { id: `${d.id}-multiply`, title: "Σ_j q_j ⊠ RotL(k_j)", stepTitle: "Sum feature products into diagonals", shortTitle: "Score diagonals",
      description: "For each shift, multiply corresponding query and key columns and add the products across features.", revealAt: 3, rows },
    outputStage(d, decoded, diagonals, 4, "Read one cyclic diagonal per ciphertext to verify the score matrix for the first sample."),
  ] };
  const steps = graphSteps(d, graph, Q, KT, C, [{}, { ciphertextCount: 4 }, { rotations: 2 },
    { ctCtMultiplications: 4, additions: 2, relinearizations: 2, rescales: 2, multiplicativeDepth: 1 }, {}]);
  return shell(d, Q, KT, C, graph, steps, "column-packed Algorithm 5 → diagonal-packed");
}

function buildContext(d, S, V) {
  assertRect(S, 2, 2, d.leftLabel); assertRect(V, 2, 2, d.rightLabel);
  const packDot = (values) => values.flatMap((v) => [v, v]);
  const C = matmul(S, V), rows = S.map((v, i) => strip({ id: `${d.id}-s-${i}`, label: `S${i}`,
    values: packDot(v), slotsPerSampleBlock: 1, revealAt: 1, tone: i }));
  const vc = cols(V).map((v, j) => strip({ id: `${d.id}-v-${j}`, label: `V${j}`,
    values: packDot(v), slotsPerSampleBlock: 1, revealAt: 1, tone: j + 2 }));
  const width = S[0].length * BATCHES;
  const mask = strip({ id: `${d.id}-mask`, label: "Keep active block", values: Array(width).fill(1), revealAt: 3, badge: "PT" });
  const products = [], sums = [], foldRows = [];
  for (const [i, row] of rows.entries()) {
    for (const [j, col] of vc.entries()) {
      const id = `${d.id}-${i}-${j}`;
      const tone = (i + j) % 4;
      const values = multiplyVectors(valuesOf(row), valuesOf(col));
      const product = strip({ id: `${id}-product`, label: `p${i}${j}`, values: values.slice(0, width),
        slotsPerSampleBlock: 1, revealAt: 2, tone, kind: "product" });
      const repeated = addVectors([values, rotateLeft(values, -width)]);
      const repeat = strip({ id: `${id}-repeat`, label: "Duplicated lanes", values: repeated, slotsPerSampleBlock: 1, revealAt: 3, tone });
      const folded = addVectors([repeated, rotateLeft(repeated, BATCHES)]);
      const retained = multiplyVectors(folded, valuesOf(mask)).slice(0, width);
      const sum = strip({ id: `${id}-sum`, label: `${d.outputLabel}${i}${j}`, values: retained,
        slotsPerSampleBlock: 1, revealAt: 3, tone, kind: "result" });
      products.push([row, operator(`${id}-mul`, "⊠", "", 2), col, operator(`${id}-p-eq`, "=", "", 2), product]);
      foldRows.push([product, operator(`${id}-duplicate`, "⊞ RotR₄", "", 3), repeat,
        operator(`${id}-fold`, "⊞ RotL₂", "", 3), operator(`${id}-keep`, "⊙", "", 3), mask,
        operator(`${id}-sum-eq`, "=", "", 3), sum]);
      sums.push(sum);
    }
  }
  const decoded = C.map((r, i) => r.map((_, j) => sums[i * vc.length + j].cells[0].value));
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: `${d.leftLabel} × V`, stepTitle: "Input encrypted attention weights and values", shortTitle: "Softmax × V",
      description: "This toy composes a row–column dot product with NEXUS’s QuickSum to illustrate the context multiplication.", revealAt: 0,
      rows: [[matrix("A", d.leftLabel, S, 0, "CT", "row"), operator(`${d.id}-times`, "×", "", 0), matrix("B", "V", V, 0, "CT", "column", 2)]] },
    { id: `${d.id}-row`, title: "Row pack", stepTitle: "Pack attention rows against value columns", shortTitle: "Pack dot inputs",
      description: "Align each attention row with each V column in a zero-padded block with separate sample lanes.", revealAt: 1, rows: [rows, vc] },
    { id: `${d.id}-multiply`, title: "⊠", stepTitle: "Multiply each row and column slotwise", shortTitle: "Slot products",
      description: "Each product vector contains the terms for one context entry per sample.", revealAt: 2, rows: products },
    { id: `${d.id}-sum`, title: "QuickSum", stepTitle: "Fold the products with QuickSum", shortTitle: "QuickSum",
      description: "Duplicate the active block, rotate and add within the sample lanes, then mask away the duplicate block.", revealAt: 3, rows: foldRows },
    outputStage(d, decoded, sums, 4, "Read the first sample’s reduced dot products into the context matrix."),
  ] };
  const steps = graphSteps(d, graph, S, V, C, [{}, { ciphertextCount: 4 },
    { ctCtMultiplications: 4, relinearizations: 4, rescales: 4, multiplicativeDepth: 1 },
    { rotations: 8, additions: 8, masks: 4, ptCtMultiplications: 4, multiplicativeDepth: 1 }, {}]);
  return shell(d, S, V, C, graph, steps, "row-packed ciphertext MM + QuickSum");
}

function buildHiddenLinear(d, A, W) {
  assertRect(A, d.A.length, d.A[0].length, d.leftLabel);
  assertRect(W, d.B.length, d.B[0].length, d.rightLabel);
  const C = matmul(A, W);
  const inputs = cols(A).map((col, i) => batchColumn(col, `${d.id}-in-${i}`, `${d.leftLabel}${i}`, 1, i));
  const productRows = [], sumRows = [];
  const outputs = cols(W).map((weights, j) => {
    const terms = inputs.map((ct, i) => {
      const id = `${d.id}-term-${j}-${i}`;
      const weight = batchColumn(Array(A.length).fill(weights[i]), `${id}-w`, `w${i},${j}`, 2, 4, "vector", "PT");
      const values = multiplyVectors(valuesOf(ct), valuesOf(weight)).slice(0, A.length);
      const term = batchColumn(values, id, `p${i},${j}`, 2, i, "product");
      productRows.push([ct, operator(`${id}-mul`, "⊙", "", 2), weight, operator(`${id}-eq`, "=", "", 2), term]);
      return term;
    });
    const out = batchColumn(addVectors(terms.map(valuesOf)).slice(0, A.length), `${d.id}-out-${j}`, `${d.outputLabel}${j}`, 3, j, "result");
    sumRows.push([...terms, operator(`${out.id}-sum`, "Σ", "", 3), out]);
    return out;
  });
  const graph = { id: `${d.id}-graph`, algorithmId: d.id, title: d.shortLabel, subtitle: "", stages: [
    { id: `${d.id}-input`, title: `${d.leftLabel} × ${d.rightLabel}`, stepTitle: "Input encrypted activations and server weights", shortTitle: "Input",
      description: "The intermediate activation remains encrypted. This example uses the column products of the linear operation with the model owner’s plaintext weights.", revealAt: 0,
      rows: [[matrix("A", d.leftLabel, A, 0), operator(`${d.id}-times`, "×", "", 0), matrix("B", d.rightLabel, W, 0, "PT")]] },
    { id: `${d.id}-pack`, title: "Column pack", stepTitle: "Pack the encrypted input columns", shortTitle: "Pack columns",
      description: "Store each feature column in one ciphertext with a contiguous row block per sample.", revealAt: 1, rows: [inputs] },
    { id: `${d.id}-multiply`, title: "⊙ PT", stepTitle: "Apply a broadcast weight to each input column", shortTitle: "Slot products",
      description: "Repeat each weight across the active slots and multiply it by the matching encrypted activation column.", revealAt: 2, rows: productRows },
    { id: `${d.id}-sum`, title: "Σ", stepTitle: "Sum feature contributions by output column", shortTitle: "Sum products",
      description: "Add all the weighted input columns that feed the same output feature.", revealAt: 3, rows: sumRows },
    outputStage(d, transpose(outputs.map((out) => valuesOf(out).slice(0, A.length))), outputs, 4,
      "Read the first sample block of each encrypted column to check the resulting projection."),
  ] };
  const products = W.length * W[0].length;
  const steps = graphSteps(d, graph, A, W, C, [{}, { ciphertextCount: inputs.length },
    { ptCtMultiplications: products, rescales: W[0].length, multiplicativeDepth: 1 },
    { additions: products - W[0].length }, {}]);
  return shell(d, A, W, C, graph, steps, "column-packed ciphertext × plaintext projection");
}

export function buildNexusOperationTrace(id, options = {}) {
  const d = getNexusOperation(id); if (!d) throw new Error(`Unknown NEXUS operation: ${id}`);
  if (options.leftSecret != null && Boolean(options.leftSecret) !== d.leftSecret) throw new Error(`${d.shortLabel} has a fixed left operand type`);
  if (options.rightSecret != null && Boolean(options.rightSecret) !== d.rightSecret) throw new Error(`${d.shortLabel} has a fixed right operand type`);
  if (options.slotsPerCiphertext != null && Number(options.slotsPerCiphertext) !== SLOTS) throw new Error("NEXUS toy uses 8 slots");
  const A = options.A || d.A, B = options.B || d.B;
  if (d.kind === "ccmm") return buildCcmm(d, A, B);
  if (d.kind === "context") return buildContext(d, A, B);
  if (d.kind === "hidden-linear") return buildHiddenLinear(d, A, B);
  return buildOfflineOnline(d, A, B, d.kind === "decompress");
}
export function nexusOperationDefaults(id) {
  const d = getNexusOperation(id); if (!d) return null;
  return { m: d.A.length, k: d.A[0].length, n: d.B[0].length, A: clone(d.A), B: clone(d.B), leftSecret: d.leftSecret,
    rightSecret: d.rightSecret, slotsPerCiphertext: SLOTS, strategy: "column" };
}
export function parseNexusOperationInputs(raw) { return parseAndValidateInputs(raw); }
