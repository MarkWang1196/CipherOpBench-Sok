// Fixed-size THOR toy models.  The numerical pipeline is delegated to the
// paper-faithful Figure 3/4 builders; this module only gives those graphs the
// names and visual semantics of the concrete BERT matrix multiplication.

import { buildThorFigure3PcmmTrace } from "./figure3_pcmm.js";
import { buildThorFigure4CcmmTrace, THOR_CCMM_METHODS } from "./figure4_ccmm.js";
import { buildThorScoreCcmmTrace } from "./score_ccmm.js";
import { parseAndValidateInputs, parseMatrixText } from "../../core/input_parser.js";

export const THOR_OPERATION_IDS = Object.freeze({
  QUERY: "thor-query",
  KEY: "thor-key",
  VALUE: "thor-value",
  SCORE: "thor-score",
  CONTEXT: "thor-context",
  OUTPUT: "thor-output",
  FC1: "thor-fc1",
  FC2: "thor-fc2",
  POOLER: "thor-pooler",
  CLASSIFIER: "thor-classifier",
});

const W4 = Object.freeze([
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [2, 1, 0, 1],
  [3, 2, 1, 2],
]);

const X42 = Object.freeze([
  [1, 0],
  [0, 1],
  [2, 1],
  [1, 2],
]);

const Q24 = Object.freeze([
  [1, 2, 0, 1],
  [0, 1, 3, 2],
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

const K4 = Object.freeze([
  [1, 0, 2, 1],
  [0, 1, 1, 2],
  [2, 1, 0, 1],
  [1, 2, 1, 0],
]);

const VT24 = Object.freeze([
  [1, 0, 2, 1],
  [0, 1, 1, 2],
]);

const ST4 = Object.freeze([
  [0.5, 0.25, 0.125, 0.125],
  [0.25, 0.5, 0.125, 0.125],
  [0.125, 0.125, 0.5, 0.25],
  [0.125, 0.125, 0.25, 0.5],
]);

function operation(id, config) {
  return Object.freeze({ id, ...config });
}

function pcmm(id, shortLabel, leftLabel, rightLabel, outputLabel, group) {
  return operation(id, {
    shortLabel,
    group,
    kind: "pcmm",
    formula: `${leftLabel} × ${rightLabel}`,
    leftLabel,
    rightLabel,
    outputLabel,
    leftSecret: false,
    rightSecret: true,
    A: W4,
    B: X42,
  });
}

function ccmm(id, shortLabel, leftLabel, rightLabel, outputLabel, A, B) {
  return operation(id, {
    shortLabel,
    group: "attention",
    kind: "ccmm",
    formula: `${leftLabel} × ${rightLabel}`,
    leftLabel,
    rightLabel,
    outputLabel,
    leftSecret: true,
    rightSecret: true,
    A,
    B,
  });
}

export const THOR_OPERATIONS = Object.freeze({
  [THOR_OPERATION_IDS.QUERY]: pcmm(THOR_OPERATION_IDS.QUERY, "XWq", "Wq", "X", "Q", "attention"),
  [THOR_OPERATION_IDS.KEY]: pcmm(THOR_OPERATION_IDS.KEY, "XWk", "Wk", "X", "K", "attention"),
  [THOR_OPERATION_IDS.VALUE]: pcmm(THOR_OPERATION_IDS.VALUE, "XWv", "Wv", "X", "V", "attention"),
  [THOR_OPERATION_IDS.SCORE]: operation(THOR_OPERATION_IDS.SCORE, {
    shortLabel: "QKᵀ",
    group: "attention",
    kind: "score-ccmm",
    inputMode: "paired-columns",
    formula: "Q × Kᵀ",
    leftLabel: "Q",
    rightLabel: "K",
    outputLabel: "S",
    leftSecret: true,
    rightSecret: true,
    A: Q42,
    B: K42,
  }),
  [THOR_OPERATION_IDS.CONTEXT]: ccmm(THOR_OPERATION_IDS.CONTEXT, "SV", "Vᵀ", "αᵀ", "Hᵀ", VT24, ST4),
  [THOR_OPERATION_IDS.OUTPUT]: pcmm(THOR_OPERATION_IDS.OUTPUT, "HWo", "Wo", "H", "O", "attention"),
  [THOR_OPERATION_IDS.FC1]: pcmm(THOR_OPERATION_IDS.FC1, "FC1", "W1", "H", "F1", "ffn"),
  [THOR_OPERATION_IDS.FC2]: pcmm(THOR_OPERATION_IDS.FC2, "FC2", "W2", "F1", "F2", "ffn"),
  [THOR_OPERATION_IDS.POOLER]: pcmm(THOR_OPERATION_IDS.POOLER, "Pooler", "Wp", "CLS", "P", "output"),
  [THOR_OPERATION_IDS.CLASSIFIER]: pcmm(THOR_OPERATION_IDS.CLASSIFIER, "Classifier", "Wc", "P", "Y", "output"),
});

export function listThorOperations(group = null) {
  return Object.values(THOR_OPERATIONS).filter((item) => !group || item.group === group);
}

export function getThorOperation(id) {
  return THOR_OPERATIONS[id] || null;
}

function renameArtifact(artifact, descriptor) {
  const labels = {
    A: descriptor.leftLabel,
    B: descriptor.rightLabel,
    C: descriptor.outputLabel,
    "fig3-ct-b": `${descriptor.rightLabel}₀`,
    "fig3-rho4-b": `ρ⁴(${descriptor.rightLabel}₀)`,
    "fig3-p00": `${descriptor.leftLabel}₀`,
    "fig3-p10": `${descriptor.leftLabel}₁`,
    "fig3-p01": `${descriptor.leftLabel}₂`,
    "fig3-p11": `${descriptor.leftLabel}₃`,
    "fig3-partial-0": "T₀",
    "fig3-partial-1": "T₁",
    "fig3-rho-partial": "ρ(T₁)",
    "fig3-beta-mask": "M₀",
    "fig3-gamma-mask": "M₁",
    "fig3-aligned": "T₁′",
    "fig3-ct-c": descriptor.outputLabel,
    "fig4-ct-a": `${descriptor.leftLabel}₀`,
    "fig4-ct-c": descriptor.outputLabel,
  };
  artifact.meta = "";
  if (artifact.type === "operator") {
    artifact.label = "";
    artifact.symbol = artifact.symbol
      .replace(".*", "⊙")
      .replace("sum", "Σ")
      .replace("->", "→")
      .replace("rho", "ρ")
      .replace("x", "×");
    return;
  }
  if (labels[artifact.id]) artifact.label = labels[artifact.id];
  const match = artifact.id?.match(/fig4-(a|b|product)-(\d+)/);
  if (match) {
    const prefix = match[1] === "a" ? descriptor.leftLabel : match[1] === "b" ? descriptor.rightLabel : "P";
    artifact.label = `${prefix}₍${match[2]}₎`;
  }
}

function compactGraph(trace, descriptor) {
  const stageTitles = descriptor.kind === "pcmm"
    ? [
        `Setup: ${descriptor.leftLabel} × ${descriptor.rightLabel} inputs`,
        `Setup: multi-diagonal pack ${descriptor.rightLabel}`,
        "Step 1a: prepare parallel operands",
        "Step 1b: parallel submatrix products",
        "Step 2a: rotate partial products",
        "Step 2b: mask and align wraparound",
        "Step 3: aggregate aligned products",
        `Visual check: decoded ${descriptor.outputLabel}`,
      ]
    : [
        `Setup: ${descriptor.leftLabel} × ${descriptor.rightLabel} inputs`,
        "Encode multi-lower diagonals",
        "Prepare rotated ciphertext pairs",
        "Evaluate parallel CT×CT products",
        "Accumulate contraction lanes",
        "Output multi-lower-diagonal encoding",
        `Visual check: decoded ${descriptor.outputLabel}`,
      ];
  const stageDescriptions = descriptor.kind === "pcmm"
    ? [
        "Visualization setup; THOR executes the projection in plaintext-weight × ciphertext-activation orientation.",
        "Pack lower diagonals from encrypted activation blocks into ct.B.",
        "THOR Figure 3, Step 1: prepare plaintext diagonal masks and the rotated ct.B family.",
        "THOR Figure 3, Step 1: evaluate the parallel plaintext-ciphertext submatrix products.",
        "THOR Figure 3, Step 2: internally rotate the second partial-product family.",
        "THOR Figure 3, Step 2: mask the wraparound lanes and align the rotated family.",
        "THOR Figure 3, Step 3: add the aligned block-product families.",
        "Visualization-only decoding; the protocol keeps the result encrypted.",
      ]
    : [
        `Start from encrypted ${descriptor.leftLabel} and ${descriptor.rightLabel}; compute ${descriptor.outputLabel} with THOR's lower-lower ciphertext MatMul.`,
        `Encode ${descriptor.leftLabel} in THOR's multi-lower-diagonal layout.`,
        `Rotate ${descriptor.leftLabel} diagonals and pair them with the matching replicated lower diagonals of ${descriptor.rightLabel}.`,
        "Evaluate the paired lanes with parallel slot-wise ciphertext-ciphertext products.",
        "Add product ciphertexts across the contraction dimension for each output diagonal group.",
        `Keep encrypted ${descriptor.outputLabel} in multi-lower-diagonal form for the next module.`,
        `Decode ${descriptor.outputLabel} only for visualization; the protocol itself keeps the result encrypted.`,
      ];
  const stepTitles = descriptor.kind === "pcmm"
    ? [
        "Input plaintext weight and encrypted activation",
        "Pack encrypted activation diagonals",
        "Prepare parallel block operands",
        "Compute parallel block products",
        "Rotate the second partial-product family",
        "Mask and align wraparound lanes",
        "Aggregate aligned block products",
        `Verify encrypted ${descriptor.outputLabel} output`,
      ]
    : [
        "Input two encrypted matrices",
        "Encode multi-lower diagonals",
        "Prepare matching diagonal pairs",
        "Compute parallel ciphertext products",
        "Accumulate contraction lanes",
        "Keep multi-lower-diagonal output",
        `Verify encrypted ${descriptor.outputLabel} output`,
      ];
  const shortTitles = descriptor.kind === "pcmm"
    ? ["Input", "Pack ct.B", "Prepare", "Block products", "Rotate partials", "Mask + align", "Aggregate", "Output"]
    : ["Input", "Encode diagonals", "Pair diagonals", "CT products", "Accumulate", "Encoded output", "Output"];

  trace.graph.id = `${descriptor.id}-toy`;
  trace.graph.algorithmId = descriptor.id;
  trace.graph.title = descriptor.shortLabel;
  trace.graph.subtitle = "";
  trace.graph.stages.forEach((stage, index) => {
    stage.title = stageTitles[index] || "";
    stage.caption = stageDescriptions[index] || "";
    stage.stepTitle = stepTitles[index] || stage.title;
    stage.shortTitle = shortTitles[index] || stage.title;
    for (const row of stage.rows || []) {
      for (const artifact of row) renameArtifact(artifact, descriptor);
    }
  });
}

function relabelTrace(trace, descriptor) {
  const { A, B } = descriptor;
  trace.id = `${descriptor.id}-toy`;
  trace.title = descriptor.shortLabel;
  trace.parameters.algorithmId = descriptor.id;
  trace.operands.left.label = descriptor.leftLabel;
  trace.operands.right.label = descriptor.rightLabel;
  trace.output.label = descriptor.outputLabel;
  trace.plan.algorithmId = descriptor.id;
  trace.plan.visualStyle = "diagram-only";
  trace.plan.plannerSource = "THOR Figure 3/4 toy model";
  trace.plan.displayMode = "diagram-only";
  trace.plan.warnings = [];
  compactGraph(trace, descriptor);
  trace.steps.forEach((step, index) => {
    step.title = trace.graph.stages[index]?.stepTitle || trace.graph.stages[index]?.title || "";
    step.description = trace.graph.stages[index]?.caption || "";
    step.formula = { plain: "" };
    for (const tensor of Object.values(step.tensors || {})) {
      if (tensor.id === "A") tensor.label = descriptor.leftLabel;
      if (tensor.id === "B") tensor.label = descriptor.rightLabel;
      if (tensor.id === "C") tensor.label = descriptor.outputLabel;
    }
  });
  trace.operands.left.values = A;
  trace.operands.right.values = B;
  return trace;
}

export function buildThorOperationTrace(id, options = {}) {
  const descriptor = getThorOperation(id);
  if (!descriptor) throw new Error(`Unknown THOR BERT operation: ${id}`);
  if (options.leftSecret != null && Boolean(options.leftSecret) !== descriptor.leftSecret) {
    throw new Error(`${descriptor.shortLabel} requires ${descriptor.leftSecret ? "ciphertext" : "plaintext"} A`);
  }
  if (options.rightSecret != null && Boolean(options.rightSecret) !== descriptor.rightSecret) {
    throw new Error(`${descriptor.shortLabel} requires ${descriptor.rightSecret ? "ciphertext" : "plaintext"} B`);
  }

  const A = options.A || descriptor.A;
  const B = options.B || descriptor.B;
  const workingDescriptor = { ...descriptor, A, B };
  if (descriptor.kind === "score-ccmm") {
    // Dedicated Corollary 3.9 score trace (S = Q K^T, 4x4), already labeled.
    return buildThorScoreCcmmTrace({ A, B, slotsPerCiphertext: 8 });
  }
  const trace = descriptor.kind === "pcmm"
    ? buildThorFigure3PcmmTrace({ A, B, slotsPerCiphertext: 8 })
    : buildThorFigure4CcmmTrace({ A, B, slotsPerCiphertext: 8, method: THOR_CCMM_METHODS.LOWER_LOWER });
  return relabelTrace(trace, workingDescriptor);
}

export function thorOperationDefaults(id) {
  const descriptor = getThorOperation(id);
  if (!descriptor) return null;
  const paired = descriptor.inputMode === "paired-columns";
  return {
    m: descriptor.A.length,
    k: descriptor.A[0].length,
    n: paired ? descriptor.A.length : descriptor.B[0].length,
    A: descriptor.A.map((row) => [...row]),
    B: descriptor.B.map((row) => [...row]),
    leftSecret: descriptor.leftSecret,
    rightSecret: descriptor.rightSecret,
    slotsPerCiphertext: 8,
    strategy: "diagonal",
  };
}

function assertThorShape(matrix, rows, cols, label) {
  if (!Array.isArray(matrix) || matrix.length !== rows || matrix.some((row) => !Array.isArray(row) || row.length !== cols)) {
    throw new Error(`${label} must be ${rows}×${cols}`);
  }
}

export function parseThorOperationInputs(raw, descriptor) {
  if (descriptor.inputMode !== "paired-columns") return parseAndValidateInputs(raw);
  const errors = {};
  let A;
  let B;
  try {
    A = parseMatrixText(raw.valuesTextA);
    B = parseMatrixText(raw.valuesTextB);
    assertThorShape(A, 4, 2, descriptor.leftLabel);
    assertThorShape(B, 4, 2, descriptor.rightLabel);
  } catch (error) {
    errors.values = error.message;
  }
  if (Number(raw.slots) !== 8) errors.slots = "THOR score toy uses exactly 8 slots";
  if (Object.keys(errors).length) return { ok: false, errors, inputs: null };
  return {
    ok: true,
    errors: {},
    inputs: {
      m: 4,
      k: 2,
      n: 4,
      A,
      B,
      leftSecret: String(raw.leftType).toLowerCase() === "ct",
      rightSecret: String(raw.rightType).toLowerCase() === "ct",
      slotsPerCiphertext: 8,
      strategy: "diagonal",
      customLayouts: null,
      valueMode: "manual",
      seed: raw.seed,
    },
  };
}
