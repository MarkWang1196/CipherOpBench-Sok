// Generic matmul scene builder (spec §11.1).
//
// buildMatmulScene() turns any legal M×K · K×N matmul into a complete,
// validated trace. It is shape-agnostic: no fixed 4×4 / 16-slot / four-term
// assumptions live in the step generation (verified by tests). The two default
// scenes (XWq, QKᵀ) are just this builder invoked with preset operands.
//
// Algorithm — Outer-Product Planes (OPP), see packing_planner.js:
//   pack left (and right) → per tile: replicate (repCol) → encode/align
//   (repRow) → slot-wise multiply → cross-ciphertext reduce over k → output.
// Correctness is structural (each P_k slot = A[i,k]·B[k,j]) and is re-verified
// against direct matmul at the end (spec §12.9: builder throws on mismatch).

import { matmul, transpose, matricesEqual } from "./matrix.js";
import { planMatmulPacking, padMatrixTo, cropMatrix, nextPow2 } from "./packing_planner.js";
import {
  buildCiphertextStates,
  buildCiphertextStatesFromArrays,
} from "./layout_adapter.js";
import { validateTrace } from "./trace_validator.js";
import { OP_TYPES, PHASES, KINDS, SCHEMA_VERSION } from "./trace_schema.js";
import { emptyMetrics, addMetricsDelta, cloneMetrics } from "./metrics.js";

const SHORT_TITLES = {
  reference: "Matmul",
  pack: "Pack",
  replicate: "Replicate",
  encode: "Encode",
  transpose: "Transpose",
  multiply: "Multiply",
  relinearize: "Relin",
  rescale: "Rescale",
  reduce: "Reduce",
  output: "Output",
};

function numArray(n, value = 0) {
  return Array.from({ length: n }, () => value);
}

/** Slot descriptors for the replicate step: repCol(A[:,k]) for all k. */
function buildRepColArrays({ leftId, Ap, Kp, Np, tile, n }) {
  const R_t = tile.slotGrid[0];
  const C_t = tile.slotGrid[1];
  const arrays = [];
  for (let k = 0; k < Kp; k++) {
    const slots = [];
    for (let s = 0; s < n; s++) {
      const i = s % R_t;
      const j = Math.floor(s / R_t);
      const gi = tile.rowRange[0] + i;
      const gcol = tile.colRange[0] + j;
      slots.push({
        value: Ap[gi][k],
        logicalId: `${leftId}-rep:${k}:${gi}:${gcol}`,
        sourceRefs: [
          {
            tensorId: leftId,
            coords: [gi, k],
            linear: gi * Kp + k,
            replica: gcol,
            logicalId: `${leftId}:${gi * Kp + k}`,
          },
        ],
        expression: `${leftId}[${gi},${k}]`,
      });
    }
    arrays.push(slots);
  }
  return arrays;
}

/** Slot descriptors for the encode/align step: repRow(B[k,:]) for all k. */
function buildRepRowArrays({ rightId, Bp, Kp, Np, tile, n }) {
  const R_t = tile.slotGrid[0];
  const C_t = tile.slotGrid[1];
  const arrays = [];
  for (let k = 0; k < Kp; k++) {
    const slots = [];
    for (let s = 0; s < n; s++) {
      const i = s % R_t;
      const j = Math.floor(s / R_t);
      const gi = tile.rowRange[0] + i;
      const gcol = tile.colRange[0] + j;
      if (gcol >= Np) {
        slots.push({
          value: 0,
          kind: KINDS.OOB,
          logicalId: `${rightId}-row:${k}:oob:${gcol}`,
          sourceRefs: [],
          expression: "out of range",
        });
      } else {
        slots.push({
          value: Bp[k][gcol],
          logicalId: `${rightId}-row:${k}:${gi}:${gcol}`,
          sourceRefs: [
            {
              tensorId: rightId,
              coords: [k, gcol],
              linear: k * Np + gcol,
              replica: gi,
              logicalId: `${rightId}:${k * Np + gcol}`,
            },
          ],
          expression: `${rightId}[${k},${gcol}]`,
        });
      }
    }
    arrays.push(slots);
  }
  return arrays;
}

/** Slot descriptors for the multiply step: P_k = repCol_k ⊙ repRow_k. */
function buildProductArrays({ repColArrays, repRowArrays, leftId, rightId, tile, n }) {
  const R_t = tile.slotGrid[0];
  const C_t = tile.slotGrid[1];
  const arrays = [];
  for (let k = 0; k < repColArrays.length; k++) {
    const aArr = repColArrays[k];
    const bArr = repRowArrays[k];
    const slots = [];
    for (let s = 0; s < n; s++) {
      const a = aArr[s];
      const b = bArr[s];
      const i = s % R_t;
      const j = Math.floor(s / R_t);
      const gi = tile.rowRange[0] + i;
      const gcol = tile.colRange[0] + j;
      const oob = b.kind === KINDS.OOB;
      slots.push({
        value: oob ? 0 : a.value * b.value,
        kind: oob ? KINDS.MASKED : KINDS.PRODUCT,
        logicalId: `P:${k}:${gi}:${gcol}`,
        sourceRefs: [
          { kind: "slot", ciphertextId: `${leftId}-rep-${k}-ct-0`, slot: s, logicalId: a.logicalId },
          { kind: "slot", ciphertextId: `${rightId}-row-${k}-ct-0`, slot: s, logicalId: b.logicalId },
        ],
        outputCoords: oob ? null : [gi, gcol],
        contractionIndex: k,
        expression: oob ? "masked" : `${leftId}[${gi},${k}] × ${rightId}[${k},${gcol}]`,
      });
    }
    arrays.push(slots);
  }
  return arrays;
}

/** Slot descriptors for the reduce step: tile sum Σ_k P_k. */
function buildReduceSlots({ productArrays, outputId, tile, n, planeCtIdPrefix }) {
  const R_t = tile.slotGrid[0];
  const C_t = tile.slotGrid[1];
  const Kp = productArrays.length;
  const slots = [];
  for (let s = 0; s < n; s++) {
    const i = s % R_t;
    const j = Math.floor(s / R_t);
    const gi = tile.rowRange[0] + i;
    const gcol = tile.colRange[0] + j;
    let sum = 0;
    const sourceRefs = [];
    const terms = [];
    for (let k = 0; k < Kp; k++) {
      const p = productArrays[k][s];
      sum += p.value;
      sourceRefs.push({
        kind: "slot",
        ciphertextId: `${planeCtIdPrefix}-ct-0`,
        slot: s,
        logicalId: p.logicalId,
      });
      terms.push(p.expression);
    }
    slots.push({
      value: sum,
      kind: KINDS.PARTIAL_SUM,
      logicalId: `${outputId}-sum-${tile.iTile}-${tile.jTile}:${gi}:${gcol}`,
      sourceRefs,
      outputCoords: [gi, gcol],
      contractionIndex: null,
      expression: `${terms.filter(Boolean).join(" + ")} = ${sum}`,
    });
  }
  return slots;
}

function connection(c, d) {
  return { type: c.type, from: c.from, to: c.to };
}

/**
 * Build a full validated trace for one matmul.
 *
 * @param {object} cfg
 * @param {{id:string,label:string,values:number[][],secret:boolean}} cfg.left
 * @param {{id:string,label:string,values:number[][],secret:boolean}} cfg.right
 * @param {number} cfg.slotsPerCiphertext
 * @param {string} cfg.packingStrategy auto|row|column|diagonal|custom
 * @param {object|null} cfg.customLayouts
 * @param {{transposeRight?:boolean, outputId?:string, outputLabel?:string}} cfg.options
 * @returns {object} validated trace
 */
export function buildMatmulScene({
  left,
  right,
  slotsPerCiphertext,
  packingStrategy = "auto",
  customLayouts = null,
  options = {},
}) {
  const A = left.values;
  const B = right.values;
  const M = A.length;
  const K = A[0].length;
  const transposeRight = Boolean(options.transposeRight);

  // For a plain matmul the right operand is K×N. For a "Q·Kᵀ"-style preset the
  // caller passes the *source* matrix (K, shape N×K) and we transpose it, so
  // N = number of rows of that source.
  let N;
  if (transposeRight) {
    if (B[0].length !== K) {
      throw new Error(
        `transposeRight scene requires the right operand to have ${K} columns (the shared dim)`
      );
    }
    N = B.length;
  } else {
    N = B[0].length;
    if (B.length !== K) {
      throw new Error(`matmul shape mismatch: left is ${M}×${K}, right is ${B.length}×${N}`);
    }
  }

  const leftSecret = Boolean(left.secret);
  const rightSecret = Boolean(right.secret);
  const ctCt = leftSecret && rightSecret;

  const outputId = options.outputId || "Q";
  const outputLabel = options.outputLabel || "Output Q";
  const outputSecret = leftSecret || rightSecret;

  const plan = planMatmulPacking({
    m: M,
    k: K,
    n: N,
    slotsPerCiphertext,
    leftSecret,
    rightSecret,
    strategy: packingStrategy,
    customLayouts,
  });

  const { m: Mp, k: Kp, n: Np } = plan.paddedShape;
  const Ap = padMatrixTo(A, Mp, Kp);
  const rightLogical = transposeRight ? transpose(B) : B;
  const Bp = padMatrixTo(rightLogical, Kp, Np);
  const rightMultId = transposeRight ? `${right.id}T` : right.id;

  const expected = matmul(A, rightLogical);
  const n = slotsPerCiphertext;
  const tiles = plan.tiling.tiles;

  const steps = [];
  let running = emptyMetrics();

  function commitStep(step, delta) {
    step.metricsDelta = delta || emptyMetrics();
    step.metricsTotal = cloneMetrics(addMetricsDelta(running, delta || emptyMetrics()));
    running = cloneMetrics(step.metricsTotal);
    steps.push(step);
    return step;
  }

  const tensors = {
    left: {
      id: left.id,
      label: left.label,
      shape: [M, K],
      secret: leftSecret,
      values: A,
    },
    right: {
      id: rightMultId,
      label: transposeRight ? `${right.label}ᵀ` : right.label,
      shape: [K, N],
      secret: rightSecret,
      values: rightLogical,
    },
  };
  if (transposeRight) {
    tensors.rightSource = {
      id: right.id,
      label: right.label,
      shape: [K, N],
      secret: rightSecret,
      values: B,
    };
  }

  const L = left.id;
  const R = rightMultId;

  // ---- Step 0: reference ----
  {
    const terms = [];
    for (let p = 0; p < K; p++) terms.push(`${L}[0,${p}]·${R}[${p},0]`);
    commitStep(
      {
        id: "reference",
        index: 0,
        phase: PHASES.REFERENCE,
        title: "Plain matrix multiplication",
        shortTitle: SHORT_TITLES.reference,
        description: `Reference result we must reproduce: ${outputId} = ${L} · ${R}.`,
        formula: { plain: `${outputId}[0,0] = Σₖ ${L}[0,k]·${R}[k,0]\n= ${terms.join(" + ")}` },
        tensors: { ...tensors, output: null },
        ciphertexts: [],
        operation: {
          type: OP_TYPES.REFERENCE,
          inputIds: [L, R],
          outputIds: [outputId],
        },
        focus: {
          outputCoords: [0, 0],
          cellRefs: [
            ...Array.from({ length: K }, (_, p) => ({
              tensorId: L,
              coords: [0, p],
              linear: p,
            })),
            ...Array.from({ length: K }, (_, p) => ({
              tensorId: R,
              coords: [p, 0],
              linear: p * N,
            })),
          ],
          contractionIndex: null,
          planeIds: [],
        },
        connections: [],
        notes: [
          "This step is the plain-matrix reference; the animated pipeline must reach the same result.",
          "The output matrix appears (and is verified) at the Output step.",
        ],
        group: "reference",
        tile: null,
      },
      emptyMetrics()
    );
  }

  // ---- Step: pack left ----
  {
    const packed = buildCiphertextStates({
      tensorId: L,
      tensor: Ap,
      layoutString: plan.leftLayout,
      n,
      secret: leftSecret,
      stepId: "pack-left",
      idPrefix: `${L}-pack`,
      logicalIdFor: (m) => `${L}-pack:${m.coords[0]}:${m.coords[1]}`,
    });
    const secretCts = leftSecret ? packed.length : 0;
    commitStep(
      {
        id: `pack-${L}`,
        index: 1,
        phase: PHASES.PACK,
        title: `Pack ${L} into ciphertext slots`,
        shortTitle: SHORT_TITLES.pack,
        description: `Pack the padded ${L} (${Mp}×${Kp}) into ${packed.length} ciphertext(s) using layout ${plan.leftLayout}.`,
        formula: { plain: `slot(row,col) = ${L}[row,col]` },
        tensors: { ...tensors, output: null },
        ciphertexts: packed,
        operation: { type: OP_TYPES.PACK, inputIds: [L], outputIds: packed.map((c) => c.id) },
        focus: {
          outputCoords: null,
          cellRefs: Array.from({ length: K }, (_, p) => ({ tensorId: L, coords: [0, p], linear: p })),
          contractionIndex: null,
          planeIds: [],
        },
        connections: [],
        notes: [
          `Layout string: ${plan.leftLayout}`,
          leftSecret
            ? "This is a ciphertext (encrypted activations)."
            : "This is a plaintext (unencrypted).",
        ],
        group: "pack",
        tile: null,
      },
      { ciphertextCount: secretCts }
    );
  }

  // ---- CT×CT: pack the original right operand source (for the transpose story) ----
  if (ctCt && transposeRight) {
    const rightPackLayout = choosePackLayout(K, N, n);
    const packed = buildCiphertextStates({
      tensorId: right.id,
      tensor: padMatrixTo(B, nextPow2(K), nextPow2(N)),
      layoutString: rightPackLayout,
      n,
      secret: true,
      stepId: `pack-${right.id}`,
      idPrefix: `${right.id}-pack`,
      logicalIdFor: (m) => `${right.id}-pack:${m.coords[0]}:${m.coords[1]}`,
    });
    commitStep(
      {
        id: `pack-${right.id}`,
        index: steps.length,
        phase: PHASES.PACK,
        title: `Pack ${right.id}`,
        shortTitle: SHORT_TITLES.pack,
        description: `Pack the keys ${right.id} as ciphertexts (CT×CT scenario).`,
        formula: { plain: `slot(row,col) = ${right.id}[row,col]` },
        tensors: { ...tensors, output: null },
        ciphertexts: packed,
        operation: {
          type: OP_TYPES.PACK,
          inputIds: [right.id],
          outputIds: packed.map((c) => c.id),
        },
        focus: { outputCoords: null, cellRefs: [], contractionIndex: null, planeIds: [] },
        connections: [],
        notes: ["Both operands are ciphertexts in this scenario."],
        group: "pack",
        tile: null,
      },
      { ciphertextCount: packed.length }
    );

    // Transpose step: show K → Kᵀ.
    commitStep(
      {
        id: `transpose-${right.id}`,
        index: steps.length,
        phase: PHASES.TRANSPOSE,
        title: `Transpose ${right.id} → ${right.id}ᵀ`,
        shortTitle: SHORT_TITLES.transpose,
        description: `Attention needs ${right.id}ᵀ. Transpose is a pure data-movement/permute step.`,
        formula: { plain: `${right.id}ᵀ[j,d] = ${right.id}[d,j]` },
        tensors: { ...tensors, output: null },
        ciphertexts: [],
        operation: {
          type: OP_TYPES.PERMUTE,
          inputIds: [`${right.id}-pack`],
          outputIds: [`${right.id}T`],
        },
        focus: { outputCoords: null, cellRefs: [], contractionIndex: null, planeIds: [] },
        connections: [],
        notes: [
          `${right.id}ᵀ is a reshaped view; no ciphertext conversion is required for the illustration.`,
        ],
        group: "transpose",
        tile: null,
      },
      emptyMetrics()
    );
  }

  // ---- Per-tile pipeline ----
  const tileSums = new Map();

  tiles.forEach((tile, tileIdx) => {
    const tLabel = tileIdx === 0 ? "" : ` · tile (${tile.iTile},${tile.jTile})`;
    const base = {
      tensors: { ...tensors, output: null },
      group: null,
      tile: { iTile: tile.iTile, jTile: tile.jTile },
    };

    // replicate (repCol)
    const repColArrays = buildRepColArrays({ leftId: L, Ap, Kp, Np, tile, n });
    const repColCts = buildCiphertextStatesFromArrays({
      id: `${L}-rep`,
      secret: leftSecret,
      stepId: `replicate-${tileIdx}`,
      arrays: repColArrays,
      labelFor: (k) => `Replicated ${L} column plane k=${k}`,
    });
    commitStep(
      {
        ...base,
        id: `replicate-${tileIdx}`,
        index: steps.length,
        phase: PHASES.REPLICATE,
        title: `ApplyReplicate: broadcast ${L} columns${tLabel}`,
        shortTitle: `Replicate${tileIdx ? "" : ""}`,
        description: `For each contraction index k, broadcast column k of ${L} across every output column (1→many).`,
        formula: { plain: `repCol_k[i,j] = ${L}[i,k]   (replicated over j)` },
        ciphertexts: repColCts,
        operation: {
          type: OP_TYPES.REPLICATE,
          inputIds: [`${L}-pack`],
          outputIds: repColCts.map((c) => c.id),
        },
        focus: { outputCoords: null, cellRefs: [], contractionIndex: null, planeIds: [] },
        connections: [],
        notes: [
          "Replication is a packing-time broadcast (no ciphertext HE op).",
          "Each source cell X[i,k] has one replica per output column.",
        ],
        group: "replicate",
      },
      { ciphertextCount: leftSecret ? repColCts.length : 0 }
    );

    // encode / align (repRow)
    const repRowArrays = buildRepRowArrays({ rightId: R, Bp, Kp, Np, tile, n });
    const repRowCts = buildCiphertextStatesFromArrays({
      id: `${R}-row`,
      secret: rightSecret,
      stepId: `encode-${tileIdx}`,
      arrays: repRowArrays,
      labelFor: (k) =>
        rightSecret
          ? `Aligned ${R} row plane k=${k}`
          : `Encoded ${R} row k=${k}`,
    });
    const opTypeEncode = rightSecret ? OP_TYPES.REPLICATE : OP_TYPES.ENCODE;
    commitStep(
      {
        ...base,
        id: `encode-${tileIdx}`,
        index: steps.length,
        phase: rightSecret ? PHASES.ENCODE : PHASES.ENCODE,
        title: rightSecret
          ? `Align ${R} ciphertext rows${tLabel}`
          : `Encode plaintext ${R} aligned to slots${tLabel}`,
        shortTitle: rightSecret ? "Align" : "Encode",
        description: rightSecret
          ? `Align ciphertext row k of ${R} across all output rows (CT×CT alignment).`
          : `${R} is plaintext weights: the encode step can bake the alignment directly, with no ciphertext conversion (spec §12.6).`,
        formula: { plain: `repRow_k[i,j] = ${R}[k,j]   (replicated over i)` },
        ciphertexts: repRowCts,
        operation: {
          type: opTypeEncode,
          inputIds: [`${R}-pack`],
          outputIds: repRowCts.map((c) => c.id),
        },
        focus: { outputCoords: null, cellRefs: [], contractionIndex: null, planeIds: [] },
        connections: [],
        notes: rightSecret
          ? ["Both operands are ciphertexts; alignment happens on the ciphertext side too."]
          : [
              "Plaintext encoding cost is real (memory + encode) but requires no ciphertext layout conversion.",
            ],
        group: "encode",
      },
      emptyMetrics()
    );

    // multiply
    const productArrays = buildProductArrays({
      repColArrays,
      repRowArrays,
      leftId: L,
      rightId: R,
      tile,
      n,
    });
    const productCts = buildCiphertextStatesFromArrays({
      id: `P-${tileIdx}`,
      secret: true,
      stepId: `multiply-${tileIdx}`,
      arrays: productArrays,
      labelFor: (k) => `Product plane k=${k}`,
    });
    const multOp = ctCt ? OP_TYPES.CT_CT_MULTIPLY : OP_TYPES.CT_PT_MULTIPLY;
    const multDelta = ctCt
      ? { ctCtMultiplications: Kp, multiplicativeDepth: 1, ciphertextCount: productCts.length }
      : { ptCtMultiplications: Kp, ciphertextCount: productCts.length };
    commitStep(
      {
        ...base,
        id: `multiply-${tileIdx}`,
        index: steps.length,
        phase: PHASES.MULTIPLY,
        title: `Slot-wise ${ctCt ? "ciphertext×ciphertext" : "ciphertext×plaintext"} multiplication${tLabel}`,
        shortTitle: SHORT_TITLES.multiply,
        description: `P_k[i,j] = repCol_k[i,j] × repRow_k[i,j] = ${L}[i,k]·${R}[k,j].`,
        formula: { plain: `P_k[i,j] = ${L}[i,k] × ${R}[k,j]` },
        ciphertexts: productCts,
        operation: {
          type: multOp,
          inputIds: [...repColCts.map((c) => c.id), ...repRowCts.map((c) => c.id)],
          outputIds: productCts.map((c) => c.id),
        },
        focus: {
          outputCoords: [tile.rowRange[0], tile.colRange[0]],
          cellRefs: [],
          contractionIndex: 0,
          planeIds: productCts.map((c) => c.id),
        },
        connections: [],
        notes: [
          ctCt
            ? `CT × CT: each plane is a ciphertext-ciphertext multiplication (+1 multiplicative depth).`
            : "CT × PT: the plaintext is multiplied slot-wise into the ciphertext.",
        ],
        group: "multiply",
      },
      multDelta
    );

    // CT×CT metadata steps once (after the first tile's multiply).
    if (ctCt && tileIdx === 0) {
      commitStep(
        {
          ...base,
          id: "relinearize",
          index: steps.length,
          phase: PHASES.RELINEARIZE,
          title: "Relinearize (illustrative)",
          shortTitle: SHORT_TITLES.relinearize,
          description:
            "After CT×CT multiplication the ciphertext degree grows (e.g. 3 → 2); relinearization lowers it back. Metadata only — no real polynomial is shown.",
          formula: { plain: "Ciphertext degree: 3 → 2" },
          ciphertexts: [],
          operation: {
            type: OP_TYPES.RELINEARIZE,
            inputIds: [],
            outputIds: [],
          },
          focus: { outputCoords: null, cellRefs: [], contractionIndex: null, planeIds: [] },
          connections: [],
          notes: ["Illustrative CKKS state, not a measured backend value."],
          group: "relinearize",
        },
        { relinearizations: 1 }
      );
      commitStep(
        {
          ...base,
          id: "rescale",
          index: steps.length,
          phase: PHASES.RESCALE,
          title: "Rescale (illustrative)",
          shortTitle: SHORT_TITLES.rescale,
          description:
            "After multiplication the scale grows (Δ²); rescale brings it back to Δ and consumes one level.",
          formula: { plain: "Level: L → L-1\nScale: Δ² → Δ" },
          ciphertexts: [],
          operation: { type: OP_TYPES.RESCALE, inputIds: [], outputIds: [] },
          focus: { outputCoords: null, cellRefs: [], contractionIndex: null, planeIds: [] },
          connections: [],
          notes: ["Illustrative CKKS state."],
          group: "rescale",
        },
        { rescales: 1 }
      );
    }

    // reduce
    const reduceSlots = buildReduceSlots({
      productArrays,
      outputId,
      tile,
      n,
      planeCtIdPrefix: `P-${tileIdx}`,
    });
    const sumCts = buildCiphertextStatesFromArrays({
      id: `${outputId}-sum-${tile.iTile}-${tile.jTile}`,
      secret: true,
      stepId: `reduce-${tileIdx}`,
      arrays: [reduceSlots],
      labelFor: () => `Partial sums ${outputId} tile (${tile.iTile},${tile.jTile})`,
    });
    commitStep(
      {
        ...base,
        id: `reduce-${tileIdx}`,
        index: steps.length,
        phase: PHASES.REDUCE,
        title: `Reduce over k (sum planes)${tLabel}`,
        shortTitle: SHORT_TITLES.reduce,
        description: `Add the ${Kp} product planes across ciphertexts: ${outputId}[i,j] = Σₖ P_k[i,j].`,
        formula: { plain: `${outputId}[i,j] = Σₖ P_k[i,j] = ${L}[i,k]·${R}[k,j] summed over k` },
        ciphertexts: sumCts,
        operation: {
          type: OP_TYPES.ADD,
          inputIds: productCts.map((c) => c.id),
          outputIds: sumCts.map((c) => c.id),
        },
        focus: {
          outputCoords: [tile.rowRange[0], tile.colRange[0]],
          cellRefs: [],
          contractionIndex: null,
          planeIds: productCts.map((c) => c.id),
        },
        connections: [],
        notes: [
          `Reduction strategy: ${plan.reductionStrategy} (${Kp - 1} ciphertext additions).`,
          "This visualization adds whole ciphertexts; real circuits often use rotation-and-add inside a ciphertext instead.",
        ],
        group: "reduce",
      },
      { additions: Math.max(0, Kp - 1), ciphertextCount: sumCts.length }
    );

    tileSums.set(`${tile.iTile}:${tile.jTile}`, reduceSlots);
  });

  // ---- Output step ----
  const assembled = assembleTiles({ tiles, tileSums, Mp, Np });
  const finalTensor = cropMatrix(assembled, M, N);
  if (!matricesEqual(finalTensor, expected)) {
    throw new Error(
      `Scene generation failed verification: final result does not match direct matmul (M=${M},K=${K},N=${N})`
    );
  }

  const R_t = plan.tiling.R_t;
  const C_t = plan.tiling.C_t;
  const flatResult = [];
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      const ti = Math.floor(i / R_t);
      const tj = Math.floor(j / C_t);
      flatResult.push({
        value: finalTensor[i][j],
        logicalId: `${outputId}:${i}:${j}`,
        kind: KINDS.RESULT,
        sourceRefs: [
          {
            kind: "slot",
            ciphertextId: `${outputId}-sum-${ti}-${tj}-ct-0`,
            slot: (i % R_t) + Math.floor(j % C_t) * R_t,
            logicalId: `${outputId}-sum-${ti}-${tj}:${i}:${j}`,
          },
        ],
        outputCoords: [i, j],
        expression: `${outputId}[${i},${j}]`,
      });
    }
  }
  // Pad the last ciphertext with gap slots so every ct has exactly n slots.
  const nResultCts = Math.max(1, Math.ceil(flatResult.length / n));
  const outputArrays = [];
  for (let c = 0; c < nResultCts; c++) {
    const group = flatResult.slice(c * n, (c + 1) * n);
    while (group.length < n) {
      const s = group.length;
      group.push({
        value: 0,
        kind: KINDS.GAP,
        logicalId: `${outputId}:gap:${c}:${s}`,
        sourceRefs: [],
        outputCoords: null,
        expression: "unused slot",
      });
    }
    outputArrays.push(group);
  }
  const outputCts = buildCiphertextStatesFromArrays({
    id: outputId,
    secret: outputSecret,
    stepId: "output",
    arrays: outputArrays,
    labelFor: (c) => `Packed result ${outputId} #${c}`,
  });

  commitStep(
    {
      id: "output",
      index: steps.length,
      phase: PHASES.OUTPUT,
      title: `Output ${outputLabel}`,
      shortTitle: SHORT_TITLES.output,
      description: `Crop the padded result back to ${M}×${N} and verify against direct matmul.`,
      formula: { plain: `${outputId} = ${L} · ${R}\nVerified: equals direct matrix multiplication.` },
      tensors: {
        ...tensors,
        output: {
          id: outputId,
          label: outputLabel,
          shape: [M, N],
          secret: outputSecret,
          values: finalTensor,
        },
      },
      ciphertexts: outputCts,
      operation: { type: OP_TYPES.OUTPUT, inputIds: [], outputIds: outputCts.map((c) => c.id) },
      focus: {
        outputCoords: [0, 0],
        cellRefs: [],
        contractionIndex: null,
        planeIds: outputCts.map((c) => c.id),
      },
      connections: [],
      notes: [
        "Padding/oob slots contributed 0 and were cropped away — the logical output keeps the user's shape.",
        "Verified: animated result equals direct matrix multiplication.",
      ],
      group: "output",
      tile: null,
    },
    { ciphertextCount: outputCts.length }
  );

  const trace = {
    schemaVersion: SCHEMA_VERSION,
    id: `${left.id}-times-${rightMultId}-${M}x${K}x${N}`,
    title: `${leftSecret ? "Encrypted" : "Plaintext"} ${L} × ${rightSecret ? "Encrypted" : "Plaintext"} ${R}`,
    description: `${outputLabel} = ${L} · ${R} (${ctCt ? "CT×CT" : "CT×PT"})`,
    source: "educational",
    parameters: {
      scheme: "CKKS",
      slotsPerCiphertext: n,
      matmulType: plan.matmulType,
      matrixShape: { m: M, k: K, n: N },
      packingStrategy: packingStrategy || "auto",
      displayMode: plan.displayMode,
    },
    operands: {
      left: { id: left.id, label: left.label, secret: leftSecret, shape: [M, K], values: A },
      right: {
        id: rightMultId,
        label: transposeRight ? `${right.label}ᵀ` : right.label,
        secret: rightSecret,
        shape: [K, N],
        values: rightLogical,
      },
      ...(transposeRight
        ? { rightSource: { id: right.id, label: right.label, secret: rightSecret, shape: [K, N], values: B } }
        : {}),
    },
    plan,
    finalTensor,
    output: { id: outputId, label: outputLabel },
    steps,
  };

  const validation = validateTrace(trace);
  if (!validation.ok) {
    throw new Error(`Scene builder produced an invalid trace: ${validation.errors.join("; ")}`);
  }
  return trace;
}

/** Assemble the padded M′×N′ result from per-tile partial sums. */
function assembleTiles({ tiles, tileSums, Mp, Np }) {
  const out = Array.from({ length: Mp }, () => numArray(Np, 0));
  for (const tile of tiles) {
    const slots = tileSums.get(`${tile.iTile}:${tile.jTile}`);
    const R_t = tile.slotGrid[0];
    for (let s = 0; s < slots.length; s++) {
      const i = s % R_t;
      const j = Math.floor(s / R_t);
      const gi = tile.rowRange[0] + i;
      const gcol = tile.colRange[0] + j;
      if (gi < Mp && gcol < Np) out[gi][gcol] = slots[s].value;
    }
  }
  return out;
}

/** Generic row-major pack layout for an r×c matrix into n-slot ciphertexts. */
export function choosePackLayout(rows, cols, n) {
  const Rp = nextPow2(rows);
  const Cp = nextPow2(cols);
  if (Rp * Cp <= n) return `[0:${Rp}:1][1:${Cp}:1]`;
  const chunkRows = n / Cp;
  const rowCts = Rp / chunkRows;
  return `[0:${rowCts}:1][0:${chunkRows}:1][1:${Cp}:1]`;
}
