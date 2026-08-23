// Trace schema constants shared across the transformer visualizer.
//
// These are the *contract* of a trace: the scene builder emits a trace, the
// validator checks it, and every renderer reads only this shape. DOM state is
// never a source of business truth (spec §9: "动画不能直接依赖 DOM").

export const SCHEMA_VERSION = "1.0";

// High-level phase a step belongs to (drives grouping + display policy).
export const PHASES = Object.freeze({
  REFERENCE: "reference",
  PARTITION: "partition",
  DIAGONAL: "diagonal",
  PACK: "pack",
  REPLICATE: "replicate",
  ENCODE: "encode",
  TRANSPOSE: "transpose",
  ROTATE: "rotate",
  MASK: "mask",
  MULTIPLY: "multiply",
  RELINEARIZE: "relinearize",
  RESCALE: "rescale",
  REDUCE: "reduce",
  OUTPUT: "output",
});

// Concrete operation a step performs (drives the metrics delta + animation).
export const OP_TYPES = Object.freeze({
  REFERENCE: "reference",
  PARTITION: "partition",
  EXTRACT_DIAGONAL: "extract-diagonal",
  INTERLACE: "interlace",
  MULTI_DIAGONAL_PACK: "multi-diagonal-pack",
  PACK: "pack",
  REPLICATE: "replicate",
  ENCODE: "encode",
  PERMUTE: "permute",
  ROTATE: "rotate",
  INTERNAL_ROTATE: "internal-rotate",
  MASK: "mask",
  CT_PT_MULTIPLY: "ct-pt-multiply",
  CT_CT_MULTIPLY: "ct-ct-multiply",
  RELINEARIZE: "relinearize",
  RESCALE: "rescale",
  ADD: "add",
  AGGREGATE: "aggregate",
  DECODE: "decode",
  OUTPUT: "output",
});

// Kind of a slot: what it holds in the current step.
export const KINDS = Object.freeze({
  TENSOR: "tensor",
  GAP: "gap",
  OOB: "oob",
  PRODUCT: "product",
  PARTIAL_SUM: "partial-sum",
  RESULT: "result",
  MASKED: "masked",
});

// Where the trace/plan came from. Educational Auto results must never be
// presented as "Rotom compiler optimized layouts" (spec §11.7).
export const PLANNER_SOURCES = Object.freeze({
  EDUCATIONAL_AUTO: "educational-auto",
  USER_CUSTOM: "user-custom",
  ROTOM_COMPILER: "rotom-compiler",
});

export const PACKING_STRATEGIES = Object.freeze({
  AUTO: "auto",
  ROW: "row",
  COLUMN: "column",
  DIAGONAL: "diagonal",
  CUSTOM: "custom",
});

export const MATMUL_TYPES = Object.freeze({
  CT_PT: "ct-pt",
  PT_CT: "pt-ct",
  CT_CT: "ct-ct",
});

export const DISPLAY_MODES = Object.freeze({
  FULL: "full",
  TILE: "tile",
  SUMMARY: "summary",
});

export const CONNECTION_TYPES = Object.freeze({
  CELL_TO_SLOT: "cell-to-slot",
  SLOT_TO_PRODUCT: "slot-to-product",
  CONTRIBUTION_TO_PARTIAL: "contribution-to-partial",
  PARTIAL_TO_RESULT: "partial-to-result",
});

export const REDUCTION_STRATEGIES = Object.freeze({
  CROSS_CIPHERTEXT_ADD: "cross-ciphertext-add",
  ROTATION_ADD: "rotation-add",
});

// New empty metric block (spec §9.8). All fields are integers; "counts" are
// illustrative (educational) unless the trace comes from the Rotom compiler.
export function emptyMetrics() {
  return {
    rotations: 0,
    additions: 0,
    ptCtMultiplications: 0,
    ctCtMultiplications: 0,
    masks: 0,
    relinearizations: 0,
    rescales: 0,
    multiplicativeDepth: 0,
    ciphertextCount: 0,
  };
}
