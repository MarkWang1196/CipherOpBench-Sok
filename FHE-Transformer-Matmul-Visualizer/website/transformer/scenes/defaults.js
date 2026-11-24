// Default scenes (spec §12, §13). These are *presets* only: the two default
// scenes are generated at runtime by the generic buildMatmulScene(), so there
// is no 4×4 hardcoding in the planner or scene builder. The 4×4 data here is a
// convenient, verifiable default (spec §12.1: "4×4 仅作为默认演示和回归测试").

export const DEFAULT_X = [
  [1, 2, 0, 1],
  [0, 1, 3, 2],
  [2, 0, 1, 1],
  [1, 1, 2, 0],
];

export const DEFAULT_WQ = [
  [1, 0, 2, 1],
  [0, 1, 1, 2],
  [2, 1, 0, 1],
  [1, 2, 1, 0],
];

export const DEFAULT_K = [
  [1, 1, 0, 2],
  [0, 2, 1, 0],
  [2, 0, 1, 1],
  [1, 1, 0, 2],
];

/**
 * Scene descriptors. `transposeRight` turns a CT×CT "Q · Kᵀ" into the generic
 * builder's right operand by transposing K first (the trace then shows the
 * transpose/permute step).
 */
export const SCENES = Object.freeze({
  xwq: {
    id: "xwq",
    label: "Q = X · Wq",
    shortLabel: "XWq",
    description: "CT × PT query projection",
    left: { id: "X", label: "Encrypted activations X", values: DEFAULT_X, secret: true },
    right: { id: "Wq", label: "Query weights Wq", values: DEFAULT_WQ, secret: false },
    outputId: "Q",
    outputLabel: "Queries Q",
    transposeRight: false,
    slotsPerCiphertext: 16,
    strategy: "auto",
  },
  qkt: {
    id: "qkt",
    label: "S = Q · Kᵀ",
    shortLabel: "QKᵀ",
    description: "CT × CT attention scores",
    left: { id: "Q", label: "Queries Q", values: DEFAULT_X, secret: true },
    right: { id: "K", label: "Keys K", values: DEFAULT_K, secret: true },
    outputId: "S",
    outputLabel: "Attention scores S",
    transposeRight: true,
    slotsPerCiphertext: 16,
    strategy: "auto",
  },
});

export const SCENE_KEYS = Object.keys(SCENES);
