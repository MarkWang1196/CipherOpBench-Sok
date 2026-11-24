// Adapter between the shared packing engine (`website/layout_engine.js`) and the
// visualizer's trace SlotState / CiphertextState schema (spec §10, §9.4-§9.5).
//
// This is the ONLY module that calls `Layout.fromString` / `applyLayout`.
// It owns:
//   * the id vs logicalId split (spec §15.2): `id` is unique per step,
//     `logicalId` is stable across steps and used for FLIP matching.
//   * replica numbering (spec §10): replicated slots of the same source get
//     increasing replica indices in visit order.
//   * the "gap"/"oob" slot kinds reported by the engine.

import { Layout, applyLayout } from "../../layout_engine.js";

/** Illustrative ciphertext metadata (not real CKKS state). */
const ILLUSTRATIVE = { level: 5, scale: "Δ", degree: 2, illustrative: true };

/**
 * Build CiphertextState[] for a tensor packed by a layout string.
 *
 * @param {object} cfg
 * @param {string} cfg.tensorId  logical tensor id, e.g. "X"
 * @param {Array}  cfg.tensor    the tensor to pack (any shape applyLayout supports)
 * @param {string} cfg.layoutString packing-layout DSL string
 * @param {number} cfg.n         slots per ciphertext
 * @param {boolean} cfg.secret   ciphertext (true) vs plaintext (false)
 * @param {string} cfg.stepId    current step id (used in slot ids)
 * @param {string} cfg.idPrefix  ciphertext id prefix, e.g. "X-pack"
 * @param {(mapping, ctIndex, slotIndex)=>string} cfg.logicalIdFor
 * @param {(mapping, ctIndex, slotIndex)=>Array} cfg.sourceRefFor
 * @param {(mapping, ctIndex, slotIndex)=>string} cfg.expressionFor
 */
export function buildCiphertextStates({
  tensorId,
  tensor,
  layoutString,
  n,
  secret,
  stepId,
  idPrefix,
  logicalIdFor,
  sourceRefFor,
  expressionFor,
}) {
  const layout = Layout.fromString(layoutString, n, Boolean(secret));
  const [packed, mappings] = applyLayout(tensor, layout, { returnSlotMapping: true });

  return packed.map((values, ctIndex) => {
    const ctId = `${idPrefix}-ct-${ctIndex}`;
    return {
      id: ctId,
      label: `${secret ? "Ciphertext" : "Plaintext"} ${tensorId} #${ctIndex}`,
      secret: Boolean(secret),
      layout: layout.layoutStr(),
      ...ILLUSTRATIVE,
      slots: values.map((value, slotIndex) => {
        const mapping = mappings[ctIndex][slotIndex];
        const kind = mapping.kind;
        const isTensor = kind === "tensor";
        const sourceRefs = isTensor
          ? sourceRefFor
            ? sourceRefFor(mapping, ctIndex, slotIndex)
            : [
                {
                  tensorId,
                  coords: mapping.coords,
                  linear: mapping.linear,
                  logicalId: `${tensorId}:${mapping.linear}`,
                },
              ]
          : [];
        const logicalId = isTensor
          ? logicalIdFor
            ? logicalIdFor(mapping, ctIndex, slotIndex)
            : `${idPrefix}:${mapping.linear}`
          : `${idPrefix}:${kind}:${ctIndex}:${slotIndex}`;
        const expression = isTensor
          ? expressionFor
            ? expressionFor(mapping, ctIndex, slotIndex)
            : mapping.label || ""
          : mapping.label || "";
        return {
          id: `${stepId}:${idPrefix}:ct-${ctIndex}:slot-${slotIndex}`,
          logicalId,
          ciphertextId: ctId,
          slot: slotIndex,
          value,
          kind,
          sourceRefs,
          outputCoords: null,
          contractionIndex: null,
          expression,
        };
      }),
    };
  });
}

/**
 * Build CiphertextState[] from explicit slot descriptor arrays.
 *
 * Used for steps the engine can't express cleanly (plaintext "encode to
 * align", slot-wise products, reduction sums). Each descriptor is:
 *   { value, logicalId?, kind?, sourceRefs?, outputCoords?,
 *     contractionIndex?, expression? }
 * The caller supplies full provenance — the adapter only wraps it in the
 * schema, so every step stays independently reconstructable from the trace.
 *
 * @param {object} cfg
 * @param {string} cfg.id        logical ciphertext family, e.g. "P"
 * @param {boolean} cfg.secret
 * @param {string} cfg.stepId
 * @param {string|null} cfg.layout  optional layout string (for display)
 * @param {Array<Array<object>>} cfg.arrays  one array of slot descriptors per ciphertext
 * @param {(ctIndex)=>string} cfg.labelFor  optional per-ct label
 */
export function buildCiphertextStatesFromArrays({
  id,
  secret,
  stepId,
  layout = null,
  arrays,
  labelFor,
}) {
  return arrays.map((slots, ctIndex) => {
    const ctId = `${id}-ct-${ctIndex}`;
    return {
      id: ctId,
      label: labelFor
        ? labelFor(ctIndex)
        : `${secret ? "Ciphertext" : "Plaintext"} ${id} #${ctIndex}`,
      secret: Boolean(secret),
      layout,
      ...ILLUSTRATIVE,
      slots: slots.map((s, slotIndex) => ({
        id: `${stepId}:${id}:ct-${ctIndex}:slot-${slotIndex}`,
        logicalId: s.logicalId || `${id}:${ctIndex}:${slotIndex}`,
        ciphertextId: ctId,
        slot: slotIndex,
        value: s.value,
        kind: s.kind || "tensor",
        sourceRefs: s.sourceRefs || [],
        outputCoords: s.outputCoords != null ? s.outputCoords : null,
        contractionIndex: s.contractionIndex != null ? s.contractionIndex : null,
        expression: s.expression != null ? s.expression : "",
      })),
    };
  });
}
