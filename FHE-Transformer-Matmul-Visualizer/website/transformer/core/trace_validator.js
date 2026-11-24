// Trace validator (spec §19). The scene builder runs this over every trace it
// produces; tests exercise it with both valid and tampered traces.

import { SCHEMA_VERSION } from "./trace_schema.js";
import { matmul, matricesEqual, squaredL2Distances } from "./matrix.js";
import { isTotalGEQ } from "./metrics.js";

/**
 * Validate a complete trace.
 * @returns {{ok:boolean, errors:string[]}}
 */
export function validateTrace(trace) {
  const errors = [];

  if (!trace || typeof trace !== "object") {
    return { ok: false, errors: ["trace is not an object"] };
  }
  if (trace.schemaVersion !== SCHEMA_VERSION) {
    errors.push(`schemaVersion must be ${SCHEMA_VERSION}`);
  }
  if (!trace.id || typeof trace.id !== "string") errors.push("trace.id is required");
  if (!Array.isArray(trace.steps) || trace.steps.length < 2) {
    errors.push("trace.steps must be a non-empty array (at least reference + output)");
  }

  const params = trace.parameters || {};
  const { m, k, n } = params.matrixShape || {};
  const nSlots = params.slotsPerCiphertext;

  if (!(m > 0 && k > 0 && n > 0)) errors.push("parameters.matrixShape must be positive {m,k,n}");
  if (!Number.isInteger(nSlots) || nSlots <= 0) {
    errors.push("parameters.slotsPerCiphertext is required");
  }

  // Step ids unique + contiguous indices.
  const seenIds = new Set();
  (trace.steps || []).forEach((step, idx) => {
    if (!step || typeof step !== "object") {
      errors.push(`step[${idx}] is not an object`);
      return;
    }
    if (typeof step.id !== "string" || !step.id) errors.push(`step[${idx}].id is required`);
    if (seenIds.has(step.id)) errors.push(`duplicate step id "${step.id}"`);
    seenIds.add(step.id);
    if (step.index !== idx) errors.push(`step[${idx}].index must equal ${idx} (got ${step.index})`);
    if (step.phase == null) errors.push(`step[${idx}].phase is required`);
  });

  // Slot range + ciphertext slot counts + coords in range.
  for (const step of trace.steps || []) {
    for (const ct of step.ciphertexts || []) {
      if (!ct || !Array.isArray(ct.slots)) {
        errors.push(`step "${step.id}": ciphertext has no slots array`);
        continue;
      }
      if (ct.slots.length !== nSlots) {
        errors.push(
          `step "${step.id}": ciphertext "${ct.id}" has ${ct.slots.length} slots, expected ${nSlots}`
        );
      }
      for (const s of ct.slots) {
        if (!Number.isInteger(s.slot) || s.slot < 0 || s.slot >= nSlots) {
          errors.push(`step "${step.id}": slot index ${s.slot} out of range`);
        }
        if (s.sourceRefs && Array.isArray(s.sourceRefs)) {
          for (const ref of s.sourceRefs) {
            if (ref.tensorId && ref.coords) {
              const shape = trace.operands?.[ref.tensorId]?.shape || null;
              if (shape) {
                for (let d = 0; d < ref.coords.length; d++) {
                  if (ref.coords[d] >= shape[d]) {
                    errors.push(
                      `step "${step.id}": sourceRef coords ${JSON.stringify(ref.coords)} out of range for ${ref.tensorId} shape ${JSON.stringify(shape)}`
                    );
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // Metrics totals monotone.
  let prevTotal = null;
  for (const step of trace.steps || []) {
    if (step.metricsTotal) {
      if (prevTotal && !isTotalGEQ(prevTotal, step.metricsTotal)) {
        errors.push(`step "${step.id}": metricsTotal decreased`);
      }
      prevTotal = step.metricsTotal;
    }
  }

  // Check against the direct numerical operation on the displayed inputs.
  const left = trace.operands?.left;
  const right = trace.operands?.right;
  if (left && right && Array.isArray(left.values) && Array.isArray(right.values)) {
    try {
      const operation = params.resultOperation === "squared-l2" ? squaredL2Distances : matmul;
      const result = operation(left.values, right.values);
      const expected = params.resultScale == null ? result : result.map((row) => row.map((value) => value * params.resultScale));
      if (!matricesEqual(trace.finalTensor, expected)) {
        errors.push(`trace.finalTensor does not match direct ${operation === matmul ? "matmul(A, B)" : "squared L2 distances"}`);
      }
    } catch (err) {
      errors.push(`final check failed: ${err.message}`);
    }
  }

  return { ok: errors.length === 0, errors };
}
