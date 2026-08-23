import test from "node:test";
import assert from "node:assert/strict";

import { matricesEqual } from "../transformer/core/matrix.js";
import { validateTrace } from "../transformer/core/trace_validator.js";
import {
  NEXUS_OPERATION_IDS,
  buildNexusOperationTrace,
  getNexusOperation,
  listNexusOperations,
} from "../transformer/algorithms/nexus/transformer_operations.js";

test("NEXUS registry covers three paper primitives and all eight Transformer matrix multiplications", () => {
  assert.equal(listNexusOperations().length, 11);
  for (const id of Object.values(NEXUS_OPERATION_IDS)) assert.ok(getNexusOperation(id), id);
});

test("all NEXUS traces are valid editable two-batch eight-slot graphical toys", () => {
  for (const operation of listNexusOperations()) {
    const trace = buildNexusOperationTrace(operation.id);
    const validation = validateTrace(trace);
    assert.equal(validation.ok, true, `${operation.id}: ${validation.errors.join(", ")}`);
    assert.equal(trace.parameters.batchCount, 2);
    assert.equal(trace.parameters.slotsPerCiphertext, 8);
    assert.equal(trace.plan.method, "nexus");
    assert.equal(trace.plan.visualStyle, "diagram-only");
  }
});

test("NEXUS compression expands polynomial coefficients into repeated SIMD weight ciphertexts", () => {
  const trace = buildNexusOperationTrace(NEXUS_OPERATION_IDS.DECOMPRESS);
  assert.deepEqual(trace.graph.stages.map((stage) => stage.title), [
    "W", "Compress", "⊞ Subs · x⁻²ⁱ", "Decompress", "ENC([w,…,w])",
  ]);
  const repeated = trace.graph.stages[4].rows[0];
  for (const ciphertext of repeated) {
    assert.equal(new Set(ciphertext.cells.map((cell) => cell.value)).size, 1);
  }
});

test("NEXUS colors follow column, row-packed, and compressed-weight identities", () => {
  const linear = buildNexusOperationTrace(NEXUS_OPERATION_IDS.QUERY);
  const linearInput = linear.graph.stages[0].rows[0];
  assert.deepEqual(linearInput[0].rows.flat().map((cell) => cell.tone), [0, 1, 2, 3, 0, 1, 2, 3]);
  assert.deepEqual(linearInput[2].rows.flat().map((cell) => cell.tone), [0, 1, 0, 1, 0, 1, 0, 1]);

  const score = buildNexusOperationTrace(NEXUS_OPERATION_IDS.SCORE);
  const scoreInput = score.graph.stages[0].rows[0];
  assert.deepEqual(scoreInput[0].rows.flat().map((cell) => cell.tone), [0, 1, 0, 1]);
  assert.deepEqual(scoreInput[2].rows.flat().map((cell) => cell.tone), [2, 3, 2, 3]);

  const context = buildNexusOperationTrace(NEXUS_OPERATION_IDS.CONTEXT);
  const contextInput = context.graph.stages[0].rows[0];
  assert.deepEqual(contextInput[0].rows.flat().map((cell) => cell.tone), [0, 0, 1, 1]);
  assert.deepEqual(contextInput[2].rows.flat().map((cell) => cell.tone), [2, 3, 2, 3]);
});

test("NEXUS offline-online MatrixMul reconstructs AW from masked online product and encrypted UW", () => {
  const trace = buildNexusOperationTrace(NEXUS_OPERATION_IDS.OFFLINE_ONLINE);
  assert.deepEqual(trace.graph.stages.map((stage) => stage.title), [
    "A × W", "Compress", "⊞ Subs · x⁻²ⁱ", "Decompress", "Offline U·W",
    "Online (A−U)W ⊞ ENC(UW)", "Q",
  ]);
  assert.ok(matricesEqual(trace.finalTensor, [[1, 3], [2, 4]]));
});

test("NEXUS Algorithm 5 rotates K columns and produces diagonal-packed QK transpose", () => {
  const trace = buildNexusOperationTrace(NEXUS_OPERATION_IDS.SCORE);
  assert.deepEqual(trace.graph.stages.map((stage) => stage.title), [
    "Q × Kᵀ", "Column pack", "RotL₂(K)", "Σ_j q_j ⊠ RotL(k_j)", "P",
  ]);
  assert.ok(matricesEqual(trace.finalTensor, [[1, 5], [0, 2]]));
});

test("NEXUS row-packed Softmax times V uses slot products and QuickSum", () => {
  const trace = buildNexusOperationTrace(NEXUS_OPERATION_IDS.CONTEXT);
  assert.deepEqual(trace.graph.stages.map((stage) => stage.title), [
    "Softmax(P) × V", "Row pack", "⊠", "QuickSum", "H",
  ]);
  assert.ok(matricesEqual(trace.finalTensor, [[1.25, 0.25], [1.8, 0.8]]));
});

test("each NEXUS ciphertext is rendered as one uniform color block", () => {
  for (const operation of listNexusOperations()) {
    const trace = buildNexusOperationTrace(operation.id);
    for (const stage of trace.graph.stages) for (const row of stage.rows) for (const artifact of row) {
      if (artifact.badge !== "CT" || !artifact.cells) continue;
      assert.equal(new Set(artifact.cells.map((cell) => cell.tone)).size, 1, `${operation.id}/${artifact.id}`);
    }
  }
});

test("editable NEXUS values rebuild the final result", () => {
  const trace = buildNexusOperationTrace(NEXUS_OPERATION_IDS.FC1, { A: [[2, 0], [0, 2]] });
  assert.ok(matricesEqual(trace.finalTensor, [[2, 0, 2, 0], [0, 2, 2, 4]]));
});
