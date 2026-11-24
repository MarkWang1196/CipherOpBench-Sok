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
  const edited = buildNexusOperationTrace(NEXUS_OPERATION_IDS.DECOMPRESS, { B: [[1.5, -2.5], [3.25, 9]] });
  const oddBranch = edited.graph.stages[2].rows[0].at(-1);
  assert.deepEqual(oddBranch.cells.map((cell) => cell.value), [6.5, 0, 18, 0]);
  assert.deepEqual(edited.graph.stages[4].rows[0].map((ct) => ct.cells[0].value), [1.5, 3.25, -2.5, 9]);
});

test("NEXUS colors follow column, row-packed, and compressed-weight identities", () => {
  const linear = buildNexusOperationTrace(NEXUS_OPERATION_IDS.QUERY);
  const linearInput = linear.graph.stages[0].rows[0];
  assert.deepEqual(linearInput[0].rows.flat().map((cell) => cell.tone), [0, 1, 2, 3, 0, 1, 2, 3]);
  assert.deepEqual(linearInput[2].rows.flat().map((cell) => cell.tone), [0, 1, 0, 1, 0, 1, 0, 1]);

  const score = buildNexusOperationTrace(NEXUS_OPERATION_IDS.SCORE);
  const scoreInput = score.graph.stages[0].rows[0];
  assert.deepEqual(scoreInput[0].rows.flat().map((cell) => cell.tone), [0, 1, 0, 1]);
  assert.deepEqual(scoreInput[2].rows.flat().map((cell) => cell.tone), [2, 2, 3, 3]);

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
  assert.ok(matricesEqual(trace.finalTensor, [[5, 8], [12, 3]]));
});

test("NEXUS Algorithm 5 rotates K columns and produces diagonal-packed QK transpose", () => {
  const trace = buildNexusOperationTrace(NEXUS_OPERATION_IDS.SCORE);
  assert.deepEqual(trace.graph.stages.map((stage) => stage.title), [
    "Q × Kᵀ", "Column pack", "RotL₂(K)", "Σ_j q_j ⊠ RotL(k_j)", "P",
  ]);
  assert.ok(matricesEqual(trace.finalTensor, [[6, 8], [1, 3]]));
  const keySlots = trace.graph.stages[2].rows.map(([key, , rotated]) => [
    key.cells.map((cell) => cell.value),
    rotated.cells.map((cell) => cell.value),
  ]);
  assert.deepEqual(keySlots, [
    [[1, 1, 3, 3, 1, 1, 3, 3], [3, 3, 1, 1, 3, 3, 1, 1]],
    [[4, 4, 2, 2, 4, 4, 2, 2], [2, 2, 4, 4, 2, 2, 4, 4]],
  ]);
  for (const [key, , rotated] of trace.graph.stages[2].rows) {
    assert.deepEqual(key.cells.map((cell) => cell.sampleLane), [0, 1, 0, 1, 0, 1, 0, 1]);
    assert.deepEqual(rotated.cells.map((cell) => cell.sampleLane), [0, 1, 0, 1, 0, 1, 0, 1]);
  }
  const edited = buildNexusOperationTrace(NEXUS_OPERATION_IDS.SCORE, { A: [[2, 3], [7, 11]], B: [[5, 13], [17, 19]] });
  const offDiagonal = edited.graph.stages[3].rows[1].at(-1);
  assert.deepEqual(offDiagonal.cells.map((cell) => cell.value), [91, 91, 178, 178, 91, 91, 178, 178]);
});

test("NEXUS row-packed Softmax times V uses slot products and QuickSum", () => {
  const trace = buildNexusOperationTrace(NEXUS_OPERATION_IDS.CONTEXT);
  assert.deepEqual(trace.graph.stages.map((stage) => stage.title), [
    "Softmax(P) × V", "Row pack", "⊠", "QuickSum", "H",
  ]);
  assert.ok(matricesEqual(trace.finalTensor, [[1.75, 1.5], [1.375, 2.25]]));
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
  assert.equal(trace.parameters.matmulType, "ct-pt");
  assert.ok(matricesEqual(trace.finalTensor, [[2, 0, 4, 6], [4, 2, 6, 0]]));
});
