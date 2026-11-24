import test from "node:test";
import assert from "node:assert/strict";

import { validateTrace } from "../transformer/core/trace_validator.js";
import { matricesEqual } from "../transformer/core/matrix.js";
import {
  STIP_OPERATION_IDS,
  buildStipOperationTrace,
  getStipOperation,
  listStipOperations,
} from "../transformer/algorithms/stip/transformer_operations.js";

test("STIP registry covers five packing primitives and every Transformer matrix multiplication", () => {
  assert.equal(listStipOperations().length, 14);
  for (const id of Object.values(STIP_OPERATION_IDS)) assert.ok(getStipOperation(id), id);
});

test("all STIP traces are valid editable two-batch eight-slot toys", () => {
  for (const operation of listStipOperations()) {
    const trace = buildStipOperationTrace(operation.id);
    const validation = validateTrace(trace);
    assert.equal(validation.ok, true, `${operation.id}: ${validation.errors.join(", ")}`);
    assert.equal(trace.parameters.batchCount, 2);
    assert.equal(trace.parameters.slotsPerCiphertext, 8);
    assert.equal(trace.plan.method, "stip");
    assert.equal(trace.plan.visualStyle, "diagram-only");
  }
});

test("RIHP packs two score diagonals in real and imaginary parts before conjugate unpack", () => {
  const trace = buildStipOperationTrace(STIP_OPERATION_IDS.SCORE);
  assert.deepEqual(trace.graph.stages.map((stage) => stage.title), [
    "Q × Kᵀ", "Col", "K + i·RotL₁(K)", "RIHP ⊙ Σ", "½(C+Conj C) · −½i(C−Conj C)", "P",
  ]);
  assert.ok(matricesEqual(trace.finalTensor, [[6, 8], [1, 3]]));
});

test("main STIP L2 path produces diagonal-packed squared distances", () => {
  const trace = buildStipOperationTrace(STIP_OPERATION_IDS.DISTANCE);
  assert.deepEqual(trace.graph.stages.map((stage) => stage.title), [
    "Squared distance between Q and K", "RIHP", "z ⊙ Conj(z)", "Σ q̂ ⊙ Rot(k̂*)", "Re(S_Q + Rot(S_K) − 2C)", "D²",
  ]);
  assert.ok(matricesEqual(trace.finalTensor, [[10, 2], [16, 8]]));
  const edited = buildStipOperationTrace(STIP_OPERATION_IDS.L2, { A: [[2, 3], [7, 11]], B: [[5, 13], [17, 19]] });
  const cross = edited.graph.stages[3].rows[1].at(-1);
  assert.equal(cross.cells[0].value, 91);
  assert.equal(cross.cells[0].imaginary, 13);
});

test("input projection shows sequence-major AMCP, group rotation, DHP and conjugate unpack", () => {
  const trace = buildStipOperationTrace(STIP_OPERATION_IDS.QUERY);
  assert.deepEqual(trace.graph.stages.map((stage) => stage.title), [
    "X × Wq", "AMCP", "RotL₂ · u + RotR₂ · v", "DHP ⊙ Σ", "Conj", "Q",
  ]);
  const amcp = trace.graph.stages[1].rows[0][0];
  assert.deepEqual(amcp.cells.map((cell) => cell.label), ["1", "1", "3", "3", "4", "4", "0", "0"]);
  assert.deepEqual(amcp.cells.map((cell) => cell.sampleLane), [0, 1, 0, 1, 0, 1, 0, 1]);
  assert.deepEqual(amcp.cells.map((cell) => cell.featureIndex), [0, 0, 1, 1, 0, 0, 1, 1]);
});

test("STIP colors follow column, diagonal, and folded-head identities", () => {
  const projection = buildStipOperationTrace(STIP_OPERATION_IDS.QUERY);
  const projectionInput = projection.graph.stages[0].rows[0];
  assert.deepEqual(projectionInput[0].rows.flat().map((cell) => cell.tone), [0, 1, 2, 3, 0, 1, 2, 3]);
  assert.deepEqual(projectionInput[2].rows.flat().map((cell) => cell.tone), [0, 1, 0, 1, 0, 1, 0, 1]);

  const score = buildStipOperationTrace(STIP_OPERATION_IDS.SCORE);
  const scoreInput = score.graph.stages[0].rows[0];
  assert.deepEqual(scoreInput[0].rows.flat().map((cell) => cell.tone), [0, 1, 0, 1]);
  assert.deepEqual(scoreInput[2].rows.flat().map((cell) => cell.tone), [2, 2, 3, 3]);

  const context = buildStipOperationTrace(STIP_OPERATION_IDS.CONTEXT);
  const contextInput = context.graph.stages[0].rows[0];
  assert.deepEqual(contextInput[0].rows.flat().map((cell) => cell.tone), [0, 1, 1, 0]);
  assert.deepEqual(contextInput[2].rows.flat().map((cell) => cell.tone), [2, 3, 2, 3]);
});

test("GK·V uses RIHP value-column pairing and BSGS", () => {
  const trace = buildStipOperationTrace(STIP_OPERATION_IDS.CONTEXT);
  assert.deepEqual(trace.graph.stages.map((stage) => stage.title), [
    "GK(D²) × V", "Diag + RIHP", "BSGS β", "Σ_g Rot(Σ_b G ⊙ β)", "Conj", "H",
  ]);
  assert.ok(matricesEqual(trace.finalTensor, [[1.875, 3.125], [1.25, 2.5]]));
});

test("each STIP ciphertext is rendered as one uniform color block", () => {
  for (const operation of listStipOperations()) {
    const trace = buildStipOperationTrace(operation.id);
    for (const stage of trace.graph.stages) for (const row of stage.rows) for (const artifact of row) {
      if (artifact.badge !== "CT" || !artifact.cells) continue;
      assert.equal(new Set(artifact.cells.map((cell) => cell.tone)).size, 1, `${operation.id}/${artifact.id}`);
    }
  }
});

test("editable STIP values rebuild the result", () => {
  const trace = buildStipOperationTrace(STIP_OPERATION_IDS.FC1, { A: [[2, 0], [0, 2]] });
  assert.ok(matricesEqual(trace.finalTensor, [[2, 0, 4, 6], [4, 2, 6, 0]]));
});
