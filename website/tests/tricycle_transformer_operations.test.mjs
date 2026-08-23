import test from "node:test";
import assert from "node:assert/strict";

import { matricesEqual } from "../transformer/core/matrix.js";
import { validateTrace } from "../transformer/core/trace_validator.js";
import {
  TRICYCLE_OPERATION_IDS,
  buildTricycleOperationTrace,
  getTricycleOperation,
  listTricycleOperations,
} from "../transformer/algorithms/tricycle/transformer_operations.js";

test("Tricycle registry covers four paper primitives and all eleven implemented model matrix multiplications", () => {
  assert.equal(listTricycleOperations().length, 15);
  for (const id of Object.values(TRICYCLE_OPERATION_IDS)) assert.ok(getTricycleOperation(id), id);
});

test("all Tricycle traces are valid editable 32-slot graphical toys", () => {
  for (const operation of listTricycleOperations()) {
    const trace = buildTricycleOperationTrace(operation.id);
    const validation = validateTrace(trace);
    assert.equal(validation.ok, true, `${operation.id}: ${validation.errors.join(", ")}`);
    assert.equal(trace.parameters.slotsPerCiphertext, 32);
    assert.equal(trace.plan.method, "tricycle");
    assert.equal(trace.plan.visualStyle, "diagram-only");
    assert.equal(trace.plan.packingStrategyRequested, "tricyclic");
  }
});

test("tricyclic encoding follows slot k to tensor indices modulo h,m,n", () => {
  const trace = buildTricycleOperationTrace(TRICYCLE_OPERATION_IDS.QUERY);
  const phiX = trace.graph.stages[1].rows[0][0];
  assert.match(phiX.cells[0].label, /h0r0c0/);
  assert.match(phiX.cells[1].label, /h0r1c1/);
  assert.match(phiX.cells[2].label, /h0r2c2/);
  assert.match(phiX.cells[3].label, /h0r0c3/);
  assert.match(phiX.cells[5].label, /h0r2c0/);
});

test("Tricycle colors follow tricyclic columns, head pairs, and packed attention operands", () => {
  const query = buildTricycleOperationTrace(TRICYCLE_OPERATION_IDS.QUERY);
  const queryInput = query.graph.stages[0].rows[0];
  assert.deepEqual(queryInput[0].rows.flat().map((cell) => cell.tone), [0, 1, 2, 3, 0, 0, 1, 2, 3, 0, 0, 1, 2, 3, 0]);
  assert.deepEqual(queryInput[2].rows.flat().map((cell) => cell.tone), [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);

  const score = buildTricycleOperationTrace(TRICYCLE_OPERATION_IDS.SCORE);
  const scoreInput = score.graph.stages[0].rows[0];
  assert.deepEqual(scoreInput[0].rows.flat().map((cell) => cell.tone), [0, 1, 0, 1, 0, 1]);
  assert.deepEqual(scoreInput[2].rows.flat().map((cell) => cell.tone), [2, 3, 0, 1, 2, 2, 3, 0, 1, 2]);

  const context = buildTricycleOperationTrace(TRICYCLE_OPERATION_IDS.CONTEXT);
  const contextInput = context.graph.stages[0].rows[0];
  assert.deepEqual(contextInput[0].rows.flat().map((cell) => cell.tone), [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]);
  assert.deepEqual(contextInput[2].rows.flat().map((cell) => cell.tone), [2, 3, 2, 3, 2, 3, 2, 3, 2, 3]);
});

test("QK transpose is free, masks Q rotations, and relinearizes only after baby-step summation", () => {
  const trace = buildTricycleOperationTrace(TRICYCLE_OPERATION_IDS.SCORE);
  assert.deepEqual(trace.graph.stages.map((stage) => stage.title), [
    "Q × K′ᵀ", "φ(h,m,n)", "φ(K′) = φ(K′ᵀ)", "ρ(Q) ⊙ M/√dk",
    "Right-BSGS ⊠", "Σbaby → Relinearize → ρgiant", "½(C + Conj(C))", "P",
  ]);
  const [before, , after] = trace.graph.stages[2].rows[0];
  assert.deepEqual(before.cells.map((cell) => cell.value), after.cells.map((cell) => cell.value));
  assert.equal(trace.steps[4].metricsDelta.relinearizations, 0);
  assert.equal(trace.steps[5].metricsDelta.relinearizations, 1);
});

test("KV reshape is the tall-identity PT×CT packing change and adds two zero rows", () => {
  const trace = buildTricycleOperationTrace(TRICYCLE_OPERATION_IDS.KV_RESHAPE);
  assert.equal(trace.parameters.matmulType, "pt-ct");
  assert.deepEqual(trace.finalTensor.slice(0, 3), [[1, 0], [0, 1], [1, 1]]);
  assert.deepEqual(trace.finalTensor.slice(3), [[0, 0], [0, 0]]);
  assert.match(trace.plan.packingStrategyResolved, /tall-identity reshape/);
});

test("Softmax times V masks V and uses left-BSGS CT×CT batch multiplication", () => {
  const trace = buildTricycleOperationTrace(TRICYCLE_OPERATION_IDS.CONTEXT);
  assert.equal(trace.parameters.matmulType, "ct-ct");
  assert.match(trace.graph.stages[2].title, /V′/);
  assert.equal(trace.graph.stages[3].title, "Left-BSGS ⊠");
  assert.equal(trace.parameters.batchCount, 2);
});

test("attention output uses concatenated multiplication and conjugation across heads", () => {
  const trace = buildTricycleOperationTrace(TRICYCLE_OPERATION_IDS.OUTPUT);
  assert.match(trace.graph.stages[3].title, /Σhead/);
  assert.equal(trace.graph.stages[5].title, "C + Conj(C)");
  assert.match(trace.plan.packingStrategyResolved, /across heads/);
});

test("FF1 decomplexifies before GELU while FF2 recomplexifies then extracts real output", () => {
  const ff1 = buildTricycleOperationTrace(TRICYCLE_OPERATION_IDS.FC1);
  const ff2 = buildTricycleOperationTrace(TRICYCLE_OPERATION_IDS.FC2);
  assert.ok(ff1.graph.stages.some((stage) => /−iC/.test(stage.title)));
  assert.ok(ff2.graph.stages.some((stage) => stage.title === "FL+iFR"));
  assert.ok(ff2.graph.stages.some((stage) => stage.title === "C + Conj(C)"));
});

test("optimized block BSGS reuses baby rotations and aggregates equal giant steps", () => {
  const trace = buildTricycleOperationTrace(TRICYCLE_OPERATION_IDS.BLOCK_BSGS);
  assert.deepEqual(trace.graph.stages.map((stage) => stage.title), [
    "4×4 · 4×2", "2×2 blocks", "Shared baby rotations", "Aggregate equal giant steps", "ρgiant → lazy Relin", "Z",
  ]);
  assert.ok(matricesEqual(trace.finalTensor, [[1, 3], [2, 4], [2, 3], [2, 2]]));
});

test("each displayed Tricycle ciphertext is one uniform color block", () => {
  for (const operation of listTricycleOperations()) {
    const trace = buildTricycleOperationTrace(operation.id);
    for (const stage of trace.graph.stages) for (const row of stage.rows) for (const artifact of row) {
      if (artifact.badge !== "CT" || !artifact.cells) continue;
      assert.equal(new Set(artifact.cells.map((cell) => cell.tone)).size, 1, `${operation.id}/${artifact.id}`);
    }
  }
});

test("editable Tricycle values rebuild the direct matrix result", () => {
  const trace = buildTricycleOperationTrace(TRICYCLE_OPERATION_IDS.CLASSIFIER, { A: [[2, 0, 1]] });
  assert.ok(matricesEqual(trace.finalTensor, [[3, 1]]));
});
