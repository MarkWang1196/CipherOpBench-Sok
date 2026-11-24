import test from "node:test";
import assert from "node:assert/strict";

import { matricesEqual, matmul, transpose } from "../transformer/core/matrix.js";
import { validateTrace } from "../transformer/core/trace_validator.js";
import {
  EUSTON_OPERATION_IDS,
  buildEustonOperationTrace,
  getEustonOperation,
  listEustonOperations,
  parseEustonOperationInputs,
} from "../transformer/algorithms/euston/transformer_operations.js";

test("Euston registry covers four paper primitives and eight Transformer matrix multiplications", () => {
  assert.equal(listEustonOperations().length, 12);
  for (const id of Object.values(EUSTON_OPERATION_IDS)) assert.ok(getEustonOperation(id));
});

test("all Euston traces are valid editable two-batch toys", () => {
  for (const operation of listEustonOperations()) {
    const trace = buildEustonOperationTrace(operation.id);
    assert.equal(validateTrace(trace).ok, true, operation.id);
    assert.equal(trace.parameters.batchCount, 2);
    assert.equal(trace.parameters.slotsPerCiphertext, 8);
    assert.equal(trace.plan.visualStyle, "diagram-only");
    const B = operation.kind === "cccrmm" ? transpose(operation.B) : operation.B;
    assert.ok(matricesEqual(trace.finalTensor, matmul(operation.A, B)), operation.id);
  }
});

test("PCCDMM performs SecComp, SecDecomp and column-wise PT x CT multiplication", () => {
  const trace = buildEustonOperationTrace(EUSTON_OPERATION_IDS.PCCDMM);
  assert.deepEqual(trace.graph.stages.map((stage) => stage.title), ["U × D", "SecComp", "SecDecomp", "PCCDMM", "E"]);
  assert.equal(trace.parameters.matmulType, "pt-ct");
  assert.ok(matricesEqual(trace.finalTensor, [[2, 8], [6, 0]]));
  const secondColumn = trace.graph.stages.at(-1).rows[0][1];
  assert.deepEqual(secondColumn.cells.map((cell) => cell.label), ["8", "0", "8", "0", "·", "·", "·", "·"]);
  assert.deepEqual(secondColumn.cells.map((cell) => cell.sampleLane), [0, 0, 1, 1, undefined, undefined, undefined, undefined]);
});

test("Euston colors follow column, row, and diagonal packing identities", () => {
  const pccd = buildEustonOperationTrace(EUSTON_OPERATION_IDS.PCCDMM);
  const pccdInput = pccd.graph.stages[0].rows[0];
  assert.deepEqual(pccdInput[0].rows.flat().map((cell) => cell.tone), [0, 1, 0, 1]);
  assert.deepEqual(pccdInput[2].rows.flat().map((cell) => cell.tone), [0, 1, 1, 0]);

  const score = buildEustonOperationTrace(EUSTON_OPERATION_IDS.SCORE);
  const scoreInput = score.graph.stages[0].rows[0];
  assert.deepEqual(scoreInput[0].rows.flat().map((cell) => cell.tone), [0, 1, 0, 1]);
  assert.deepEqual(scoreInput[2].rows.flat().map((cell) => cell.tone), [2, 2, 3, 3]);

  const context = buildEustonOperationTrace(EUSTON_OPERATION_IDS.CONTEXT);
  const contextInput = context.graph.stages[0].rows[0];
  assert.deepEqual(contextInput[0].rows.flat().map((cell) => cell.tone), [0, 1, 1, 0]);
  assert.deepEqual(contextInput[2].rows.flat().map((cell) => cell.tone), [0, 1, 0, 1]);
});

test("IWMM visualizes SVD offline precomputation and online one-time-pad correction", () => {
  const trace = buildEustonOperationTrace(EUSTON_OPERATION_IDS.QUERY);
  assert.deepEqual(trace.graph.stages.map((stage) => stage.title), [
    "A × Wq", "R = U · D · H", "① PCCDMM", "② CCPUMM", "③ X = A − R; compute XW", "④ AW = RW + XW",
  ]);
  assert.match(trace.steps[4].description, /mask A with R/);
  assert.ok(matricesEqual(trace.finalTensor, [[11, 7], [4, 8]]));
});

test("CCCRMM keeps batch components isolated and emits diagonal-packed scores", () => {
  const trace = buildEustonOperationTrace(EUSTON_OPERATION_IDS.SCORE);
  assert.deepEqual(trace.graph.stages.map((stage) => stage.title), ["Q × Kᵀ", "Column Pack", "① Component ρ", "② CCCRMM ⊙ Σ", "P"]);
  assert.ok(matricesEqual(trace.finalTensor, [[6, 8], [1, 3]]));
  const rotated = trace.graph.stages[2].rows[0].find((artifact) => artifact.label === "RotL₁(k1)");
  assert.deepEqual(rotated.cells.map((cell) => cell.sampleLane), [0, 1, 1, undefined, undefined, undefined, undefined, 0]);
  assert.deepEqual(rotated.cells.map((cell) => cell.label), ["3", "1", "3", "·", "·", "·", "·", "1"]);
  const edited = buildEustonOperationTrace(EUSTON_OPERATION_IDS.SCORE, { B: [[3, 7], [11, 13]] });
  const shifted = edited.graph.stages[2].rows.flat().find((a) => a.label === "k1⇐1");
  assert.deepEqual(shifted.cells.map((cell) => cell.value), [11, 3, 11, 3, 0, 0, 0, 0]);
});

test("CDCCMM consumes diagonal-packed Softmax and emits column-packed attention", () => {
  const trace = buildEustonOperationTrace(EUSTON_OPERATION_IDS.CONTEXT);
  assert.deepEqual(trace.graph.stages.map((stage) => stage.title), ["Softmax(P) × V", "Diag + Column Pack", "① Component ρ", "② CDCCMM ⊙ Σ", "Y"]);
  assert.ok(matricesEqual(trace.finalTensor, [[1.75, 1.5], [1.375, 2.25]]));
});

test("each Euston ciphertext uses one uniform color block", () => {
  for (const operation of listEustonOperations()) {
    const trace = buildEustonOperationTrace(operation.id);
    for (const stage of trace.graph.stages) for (const row of stage.rows) for (const artifact of row) {
      if (artifact.badge !== "CT" || !artifact.cells) continue;
      assert.equal(new Set(artifact.cells.map((cell) => cell.tone)).size, 1, `${operation.id}/${artifact.id}`);
    }
  }
});

test("editable values rebuild output and PCCDMM rejects non-diagonal D", () => {
  const changed = buildEustonOperationTrace(EUSTON_OPERATION_IDS.FC1, { A: [[2, 0], [0, 2]] });
  assert.ok(matricesEqual(changed.finalTensor, [[2, 0, 4, 6], [4, 2, 6, 0]]));
  const d = getEustonOperation(EUSTON_OPERATION_IDS.PCCDMM);
  const parsed = parseEustonOperationInputs({
    m: "2", k: "2", n: "2", valuesTextA: "1 2; 0 1", valuesTextB: "2 1; 0 3",
    leftType: "pt", rightType: "ct", slots: "8", strategy: "column", seed: "1", valueMode: "manual",
  }, d);
  assert.equal(parsed.ok, false);
  assert.match(parsed.errors.values, /diagonal/);
});
