import { test } from "node:test";
import assert from "node:assert/strict";
import {
  THOR_OPERATION_IDS,
  THOR_OPERATIONS,
  buildThorOperationTrace,
  listThorOperations,
  thorOperationDefaults,
} from "../transformer/algorithms/thor/bert_operations.js";
import { validateTrace } from "../transformer/core/trace_validator.js";
import { matmul } from "../transformer/core/matrix.js";

test("THOR registry covers every BERT matrix multiplication", () => {
  assert.deepEqual(
    listThorOperations().map((item) => item.id),
    [
      THOR_OPERATION_IDS.QUERY,
      THOR_OPERATION_IDS.KEY,
      THOR_OPERATION_IDS.VALUE,
      THOR_OPERATION_IDS.SCORE,
      THOR_OPERATION_IDS.CONTEXT,
      THOR_OPERATION_IDS.OUTPUT,
      THOR_OPERATION_IDS.FC1,
      THOR_OPERATION_IDS.FC2,
      THOR_OPERATION_IDS.POOLER,
      THOR_OPERATION_IDS.CLASSIFIER,
    ]
  );
});

test("all operation traces are numerically valid diagram-only toy models", () => {
  for (const operation of listThorOperations()) {
    const trace = buildThorOperationTrace(operation.id);
    const result = validateTrace(trace);
    assert.equal(result.ok, true, `${operation.id}: ${result.errors.join("; ")}`);
    assert.equal(trace.graph.stages.length, trace.steps.length);
    assert.equal(trace.plan.visualStyle, "diagram-only");
    assert.ok(trace.graph.stages.every((stage) => stage.caption.length > 0));
    assert.deepEqual(
      trace.steps.map((step) => step.title),
      trace.graph.stages.map((stage) => stage.stepTitle || stage.title),
    );
    assert.equal(trace.plan.thorRuntime, undefined);
  }
});

test("XWq uses the requested W(4x4) times X(4x2) PTxCT geometry", () => {
  const operation = THOR_OPERATIONS[THOR_OPERATION_IDS.QUERY];
  const defaults = thorOperationDefaults(operation.id);
  const trace = buildThorOperationTrace(operation.id);
  assert.deepEqual([defaults.m, defaults.k, defaults.n], [4, 4, 2]);
  assert.equal(operation.leftLabel, "Wq");
  assert.equal(operation.rightLabel, "X");
  assert.equal(trace.parameters.matmulType, "pt-ct");
  assert.deepEqual(trace.output.shape, [4, 2]);
});

test("QKT uses Corollary 3.9 (square 4x4 score); SV uses Figure 4 CTxCT geometry", () => {
  const score = buildThorOperationTrace(THOR_OPERATION_IDS.SCORE);
  const context = buildThorOperationTrace(THOR_OPERATION_IDS.CONTEXT);
  assert.equal(score.parameters.matmulType, "ct-ct");
  assert.equal(context.parameters.matmulType, "ct-ct");
  // The score is Case 2 -> Corollary 3.9: Q,K in R^{4x2}, S = Q K^T in R^{4x4}.
  assert.deepEqual(score.parameters.matrixShape, { m: 4, k: 2, n: 4 });
  assert.deepEqual(score.output.shape, [4, 4]);
  assert.match(score.plan.plannerSource, /Corollary 3\.9/);
  assert.equal(score.operands.left.label, "Q");
  assert.equal(score.operands.right.label, "Kᵀ");
  assert.deepEqual(score.finalTensor, matmul(score.operands.left.values, score.operands.right.values));
  // The attention head is Case 3 -> Corollary 3.8 geometry (V^T x alpha^T -> H^T).
  assert.deepEqual(context.parameters.matrixShape, { m: 2, k: 4, n: 4 });
  assert.equal(context.operands.left.label, "Vᵀ");
  assert.equal(context.operands.right.label, "αᵀ");
});

test("preserves paper block colors through THOR operation relabeling", () => {
  const pcmm = buildThorOperationTrace(THOR_OPERATION_IDS.QUERY);
  const pcmmInput = pcmm.graph.stages[0].rows[0];
  const pcmmA = pcmmInput[0].rows.flat().map((cell) => cell.tone);
  const pcmmB = pcmmInput[2].rows.flat().map((cell) => cell.tone);
  assert.deepEqual(pcmmA, [0, 1, 2, 3, 1, 0, 3, 2, 0, 1, 2, 3, 1, 0, 3, 2]);
  assert.deepEqual(pcmmB, [0, 1, 1, 0, 2, 3, 3, 2]);

  const ccmm = buildThorOperationTrace(THOR_OPERATION_IDS.CONTEXT);
  const ccmmInput = ccmm.graph.stages[0].rows[0];
  const ccmmA = ccmmInput[0].rows.flat().map((cell) => cell.tone);
  const ccmmB = ccmmInput[2].rows.flat().map((cell) => cell.tone);
  assert.deepEqual(ccmmA, [0, 1, 2, 3, 0, 1, 2, 3]);
  assert.deepEqual(ccmmB, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
});

test("editable values change the final result while semantic types stay fixed", () => {
  const defaults = thorOperationDefaults(THOR_OPERATION_IDS.QUERY);
  assert.throws(
    () => buildThorOperationTrace(THOR_OPERATION_IDS.QUERY, { ...defaults, leftSecret: true }),
    /requires plaintext A/
  );
  const editedA = defaults.A.map((row) => [...row]);
  editedA[0][0] = 9;
  const base = buildThorOperationTrace(THOR_OPERATION_IDS.QUERY, defaults);
  const edited = buildThorOperationTrace(THOR_OPERATION_IDS.QUERY, { ...defaults, A: editedA });
  assert.notDeepEqual(edited.finalTensor, base.finalTensor);
});
