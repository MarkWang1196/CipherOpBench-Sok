import { test } from "node:test";
import assert from "node:assert/strict";
import {
  BLB_OPERATION_IDS,
  buildBlbOperationTrace,
  listBlbOperations,
  blbOperationDefaults,
  parseBlbOperationInputs,
} from "../transformer/algorithms/blb/bert_operations.js";
import { matmul, transpose } from "../transformer/core/matrix.js";
import { validateTrace } from "../transformer/core/trace_validator.js";

const ALL_IDS = Object.values(BLB_OPERATION_IDS);

test("BLB registry covers every Transformer matrix multiplication", () => {
  assert.deepEqual(listBlbOperations().map((item) => item.id), ALL_IDS);
});

test("all BLB traces are valid graphical 16-slot toys", () => {
  for (const operation of listBlbOperations()) {
    const trace = buildBlbOperationTrace(operation.id);
    const result = validateTrace(trace);
    assert.equal(result.ok, true, `${operation.id}: ${result.errors.join("; ")}`);
    assert.equal(trace.plan.method, "blb");
    assert.equal(trace.plan.visualStyle, "diagram-only");
    assert.equal(trace.parameters.slotsPerCiphertext, 16);
    assert.equal(trace.graph.stages.length, trace.steps.length);
    assert.ok(trace.graph.stages.every((stage) => stage.caption === ""));
  }
});

test("QKV and FFN use spatial-first BSGS ciphertext-plaintext MatMul", () => {
  for (const id of [BLB_OPERATION_IDS.QUERY, BLB_OPERATION_IDS.KEY, BLB_OPERATION_IDS.VALUE, BLB_OPERATION_IDS.FC1, BLB_OPERATION_IDS.FC2]) {
    const trace = buildBlbOperationTrace(id);
    assert.equal(trace.parameters.matmulType, "ct-pt");
    assert.equal(trace.plan.packingStrategyResolved, "Spatial-first BSGS");
    assert.equal(trace.graph.stages[1].title, "Spatial-first");
    assert.equal(trace.graph.stages[2].title, "Baby ρ");
    assert.deepEqual(trace.finalTensor, matmul(trace.operands.left.values, trace.operands.right.values));
  }
});

test("BLB colors follow spatial columns, weight columns, and partial-sum provenance", () => {
  const linear = buildBlbOperationTrace(BLB_OPERATION_IDS.QUERY);
  const input = linear.graph.stages[0].rows[0];
  assert.deepEqual(input[0].rows.flat().map((cell) => cell.tone), [
    0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
  ]);
  assert.deepEqual(input[2].rows.flat().map((cell) => cell.tone), [0, 1, 0, 1, 0, 1, 0, 1]);
  assert.deepEqual(linear.graph.stages[3].rows.flat().map((artifact) => artifact.cells[0].tone), [0, 1, 2, 3, 0, 1, 2, 3]);

  const score = buildBlbOperationTrace(BLB_OPERATION_IDS.SCORE);
  const scoreInput = score.graph.stages[0].rows[0];
  assert.deepEqual(scoreInput[0].rows.flat().map((cell) => cell.tone), [0, 1, 0, 1, 0, 1, 0, 1]);
  assert.deepEqual(scoreInput[2].rows.flat().map((cell) => cell.tone), [2, 3, 2, 3, 2, 3, 2, 3]);
});

test("QK transpose follows BLB MHP three-step ciphertext MatMul", () => {
  const defaults = blbOperationDefaults(BLB_OPERATION_IDS.SCORE);
  const trace = buildBlbOperationTrace(BLB_OPERATION_IDS.SCORE, defaults);
  assert.deepEqual(trace.operands.right.values, transpose(defaults.B));
  assert.deepEqual(trace.finalTensor, matmul(defaults.A, transpose(defaults.B)));
  assert.deepEqual(trace.graph.stages.slice(1, 5).map((stage) => stage.title), [
    "MHP", "Inner ρ + mask", "⊙", "Inner ρ + Σ → Diag",
  ]);
  assert.equal(trace.steps.at(-1).metricsTotal.ctCtMultiplications, 4);
  assert.equal(trace.steps.at(-1).metricsTotal.rotations, 15);
  assert.equal(trace.parameters.batchCount, 2);
  assert.equal(trace.plan.batchLabel, "heads");
});

test("Softmax times V crosses MPC to CKKS, pads V, and returns the logical crop", () => {
  const trace = buildBlbOperationTrace(BLB_OPERATION_IDS.CONTEXT);
  assert.equal(trace.graph.stages[0].rows[0][0].badge, "MPC");
  assert.equal(trace.graph.stages[1].title, "MPC → CKKS");
  assert.equal(trace.graph.stages.at(-1).rows[0][0].rows[0].length, 4);
  assert.deepEqual(trace.finalTensor, matmul(trace.operands.left.values, trace.operands.right.values));
  assert.equal(trace.finalTensor[0].length, 2);
});

test("output projection consumes diagonal ciphertexts and restores spatial-first packing", () => {
  const trace = buildBlbOperationTrace(BLB_OPERATION_IDS.OUTPUT);
  assert.equal(trace.graph.stages[1].title, "Diag");
  assert.equal(trace.graph.stages[2].title, "Wo ⧉");
  assert.equal(trace.graph.stages.at(-1).title, "Spatial-first");
  assert.deepEqual(trace.finalTensor, matmul(trace.operands.left.values, trace.operands.right.values));
});

test("HWo diagonal products are real and sum to the spatial-first output", () => {
  const tr = buildBlbOperationTrace(BLB_OPERATION_IDS.OUTPUT);
  const st = tr.graph.stages;
  const O = tr.finalTensor;
  const at = (id) => st.flatMap((s) => s.rows.flat()).find((a) => a.id === id).cells.map((c) => c.value);
  const D = [0, 1, 2, 3].map((t) => at(`blb-output-d-${t}`));
  const M = [0, 1, 2, 3].map((t) => at(`blb-output-w-${t}`));
  const P = [0, 1, 2, 3].map((t) => at(`blb-output-p-${t}`));
  for (let t = 0; t < 4; t += 1) assert.deepEqual(P[t], D[t].map((v, s) => v * M[t][s]), `product ${t}`);
  assert.equal(new Set(P.map(JSON.stringify)).size, 4);
  const total = P[0].map((_, s) => P.reduce((a, p) => a + p[s], 0));
  const flat = [O[0][0], O[1][0], O[2][0], O[3][0], O[0][1], O[1][1], O[2][1], O[3][1], 0, 0, 0, 0, 0, 0, 0, 0];
  assert.deepEqual(total, flat);
});

test("QK^T and SV show real pointwise products, not a false equality to the diagonals", () => {
  for (const [id, prefix] of [[BLB_OPERATION_IDS.SCORE, "blb-score"], [BLB_OPERATION_IDS.CONTEXT, "blb-context"]]) {
    const tr = buildBlbOperationTrace(id);
    const st = tr.graph.stages;
    const at = (aid) => st.flatMap((s) => s.rows.flat()).find((a) => a.id === aid).cells.map((c) => c.value);
    const leftId = id === BLB_OPERATION_IDS.SCORE ? `${prefix}-q` : `${prefix}-s`;
    const left = at(leftId);
    for (let t = 0; t < 4; t += 1) {
      const rotId = id === BLB_OPERATION_IDS.SCORE ? `${prefix}-r-${t}` : `${prefix}-rv-${t}`;
      const prodId = `${prefix}-pr-${t}`;
      const rot = at(rotId);
      const P = at(prodId);
      assert.deepEqual(P, left.map((v, s) => v * rot[s]), `${id} product ${t}`);
    }
  }
});

test("pooler and classifier are shown as MPC share MatMul, not CKKS ciphertext MatMul", () => {
  for (const id of [BLB_OPERATION_IDS.POOLER, BLB_OPERATION_IDS.CLASSIFIER]) {
    const trace = buildBlbOperationTrace(id);
    assert.equal(trace.parameters.matmulType, "mpc-mpc");
    assert.equal(trace.plan.packingStrategyResolved, "MPC secret-share MatMul");
    assert.ok(trace.graph.stages.flatMap((stage) => stage.rows.flat()).filter((artifact) => artifact.type === "matrix").every((artifact) => artifact.badge === "MPC"));
  }
});

test("one ciphertext always has one uniform color block", () => {
  for (const id of [BLB_OPERATION_IDS.QUERY, BLB_OPERATION_IDS.SCORE, BLB_OPERATION_IDS.CONTEXT, BLB_OPERATION_IDS.OUTPUT]) {
    const trace = buildBlbOperationTrace(id);
    for (const artifact of trace.graph.stages.flatMap((stage) => stage.rows.flat())) {
      if (artifact.badge === "CT" && artifact.cells) {
        assert.equal(artifact.cells.length, 16, `${id}/${artifact.id}`);
        assert.equal(new Set(artifact.cells.map((cell) => cell.tone)).size, 1, `${id}/${artifact.id}`);
      }
    }
  }
});

test("paired-column parser keeps editable Q and K", () => {
  const operation = listBlbOperations().find((item) => item.id === BLB_OPERATION_IDS.SCORE);
  const defaults = blbOperationDefaults(operation.id);
  const raw = {
    valuesTextA: defaults.A.map((row) => row.join(" ")).join("\n"),
    valuesTextB: defaults.B.map((row) => row.join(" ")).join("\n"),
    slots: "16", leftType: "ct", rightType: "ct", seed: "1",
  };
  const parsed = parseBlbOperationInputs(raw, operation);
  assert.equal(parsed.ok, true);
  assert.deepEqual(parsed.inputs.B, defaults.B);
});
