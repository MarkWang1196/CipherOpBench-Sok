import { test } from "node:test";
import assert from "node:assert/strict";
import {
  MOAI_OPERATION_IDS,
  buildMoaiOperationTrace,
  listMoaiOperations,
  moaiOperationDefaults,
  parseMoaiOperationInputs,
} from "../transformer/algorithms/moai/bert_operations.js";
import { matmul, transpose } from "../transformer/core/matrix.js";
import { validateTrace } from "../transformer/core/trace_validator.js";

const ALL_IDS = [
  MOAI_OPERATION_IDS.QUERY,
  MOAI_OPERATION_IDS.KEY,
  MOAI_OPERATION_IDS.VALUE,
  MOAI_OPERATION_IDS.SCORE,
  MOAI_OPERATION_IDS.CONTEXT,
  MOAI_OPERATION_IDS.OUTPUT,
  MOAI_OPERATION_IDS.FC1,
  MOAI_OPERATION_IDS.FC2,
  MOAI_OPERATION_IDS.POOLER,
  MOAI_OPERATION_IDS.CLASSIFIER,
];

test("MOAI registry covers every Transformer matrix multiplication", () => {
  assert.deepEqual(listMoaiOperations().map((item) => item.id), ALL_IDS);
});

test("all MOAI traces are valid, graphical, editable two-batch toys", () => {
  for (const operation of listMoaiOperations()) {
    const trace = buildMoaiOperationTrace(operation.id);
    const result = validateTrace(trace);
    assert.equal(result.ok, true, `${operation.id}: ${result.errors.join("; ")}`);
    assert.equal(trace.plan.method, "moai");
    assert.equal(trace.plan.visualStyle, "diagram-only");
    assert.equal(trace.parameters.batchCount, 2);
    assert.equal(trace.parameters.slotsPerCiphertext, 8);
    assert.equal(trace.graph.stages.length, trace.steps.length);
    assert.ok(trace.graph.stages.every((stage) => stage.caption === ""));
  }
});

test("Algorithm 2 CT-PT path is Col to Col and performs no rotations", () => {
  const trace = buildMoaiOperationTrace(MOAI_OPERATION_IDS.QUERY);
  assert.equal(trace.parameters.matmulType, "ct-pt");
  assert.equal(trace.plan.packingStrategyResolved, "Col→Col");
  assert.equal(trace.steps.at(-1).metricsTotal.rotations, 0);
  assert.equal(trace.steps.at(-1).metricsTotal.ptCtMultiplications, 8);
  assert.deepEqual(trace.finalTensor, matmul(trace.operands.left.values, trace.operands.right.values));
});

test("Algorithms 3/6 accept Q and K columns, rotate by interleaved batch steps, and output diagonals", () => {
  const defaults = moaiOperationDefaults(MOAI_OPERATION_IDS.SCORE);
  assert.deepEqual([defaults.A.length, defaults.A[0].length], [4, 2]);
  assert.deepEqual([defaults.B.length, defaults.B[0].length], [4, 2]);
  const trace = buildMoaiOperationTrace(MOAI_OPERATION_IDS.SCORE, defaults);
  assert.deepEqual(trace.operands.right.values, transpose(defaults.B));
  assert.deepEqual(trace.finalTensor, matmul(defaults.A, transpose(defaults.B)));
  assert.equal(trace.plan.packingStrategyResolved, "Col×Col→Diag");
  assert.equal(trace.graph.stages[2].title, "ρ {0,2,4,6}");
  assert.equal(trace.steps.at(-1).metricsTotal.rotations, 6);
  const diagonalOutput = trace.graph.stages.at(-1).rows.flat().find((artifact) => artifact.id === "C");
  assert.ok(diagonalOutput.rows.flat().every((cell) => cell.tone >= 0 && cell.tone <= 3));
});

test("Algorithms 4/7 use BSGS Diag by Col and reconstruct S times V", () => {
  const trace = buildMoaiOperationTrace(MOAI_OPERATION_IDS.CONTEXT);
  assert.equal(trace.plan.packingStrategyResolved, "Diag×Col→Col");
  assert.deepEqual(trace.finalTensor, matmul(trace.operands.left.values, trace.operands.right.values));
  assert.equal(trace.steps.at(-1).metricsTotal.rotations, 6);
  assert.equal(trace.steps.at(-1).metricsTotal.ctCtMultiplications, 8);
  assert.match(trace.graph.stages[2].title, /Baby/);
  assert.match(trace.graph.stages[5].title, /Giant/);
});

test("SV giant-step groups keep both batches and sum to the output columns", () => {
  const tr = buildMoaiOperationTrace(MOAI_OPERATION_IDS.CONTEXT);
  const H = tr.finalTensor;
  const st = tr.graph.stages;
  const at = (id) => st.flatMap((s) => s.rows.flat()).find((a) => a.id === id).cells.map((c) => c.value);
  for (let col = 0; col < 2; col += 1) {
    const out = at(`moai-context-out-${col}`);
    const expect = H.map((r) => [r[col], r[col]]).flat();
    assert.deepEqual(out, expect, `output column ${col} keeps both batches`);
    const g = [0, 1].map((a) => at(`moai-context-g-${col}-${a}`));
    assert.equal(g[0].length, 8);
    assert.deepEqual(g[0].map((_, s) => g[0][s] + g[1][s]), out, `groups sum for column ${col}`);
  }
});

test("MOAI ciphertext strips expose strict interleaved batch slots", () => {
  for (const id of [MOAI_OPERATION_IDS.QUERY, MOAI_OPERATION_IDS.SCORE, MOAI_OPERATION_IDS.CONTEXT]) {
    const trace = buildMoaiOperationTrace(id);
    for (const stage of trace.graph.stages) {
      for (const row of stage.rows) {
        for (const artifact of row) {
          if (artifact.badge !== "CT" || !artifact.cells) continue;
          assert.equal(artifact.cells.length, 8, `${id}/${artifact.id}`);
          assert.ok(artifact.cells.every((cell, index) => cell.label.endsWith(index % 2 ? "₁" : "₀")));
          assert.equal(new Set(artifact.cells.map((cell) => cell.tone)).size, 1);
        }
      }
    }
  }
});

test("MOAI colors follow packed ciphertext columns through rotations and products", () => {
  const query = buildMoaiOperationTrace(MOAI_OPERATION_IDS.QUERY);
  const queryInput = query.graph.stages[0].rows[0];
  assert.deepEqual(queryInput[0].rows.flat().map((cell) => cell.tone), [
    0, 1, 2, 3, 0, 1, 2, 3,
    0, 1, 2, 3, 0, 1, 2, 3,
  ]);
  assert.deepEqual(queryInput[2].rows.flat().map((cell) => cell.tone), [
    0, 1, 0, 1, 0, 1, 0, 1,
  ]);
  assert.deepEqual(query.graph.stages[2].rows.flat().filter((artifact) => artifact.id.includes("-p-"))
    .map((artifact) => artifact.cells[0].tone), [0, 1, 2, 3]);

  const score = buildMoaiOperationTrace(MOAI_OPERATION_IDS.SCORE);
  const scoreInput = score.graph.stages[0].rows[0];
  assert.deepEqual(scoreInput[2].rows.flat().map((cell) => cell.tone), [
    2, 3, 2, 3, 2, 3, 2, 3,
  ]);
  assert.ok(score.graph.stages[2].rows.flat().filter((artifact) => artifact.id.includes("-rk-"))
    .every((artifact) => [2, 3].includes(artifact.cells[0].tone)));
  assert.ok(score.graph.stages[3].rows.flat().filter((artifact) => artifact.id.includes("-p-"))
    .every((artifact) => [2, 3].includes(artifact.cells[0].tone)));

  const context = buildMoaiOperationTrace(MOAI_OPERATION_IDS.CONTEXT);
  const contextInput = context.graph.stages[0].rows[0];
  assert.deepEqual(contextInput[2].rows.flat().map((cell) => cell.tone), [
    2, 3, 2, 3, 2, 3, 2, 3,
  ]);
  assert.ok(context.graph.stages[4].rows.flat().filter((artifact) => artifact.id.includes("-p-"))
    .every((artifact) => [2, 3].includes(artifact.cells[0].tone)));
  assert.deepEqual(context.graph.stages.at(-1).rows.flat().find((artifact) => artifact.id === "C").rows
    .flat().map((cell) => cell.tone), [2, 3, 2, 3, 2, 3, 2, 3]);
});

test("paired-column parser keeps editable Q and K rather than asking for K transpose", () => {
  const operation = listMoaiOperations().find((item) => item.id === MOAI_OPERATION_IDS.SCORE);
  const defaults = moaiOperationDefaults(operation.id);
  const raw = {
    valuesTextA: defaults.A.map((row) => row.join(" ")).join("\n"),
    valuesTextB: defaults.B.map((row) => row.join(" ")).join("\n"),
    slots: "8", leftType: "ct", rightType: "ct", seed: "1",
  };
  const parsed = parseMoaiOperationInputs(raw, operation);
  assert.equal(parsed.ok, true);
  assert.deepEqual(parsed.inputs.B, defaults.B);
  const editedA = defaults.A.map((row) => [...row]);
  editedA[0][0] = 9;
  assert.notDeepEqual(
    buildMoaiOperationTrace(operation.id, { ...defaults, A: editedA }).finalTensor,
    buildMoaiOperationTrace(operation.id, defaults).finalTensor
  );
});
