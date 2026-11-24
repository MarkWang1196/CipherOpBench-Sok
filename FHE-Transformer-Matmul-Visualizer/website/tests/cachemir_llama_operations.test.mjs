import { test } from "node:test";
import assert from "node:assert/strict";
import {
  CACHEMIR_OPERATION_IDS,
  buildCachemirOperationTrace,
  listCachemirOperations,
  cachemirOperationDefaults,
  parseCachemirOperationInputs,
} from "../transformer/algorithms/cachemir/llama_operations.js";
import { matmul, transpose } from "../transformer/core/matrix.js";
import { validateTrace } from "../transformer/core/trace_validator.js";

test("Cachemir registry covers every visualized decoder matrix multiplication", () => {
  assert.deepEqual(listCachemirOperations().map((item) => item.id), Object.values(CACHEMIR_OPERATION_IDS));
});

test("all Cachemir traces are valid editable 8-slot decoding toys", () => {
  for (const operation of listCachemirOperations()) {
    const trace = buildCachemirOperationTrace(operation.id);
    const result = validateTrace(trace);
    assert.equal(result.ok, true, `${operation.id}: ${result.errors.join("; ")}`);
    assert.equal(trace.plan.method, "cachemir");
    assert.equal(trace.plan.visualStyle, "diagram-only");
    assert.equal(trace.parameters.slotsPerCiphertext, 8);
    assert.equal(trace.parameters.batchCount, 1);
    assert.equal(trace.graph.stages.length, trace.steps.length);
    assert.ok(trace.graph.stages.every((stage) => stage.caption === ""));
  }
});

test("QKV, output, Up, Gate and Down use interleaved replicated VMM", () => {
  const ids = [CACHEMIR_OPERATION_IDS.QUERY, CACHEMIR_OPERATION_IDS.KEY, CACHEMIR_OPERATION_IDS.VALUE,
    CACHEMIR_OPERATION_IDS.OUTPUT, CACHEMIR_OPERATION_IDS.UP, CACHEMIR_OPERATION_IDS.GATE, CACHEMIR_OPERATION_IDS.DOWN];
  for (const id of ids) {
    const trace = buildCachemirOperationTrace(id);
    assert.equal(trace.parameters.matmulType, "ct-pt");
    assert.equal(trace.plan.packingStrategyResolved, "Interleaved replicated VMM");
    const square = trace.parameters.matrixShape.k === 4 && trace.parameters.matrixShape.n === 4;
    assert.deepEqual(trace.graph.stages.slice(1, 5).map((stage) => stage.title), square
      ? ["Interleave", "① ρ +", "② ρ ⊙ diag(W)", "③ Σ + ρ1"]
      : ["Interleave", "① ρ +", "② Generalized ρ ⊙ pⱼ,ₖ", "③ Σ"]);
    assert.deepEqual(trace.finalTensor, matmul(trace.operands.left.values, trace.operands.right.values));
    assert.equal(trace.steps.at(-1).metricsTotal.multiplicativeDepth, 1);
    if (square) {
      assert.equal(trace.steps.at(-1).metricsTotal.rotations, 3);
      assert.equal(trace.steps.at(-1).metricsTotal.ptCtMultiplications, 2);
    }
  }
});

test("Cachemir colors follow weight diagonals and KV-cache packing identities", () => {
  const vmm = buildCachemirOperationTrace(CACHEMIR_OPERATION_IDS.QUERY);
  const vmmInput = vmm.graph.stages[0].rows[0];
  assert.deepEqual(vmmInput[0].rows.flat().map((cell) => cell.tone), [0, 0, 0, 0]);
  assert.deepEqual(vmmInput[2].rows.flat().map((cell) => cell.tone), [0, 1, 2, 3, 3, 0, 1, 2,
    2, 3, 0, 1, 1, 2, 3, 0]);
  assert.deepEqual(vmm.graph.stages[4].rows.flat().filter((artifact) => artifact.id.includes("-p-")
    ).map((artifact) => artifact.cells[0].tone), [0, 1]);

  const score = buildCachemirOperationTrace(CACHEMIR_OPERATION_IDS.SCORE);
  const scoreInput = score.graph.stages[0].rows[0];
  assert.deepEqual(scoreInput[2].rows.flat().map((cell) => cell.tone), [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0]);

  const context = buildCachemirOperationTrace(CACHEMIR_OPERATION_IDS.CONTEXT);
  const contextInput = context.graph.stages[0].rows[0];
  assert.deepEqual(contextInput[2].rows.flat().map((cell) => cell.tone), [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);
});

test("QK transpose packs two cached tokens per ciphertext and appends the next key", () => {
  const defaults = cachemirOperationDefaults(CACHEMIR_OPERATION_IDS.SCORE);
  const trace = buildCachemirOperationTrace(CACHEMIR_OPERATION_IDS.SCORE, defaults);
  assert.deepEqual(trace.operands.right.values, transpose(defaults.B));
  assert.deepEqual(trace.finalTensor, matmul(defaults.A, transpose(defaults.B)));
  assert.equal(trace.graph.stages[1].title, "K Cache · t=2");
  assert.equal(trace.graph.stages[1].rows[1].length, 3);
  assert.equal(trace.graph.stages.at(-1).title, "K Cache ⊕ K[5]");
  assert.equal(trace.steps.at(-1).metricsTotal.ctCtMultiplications, 3);
  assert.equal(trace.steps[5].metricsDelta.rotations, 2);
});

test("Softmax times V uses four distributed cache ciphertexts and emits interleaved Att", () => {
  const trace = buildCachemirOperationTrace(CACHEMIR_OPERATION_IDS.CONTEXT);
  assert.equal(trace.graph.stages[1].title, "V Cache");
  assert.equal(trace.graph.stages[1].rows[1].length, 4);
  assert.equal(trace.graph.stages[2].title, "① ρ{0,2,4,6}");
  assert.equal(trace.graph.stages.at(-1).title, "V Cache ⊕ V[5]");
  assert.equal(trace.steps.at(-1).metricsTotal.ctCtMultiplications, 4);
  assert.equal(trace.steps.at(-1).metricsTotal.rotations, 4);
  assert.deepEqual(trace.finalTensor, matmul(trace.operands.left.values, trace.operands.right.values));
});

test("fused extraction names match Cachemir's downstream operators", () => {
  assert.match(buildCachemirOperationTrace(CACHEMIR_OPERATION_IDS.QUERY).graph.stages.at(-1).title, /RoPE/);
  assert.match(buildCachemirOperationTrace(CACHEMIR_OPERATION_IDS.VALUE).graph.stages.at(-1).title, /V-cache/);
  assert.match(buildCachemirOperationTrace(CACHEMIR_OPERATION_IDS.GATE).graph.stages.at(-1).title, /SiLU/);
  assert.match(buildCachemirOperationTrace(CACHEMIR_OPERATION_IDS.DOWN).graph.stages.at(-1).title, /Norm/);
});

test("every Cachemir ciphertext strip has one uniform color", () => {
  for (const operation of listCachemirOperations()) {
    const trace = buildCachemirOperationTrace(operation.id);
    for (const artifact of trace.graph.stages.flatMap((stage) => stage.rows.flat())) {
      if (artifact.badge === "CT" && artifact.cells) {
        assert.equal(artifact.cells.length, 8, `${operation.id}/${artifact.id}`);
        assert.equal(new Set(artifact.cells.map((cell) => cell.tone)).size, 1, `${operation.id}/${artifact.id}`);
      }
    }
  }
});

test("VMM products and reductions genuinely produce every displayed intermediate", () => {
  const vmmIds = [CACHEMIR_OPERATION_IDS.QUERY, CACHEMIR_OPERATION_IDS.KEY, CACHEMIR_OPERATION_IDS.VALUE,
    CACHEMIR_OPERATION_IDS.OUTPUT, CACHEMIR_OPERATION_IDS.UP, CACHEMIR_OPERATION_IDS.GATE, CACHEMIR_OPERATION_IDS.DOWN];
  for (const id of vmmIds) {
    const tr = buildCachemirOperationTrace(id);
    const seen = new Set();
    const prods = tr.graph.stages.flatMap((s) => s.rows.flat())
      .filter((a) => a.id.includes("-p-") && !seen.has(a.id) && seen.add(a.id));
    const artifacts = tr.graph.stages.flatMap((s) => s.rows.flat());
    const reduced = artifacts
      .find((a) => a.id === `${id}-reduce`).cells.map((c) => c.value);
    const total = prods.reduce((acc, p) => acc.map((v, s) => v + p.cells[s].value), new Array(8).fill(0));
    const square = tr.parameters.matrixShape.k === 4 && tr.parameters.matrixShape.n === 4;
    if (square) {
      const accumulated = artifacts.find((a) => a.id === `${id}-accumulate`).cells.map((c) => c.value);
      assert.deepEqual(total, accumulated, `${id} product sum`);
    }
    assert.deepEqual(reduced, square ? total.map((v, s) => v + total[(s + 1) % 8]) : total, `${id} reduction`);
  }
});

test("QK^T products are real and fold (log d) into the replicated scores", () => {
  const tr = buildCachemirOperationTrace(CACHEMIR_OPERATION_IDS.SCORE);
  const scores = tr.finalTensor[0];
  const qRep = tr.graph.stages.flatMap((s) => s.rows.flat()).find((a) => a.id === "cachemir-score-qrep").cells.map((c) => c.value);
  for (let i = 0; i < 3; i += 1) {
    const cache = tr.graph.stages.flatMap((s) => s.rows.flat()).find((a) => a.id === `cachemir-score-kc-${i}`).cells.map((c) => c.value);
    const P = tr.graph.stages.flatMap((s) => s.rows.flat()).find((a) => a.id === `cachemir-score-prod-${i}`).cells.map((c) => c.value);
    assert.deepEqual(P, qRep.map((v, j) => v * cache[j]));
    const even = P.filter((_, j) => j % 2 === 0).reduce((a, v) => a + v, 0);
    const odd = P.filter((_, j) => j % 2 === 1).reduce((a, v) => a + v, 0);
    assert.ok(Math.abs(even - (scores[2 * i] ?? 0)) < 1e-9, `pair ${i} even`);
    assert.ok(Math.abs(odd - (scores[2 * i + 1] ?? 0)) < 1e-9, `pair ${i} odd`);
  }
  const packed = [0, 1, 2].map((i) => tr.graph.stages.flatMap((s) => s.rows.flat())
    .find((a) => a.id === `cachemir-score-packed-${i}`).cells.map((c) => c.value));
  const packedSum = packed.reduce((sum, item) => sum.map((v, s) => v + item[s]), new Array(8).fill(0));
  assert.deepEqual(packedSum, [...scores, 0, 0, 0]);
});

test("Softmax times V: partials are real and the fold recovers Att", () => {
  const tr = buildCachemirOperationTrace(CACHEMIR_OPERATION_IDS.CONTEXT);
  const att = tr.finalTensor[0];
  const st = tr.graph.stages;
  const at = (id) => st.flatMap((s) => s.rows.flat()).find((a) => a.id === id).cells.map((c) => c.value);
  const partials = [0, 1, 2, 3].map((i) => at(`cachemir-context-p-${i}`));
  const sRot = [0, 1, 2, 3].map((i) => at(`cachemir-context-sr-${i}`));
  const vc = [0, 1, 2, 3].map((i) => at(`cachemir-context-vc-${i}`));
  for (let i = 0; i < 4; i += 1) assert.deepEqual(partials[i], sRot[i].map((v, j) => v * vc[i][j]));
  const total = partials.reduce((a, p) => a.map((v, s) => v + p[s]), new Array(8).fill(0));
  const folded = total.map((_, s) => total[s] + total[(s + 1) % 8]);
  assert.deepEqual(at("cachemir-context-att-pre"), folded);
  assert.ok(folded.filter((_, s) => s % 2 === 0).every((v, i) => Math.abs(v - att[i]) < 1e-9));
  assert.deepEqual(at("cachemir-context-att"), att.flatMap((v) => [v, 0]));

  const vnew = at("cachemir-context-vnew").filter((_, s) => s % 2 === 1);
  const extendedV = [...tr.operands.right.values, vnew];
  for (let k = 0; k < 4; k += 1) {
    const updated = at(`cachemir-context-vupdated-${k}`);
    for (let s = 0; s < 8; s += 1) {
      const col = Math.floor(s / 2), token = (s + 2 * k) % 8;
      assert.equal(updated[s], token < extendedV.length ? extendedV[token][col] : 0, `V${k}[${s}]`);
    }
  }
});

test("cache-row parser keeps editable q and K cache rows", () => {
  const operation = listCachemirOperations().find((item) => item.id === CACHEMIR_OPERATION_IDS.SCORE);
  const defaults = cachemirOperationDefaults(operation.id);
  const parsed = parseCachemirOperationInputs({
    valuesTextA: defaults.A.map((row) => row.join(" ")).join("\n"),
    valuesTextB: defaults.B.map((row) => row.join(" ")).join("\n"),
    slots: "8", leftType: "ct", rightType: "ct", seed: "1",
  }, operation);
  assert.equal(parsed.ok, true);
  assert.deepEqual(parsed.inputs.B, defaults.B);
});
