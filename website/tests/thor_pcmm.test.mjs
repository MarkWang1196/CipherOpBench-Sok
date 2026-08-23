import { test } from "node:test";
import assert from "node:assert/strict";
import {
  THOR_PCMM_PRESET,
  buildThorPcmmDynamicTrace,
  buildThorPcmmFigure3Trace,
  cyclicDiagonal,
  segmentedRotateLeft,
} from "../transformer/algorithms/thor/pcmm.js";
import { matmul, matricesEqual } from "../transformer/core/matrix.js";
import { validateTrace } from "../transformer/core/trace_validator.js";

test("cyclicDiagonal preserves THOR diagonal coordinates", () => {
  const A = THOR_PCMM_PRESET.A;
  assert.deepEqual(cyclicDiagonal(A, 0), [1, 6, 0, 2]);
  assert.deepEqual(cyclicDiagonal(A, 1), [2, 7, 1, 3]);
  assert.deepEqual(cyclicDiagonal(A, 3), [4, 5, 1, 1]);
});

test("segmentedRotateLeft never leaks across packed columns", () => {
  const packed = [1, 2, 3, 4, 5, 6, 7, 8];
  assert.deepEqual(segmentedRotateLeft(packed, 1, 4), [2, 3, 4, 1, 6, 7, 8, 5]);
  assert.deepEqual(segmentedRotateLeft(packed, 3, 4), [4, 1, 2, 3, 8, 5, 6, 7]);
});

test("THOR PC-MM teaching trace equals direct matrix multiplication", () => {
  const trace = buildThorPcmmFigure3Trace();
  const expected = matmul(THOR_PCMM_PRESET.A, THOR_PCMM_PRESET.B);
  assert.ok(matricesEqual(trace.finalTensor, expected));
  assert.deepEqual(trace.finalTensor, [[11, 13], [27, 29], [3, 3], [7, 7]]);
  assert.equal(trace.parameters.matmulType, "pt-ct");
  assert.equal(trace.parameters.slotsPerCiphertext, 8);
  assert.equal(trace.steps.at(-1).metricsTotal.ptCtMultiplications, 4);
  assert.equal(trace.steps.at(-1).metricsTotal.rotations, 6);
  assert.equal(trace.steps.at(-1).metricsTotal.masks, 6);
});

test("THOR story graph has ordered, uniquely identified stages", () => {
  const graph = buildThorPcmmFigure3Trace().graph;
  assert.equal(graph.algorithmId, "thor-pcmm");
  assert.deepEqual(graph.stages.map((stage) => stage.revealAt), [0, 1, 2, 3, 4, 5, 6]);
  assert.equal(new Set(graph.stages.map((stage) => stage.id)).size, graph.stages.length);
  assert.ok(graph.stages.every((stage) => stage.rows.length > 0));
});

test("dynamic THOR PC-MM supports rectangular shapes, padding and column tiles", () => {
  const cases = [
    {
      A: [[1, 2, 3], [4, 5, 6]],
      B: [[1, 0, 2, 1], [0, 1, 1, 0], [2, 1, 0, 1]],
      slotsPerCiphertext: 8,
      padded: { m: 4, k: 4, n: 4 },
      tiles: 2,
    },
    {
      A: [[1, 2, 3], [4, 5, 6], [1, 0, 1], [2, 1, 0], [3, 2, 1]],
      B: [[1, 0], [0, 1], [2, 1]],
      slotsPerCiphertext: 8,
      padded: { m: 8, k: 8, n: 2 },
      tiles: 2,
    },
    {
      A: [[1, 2, 0, 1, 3], [0, 1, 3, 2, 1], [2, 0, 1, 1, 0]],
      B: [[1, 0, 2], [0, 1, 1], [2, 1, 0], [1, 2, 1], [1, 0, 1]],
      slotsPerCiphertext: 16,
      padded: { m: 8, k: 8, n: 4 },
      tiles: 2,
    },
  ];
  for (const cfg of cases) {
    const trace = buildThorPcmmDynamicTrace(cfg);
    assert.ok(matricesEqual(trace.finalTensor, matmul(cfg.A, cfg.B)));
    assert.deepEqual(trace.plan.paddedShape, cfg.padded);
    assert.equal(trace.plan.tileGrid.n, cfg.tiles);
    assert.equal(validateTrace(trace).ok, true, validateTrace(trace).errors.join("\n"));
  }
});
