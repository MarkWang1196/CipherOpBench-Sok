import { test } from "node:test";
import assert from "node:assert/strict";
import { buildThorFigure3PcmmTrace } from "../transformer/algorithms/thor/figure3_pcmm.js";
import { buildThorFigure4CcmmTrace } from "../transformer/algorithms/thor/figure4_ccmm.js";
import { validateTrace } from "../transformer/core/trace_validator.js";

function artifact(trace, id) {
  for (const stage of trace.graph.stages) {
    for (const row of stage.rows) {
      const found = row.find((item) => item.id === id);
      if (found) return found;
    }
  }
  return null;
}

test("Figure 3 uses the paper multi-diagonal B ordering and color ordering", () => {
  const trace = buildThorFigure3PcmmTrace();
  const ctB = artifact(trace, "fig3-ct-b");
  assert.deepEqual(ctB.cells.map((cell) => cell.value), [1, 2, 1, 2, 0, 1, 0, 1]);
  assert.deepEqual(ctB.cells.map((cell) => cell.tone), [0, 2, 0, 2, 1, 3, 1, 3]);
  const output = artifact(trace, "fig3-ct-c");
  assert.deepEqual(output.cells.map((cell) => cell.value), [11, 3, 29, 7, 27, 7, 13, 3]);
  assert.deepEqual(output.cells.map((cell) => cell.tone), [0, 2, 0, 2, 1, 3, 1, 3]);
  assert.equal(validateTrace(trace).ok, true);
});

test("Figure 3 rotation carries colors with elements and masked slots are neutral", () => {
  const trace = buildThorFigure3PcmmTrace();
  const rotated = artifact(trace, "fig3-rho-partial");
  assert.deepEqual(rotated.cells.map((cell) => cell.tone), [2, 0, 2, 1, 3, 1, 3, 0]);
  const beta = artifact(trace, "fig3-beta-mask");
  assert.deepEqual(beta.cells.map((cell) => cell.tone), [2, 4, 2, 4, 3, 4, 3, 4]);
});

test("Figure 4 lower-lower and upper-lower produce the same encoded result", () => {
  const lower = buildThorFigure4CcmmTrace({ method: "lower-lower" });
  const upper = buildThorFigure4CcmmTrace({ method: "upper-lower" });
  assert.deepEqual(lower.finalTensor, upper.finalTensor);
  assert.deepEqual(artifact(lower, "fig4-ct-c").cells.map((cell) => cell.value), [11, 29, 8, 24, 27, 13, 24, 8]);
  assert.equal(validateTrace(lower).ok, true);
  assert.equal(validateTrace(upper).ok, true);
});
