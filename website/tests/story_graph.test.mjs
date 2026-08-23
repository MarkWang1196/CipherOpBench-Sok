import { test } from "node:test";
import assert from "node:assert/strict";
import { buildMatmulScene } from "../transformer/core/scene_builder.js";
import { buildGenericStoryGraph } from "../transformer/render/story_view.js";

function traceWithFirstValue(value) {
  return buildMatmulScene({
    left: { id: "A", label: "A", values: [[value, 2], [3, 4]], secret: true },
    right: { id: "B", label: "B", values: [[1, 0], [0, 1]], secret: false },
    slotsPerCiphertext: 8,
    packingStrategy: "auto",
    options: { outputId: "C", outputLabel: "C" },
  });
}

test("generic story graph is rebuilt from edited matrix values", () => {
  const before = traceWithFirstValue(1);
  const after = traceWithFirstValue(9);
  // Scene ids can remain identical across edits; the rendered data must not.
  assert.equal(before.id, after.id);
  const beforeCell = buildGenericStoryGraph(before).stages[0].rows[0][0].rows[0][0].value;
  const afterCell = buildGenericStoryGraph(after).stages[0].rows[0][0].rows[0][0].value;
  assert.equal(beforeCell, 1);
  assert.equal(afterCell, 9);
});

