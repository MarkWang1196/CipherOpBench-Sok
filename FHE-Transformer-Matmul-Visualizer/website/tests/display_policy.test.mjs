// Tests for the Full / Tile / Summary display policy (spec §11.6, §20.3).

import { test } from "node:test";
import assert from "node:assert/strict";
import { chooseDisplayMode, displayModeLabel } from "../transformer/core/display_policy.js";
import { DISPLAY_MODES } from "../transformer/core/trace_schema.js";

test("small shapes → Full", () => {
  assert.equal(
    chooseDisplayMode({ m: 4, k: 4, n: 4, planeCount: 4, tileGrid: { m: 1, n: 1 } }),
    DISPLAY_MODES.FULL
  );
});

test("medium or multi-tile → Tile", () => {
  assert.equal(
    chooseDisplayMode({ m: 8, k: 8, n: 8, planeCount: 16, tileGrid: { m: 1, n: 4 } }),
    DISPLAY_MODES.TILE
  );
  assert.equal(
    chooseDisplayMode({ m: 12, k: 12, n: 12, planeCount: 40, tileGrid: { m: 2, n: 2 } }),
    DISPLAY_MODES.TILE
  );
});

test("large → Summary", () => {
  assert.equal(
    chooseDisplayMode({ m: 24, k: 24, n: 24, planeCount: 512, tileGrid: { m: 2, n: 8 } }),
    DISPLAY_MODES.SUMMARY
  );
});

test("displayModeLabel returns readable labels", () => {
  assert.equal(displayModeLabel(DISPLAY_MODES.FULL), "Full");
  assert.equal(displayModeLabel(DISPLAY_MODES.SUMMARY), "Summary");
});
