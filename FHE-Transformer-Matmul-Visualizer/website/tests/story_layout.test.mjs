import { test } from "node:test";
import assert from "node:assert/strict";
import {
  groupStoryExpression,
  nearestStoryStageIndex,
} from "../transformer/render/story_view.js";
import {
  listThorOperations,
  buildThorOperationTrace,
} from "../transformer/algorithms/thor/bert_operations.js";
import {
  listMoaiOperations,
  buildMoaiOperationTrace,
} from "../transformer/algorithms/moai/bert_operations.js";
import {
  listBlbOperations,
  buildBlbOperationTrace,
} from "../transformer/algorithms/blb/bert_operations.js";
import {
  listCachemirOperations,
  buildCachemirOperationTrace,
} from "../transformer/algorithms/cachemir/llama_operations.js";

test("expression wrapping keeps every operator attached to its following artifact", () => {
  const A = { id: "A", type: "matrix" };
  const times = { id: "times", type: "operator" };
  const B = { id: "B", type: "matrix" };
  const equals = { id: "equals", type: "operator" };
  const C = { id: "C", type: "result" };
  assert.deepEqual(groupStoryExpression([A, times, B, equals, C]), [[A], [times, B], [equals, C]]);
});

test("drag snapping selects the stage nearest the viewport center", () => {
  const stages = [
    { offsetLeft: 100, offsetWidth: 200 },
    { offsetLeft: 340, offsetWidth: 300 },
    { offsetLeft: 680, offsetWidth: 220 },
  ];
  assert.equal(nearestStoryStageIndex(210, stages), 0);
  assert.equal(nearestStoryStageIndex(500, stages), 1);
  assert.equal(nearestStoryStageIndex(820, stages), 2);
  assert.equal(nearestStoryStageIndex(100, []), -1);
});

test("stacked stages use their vertical bounds when a card is taller than the viewport", () => {
  const stages = [
    { offsetLeft: 20, offsetWidth: 420, offsetTop: 10, offsetHeight: 100 },
    { offsetLeft: 20, offsetWidth: 420, offsetTop: 130, offsetHeight: 600 },
    { offsetLeft: 20, offsetWidth: 420, offsetTop: 750, offsetHeight: 100 },
  ];
  const axis = { start: "offsetTop", size: "offsetHeight" };
  assert.equal(nearestStoryStageIndex(155, stages, axis), 1);
  assert.equal(nearestStoryStageIndex(710, stages, axis), 1);
  assert.equal(nearestStoryStageIndex(810, stages, axis), 2);
});

test("all THOR, MOAI, BLB and Cachemir rows preserve their artifact order when wrapped", () => {
  const methods = [
    [listThorOperations, buildThorOperationTrace],
    [listMoaiOperations, buildMoaiOperationTrace],
    [listBlbOperations, buildBlbOperationTrace],
    [listCachemirOperations, buildCachemirOperationTrace],
  ];
  let stageCount = 0;
  for (const [list, build] of methods) {
    for (const operation of list()) {
      for (const stage of build(operation.id).graph.stages) {
        stageCount += 1;
        for (const row of stage.rows) {
          const groups = groupStoryExpression(row);
          assert.deepEqual(groups.flat(), row, `${operation.id}/${stage.id}`);
          assert.ok(groups.every((group) => group.length <= 2), `${operation.id}/${stage.id}`);
          assert.ok(groups.every((group) => group.length === 1 || group[0].type === "operator"), `${operation.id}/${stage.id}`);
        }
      }
    }
  }
  assert.equal(stageCount, 252);
});
