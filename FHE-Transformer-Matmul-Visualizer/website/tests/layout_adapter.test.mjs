// Tests for the layout adapter: slot maps produced by the real Rotom layout
// engine (spec §19, §10). Expected values below were verified by running the
// engine directly against `website/layout_engine.js`.

import { test } from "node:test";
import assert from "node:assert/strict";
import { buildCiphertextStates, buildCiphertextStatesFromArrays } from "../transformer/core/layout_adapter.js";

const T = [
  [0, 1, 2, 3],
  [4, 5, 6, 7],
  [8, 9, 10, 11],
  [12, 13, 14, 15],
];

function values(cts) {
  return cts.map((ct) => ct.slots.map((s) => s.value));
}

function kinds(ct) {
  return ct.slots.map((s) => s.kind);
}

test("row-major [0:4:1][1:4:1]: one ciphertext, row-fastest", () => {
  const cts = buildCiphertextStates({
    tensorId: "X",
    tensor: T,
    layoutString: "[0:4:1][1:4:1]",
    n: 16,
    secret: false,
    stepId: "s1",
    idPrefix: "X-pack",
  });
  assert.equal(cts.length, 1);
  assert.deepEqual(values(cts), [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]);
  const slot5 = cts[0].slots[5];
  assert.deepEqual(slot5.sourceRefs[0].coords, [1, 1]);
  assert.equal(slot5.sourceRefs[0].linear, 5);
});

test("column-major [1:4:1][0:4:1]: one ciphertext, column-major", () => {
  const cts = buildCiphertextStates({
    tensorId: "X",
    tensor: T,
    layoutString: "[1:4:1][0:4:1]",
    n: 16,
    secret: false,
    stepId: "s1",
    idPrefix: "X-pack",
  });
  assert.deepEqual(values(cts), [[0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]]);
  assert.deepEqual(cts[0].slots[1].sourceRefs[0].coords, [1, 0]);
});

test("roll(0,1) [0:4:1][1:4:1]: verified cyclic mapping", () => {
  const cts = buildCiphertextStates({
    tensorId: "X",
    tensor: T,
    layoutString: "roll(0,1) [0:4:1][1:4:1]",
    n: 16,
    secret: false,
    stepId: "s1",
    idPrefix: "X-roll",
  });
  assert.deepEqual(values(cts), [[0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11]]);
  // slot 2 → T[2,2]
  assert.deepEqual(cts[0].slots[2].sourceRefs[0].coords, [2, 2]);
  // slot 7 → T[0,3]
  assert.deepEqual(cts[0].slots[7].sourceRefs[0].coords, [0, 3]);
});

test("replicated [R:4:1];[0:4:1][1:4:1]: four ciphertexts, same content", () => {
  const cts = buildCiphertextStates({
    tensorId: "X",
    tensor: T,
    layoutString: "[R:4:1];[0:4:1][1:4:1]",
    n: 16,
    secret: false,
    stepId: "s1",
    idPrefix: "X-rep",
  });
  assert.equal(cts.length, 4);
  for (const ct of cts) {
    assert.deepEqual(ct.slots.map((s) => s.value), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
  }
});

test("replicated [R:4:1];[1:4:1][0:4:1]: four ciphertexts, column-major content", () => {
  const cts = buildCiphertextStates({
    tensorId: "X",
    tensor: T,
    layoutString: "[R:4:1];[1:4:1][0:4:1]",
    n: 16,
    secret: false,
    stepId: "s1",
    idPrefix: "X-rep",
  });
  assert.equal(cts.length, 4);
  assert.deepEqual(values(cts)[0], [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]);
});

test("oob slots are surfaced for a tensor smaller than the layout extent", () => {
  // A 2×2 tensor packed with a 4×4 layout: slots 0,1,4,5 are the real cells;
  // everything else is out-of-range (coords beyond the 2×2 tensor).
  const cts = buildCiphertextStates({
    tensorId: "X",
    tensor: [[1, 2], [3, 4]],
    layoutString: "[0:4:1][1:4:1]",
    n: 16,
    secret: false,
    stepId: "s1",
    idPrefix: "X-pack",
  });
  assert.equal(cts.length, 1);
  const ks = kinds(cts[0]);
  assert.equal(ks[0], "tensor"); // X[0,0]
  assert.equal(ks[2], "oob"); // coords [0,2]
  assert.equal(ks[4], "tensor"); // X[1,0]
  assert.equal(ks[6], "oob"); // coords [1,2]
  assert.equal(ks[15], "oob");
});

test("gap slots appear when n exceeds the layout product", () => {
  const cts = buildCiphertextStates({
    tensorId: "A",
    tensor: [[1, 0], [0, 1]],
    layoutString: "[0:2:1][1:2:1]",
    n: 8,
    secret: false,
    stepId: "s1",
    idPrefix: "A-pack",
  });
  const ks = kinds(cts[0]);
  assert.deepEqual(ks.slice(0, 4), ["tensor", "tensor", "tensor", "tensor"]);
  assert.deepEqual(ks.slice(4), ["gap", "gap", "gap", "gap"]);
});

test("logicalId is stable across steps; id includes the stepId (spec §15.2)", () => {
  const mk = (stepId) =>
    buildCiphertextStates({
      tensorId: "X",
      tensor: T,
      layoutString: "[0:4:1][1:4:1]",
      n: 16,
      secret: false,
      stepId,
      idPrefix: "X-pack",
      logicalIdFor: (m) => `X-pack:${m.coords[0]}:${m.coords[1]}`,
    });
  const a = mk("step-1");
  const b = mk("step-2");
  assert.equal(a[0].slots[6].logicalId, b[0].slots[6].logicalId); // stable
  assert.match(a[0].slots[6].id, /^step-1:/);
  assert.match(b[0].slots[6].id, /^step-2:/);
});

test("same value at different coordinates is not conflated", () => {
  const tensor = [
    [5, 1],
    [2, 5],
  ];
  const cts = buildCiphertextStates({
    tensorId: "X",
    tensor,
    layoutString: "[0:2:1][1:2:1]",
    n: 4,
    secret: false,
    stepId: "s1",
    idPrefix: "X-pack",
  });
  // X[0,0]=5 and X[1,1]=5 must have different sourceRefs.
  assert.notDeepEqual(cts[0].slots[0].sourceRefs[0].coords, cts[0].slots[3].sourceRefs[0].coords);
});

test("buildCiphertextStatesFromArrays wraps explicit slot arrays", () => {
  const cts = buildCiphertextStatesFromArrays({
    id: "P",
    secret: true,
    stepId: "s2",
    arrays: [
      [
        { value: 8, logicalId: "P:0:0:0", kind: "product", outputCoords: [0, 0], contractionIndex: 0, expression: "X[0,0] × W[0,0]" },
        { value: 0, kind: "gap", logicalId: "P:gap:0" },
      ],
    ],
  });
  assert.equal(cts[0].slots.length, 2);
  assert.equal(cts[0].slots[0].kind, "product");
  assert.equal(cts[0].slots[0].expression, "X[0,0] × W[0,0]");
  assert.equal(cts[0].slots[1].kind, "gap");
});
