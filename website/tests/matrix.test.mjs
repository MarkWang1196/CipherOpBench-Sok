// Tests for the pure matrix semantics module (spec §20.1).

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  matmul,
  transpose,
  buildContributions,
  groupContributionsByOutput,
  matricesEqual,
} from "../transformer/core/matrix.js";

test("transpose flips dimensions", () => {
  assert.deepEqual(transpose([[1, 2, 3], [4, 5, 6]]), [[1, 4], [2, 5], [3, 6]]);
  assert.deepEqual(transpose([[7]]), [[7]]);
});

test("matmul correctness: square, rectangular, single row/col, K=1", () => {
  assert.deepEqual(matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]), [[19, 22], [43, 50]]);
  assert.deepEqual(matmul([[1, 2, 3]], [[1], [2], [3]]), [[14]]);
  assert.deepEqual(matmul([[1], [2]], [[3, 4]]), [[3, 4], [6, 8]]);
  assert.deepEqual(matmul([[2], [5]], [[3]]), [[6], [15]]);
  // 4x4 reference from the spec default.
  const X = [[1, 2, 0, 1], [0, 1, 3, 2], [2, 0, 1, 1], [1, 1, 2, 0]];
  const Wq = [[1, 0, 2, 1], [0, 1, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]];
  assert.deepEqual(matmul(X, Wq), [[2, 4, 5, 5], [8, 8, 3, 5], [5, 3, 5, 3], [5, 3, 3, 5]]);
});

test("matmul rejects shape mismatches and non-rectangular input", () => {
  assert.throws(() => matmul([[1, 2]], [[1], [2], [3]]), /shape mismatch/);
  assert.throws(() => matmul([[1, 2], [3]], [[1], [2]]), /rectangular/);
  assert.throws(() => matmul([], [[1]]), /non-empty/);
});

test("buildContributions produces exactly m*n*k contributions", () => {
  const A = [[1, 2], [3, 4]];
  const B = [[5, 6, 7], [8, 9, 10]];
  const c = buildContributions(A, B);
  assert.equal(c.length, 2 * 3 * 2); // 12
  // Each output has k contributions.
  const groups = groupContributionsByOutput(c);
  assert.equal(groups.size, 6);
  for (const [key, group] of groups) {
    assert.equal(group.length, 2, `output ${key} should have 2 contributions`);
  }
});

test("contributions sum to the output", () => {
  const A = [[1, 2, 0, 1], [0, 1, 3, 2], [2, 0, 1, 1], [1, 1, 2, 0]];
  const Wq = [[1, 0, 2, 1], [0, 1, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]];
  const expected = matmul(A, Wq);
  const groups = groupContributionsByOutput(buildContributions(A, Wq));
  for (const [key, group] of groups) {
    const [i, j] = key.split(":").map(Number);
    const sum = group.reduce((acc, c) => acc + c.value, 0);
    assert.equal(sum, expected[i][j], `C[${i},${j}]`);
  }
});

test("contributions are identity-based, not value-based (spec §25.2)", () => {
  // Two different coordinates with the same value must stay distinct.
  const A = [[5, 5]];
  const B = [[1], [2]];
  const c = buildContributions(A, B);
  assert.equal(c[0].left.coords.join(","), "0,0");
  assert.equal(c[1].left.coords.join(","), "0,1");
  assert.notEqual(c[0].id, c[1].id);
});

test("matricesEqual compares numerically", () => {
  assert.equal(matricesEqual([[1, 2]], [[1, 2]]), true);
  assert.equal(matricesEqual([[1, 2]], [[1, 3]]), false);
  assert.equal(matricesEqual([[1, 2]], [[1, 2, 3]]), false);
});
