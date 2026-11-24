import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ALGORITHM_IDS,
  buildAlgorithmTrace,
  getAlgorithm,
  listAlgorithms,
} from "../transformer/algorithms/algorithm_registry.js";

test("algorithm registry exposes generic OPP and all paper adapters", () => {
  const ids = listAlgorithms().map((item) => item.id);
  assert.deepEqual(ids.slice(0, 3), [ALGORITHM_IDS.ROTOM_OPP, ALGORITHM_IDS.THOR_PCMM, ALGORITHM_IDS.THOR_CCMM]);
  assert.deepEqual(ids.slice(3, 13), [
    ALGORITHM_IDS.QUERY,
    ALGORITHM_IDS.KEY,
    ALGORITHM_IDS.VALUE,
    ALGORITHM_IDS.SCORE,
    ALGORITHM_IDS.CONTEXT,
    ALGORITHM_IDS.OUTPUT,
    ALGORITHM_IDS.FC1,
    ALGORITHM_IDS.FC2,
    ALGORITHM_IDS.POOLER,
    ALGORITHM_IDS.CLASSIFIER,
  ]);
  assert.deepEqual(ids.slice(13, 23), [
    ALGORITHM_IDS.MOAI_QUERY,
    ALGORITHM_IDS.MOAI_KEY,
    ALGORITHM_IDS.MOAI_VALUE,
    ALGORITHM_IDS.MOAI_SCORE,
    ALGORITHM_IDS.MOAI_CONTEXT,
    ALGORITHM_IDS.MOAI_OUTPUT,
    ALGORITHM_IDS.MOAI_FC1,
    ALGORITHM_IDS.MOAI_FC2,
    ALGORITHM_IDS.MOAI_POOLER,
    ALGORITHM_IDS.MOAI_CLASSIFIER,
  ]);
  assert.deepEqual(ids.slice(23, 33), [
    ALGORITHM_IDS.BLB_QUERY,
    ALGORITHM_IDS.BLB_KEY,
    ALGORITHM_IDS.BLB_VALUE,
    ALGORITHM_IDS.BLB_SCORE,
    ALGORITHM_IDS.BLB_CONTEXT,
    ALGORITHM_IDS.BLB_OUTPUT,
    ALGORITHM_IDS.BLB_FC1,
    ALGORITHM_IDS.BLB_FC2,
    ALGORITHM_IDS.BLB_POOLER,
    ALGORITHM_IDS.BLB_CLASSIFIER,
  ]);
  assert.deepEqual(ids.slice(33, 42), [
    ALGORITHM_IDS.CACHEMIR_QUERY,
    ALGORITHM_IDS.CACHEMIR_KEY,
    ALGORITHM_IDS.CACHEMIR_VALUE,
    ALGORITHM_IDS.CACHEMIR_SCORE,
    ALGORITHM_IDS.CACHEMIR_CONTEXT,
    ALGORITHM_IDS.CACHEMIR_OUTPUT,
    ALGORITHM_IDS.CACHEMIR_UP,
    ALGORITHM_IDS.CACHEMIR_GATE,
    ALGORITHM_IDS.CACHEMIR_DOWN,
  ]);
  assert.deepEqual(ids.slice(42, 54), [
    ALGORITHM_IDS.EUSTON_PCCDMM,
    ALGORITHM_IDS.EUSTON_CCPUMM,
    ALGORITHM_IDS.EUSTON_CCCRMM,
    ALGORITHM_IDS.EUSTON_CDCCMM,
    ALGORITHM_IDS.EUSTON_QUERY,
    ALGORITHM_IDS.EUSTON_KEY,
    ALGORITHM_IDS.EUSTON_VALUE,
    ALGORITHM_IDS.EUSTON_SCORE,
    ALGORITHM_IDS.EUSTON_CONTEXT,
    ALGORITHM_IDS.EUSTON_OUTPUT,
    ALGORITHM_IDS.EUSTON_FC1,
    ALGORITHM_IDS.EUSTON_FC2,
  ]);
  assert.deepEqual(ids.slice(54, 68), [
    ALGORITHM_IDS.STIP_RIHP,
    ALGORITHM_IDS.STIP_L2,
    ALGORITHM_IDS.STIP_DHP,
    ALGORITHM_IDS.STIP_AMCP,
    ALGORITHM_IDS.STIP_BSGS,
    ALGORITHM_IDS.STIP_QUERY,
    ALGORITHM_IDS.STIP_KEY,
    ALGORITHM_IDS.STIP_VALUE,
    ALGORITHM_IDS.STIP_SCORE,
    ALGORITHM_IDS.STIP_DISTANCE,
    ALGORITHM_IDS.STIP_CONTEXT,
    ALGORITHM_IDS.STIP_OUTPUT,
    ALGORITHM_IDS.STIP_FC1,
    ALGORITHM_IDS.STIP_FC2,
  ]);
  assert.deepEqual(ids.slice(68), [
    ALGORITHM_IDS.NEXUS_DECOMPRESS,
    ALGORITHM_IDS.NEXUS_OFFLINE_ONLINE,
    ALGORITHM_IDS.NEXUS_COLUMN_CCMM,
    ALGORITHM_IDS.NEXUS_QUERY,
    ALGORITHM_IDS.NEXUS_KEY,
    ALGORITHM_IDS.NEXUS_VALUE,
    ALGORITHM_IDS.NEXUS_SCORE,
    ALGORITHM_IDS.NEXUS_CONTEXT,
    ALGORITHM_IDS.NEXUS_OUTPUT,
    ALGORITHM_IDS.NEXUS_FC1,
    ALGORITHM_IDS.NEXUS_FC2,
  ]);
  assert.equal(getAlgorithm(ALGORITHM_IDS.THOR_PCMM).label, "THOR Figure 3 PC-MM");
  assert.equal(getAlgorithm(ALGORITHM_IDS.THOR_CCMM).label, "THOR Figure 4 CC-MM");
});

test("algorithm registry builds graphical Cachemir operations", () => {
  const score = buildAlgorithmTrace(ALGORITHM_IDS.CACHEMIR_SCORE);
  const context = buildAlgorithmTrace(ALGORITHM_IDS.CACHEMIR_CONTEXT);
  assert.equal(score.plan.method, "cachemir");
  assert.equal(score.plan.packingStrategyResolved, "Interleaved K-cache VMM");
  assert.equal(context.plan.packingStrategyResolved, "Interleaved V-cache VMM");
  assert.match(score.graph.stages.at(-1).title, /K Cache/);
  assert.match(context.graph.stages.at(-1).title, /V Cache/);
});

test("algorithm registry builds graphical Euston operations", () => {
  const score = buildAlgorithmTrace(ALGORITHM_IDS.EUSTON_SCORE);
  const context = buildAlgorithmTrace(ALGORITHM_IDS.EUSTON_CONTEXT);
  assert.equal(score.plan.method, "euston");
  assert.match(score.plan.packingStrategyResolved, /diagonal-packed/);
  assert.match(context.plan.packingStrategyResolved, /column-packed/);
  assert.equal(context.parameters.batchCount, 2);
});

test("algorithm registry builds graphical STIP operations", () => {
  const score = buildAlgorithmTrace(ALGORITHM_IDS.STIP_SCORE);
  assert.equal(score.plan.method, "stip");
  assert.equal(score.graph.title, "QKᵀ");
  assert.equal(score.finalTensor[0][1], 8);
});

test("algorithm registry builds graphical NEXUS operations", () => {
  const score = buildAlgorithmTrace(ALGORITHM_IDS.NEXUS_SCORE);
  const linear = buildAlgorithmTrace(ALGORITHM_IDS.NEXUS_QUERY);
  assert.equal(score.plan.method, "nexus");
  assert.match(score.plan.packingStrategyResolved, /Algorithm 5/);
  assert.match(linear.plan.packingStrategyResolved, /offline-online/);
  assert.equal(score.finalTensor[0][1], 8);
});

test("algorithm registry builds graphical BLB operations", () => {
  const score = buildAlgorithmTrace(ALGORITHM_IDS.BLB_SCORE);
  const context = buildAlgorithmTrace(ALGORITHM_IDS.BLB_CONTEXT);
  assert.equal(score.plan.method, "blb");
  assert.equal(score.plan.packingStrategyResolved, "MHP spatial×reduce→diagonal");
  assert.match(context.graph.stages[1].title, /MPC → CKKS/);
  assert.equal(context.parameters.batchCount, 2);
});

test("algorithm registry builds graphical MOAI operations", () => {
  const score = buildAlgorithmTrace(ALGORITHM_IDS.MOAI_SCORE);
  const context = buildAlgorithmTrace(ALGORITHM_IDS.MOAI_CONTEXT);
  assert.equal(score.plan.method, "moai");
  assert.equal(score.plan.packingStrategyResolved, "Col×Col→Diag");
  assert.equal(context.plan.packingStrategyResolved, "Diag×Col→Col");
  assert.equal(context.parameters.batchCount, 2);
});

test("algorithm registry builds graphical THOR toy operations", () => {
  const score = buildAlgorithmTrace(ALGORITHM_IDS.SCORE);
  const context = buildAlgorithmTrace(ALGORITHM_IDS.CONTEXT);
  assert.equal(score.parameters.matmulType, "ct-ct");
  assert.equal(context.plan.visualStyle, "diagram-only");
  assert.equal(context.graph.title, "SV");
});

test("algorithm registry builds the THOR trace", () => {
  const trace = buildAlgorithmTrace(ALGORITHM_IDS.THOR_PCMM);
  assert.equal(trace.id, "thor-pcmm-figure3-paper");
  assert.equal(trace.graph.stages.length, trace.steps.length);
});

test("algorithm registry builds both Figure 4 CC-MM approaches", () => {
  const lower = buildAlgorithmTrace(ALGORITHM_IDS.THOR_CCMM, { method: "lower-lower" });
  const upper = buildAlgorithmTrace(ALGORITHM_IDS.THOR_CCMM, { method: "upper-lower" });
  assert.deepEqual(lower.finalTensor, upper.finalTensor);
  assert.equal(lower.graph.stages.length, lower.steps.length);
});

test("algorithm registry rejects unknown adapters", () => {
  assert.throws(() => buildAlgorithmTrace("unknown-paper"), /Unknown visualization algorithm/);
});
