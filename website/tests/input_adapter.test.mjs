import { test } from "node:test";
import assert from "node:assert/strict";
import { ALGORITHM_IDS } from "../transformer/algorithms/algorithm_registry.js";
import { buildTraceFromInputs } from "../transformer/algorithms/input_adapter.js";
import { THOR_FIGURE3_PRESET } from "../transformer/algorithms/thor/figure3_pcmm.js";
import { THOR_FIGURE4_PRESET } from "../transformer/algorithms/thor/figure4_ccmm.js";
import { MOAI_OPERATION_IDS, moaiOperationDefaults } from "../transformer/algorithms/moai/bert_operations.js";
import { BLB_OPERATION_IDS, blbOperationDefaults } from "../transformer/algorithms/blb/bert_operations.js";
import { CACHEMIR_OPERATION_IDS, cachemirOperationDefaults } from "../transformer/algorithms/cachemir/llama_operations.js";
import { EUSTON_OPERATION_IDS, eustonOperationDefaults } from "../transformer/algorithms/euston/transformer_operations.js";
import { STIP_OPERATION_IDS, stipOperationDefaults } from "../transformer/algorithms/stip/transformer_operations.js";
import { NEXUS_OPERATION_IDS, nexusOperationDefaults } from "../transformer/algorithms/nexus/transformer_operations.js";
import { TRICYCLE_OPERATION_IDS, tricycleOperationDefaults } from "../transformer/algorithms/tricycle/transformer_operations.js";

function thorInputs(overrides = {}) {
  return {
    m: 4,
    k: 4,
    n: 2,
    A: THOR_FIGURE3_PRESET.A.map((row) => [...row]),
    B: THOR_FIGURE3_PRESET.B.map((row) => [...row]),
    leftSecret: false,
    rightSecret: true,
    slotsPerCiphertext: 8,
    strategy: "diagonal",
    customLayouts: null,
    ...overrides,
  };
}

test("THOR form edits rebuild the THOR graph and final result", () => {
  const original = buildTraceFromInputs(ALGORITHM_IDS.THOR_PCMM, thorInputs());
  const editedA = THOR_FIGURE3_PRESET.A.map((row) => [...row]);
  editedA[0][0] = 9;
  const edited = buildTraceFromInputs(ALGORITHM_IDS.THOR_PCMM, thorInputs({ A: editedA }));

  assert.equal(original.graph.algorithmId, "thor-pcmm");
  assert.equal(edited.graph.algorithmId, "thor-pcmm");
  assert.equal(original.graph.stages[0].rows[0][0].rows[0][0].value, 1);
  assert.equal(edited.graph.stages[0].rows[0][0].rows[0][0].value, 9);
  assert.notDeepEqual(edited.finalTensor, original.finalTensor);
  assert.deepEqual(edited.finalTensor[0], [19, 13]);
});

test("THOR Figure 3 adapter enforces the paper geometry and PT x CT types", () => {
  assert.throws(
    () => buildTraceFromInputs(ALGORITHM_IDS.THOR_PCMM, {
      ...thorInputs(),
      m: 2,
      k: 3,
      n: 4,
      A: [[1, 2, 3], [4, 5, 6]],
      B: [[1, 0, 2, 1], [0, 1, 1, 0], [2, 1, 0, 1]],
    }),
    /Figure 3 requires A to be 4x4/
  );
  assert.throws(
    () => buildTraceFromInputs(ALGORITHM_IDS.THOR_PCMM, thorInputs({ leftSecret: true })),
    /requires plaintext A and ciphertext B/
  );
});

test("THOR input adapter rejects unsupported packing and insufficient slots", () => {
  assert.throws(
    () => buildTraceFromInputs(ALGORITHM_IDS.THOR_PCMM, thorInputs({ strategy: "row" })),
    /supports Auto or Diagonal packing/
  );
  assert.throws(() => buildTraceFromInputs(ALGORITHM_IDS.THOR_PCMM, thorInputs({ slotsPerCiphertext: 16 })), /exactly 8 slots/);
});

test("THOR Figure 4 adapter switches approaches and responds to edited values", () => {
  const base = {
    m: 2, k: 4, n: 4,
    A: THOR_FIGURE4_PRESET.A.map((row) => [...row]),
    B: THOR_FIGURE4_PRESET.B.map((row) => [...row]),
    leftSecret: true, rightSecret: true,
    slotsPerCiphertext: 8, strategy: "diagonal", customLayouts: null,
    thorCcmmMethod: "lower-lower",
  };
  const lower = buildTraceFromInputs(ALGORITHM_IDS.THOR_CCMM, base);
  const upper = buildTraceFromInputs(ALGORITHM_IDS.THOR_CCMM, { ...base, thorCcmmMethod: "upper-lower" });
  assert.deepEqual(lower.finalTensor, upper.finalTensor);
  const editedA = base.A.map((row) => [...row]);
  editedA[0][0] = 9;
  const edited = buildTraceFromInputs(ALGORITHM_IDS.THOR_CCMM, { ...base, A: editedA });
  assert.notDeepEqual(edited.finalTensor, lower.finalTensor);
  assert.throws(
    () => buildTraceFromInputs(ALGORITHM_IDS.THOR_CCMM, { ...base, rightSecret: false }),
    /requires ciphertext A and ciphertext B/
  );
});

test("MOAI adapter preserves its paper packing and semantic operand types", () => {
  const defaults = moaiOperationDefaults(MOAI_OPERATION_IDS.CONTEXT);
  const trace = buildTraceFromInputs(MOAI_OPERATION_IDS.CONTEXT, {
    ...defaults,
    customLayouts: null,
  });
  assert.equal(trace.plan.method, "moai");
  assert.equal(trace.plan.packingStrategyResolved, "Diag×Col→Col");
  assert.throws(
    () => buildTraceFromInputs(MOAI_OPERATION_IDS.CONTEXT, { ...defaults, rightSecret: false }),
    /requires CT × CT/
  );
  assert.throws(
    () => buildTraceFromInputs(MOAI_OPERATION_IDS.CONTEXT, { ...defaults, strategy: "diagonal" }),
    /uses column packing/
  );
});

test("BLB adapter preserves spatial-first packing and fixed hybrid operand types", () => {
  const defaults = blbOperationDefaults(BLB_OPERATION_IDS.CONTEXT);
  const trace = buildTraceFromInputs(BLB_OPERATION_IDS.CONTEXT, { ...defaults, customLayouts: null });
  assert.equal(trace.plan.method, "blb");
  assert.equal(trace.plan.packingStrategyResolved, "MPC→MHP spatial×reduce→diagonal");
  assert.throws(
    () => buildTraceFromInputs(BLB_OPERATION_IDS.CONTEXT, { ...defaults, rightSecret: false }),
    /requires encrypted\/shared × encrypted\/shared/
  );
  assert.throws(
    () => buildTraceFromInputs(BLB_OPERATION_IDS.CONTEXT, { ...defaults, strategy: "diagonal" }),
    /uses spatial-first packing/
  );
});

test("Cachemir adapter preserves interleaved replicated packing and operand types", () => {
  const defaults = cachemirOperationDefaults(CACHEMIR_OPERATION_IDS.SCORE);
  const trace = buildTraceFromInputs(CACHEMIR_OPERATION_IDS.SCORE, { ...defaults, customLayouts: null });
  assert.equal(trace.plan.method, "cachemir");
  assert.equal(trace.plan.packingStrategyResolved, "Interleaved K-cache VMM");
  assert.throws(
    () => buildTraceFromInputs(CACHEMIR_OPERATION_IDS.SCORE, { ...defaults, rightSecret: false }),
    /requires CT × CT/
  );
  assert.throws(
    () => buildTraceFromInputs(CACHEMIR_OPERATION_IDS.SCORE, { ...defaults, strategy: "diagonal" }),
    /uses interleaved replicated packing/
  );
});

test("Euston adapter preserves batched column packing and fixed operand types", () => {
  const defaults = eustonOperationDefaults(EUSTON_OPERATION_IDS.SCORE);
  const trace = buildTraceFromInputs(EUSTON_OPERATION_IDS.SCORE, { ...defaults, customLayouts: null });
  assert.equal(trace.plan.method, "euston");
  assert.match(trace.plan.packingStrategyResolved, /column-packed/);
  assert.throws(
    () => buildTraceFromInputs(EUSTON_OPERATION_IDS.SCORE, { ...defaults, rightSecret: false }),
    /requires CT × CT/
  );
  assert.throws(
    () => buildTraceFromInputs(EUSTON_OPERATION_IDS.SCORE, { ...defaults, strategy: "diagonal" }),
    /uses batched column\/diagonal packing/
  );
});

test("STIP adapter preserves compact packing and fixed operand types", () => {
  const defaults = stipOperationDefaults(STIP_OPERATION_IDS.CONTEXT);
  const trace = buildTraceFromInputs(STIP_OPERATION_IDS.CONTEXT, { ...defaults, customLayouts: null });
  assert.equal(trace.plan.method, "stip");
  assert.match(trace.plan.packingStrategyResolved, /RIHP columns with BSGS/);
  assert.throws(
    () => buildTraceFromInputs(STIP_OPERATION_IDS.CONTEXT, { ...defaults, rightSecret: false }),
    /requires CT × CT/
  );
  assert.throws(
    () => buildTraceFromInputs(STIP_OPERATION_IDS.CONTEXT, { ...defaults, strategy: "diagonal" }),
    /uses compact column\/diagonal packing/
  );
});

test("NEXUS adapter preserves batched column packing and fixed operand types", () => {
  const defaults = nexusOperationDefaults(NEXUS_OPERATION_IDS.SCORE);
  const trace = buildTraceFromInputs(NEXUS_OPERATION_IDS.SCORE, { ...defaults, customLayouts: null });
  assert.equal(trace.plan.method, "nexus");
  assert.match(trace.plan.packingStrategyResolved, /Algorithm 5/);
  assert.throws(
    () => buildTraceFromInputs(NEXUS_OPERATION_IDS.SCORE, { ...defaults, rightSecret: false }),
    /requires CT × CT/
  );
  assert.throws(
    () => buildTraceFromInputs(NEXUS_OPERATION_IDS.SCORE, { ...defaults, strategy: "diagonal" }),
    /uses batched column\/diagonal packing/
  );
});

test("Tricycle adapter preserves tricyclic packing and fixed operand types", () => {
  const defaults = tricycleOperationDefaults(TRICYCLE_OPERATION_IDS.SCORE);
  const trace = buildTraceFromInputs(TRICYCLE_OPERATION_IDS.SCORE, { ...defaults, customLayouts: null });
  assert.equal(trace.plan.method, "tricycle");
  assert.match(trace.plan.packingStrategyResolved, /tricyclic CT×CT batch MM/);
  assert.throws(
    () => buildTraceFromInputs(TRICYCLE_OPERATION_IDS.SCORE, { ...defaults, rightSecret: false }),
    /requires CT × CT/
  );
  assert.throws(
    () => buildTraceFromInputs(TRICYCLE_OPERATION_IDS.SCORE, { ...defaults, strategy: "column" }),
    /uses tricyclic packing/
  );
});
