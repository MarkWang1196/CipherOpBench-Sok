import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ALGORITHM_IDS,
  buildAlgorithmTrace,
  listAlgorithms,
} from "../transformer/algorithms/algorithm_registry.js";
import { validateTrace } from "../transformer/core/trace_validator.js";

const PAPER_QUERY_SEQUENCES = [
  {
    method: "THOR",
    id: ALGORITHM_IDS.QUERY,
    source: /THOR Figure 3\/4/,
    titles: [
      "Setup: Wq × X inputs",
      "Setup: multi-diagonal pack X",
      "Step 1a: prepare parallel operands",
      "Step 1b: parallel submatrix products",
      "Step 2a: rotate partial products",
      "Step 2b: mask and align wraparound",
      "Step 3: aggregate aligned products",
      "Visual check: decoded Q",
    ],
    stepTitles: [
      "Input plaintext weight and encrypted activation",
      "Pack encrypted activation diagonals",
      "Prepare parallel block operands",
      "Compute parallel block products",
      "Rotate the second partial-product family",
      "Mask and align wraparound lanes",
      "Aggregate aligned block products",
      "Verify encrypted Q output",
    ],
  },
  {
    method: "MOAI",
    id: ALGORITHM_IDS.MOAI_QUERY,
    source: /MOAI Algorithms 2-7/,
    titles: ["X × Wq", "Col", "Q₀", "Q₁", "Σ", "Q"],
    stepTitles: ["Input encrypted X and plaintext W", "Interleaved column-pack X", "Form Q column 0", "Form Q column 1", "Sum products by output column", "Return column-packed output"],
  },
  {
    method: "BLB",
    id: ALGORITHM_IDS.BLB_QUERY,
    source: /BLB Sections 4-5/,
    titles: ["X × Wq", "Spatial-first", "Baby ρ", "⊙ PT", "Giant ρ + Σ", "Q"],
    stepTitles: ["Input encrypted activation and plaintext weight", "Pack activation in spatial-first order", "Generate BSGS baby rotations", "Multiply plaintext weight lanes", "Apply giant rotations and sum", "Return spatial-first output"],
  },
  {
    method: "Cachemir",
    id: ALGORITHM_IDS.CACHEMIR_QUERY,
    source: /Cachemir Figures 4-6/,
    titles: ["x × Wq", "Interleave", "① ρ +", "② ρ ⊙ diag(W)", "③ Σ + ρ1", "④ Mask ⊙ RoPE"],
    stepTitles: ["Input encrypted vector and plaintext weight", "Interleave activation with empty slots", "Replicate activation by rotate-and-add", "Multiply rotated lanes by weight diagonals", "Sum products and fold output lanes", "Mask output and fuse RoPE"],
  },
  {
    method: "Euston",
    id: ALGORITHM_IDS.EUSTON_QUERY,
    source: /Euston Sections 4\.1-4\.2 and Figure 2/,
    titles: ["A × Wq", "R = U · D · H", "① PCCDMM", "② CCPUMM", "③ X = A − R; compute XW", "④ AW = RW + XW"],
    stepTitles: ["Set up the private input projection", "Factor the offline mask and encrypt D", "Build the encrypted columns of UD",
      "Precompute the encrypted mask product RW", "Multiply the masked input online", "Restore the projection with the mask product"],
  },
  {
    method: "STIP",
    id: ALGORITHM_IDS.STIP_QUERY,
    source: /STIP Algorithms 1-9/,
    titles: ["X × Wq", "AMCP", "RotL₂ · u + RotR₂ · v", "DHP ⊙ Σ", "Conj", "Q"],
    stepTitles: ["Input activation and projection weights", "Interleave features within each token", "Align the two feature lanes",
      "Project two heads using complex weights", "Separate the two projected heads", "Verify the packed Q output"],
  },
  {
    method: "NEXUS",
    id: ALGORITHM_IDS.NEXUS_QUERY,
    source: /NEXUS Figures 3-5, Algorithms 1\/5/,
    titles: ["X × Wq", "Compress", "⊞ Subs · x⁻²ⁱ", "Decompress", "Offline U·W", "Online (A−U)W ⊞ ENC(UW)", "Q"],
    stepTitles: ["Set up the offline–online projection", "Encode weights as polynomial coefficients", "Split the even and odd coefficients",
      "Extract repeated SIMD weights", "Precompute the encrypted correction UW", "Add the online masked product", "Check the encrypted Q output"],
  },
];

for (const paper of PAPER_QUERY_SEQUENCES) {
  test(`${paper.method} query stages remain aligned with the audited paper sequence`, () => {
    const trace = buildAlgorithmTrace(paper.id);
    const graphTitles = trace.graph.stages.map((stage) => stage.title);
    const stepTitles = trace.steps.map((step) => step.title);

    assert.deepEqual(graphTitles, paper.titles);
    assert.deepEqual(stepTitles, paper.stepTitles);
    assert.ok(trace.steps.every((step) => step.description.length > 0));
    assert.ok(trace.graph.stages.every((stage) => stage.shortTitle.length > 0));
    assert.match(trace.plan.plannerSource, paper.source);
  });
}

for (const method of ["euston", "stip", "nexus"]) {
  test(`${method} describes every step and renders the decoded result for default and edited inputs`, () => {
    const algorithms = listAlgorithms().filter(({ id }) => id.startsWith(`${method}-`));
    for (const { id } of algorithms) {
      const original = buildAlgorithmTrace(id);
      const { left, right } = original.steps[0].tensors;
      const A = left.values.map((row, i) => row.map((_, j) => (i + 2) * 2.7 - (j + 1) * 3.3));
      const B = right.values.map((row, i) => row.map((_, j) =>
        id === ALGORITHM_IDS.EUSTON_PCCDMM && i !== j ? 0 : i * 3.75 - j * 2.1 + 1.8));
      for (const trace of [original, buildAlgorithmTrace(id, { A, B })]) {
        const { ok, errors } = validateTrace(trace);
        assert.ok(ok, `${id}: ${errors.join(", ")}`);
        assert.equal(trace.steps.length, trace.graph.stages.length, id);
        trace.steps.forEach((step, index) => {
          const stage = trace.graph.stages[index];
          assert.equal(stage.revealAt, index, id);
          assert.equal(step.title, stage.stepTitle, id);
          assert.equal(step.description, stage.description, id);
          assert.ok(step.title.trim() && step.description.trim() && stage.shortTitle.trim(), `${id} step ${index + 1}`);
        });
        if (id === ALGORITHM_IDS.NEXUS_DECOMPRESS) continue;
        const decoded = trace.graph.stages.at(-1).rows.flat().find((artifact) => artifact.id === "C");
        assert.ok(decoded, id);
        assert.equal(decoded.rows.length, trace.finalTensor.length, id);
        for (const [i, row] of decoded.rows.entries()) {
          assert.equal(row.length, trace.finalTensor[i].length, id);
          for (const [j, cell] of row.entries()) {
            const expected = trace.finalTensor[i][j];
            assert.ok(Math.abs(cell.value - expected) < 1e-10 * (1 + Math.abs(expected)), `${id} output [${i},${j}]`);
          }
        }
      }
    }
  });
}
