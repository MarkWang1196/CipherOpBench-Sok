// Timeline controller tests (spec §20.5) with injected render/animator stubs —
// no DOM involved.

import { test } from "node:test";
import assert from "node:assert/strict";
import { TimelineController } from "../transformer/animation/timeline_controller.js";
import { buildMatmulScene } from "../transformer/core/scene_builder.js";
import { SCENES } from "../transformer/scenes/defaults.js";

function makeTrace() {
  return buildMatmulScene({
    left: SCENES.xwq.left,
    right: SCENES.xwq.right,
    slotsPerCiphertext: 16,
    packingStrategy: "auto",
    options: {},
  });
}

function makeStubs() {
  const rendered = [];
  const animator = {
    calls: [],
    animate: async ({ from, to }) => {
      animator.calls.push({ op: "animate", from, to });
    },
    cancel: () => {
      animator.calls.push({ op: "cancel" });
    },
  };
  return { rendered, animator };
}

function makeController(animator = null, opts = {}) {
  const trace = makeTrace();
  const rendered = [];
  const ctrl = new TimelineController(
    trace,
    { renderStep: (t, i) => rendered.push(i) },
    animator,
    opts
  );
  return { ctrl, trace, rendered };
}

test("constructor renders step 0", () => {
  const { rendered, ctrl } = makeController();
  assert.deepEqual(rendered, [0]);
  assert.equal(ctrl.currentStep, 0);
  ctrl.destroy();
});

test("next/previous respect bounds", async () => {
  const { ctrl, trace } = makeController();
  await ctrl.previous();
  assert.equal(ctrl.currentStep, 0);
  await ctrl.next();
  assert.equal(ctrl.currentStep, 1);
  await ctrl.seek(trace.steps.length - 1);
  await ctrl.next();
  assert.equal(ctrl.currentStep, trace.steps.length - 1); // clamped at the end
  ctrl.destroy();
});

test("play advances and stops at the last step", async () => {
  const { ctrl } = makeController(null, { reducedMotion: true });
  ctrl.play();
  assert.equal(ctrl.isPlaying, true);
  // Reduced-motion delay is small; wait enough for several steps.
  await new Promise((r) => setTimeout(r, 400));
  assert.equal(ctrl.isPlaying, false, "play should auto-stop at the last step");
  assert.equal(ctrl.currentStep, ctrl.totalSteps - 1);
  ctrl.destroy();
});

test("pause stops advancing", async () => {
  const { ctrl } = makeController(null, { reducedMotion: true });
  ctrl.play();
  await new Promise((r) => setTimeout(r, 120));
  ctrl.pause();
  const frozen = ctrl.currentStep;
  assert.equal(ctrl.isPlaying, false);
  await new Promise((r) => setTimeout(r, 120));
  assert.equal(ctrl.currentStep, frozen);
  ctrl.destroy();
});

test("seek is reversible", async () => {
  const { ctrl } = makeController();
  await ctrl.seek(3);
  assert.equal(ctrl.currentStep, 3);
  await ctrl.seek(1);
  assert.equal(ctrl.currentStep, 1);
  ctrl.destroy();
});

test("rapid seek cancels the previous animation and only the last completes", async () => {
  const { animator } = makeStubs();
  const { ctrl } = makeController(animator);
  const p1 = ctrl.next();
  const p2 = ctrl.next(); // supersedes p1
  await Promise.all([p1, p2]);
  assert.equal(ctrl.currentStep, 2);
  // The first transition must have been superseded (cancel called at least once).
  assert.ok(animator.calls.filter((c) => c.op === "cancel").length >= 1);
  ctrl.destroy();
});

test("scene switch resets the controller (destroy then fresh)", () => {
  const { ctrl } = makeController();
  const before = ctrl.currentStep;
  ctrl.destroy();
  const { ctrl: ctrl2, rendered } = makeController();
  assert.equal(ctrl2.currentStep, 0);
  assert.deepEqual(rendered, [0]);
  assert.notEqual(ctrl2, ctrl);
  assert.ok(before >= 0);
  ctrl2.destroy();
});

test("setSpeed changes cadence without losing state", () => {
  const { ctrl } = makeController();
  ctrl.setSpeed(2);
  assert.equal(ctrl.speed, 2);
  assert.equal(ctrl.currentStep, 0);
  ctrl.destroy();
});

test("destroy stops playback", async () => {
  const { ctrl } = makeController(null, { reducedMotion: true });
  ctrl.play();
  ctrl.destroy();
  await new Promise((r) => setTimeout(r, 120));
  assert.equal(ctrl.isPlaying, false);
  assert.equal(ctrl.totalSteps > 0, true);
});
