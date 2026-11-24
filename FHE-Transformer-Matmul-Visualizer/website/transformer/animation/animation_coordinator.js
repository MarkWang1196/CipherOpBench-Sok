// Animation coordinator (spec §16.1): capture → render target → capture →
// FLIP matching logical IDs → commit. Owns the live Web Animations so they can
// be cancelled on rapid seek / scene switch.

import { animateFlip, findLogical, prefersReducedMotion } from "./flip.js";

export class AnimationCoordinator {
  /**
   * @param {object} cfg
   * @param {Element} cfg.container root that holds all `[data-logical-id]` nodes
   * @param {(trace, index)=>void} cfg.renderStep
   * @param {boolean|null} [cfg.reducedMotion] override prefers-reduced-motion
   * @param {()=>void} [cfg.onSettled] called after an animation settles (or
   *   immediately when reduced motion skips the animation) — used to redraw
   *   the connection overlay so it matches the committed state.
   */
  constructor({ container, renderStep, reducedMotion = null, onSettled = null }) {
    this.container = container;
    this.renderStep = renderStep;
    this.reduced = reducedMotion != null ? reducedMotion : prefersReducedMotion();
    this.onSettled = onSettled;
    this._animations = [];
  }

  cancel() {
    for (const a of this._animations) {
      try {
        a.cancel();
      } catch {
        /* already finished */
      }
    }
    this._animations = [];
  }

  _capture() {
    const map = new Map();
    const nodes = this.container.querySelectorAll("[data-logical-id]");
    for (const el of nodes) {
      const id = el.getAttribute("data-logical-id");
      if (id) map.set(id, el.getBoundingClientRect());
    }
    return map;
  }

  /**
   * Render `to` and animate the delta from `from`. Resolves when the
   * animation completes (or immediately for reduced motion).
   */
  async animate({ trace, from, to, speed = 1, reducedMotion }) {
    this.cancel();
    const reduced = reducedMotion != null ? reducedMotion : this.reduced;
    const prev = this._capture();
    this.renderStep(trace, to);
    const next = this._capture();
    if (reduced) {
      this.onSettled?.();
      return; // render only, no motion
    }

    const duration = Math.max(120, Math.min(900, 600 / (speed || 1)));

    const flip = animateFlip(prev, next, (id) => findLogical(this.container, id), { duration });
    this._animations.push(...flip.animations);

    // New elements: rise + fade in, or slide in from their source element.
    const newAnimations = [];
    const animatedEls = new Set();
    for (const [logicalId] of next) {
      if (prev.has(logicalId)) continue;
      const el = findLogical(this.container, logicalId);
      if (!el) continue;
      const srcId = el.getAttribute("data-source-logical");
      const srcRect = srcId ? prev.get(srcId) : null;
      let anim;
      if (srcRect) {
        const rect = el.getBoundingClientRect();
        anim = el.animate(
          [
            { transform: `translate(${srcRect.left - rect.left}px, ${srcRect.top - rect.top}px)`, opacity: 0.15 },
            { transform: "translate(0,0)", opacity: 1 },
          ],
          { duration, easing: "cubic-bezier(.2,.8,.2,1)", fill: "both" }
        );
      } else {
        anim = el.animate(
          [
            { opacity: 0, transform: "translateY(8px)" },
            { opacity: 1, transform: "translateY(0)" },
          ],
          { duration: duration * 0.6, easing: "ease-out", fill: "both" }
        );
      }
      this._animations.push(anim);
      newAnimations.push(anim.finished);
      animatedEls.add(el);
    }

    await Promise.allSettled([flip.finished, ...newAnimations]);

    // Clear fill styles so elements rest at their natural positions.
    for (const el of animatedEls) {
      el.style.transform = "";
      el.style.opacity = "";
    }
    this.onSettled?.();
  }
}
