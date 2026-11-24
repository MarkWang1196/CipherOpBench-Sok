// FLIP helpers (spec §16.2). DOM-only; used by the animation coordinator.
//
// Elements are matched across steps by their stable `data-logical-id`
// attribute. The renderer writes that attribute; the coordinator captures
// rects before/after a render and animates the deltas here.

export function prefersReducedMotion() {
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

/**
 * Animate elements present in both rect maps by translating from their old to
 * their new position (First-Last-Invert-Play).
 *
 * @param {Map<string, DOMRect>} previousRects
 * @param {Map<string, DOMRect>} nextRects
 * @param {(logicalId:string)=>Element|null} getElement
 * @param {{duration?:number}} [options]
 * @returns {{finished: Promise, animations: Animation[]}}
 */
export function animateFlip(previousRects, nextRects, getElement, { duration = 600 } = {}) {
  const animations = [];
  for (const [logicalId, newRect] of nextRects) {
    const oldRect = previousRects.get(logicalId);
    const el = getElement(logicalId);
    if (!oldRect || !el) continue;
    const dx = oldRect.left - newRect.left;
    const dy = oldRect.top - newRect.top;
    if (!dx && !dy) continue;
    animations.push(
      el.animate(
        [
          { transform: `translate(${dx}px, ${dy}px)` },
          { transform: "translate(0, 0)" },
        ],
        { duration, easing: "cubic-bezier(.2,.8,.2,1)", fill: "both" }
      )
    );
  }
  const finished = Promise.allSettled(animations.map((a) => a.finished));
  return { finished, animations };
}

/** Escape a logicalId so it is safe inside a CSS attribute selector. */
export function cssEscape(id) {
  if (typeof CSS !== "undefined" && typeof CSS.escape === "function") return CSS.escape(id);
  return String(id).replace(/["\\]/g, "\\$&");
}

/** Find the element carrying a given data-logical-id. */
export function findLogical(container, logicalId) {
  if (!container || logicalId == null) return null;
  try {
    return container.querySelector(`[data-logical-id="${cssEscape(logicalId)}"]`);
  } catch {
    return null;
  }
}
