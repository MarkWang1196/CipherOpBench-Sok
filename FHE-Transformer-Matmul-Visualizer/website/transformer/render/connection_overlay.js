// SVG connection overlay (spec §14.3, §18.2). Draws a *bounded* set of lines:
// only the current focus's provenance chain (tensor cell → slot, slot →
// product, product → partial sum). Never draws the whole trace — that keeps
// the overlay cheap and readable.

import { findLogical } from "../animation/flip.js";

const NS = "http://www.w3.org/2000/svg";

function computeLines(step, focus) {
  const lines = [];
  if (!step || !step.ciphertexts) return lines;

  // Collect slot states by logicalId.
  const byLogical = new Map();
  for (const ct of step.ciphertexts) {
    for (const s of ct.slots) byLogical.set(s.logicalId, s);
  }

  if (focus && focus.outputCoords) {
    const [oi, oj] = focus.outputCoords;
    // Slots that belong to this output cell (products / partial sums).
    for (const ct of step.ciphertexts) {
      for (const s of ct.slots) {
        if (s.outputCoords && s.outputCoords[0] === oi && s.outputCoords[1] === oj) {
          for (const ref of s.sourceRefs || []) {
            if (ref.logicalId) lines.push({ from: ref.logicalId, to: s.logicalId });
          }
        }
      }
    }
  } else if (focus && focus.cellRefs && focus.cellRefs.length) {
    // Tensor cell → slots that derive from that cell (pack/replicate/encode).
    for (const ref of focus.cellRefs) {
      const key = `${ref.tensorId}:${ref.linear}`;
      for (const ct of step.ciphertexts) {
        for (const s of ct.slots) {
          for (const sr of s.sourceRefs || []) {
            if (sr.tensorId === ref.tensorId && sr.linear === ref.linear) {
              lines.push({ from: key, to: s.logicalId });
            }
          }
        }
      }
    }
  }
  return lines;
}

/**
 * Redraw the SVG overlay for the current step/focus.
 * @param {SVGSVGElement} svg
 * @param {{step:object, container:Element, focus:object|null, palette?:object}} cfg
 */
export function renderConnections(svg, { step, container, focus, palette }) {
  svg.innerHTML = "";
  const view = container.getBoundingClientRect();
  if (!view.width || !view.height) return;

  const p = palette || { stroke: "rgba(56,189,248,0.5)", strokeWidth: 1.5 };

  svg.setAttribute("viewBox", `0 0 ${view.width} ${view.height}`);
  svg.style.width = `${view.width}px`;
  svg.style.height = `${view.height}px`;

  const lines = computeLines(step, focus);
  const drawn = new Set();
  for (const line of lines) {
    const a = findLogical(container, line.from);
    const b = findLogical(container, line.to);
    if (!a || !b) continue;
    const ra = a.getBoundingClientRect();
    const rb = b.getBoundingClientRect();
    if (!ra.width || !rb.width) continue;
    const x1 = ra.left - view.left + ra.width / 2;
    const y1 = ra.top - view.top + ra.height / 2;
    const x2 = rb.left - view.left + rb.width / 2;
    const y2 = rb.top - view.top + rb.height / 2;
    const key = `${x1},${y1}->${x2},${y2}`;
    if (drawn.has(key)) continue;
    drawn.add(key);
    const l = document.createElementNS(NS, "line");
    l.setAttribute("x1", String(x1));
    l.setAttribute("y1", String(y1));
    l.setAttribute("x2", String(x2));
    l.setAttribute("y2", String(y2));
    l.setAttribute("stroke", p.stroke);
    l.setAttribute("stroke-width", String(p.strokeWidth));
    l.setAttribute("stroke-dasharray", "3 3");
    l.setAttribute("aria-hidden", "true");
    svg.append(l);
  }
}
