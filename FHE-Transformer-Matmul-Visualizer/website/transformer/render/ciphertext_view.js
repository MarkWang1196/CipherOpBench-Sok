// Ciphertext slot view (spec §14.2). Renders each ciphertext/plaintext as a
// labeled strip of slots. CT vs PT differ by frame (not only hue); gap/oob
// slots use distinct fill patterns; large strips collapse but keep highlighted
// slots visible.

import { findLogical } from "../animation/flip.js";

const SLOT_WRAP = 32; // wrap slot strips at this many per row

function makeEl(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text != null) el.textContent = text;
  return el;
}

function makeSlotButton(s, collapsed) {
  const cell = document.createElement("button");
  cell.type = "button";
  cell.className = `tv-cell tv-slot tv-slot--${s.kind || "tensor"}`;
  cell.dataset.logicalId = s.logicalId;
  cell.dataset.slot = String(s.slot);
  cell.dataset.kind = s.kind || "tensor";
  if (s.outputCoords) cell.dataset.outputCoords = `${s.outputCoords[0]},${s.outputCoords[1]}`;
  const first = s.sourceRefs && s.sourceRefs[0];
  if (first && first.logicalId) cell.dataset.sourceLogical = first.logicalId;
  cell.title = `${s.expression || ""}  (slot ${s.slot})`;
  cell.setAttribute("aria-label", `slot ${s.slot}: ${s.expression || s.kind}`);
  if (s.kind === "gap") {
    cell.textContent = "·";
  } else if (s.kind === "oob") {
    cell.textContent = "✕";
  } else {
    cell.textContent = String(s.value);
  }
  return cell;
}

/**
 * Render ciphertext states.
 * @param {Element} container
 * @param {{ciphertexts: Array, focusedLogicalIds?: Set<string>}} cfg
 */
export function renderCiphertexts(container, { ciphertexts }) {
  container.innerHTML = "";
  if (!ciphertexts || !ciphertexts.length) {
    container.append(makeEl("div", "tv-pending", "No ciphertexts at this step."));
    return;
  }
  for (const ct of ciphertexts) {
    const box = makeEl("div", `tv-ct ${ct.ghost ? "tv-ct--ghost" : ""} ${ct.secret ? "tv-ct--ct" : "tv-ct--pt"}`);
    const head = makeEl("div", "tv-ct-head");
    head.append(
      makeEl("span", `tv-badge ${ct.secret ? "tv-badge-ct" : "tv-badge-pt"}`, ct.secret ? "CT" : "PT"),
      makeEl("span", "tv-ct-name", ct.label)
    );
    const meta = makeEl("span", "tv-ct-meta");
    const bits = [];
    if (ct.layout) bits.push(ct.layout);
    if (ct.illustrative) bits.push("illustrative");
    meta.textContent = bits.join(" · ");
    head.append(meta);
    box.append(head);

    const slots = ct.slots || [];
    const strip = makeEl("div", "tv-ct-strip");
    const columns = Math.min(SLOT_WRAP, slots.length);
    strip.style.gridTemplateColumns = `repeat(${columns}, var(--slot))`;

    if (slots.length > 128) {
      // Collapsible large vector; keep the highlighted slot visible by
      // rendering everything but letting the user expand (lightweight).
      const details = makeEl("details", "tv-ct-collapse");
      const summary = makeEl("summary", "tv-ct-collapse-summary", `Show ${slots.length} slots`);
      details.append(summary, strip);
      for (const s of slots) strip.append(makeSlotButton(s, true));
      box.append(details);
    } else {
      for (const s of slots) strip.append(makeSlotButton(s, false));
      box.append(strip);
    }
    container.append(box);
  }
}

/** Convenience used by the connection overlay to find slot elements. */
export function findSlotByLogical(container, logicalId) {
  return findLogical(container, logicalId);
}
