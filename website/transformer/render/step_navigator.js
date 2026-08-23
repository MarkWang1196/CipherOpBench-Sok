// Step navigator (spec §8.4): clickable step chips + range progress, synced
// to the timeline controller. The play/pause/step buttons live in the HTML bar
// and call the controller directly; this module only renders the progress UI.

function makeEl(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text != null) el.textContent = text;
  return el;
}

/**
 * Render step chips.
 * @param {Element} container
 * @param {{trace:object, current:number, onSelect:(index:number)=>void}} cfg
 */
export function renderStepChips(container, { trace, current, onSelect }) {
  container.innerHTML = "";
  const steps = trace.steps || [];
  for (let idx = 0; idx < steps.length; idx++) {
    const step = steps[idx];
    const item = makeEl("li", "tv-chip-item");
    const chip = makeEl("button", "tv-chip" + (idx === current ? " tv-chip--active" : ""));
    chip.type = "button";
    const number = makeEl("span", "tv-chip-number", idx < current ? "✓" : String(idx + 1));
    const copy = makeEl("span", "tv-chip-copy");
    const label = makeEl("span", "tv-chip-label", step.title || `Step ${idx + 1}`);
    copy.append(label);
    if (idx === current && step.description) {
      copy.append(makeEl("span", "tv-chip-description", step.description));
      chip.setAttribute("aria-expanded", "true");
    } else {
      chip.setAttribute("aria-expanded", "false");
    }
    chip.append(number, copy);
    chip.title = step.title || `Step ${idx + 1}`;
    chip.dataset.stepIndex = String(idx);
    if (idx === current) chip.setAttribute("aria-current", "step");
    chip.addEventListener("click", () => onSelect && onSelect(idx));
    item.append(chip);
    container.append(item);
  }
  // Scroll the active chip into view.
  const active = container.querySelector(".tv-chip--active");
  if (active && active.scrollIntoView) {
    active.scrollIntoView({ block: "nearest", inline: "nearest" });
  }
}

/**
 * Sync a range slider to the trace + current step.
 * @param {HTMLInputElement} range
 * @param {{trace:object, current:number, onSeek:(index:number)=>void}} cfg
 */
export function syncRange(range, { trace, current, onSeek }) {
  const max = Math.max(0, (trace.steps || []).length - 1);
  if (Number(range.max) !== max) {
    range.max = String(max);
    range.step = "1";
    range.value = String(current);
  }
  range.value = String(current);
  if (range.dataset.listener !== "1") {
    range.dataset.listener = "1";
    range.addEventListener("input", () => onSeek && onSeek(Number(range.value)));
  }
}
