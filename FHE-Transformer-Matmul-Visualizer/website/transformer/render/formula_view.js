// Formula view (spec §14.4): plain-text math for the current step. No MathJax
// in phase 1 — Unicode/HTML only.

import { PHASES } from "../core/trace_schema.js";

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

/** Human phase label for the step heading. */
export function phaseLabel(step) {
  const map = {
    [PHASES.REFERENCE]: "Reference",
    [PHASES.PARTITION]: "Block partition",
    [PHASES.DIAGONAL]: "Diagonal extraction",
    [PHASES.PACK]: "Packing",
    [PHASES.REPLICATE]: "Replication",
    [PHASES.ENCODE]: "Encoding",
    [PHASES.TRANSPOSE]: "Transpose",
    [PHASES.ROTATE]: "Internal rotation",
    [PHASES.MASK]: "Mask",
    [PHASES.MULTIPLY]: "Slot-wise multiply",
    [PHASES.RELINEARIZE]: "Relinearize",
    [PHASES.RESCALE]: "Rescale",
    [PHASES.REDUCE]: "Reduction",
    [PHASES.OUTPUT]: "Output",
  };
  return map[step.phase] || step.phase;
}

export function renderFormula(container, { step }) {
  container.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "tv-formula-box";

  const title = document.createElement("h3");
  title.className = "tv-formula-title";
  title.textContent = step.title || "Step";
  wrap.append(title);

  if (step.description) {
    const desc = document.createElement("p");
    desc.className = "tv-formula-desc";
    desc.textContent = step.description;
    wrap.append(desc);
  }

  if (step.formula && step.formula.plain) {
    const pre = document.createElement("pre");
    pre.className = "tv-formula";
    pre.textContent = step.formula.plain;
    wrap.append(pre);
  }

  if (step.notes && step.notes.length) {
    const ul = document.createElement("ul");
    ul.className = "tv-notes";
    for (const note of step.notes) {
      const li = document.createElement("li");
      li.textContent = note;
      ul.append(li);
    }
    wrap.append(ul);
  }

  container.append(wrap);
}
