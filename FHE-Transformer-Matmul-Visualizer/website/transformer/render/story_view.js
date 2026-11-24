// Persistent, paper-style algorithm canvas.  Unlike the inspector panes, the
// complete dataflow stays mounted; timeline changes only reveal/focus stages.

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function safeValues(matrix, maxRows = 8, maxCols = 8) {
  return matrix.slice(0, maxRows).map((row) => row.slice(0, maxCols));
}

function genericMatrixArtifact(id, label, operand, revealAt) {
  const values = safeValues(operand.values || []);
  return {
    id,
    type: "matrix",
    label,
    badge: operand.secret ? "CT" : "PT",
    secret: Boolean(operand.secret),
    meta: `${operand.shape?.[0] || values.length}×${operand.shape?.[1] || values[0]?.length || 0}`,
    revealAt,
    rows: values.map((row, i) => row.map((value, j) => ({
      value,
      label: String(value),
      tone: (i + j) % 4,
      provenance: [`${operand.id || id}:${i}:${j}`],
    }))),
  };
}

export function buildGenericStoryGraph(trace) {
  const left = trace.operands?.left;
  const right = trace.operands?.right;
  const stages = [];
  if (left && right) {
    stages.push({
      id: "generic-input",
      title: "Input matrices",
      caption: `${left.shape.join("×")} · ${right.shape.join("×")}`,
      revealAt: 0,
      rows: [[
        genericMatrixArtifact("generic-A", left.label || left.id, left, 0),
        { id: "generic-times", type: "operator", symbol: "×", label: "matmul", revealAt: 0 },
        genericMatrixArtifact("generic-B", right.label || right.id, right, 0),
      ]],
    });
  }

  (trace.steps || []).slice(1).forEach((step, index) => {
    const revealAt = index + 1;
    const visibleCiphertexts = (step.ciphertexts || []).slice(0, 4);
    const rows = visibleCiphertexts.length
      ? visibleCiphertexts.map((ct, ctIndex) => [{
          id: `${step.id}:${ct.id || ctIndex}`,
          type: ct.secret ? "ciphertext" : "vector",
          label: ct.label || ct.id || `vector ${ctIndex + 1}`,
          badge: ct.secret ? "CT" : "PT",
          meta: ct.layout || (ct.illustrative ? "illustrative" : ""),
          revealAt,
          cells: (ct.slots || []).slice(0, 32).map((slot) => ({
            value: slot.value,
            label: slot.kind === "gap" ? "·" : slot.kind === "oob" ? "×" : String(slot.value),
            tone: slot.contractionIndex ?? ctIndex,
            provenance: (slot.sourceRefs || []).map((ref) =>
              ref.coords ? `${ref.tensorId}:${ref.coords.join(":")}` : ref.logicalId
            ).filter(Boolean),
          })),
          truncated: (ct.slots || []).length > 32,
        }])
      : [[{
          id: `${step.id}:note`,
          type: "note",
          label: step.title,
          text: step.formula?.plain || step.description || "Continue to the next transformation.",
          revealAt,
        }]];
    stages.push({
      id: step.id,
      title: `${revealAt} · ${step.title}`,
      caption: step.description || "",
      revealAt,
      rows,
    });
  });

  return {
    id: `${trace.id}:generic-story`,
    algorithmId: trace.parameters?.algorithmId || "rotom-opp",
    title: "Generic HE matmul · persistent dataflow",
    subtitle: "The existing arbitrary-shape OPP trace, arranged horizontally without discarding earlier stages.",
    stages,
  };
}

function provenanceValue(items) {
  return [...new Set(items || [])].join(" ");
}

function renderCell(cell, artifact, index, matrix = false) {
  const rawTone = Number.isInteger(cell.tone) ? cell.tone : 4;
  const tone = rawTone >= 0 && rawTone <= 4 ? rawTone : ((rawTone % 4) + 4) % 4;
  const button = el("button", `tv-story-cell tv-story-cell--tone-${tone}`, cell.label);
  if (String(cell.label ?? "").length >= 5) button.classList.add("tv-story-cell--compact");
  button.type = "button";
  button.dataset.paperTone = String(tone);
  button.dataset.logicalId = `${artifact.id}:${index}`;
  button.dataset.provenance = provenanceValue(cell.provenance);
  const details = [];
  if (cell.sampleLane != null) {
    button.dataset.sampleLane = String(cell.sampleLane);
    details.push(`sample ${cell.sampleLane}`);
  } else if (cell.label === "·") {
    details.push("padding");
  }
  if (cell.featureIndex != null) {
    const feature = `x${cell.featureIndex}`;
    const badge = el("span", "tv-story-slot-feature", feature);
    badge.setAttribute("aria-hidden", "true");
    button.append(badge);
    details.push(`feature ${feature}`);
  }
  const value = cell.imaginary ? `${cell.value}${cell.imaginary < 0 ? "−" : "+"}${Math.abs(cell.imaginary)}i` : cell.value;
  const position = matrix ? "matrix cell" : artifact.coefficients ? "coefficient" : "slot";
  const detail = details.length ? ` (${details.join(", ")})` : "";
  button.title = `${artifact.label}[${index}] = ${value}${detail}`;
  button.setAttribute("aria-label", `${artifact.label}, ${position} ${index}, value ${value}${detail}`);
  return button;
}

function renderArtifact(artifact) {
  if (artifact.type === "operator") {
    const node = el("div", "tv-story-operator");
    node.dataset.revealAt = String(artifact.revealAt || 0);
    node.append(el("span", "tv-story-operator-symbol", artifact.symbol), el("span", "tv-story-operator-label", artifact.label));
    return node;
  }

  const card = el("article", `tv-story-artifact tv-story-artifact--${artifact.type}`);
  card.dataset.artifactId = artifact.id;
  card.dataset.revealAt = String(artifact.revealAt || 0);
  const head = el("header", "tv-story-artifact-head");
  if (artifact.badge) head.append(el("span", `tv-story-badge tv-story-badge--${artifact.badge.toLowerCase()}`, artifact.badge));
  head.append(el("strong", "tv-story-artifact-label", artifact.label));
  if (artifact.meta) head.append(el("span", "tv-story-artifact-meta", artifact.meta));
  card.append(head);

  if (artifact.type === "matrix") {
    const rows = artifact.rows || [];
    card.style.setProperty("--story-cols", String(rows[0]?.length || 1));
    const grid = el("div", "tv-story-matrix");
    grid.style.setProperty("--story-cols", String(rows[0]?.length || 1));
    rows.flat().forEach((cell, index) => grid.append(renderCell(cell, artifact, index, true)));
    card.append(grid);
  } else if (artifact.type === "batch") {
    const grid = el("div", "tv-story-batch-grid");
    for (const group of artifact.groups || []) {
      const lane = el(
        "div",
        `tv-story-batch-lane tv-story-batch-lane--tone-${group.tone ?? 4}${group.padded ? " tv-story-batch-lane--padded" : ""}`
      );
      lane.append(
        el("strong", "tv-story-batch-label", group.label),
        el("span", "tv-story-batch-meta", group.meta || "")
      );
      grid.append(lane);
    }
    card.append(grid);
  } else if (artifact.type === "note") {
    card.append(el("p", "tv-story-note", artifact.text));
  } else {
    const strip = el("div", "tv-story-vector-strip");
    (artifact.cells || []).forEach((cell, index) => strip.append(renderCell(cell, artifact, index, false)));
    if (artifact.truncated) strip.append(el("span", "tv-story-ellipsis", "…"));
    card.append(strip);
  }
  return card;
}

function estimatedArtifactWidth(artifact) {
  if (artifact.type === "operator") return 62;
  const headerWidth = 48
    + String(artifact.label || "").length * 7
    + String(artifact.meta || "").length * 6
    + (artifact.badge ? 30 : 0);
  if (artifact.type === "matrix") {
    const cols = artifact.rows?.[0]?.length || 1;
    return Math.max(150, cols * 36 + 22, headerWidth);
  }
  if (artifact.type === "note") return 300;
  if (artifact.type === "batch") return Math.max(460, Math.min(980, (artifact.groups?.length || 1) * 74));
  const cells = artifact.cells?.length || 1;
  return Math.max(220, cells * 36 + 24, headerWidth);
}

function estimatedStageWidth(stage) {
  const widestLane = Math.max(
    0,
    ...(stage.rows || []).map((row) =>
      row.reduce((width, artifact) => width + estimatedArtifactWidth(artifact), 0)
      + Math.max(0, row.length - 1) * 12
    )
  );
  // Do not cap wide stages. Long BSGS/product-sum rows need their full
  // measured width; the outer story viewport provides horizontal scrolling.
  // Capping here makes artifacts overflow the stage and paint over neighbors.
  return Math.max(410, widestLane + 42);
}

export function groupStoryExpression(row) {
  const groups = [];
  for (let i = 0; i < row.length; i += 1) {
    const artifact = row[i];
    if (artifact.type === "operator" && row[i + 1]) {
      groups.push([artifact, row[i + 1]]);
      i += 1;
    } else {
      groups.push([artifact]);
    }
  }
  return groups;
}

const HORIZONTAL = { scroll: "scrollLeft", start: "offsetLeft", size: "offsetWidth", viewport: "clientWidth", coordinate: "clientX" };
const VERTICAL = { scroll: "scrollTop", start: "offsetTop", size: "offsetHeight", viewport: "clientHeight", coordinate: "clientY" };

export function nearestStoryStageIndex(viewCenter, stages, axis = HORIZONTAL) {
  let nearestIndex = -1;
  let distance = Infinity;
  stages.forEach((stage, index) => {
    const start = stage[axis.start];
    const end = start + stage[axis.size];
    const currentDistance = Math.max(start - viewCenter, viewCenter - end, 0);
    if (currentDistance < distance) {
      nearestIndex = index;
      distance = currentDistance;
    }
  });
  return nearestIndex;
}

function renderStage(stage, index) {
  const section = el("section", "tv-story-stage");
  section.dataset.stageId = stage.id;
  section.dataset.stageNumber = String(index + 1).padStart(2, "0");
  section.dataset.revealAt = String(stage.revealAt || 0);
  section.style.setProperty("--story-stage-width", `${estimatedStageWidth(stage)}px`);
  const heading = el("div", "tv-story-stage-head");
  heading.append(el("h3", "tv-story-stage-title", stage.title));
  if (stage.caption) heading.append(el("p", "tv-story-stage-caption", stage.caption));
  section.append(heading);
  const body = el("div", "tv-story-stage-body");
  for (const row of stage.rows || []) {
    const lane = el("div", "tv-story-lane");
    for (const group of groupStoryExpression(row)) {
      const term = el("div", "tv-story-expression-term");
      for (const artifact of group) term.append(renderArtifact(artifact));
      lane.append(term);
    }
    body.append(lane);
  }
  section.append(body);
  return section;
}

function renderFlowStep(stage, index) {
  const shortTitle = stage.shortTitle || stage.stepTitle || stage.title;
  const accessibleTitle = stage.stepTitle || stage.title;
  const item = el("li", "tv-story-flow-item");
  item.dataset.revealAt = String(stage.revealAt || 0);
  const button = el("button", "tv-story-flow-button");
  button.type = "button";
  button.title = accessibleTitle;
  button.setAttribute("aria-label", `Step ${index + 1}: ${accessibleTitle}`);
  button.append(el("span", "tv-story-flow-marker", String(index + 1)));
  button.append(el("span", "tv-story-flow-label", shortTitle));
  item.append(button);
  return item;
}

export class StoryView {
  constructor({ root, viewport, canvas, title, subtitle, detail, onStepRequest = null }) {
    this.root = root;
    this.viewport = viewport;
    this.canvas = canvas;
    this.title = title;
    this.subtitle = subtitle;
    this.detail = detail;
    this.traceRef = null;
    this.graph = null;
    this.pinned = new Set();
    this.previousIndex = null;
    this.onStepRequest = typeof onStepRequest === "function" ? onStepRequest : null;
    this.flow = el("ol", "tv-story-flow");
    this.flow.setAttribute("aria-label", "Connected algorithm steps");
    this.viewport.before(this.flow);
    this._drag = null;
    this._scrollTimer = 0;
    this._focusing = false;
    this._suppressClick = false;
    this._bind();
    if (typeof ResizeObserver !== "undefined") {
      this._resizeObserver = new ResizeObserver(() => this._updateEdgeSpacing());
      this._resizeObserver.observe(this.viewport);
    }
  }

  _bind() {
    this.canvas.addEventListener("pointerover", (event) => {
      if (this.pinned.size) return;
      const cell = event.target.closest("[data-provenance]");
      this._highlight(cell ? new Set(cell.dataset.provenance.split(" ").filter(Boolean)) : new Set());
    });
    this.canvas.addEventListener("pointerleave", () => {
      if (!this.pinned.size) this._highlight(new Set());
    });
    this.canvas.addEventListener("click", (event) => {
      if (this._suppressClick) {
        this._suppressClick = false;
        event.preventDefault();
        event.stopPropagation();
        return;
      }
      const cell = event.target.closest("[data-provenance]");
      if (!cell) {
        this.pinned.clear();
        this._highlight(new Set());
        return;
      }
      const selected = new Set(cell.dataset.provenance.split(" ").filter(Boolean));
      const same = selected.size === this.pinned.size && [...selected].every((value) => this.pinned.has(value));
      this.pinned = same ? new Set() : selected;
      this._highlight(this.pinned);
    });
    this.flow.addEventListener("click", (event) => {
      const item = event.target.closest(".tv-story-flow-item[data-reveal-at]");
      if (item) this._requestStep(Number(item.dataset.revealAt));
    });
    this.viewport.addEventListener("pointerdown", (event) => this._startDrag(event));
    this.viewport.addEventListener("pointermove", (event) => this._moveDrag(event));
    this.viewport.addEventListener("pointerup", (event) => this._finishDrag(event));
    this.viewport.addEventListener("pointercancel", (event) => this._finishDrag(event));
    this.viewport.addEventListener("wheel", () => { this._focusing = false; }, { passive: true });
    this.viewport.addEventListener("scroll", () => {
      clearTimeout(this._scrollTimer);
      if (!this._focusing) this._markNearestStage();
      this._scrollTimer = setTimeout(() => {
        if (this._focusing) this._focusing = false;
        else this._commitNearestStage();
      }, 160);
    }, { passive: true });
  }

  setTrace(trace) {
    // A custom edit commonly produces a new trace with the same trace.id and
    // matrix shape. Cache by object identity, not id, or the paper canvas keeps
    // displaying stale values after a successful regeneration.
    if (this.traceRef === trace && this.graph) return;
    clearTimeout(this._scrollTimer);
    this._focusing = false;
    this._drag = null;
    this._suppressClick = false;
    this.viewport.classList.remove("tv-story-viewport--dragging");
    this.traceRef = trace;
    this.graph = trace.graph || buildGenericStoryGraph(trace);
    this.pinned.clear();
    this.title.textContent = this.graph.title;
    this.subtitle.textContent = this.graph.subtitle || "";
    this.canvas.innerHTML = "";
    this.flow.innerHTML = "";
    for (const [index, stage] of (this.graph.stages || []).entries()) {
      this.flow.append(renderFlowStep(stage, index));
      this.canvas.append(renderStage(stage, index));
    }
    this.previousIndex = null;
    this.flow.scrollLeft = 0;
    this.viewport.scrollLeft = 0;
    this.viewport.scrollTop = 0;
    if (typeof requestAnimationFrame === "function") requestAnimationFrame(() => this._updateEdgeSpacing());
    else this._updateEdgeSpacing();
  }

  renderStep(trace, index, { scroll = false } = {}) {
    this.setTrace(trace);
    const step = trace.steps[index];
    this.root.dataset.currentStep = String(index);
    for (const container of [this.flow, this.canvas]) {
      container.querySelectorAll("[data-reveal-at]").forEach((node) => {
        const revealAt = Number(node.dataset.revealAt);
        node.classList.toggle("tv-story--future", revealAt > index);
        node.classList.toggle("tv-story--current", revealAt === index);
        node.classList.toggle("tv-story--past", revealAt < index);
        if (node.classList.contains("tv-story-stage")) node.setAttribute("aria-current", revealAt === index ? "step" : "false");
      });
    }
    this.detail.innerHTML = "";
    if (step) {
      this.detail.append(el("strong", "tv-story-detail-title", step.title));
      this.detail.append(el("span", "tv-story-detail-text", step.description || ""));
      if (step.formula?.plain) this.detail.append(el("code", "tv-story-detail-formula", step.formula.plain));
    }
    const direction = this.previousIndex == null || index === this.previousIndex ? 0 : index > this.previousIndex ? 1 : -1;
    this._focusStage(index, scroll);
    this._focusFlowStep(index, scroll);
    this._animateCurrentStage(index, direction);
    this.previousIndex = index;
    if (this.pinned.size) this._highlight(this.pinned);
  }

  _focusStage(index, animated) {
    const active = this.canvas.querySelector(`.tv-story-stage[data-reveal-at="${index}"]`);
    if (!active) return;
    const axis = this._scrollAxis();
    this._updateEdgeSpacing();
    const centeredInset = (this.viewport[axis.viewport] - active[axis.size]) / 2;
    // Keep the heading visible when a stacked card is taller than its viewport.
    const inset = axis === VERTICAL ? Math.max(0, centeredInset) : centeredInset;
    const target = Math.max(0, active[axis.start] - inset);
    clearTimeout(this._scrollTimer);
    this._focusing = true;
    this.viewport.scrollTo({ [axis === HORIZONTAL ? "left" : "top"]: target, behavior: animated ? "smooth" : "instant" });
    this._scrollTimer = setTimeout(() => { this._focusing = false; }, 160);
  }

  _focusFlowStep(index, animated) {
    const active = this.flow.querySelector(`.tv-story-flow-item[data-reveal-at="${index}"]`);
    if (!active) return;
    const target = Math.max(0, active.offsetLeft - (this.flow.clientWidth - active.offsetWidth) / 2);
    this.flow.scrollTo({ left: target, behavior: animated ? "smooth" : "auto" });
  }

  _updateEdgeSpacing() {
    const stages = [...this.canvas.querySelectorAll(".tv-story-stage")];
    if (!stages.length) return;
    const leading = Math.max(16, (this.viewport.clientWidth - stages[0].offsetWidth) / 2);
    const trailing = Math.max(16, (this.viewport.clientWidth - stages.at(-1).offsetWidth) / 2);
    this.canvas.style.setProperty("--story-leading-space", `${leading}px`);
    this.canvas.style.setProperty("--story-trailing-space", `${trailing}px`);
  }

  _startDrag(event) {
    if (event.button !== 0) return;
    const axis = this._scrollAxis();
    this._drag = { pointerId: event.pointerId, axis, start: event[axis.coordinate], scroll: this.viewport[axis.scroll], moved: false };
  }

  _moveDrag(event) {
    if (!this._drag || event.pointerId !== this._drag.pointerId) return;
    const { axis, start, scroll } = this._drag;
    const delta = event[axis.coordinate] - start;
    if (!this._drag.moved && Math.abs(delta) < 6) return;
    if (!this._drag.moved) {
      this._drag.moved = true;
      this._focusing = false;
      this._suppressClick = true;
      this.viewport.classList.add("tv-story-viewport--dragging");
      this.viewport.setPointerCapture?.(event.pointerId);
    }
    event.preventDefault();
    this.viewport[axis.scroll] = scroll - delta;
  }

  _finishDrag(event) {
    if (!this._drag || event.pointerId !== this._drag.pointerId) return;
    const moved = this._drag.moved;
    this._drag = null;
    this.viewport.classList.remove("tv-story-viewport--dragging");
    this.viewport.releasePointerCapture?.(event.pointerId);
    if (moved) {
      this._commitNearestStage();
      setTimeout(() => { this._suppressClick = false; }, 0);
    }
  }

  _nearestStage() {
    const axis = this._scrollAxis();
    const center = this.viewport[axis.scroll] + this.viewport[axis.viewport] / 2;
    const stages = [...this.canvas.querySelectorAll(".tv-story-stage")];
    return stages[nearestStoryStageIndex(center, stages, axis)] || null;
  }

  _scrollAxis() {
    return getComputedStyle(this.canvas).flexDirection === "column" ? VERTICAL : HORIZONTAL;
  }

  _markNearestStage() {
    const nearest = this._nearestStage();
    this.canvas.querySelectorAll(".tv-story--drag-target").forEach((stage) => stage.classList.remove("tv-story--drag-target"));
    if (nearest && !nearest.classList.contains("tv-story--current")) nearest.classList.add("tv-story--drag-target");
  }

  _commitNearestStage() {
    const nearest = this._nearestStage();
    if (!nearest) return;
    nearest.classList.remove("tv-story--drag-target");
    this._requestStep(Number(nearest.dataset.revealAt));
  }

  _requestStep(index) {
    if (Number.isInteger(index)) this.onStepRequest?.(index);
  }

  _animateCurrentStage(index, direction) {
    const active = this.canvas.querySelector(`.tv-story-stage[data-reveal-at="${index}"]`);
    if (!active?.animate || globalThis.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
    active.getAnimations().forEach((animation) => animation.cancel());
    active.animate([
      { opacity: 0.15, transform: `translateX(${direction * 22}px) scale(0.985)` },
      { opacity: 1, transform: "translateX(0) scale(1)" },
    ], { duration: 320, easing: "cubic-bezier(0.22, 1, 0.36, 1)" });
  }

  _highlight(selected) {
    const hasSelection = selected.size > 0;
    this.canvas.classList.toggle("tv-story-has-selection", hasSelection);
    this.canvas.querySelectorAll("[data-provenance]").forEach((node) => {
      const values = node.dataset.provenance.split(" ").filter(Boolean);
      const hit = hasSelection && values.some((value) => selected.has(value));
      node.classList.toggle("tv-story-provenance-hit", hit);
    });
  }
}
