// Tensor view (spec §14.1). Renders the operand matrices (left, right, and the
// optional transposed right source) plus the output matrix.
//
// Cells are <button>s so they are keyboard-focusable; each carries the data
// attributes used by the interaction layer (tensorId / coords / linear) and a
// stable data-logical-id for FLIP matching across steps.

function makeEl(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text != null) el.textContent = text;
  return el;
}

function renderOneMatrix(container, tensor, compact) {
  const wrap = makeEl("div", "tv-tensor-group");
  const head = makeEl("div", "tv-tensor-head");
  head.append(
    makeEl("span", "tv-tensor-name", tensor.label || tensor.id),
    makeEl("span", "tv-tensor-shape", `${tensor.shape[0]}×${tensor.shape[1]}`)
  );
  wrap.append(head);

  const [rows, cols] = tensor.shape;
  const grid = makeEl("div", "tv-tensor-grid");
  grid.style.gridTemplateColumns = `repeat(${cols}, var(--cell))`;
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "tv-cell tv-tensor-cell" + (tensor.output ? " tv-cell--output" : "");
      const linear = i * cols + j;
      const hasValue = Array.isArray(tensor.values) && tensor.values[i] && tensor.values[i][j] != null;
      cell.dataset.tensorId = tensor.id;
      cell.dataset.coords = `${i},${j}`;
      cell.dataset.linear = String(linear);
      cell.dataset.logicalId = `${tensor.id}:${linear}`;
      cell.dataset.value = hasValue ? String(tensor.values[i][j]) : "";
      cell.setAttribute(
        "aria-label",
        `${tensor.id}[${i},${j}] = ${hasValue ? tensor.values[i][j] : "pending"}`
      );
      cell.textContent = hasValue ? String(tensor.values[i][j]) : "…";
      grid.append(cell);
    }
  }
  wrap.append(grid);
  container.append(wrap);
}

/**
 * Render all tensors for a step.
 * @param {Element} container
 * @param {{tensors: object, focus: object|null}} cfg
 */
export function renderTensors(container, { tensors, focus }) {
  container.innerHTML = "";
  if (!tensors || typeof tensors !== "object") return;
  // Order: left, right source (original K), right (as multiplied), output.
  if (tensors.left) renderOneMatrix(container, tensors.left);
  if (tensors.rightSource) renderOneMatrix(container, tensors.rightSource);
  if (tensors.right) renderOneMatrix(container, tensors.right);
  if (tensors.output) {
    renderOneMatrix(container, { ...tensors.output, output: true });
  } else {
    const note = makeEl("div", "tv-pending");
    note.textContent = "Output pending — appears at the Output step.";
    container.append(note);
  }
}
