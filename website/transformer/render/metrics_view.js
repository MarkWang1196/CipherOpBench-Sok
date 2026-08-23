// Metrics view (spec §14.5, §9.8): current-step delta + cumulative totals.
// Educational counts are labelled "Illustrative"; only Rotom-compiler traces
// are labelled as measured.

const METRIC_META = [
  { key: "ptCtMultiplications", label: "CT×PT mults" },
  { key: "ctCtMultiplications", label: "CT×CT mults" },
  { key: "additions", label: "Additions" },
  { key: "rotations", label: "Rotations" },
  { key: "masks", label: "Masks" },
  { key: "relinearizations", label: "Relins" },
  { key: "rescales", label: "Rescales" },
  { key: "multiplicativeDepth", label: "Depth" },
  { key: "ciphertextCount", label: "CTs (cumulative)" },
];

function makeEl(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text != null) el.textContent = text;
  return el;
}

export function renderMetrics(container, { step, trace }) {
  container.innerHTML = "";
  const wrap = makeEl("div", "tv-metrics");

  const source =
    trace && trace.plan && trace.plan.plannerSource === "rotom-compiler"
      ? "Rotom trace"
      : "Illustrative educational counts";
  wrap.append(makeEl("div", "tv-metrics-source", `Source: ${source}`));

  const grid = makeEl("div", "tv-metrics-grid");
  for (const meta of METRIC_META) {
    const delta = step.metricsDelta ? Number(step.metricsDelta[meta.key]) || 0 : 0;
    const total = step.metricsTotal ? Number(step.metricsTotal[meta.key]) || 0 : 0;
    const row = makeEl("div", "tv-metrics-row");
    row.append(
      makeEl("span", "tv-metrics-label", meta.label),
      makeEl("span", "tv-metrics-delta", delta ? `+${delta}` : ""),
      makeEl("span", "tv-metrics-total", String(total))
    );
    grid.append(row);
  }
  wrap.append(grid);

  const opType = step.operation ? step.operation.type : "";
  if (opType) {
    wrap.append(makeEl("div", "tv-metrics-op", `Operation: ${opType}`));
  }
  container.append(wrap);
}
