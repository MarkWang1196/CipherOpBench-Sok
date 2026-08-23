// Tile overview (spec §11.4): a compact grid of output tiles so the user can
// see the global tile flow and which tile the current step is operating on.
// Drill-down (choose a tile to replay) is handled by the timeline/step chips.

function makeEl(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text != null) el.textContent = text;
  return el;
}

/**
 * @param {Element} container
 * @param {{plan:object, currentStep:object|null}} cfg
 */
export function renderTiles(container, { plan, currentStep }) {
  container.innerHTML = "";
  if (!plan || !plan.tiling) return;
  const tiling = plan.tiling;
  const nTiles = tiling.nTileRows * tiling.nTileCols;
  if (nTiles <= 1) {
    container.append(makeEl("div", "tv-tiles-note", "Single tile — no tiling required."));
    return;
  }

  const curTile = currentStep && currentStep.tile
    ? `${currentStep.tile.iTile}:${currentStep.tile.jTile}`
    : null;

  const wrap = makeEl("div", "tv-tiles-wrap");
  wrap.append(makeEl("div", "tv-tiles-label", "Output tiles"));
  const grid = makeEl("div", "tv-tiles");
  grid.style.gridTemplateColumns = `repeat(${tiling.nTileCols}, minmax(0, 1fr))`;
  for (const tile of tiling.tiles) {
    const key = `${tile.iTile}:${tile.jTile}`;
    const cell = makeEl(
      "button",
      "tv-tile" + (curTile === key ? " tv-tile--active" : "")
    );
    cell.type = "button";
    cell.textContent = `(${tile.iTile},${tile.jTile})`;
    cell.title = `Tile ${key}: rows ${tile.rowRange[0]}–${tile.rowRange[1] - 1}, cols ${tile.colRange[0]}–${tile.colRange[1] - 1}`;
    grid.append(cell);
  }
  wrap.append(grid);
  container.append(wrap);
}
