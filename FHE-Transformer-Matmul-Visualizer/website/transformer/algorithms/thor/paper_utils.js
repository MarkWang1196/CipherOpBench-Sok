import { emptyMetrics } from "../../core/trace_schema.js";

export const PAPER_TONES = Object.freeze({
  green: 0,
  blue: 1,
  purple: 2,
  orange: 3,
  neutral: 4,
});

export function mod(value, modulus) {
  return ((value % modulus) + modulus) % modulus;
}

export function rotateLeft(values, amount) {
  if (!values.length) return [];
  const shift = mod(amount, values.length);
  return values.map((_, index) => values[(index + shift) % values.length]);
}

export function addVectors(vectors) {
  if (!vectors.length) return [];
  return vectors[0].map((_, index) => vectors.reduce((sum, vector) => sum + Number(vector[index]), 0));
}

export function multiplyVectors(left, right) {
  if (left.length !== right.length) throw new Error("Packed vectors must have the same length");
  return left.map((value, index) => Number(value) * Number(right[index]));
}

export function addMatrices(left, right) {
  return left.map((row, i) => row.map((value, j) => Number(value) + Number(right[i][j])));
}

export function block(matrix, blockRow, blockCol, size) {
  return Array.from({ length: size }, (_, i) =>
    Array.from({ length: size }, (_, j) => Number(matrix[blockRow * size + i][blockCol * size + j]))
  );
}

export function verticalBlock(matrix, blockRow, size) {
  return Array.from({ length: size }, (_, i) => matrix[blockRow * size + i].map(Number));
}

export function lowerDiagonal(matrix, diagonal, length = Math.max(matrix.length, matrix[0].length)) {
  const rows = matrix.length;
  const cols = matrix[0].length;
  return Array.from({ length }, (_, t) => Number(matrix[mod(diagonal + t, rows)][t % cols]));
}

export function upperDiagonal(matrix, diagonal, length = Math.max(matrix.length, matrix[0].length)) {
  const rows = matrix.length;
  const cols = matrix[0].length;
  return Array.from({ length }, (_, t) => Number(matrix[t % rows][mod(diagonal + t, cols)]));
}

export function extendedUpperDiagonal(matrix, diagonal) {
  const rows = matrix.length;
  const cols = matrix[0].length;
  const k = mod(diagonal, cols);
  if (k < rows) return upperDiagonal(matrix, k, cols);
  return rotateLeft(upperDiagonal(matrix, k % rows, cols), rows * Math.floor(k / rows));
}

export function interleave(vectors) {
  if (!vectors.length) return [];
  const width = vectors[0].length;
  return Array.from({ length: width }, (_, t) => vectors.map((vector) => vector[t])).flat();
}

export function packLowerDiagonals(matrices, startDiagonal, count) {
  const size = matrices[0][0].length;
  const packed = [];
  for (let diagonal = startDiagonal; diagonal < startDiagonal + count; diagonal += 1) {
    packed.push(...interleave(matrices.map((matrix) => lowerDiagonal(matrix, diagonal, size))));
  }
  return packed;
}

export function paperToneForPcmmA(row, col, blockSize) {
  const contractionBlock = Math.floor(col / blockSize);
  const localDiagonal = mod((row % blockSize) - (col % blockSize), blockSize);
  return mod(2 * contractionBlock + (localDiagonal === 0 ? 0 : 1), 4);
}

export function paperToneForPcmmB(row, col, blockSize) {
  const contractionBlock = Math.floor(row / blockSize);
  const localDiagonal = mod((row % blockSize) - (col % blockSize), blockSize);
  return mod(2 * contractionBlock + (localDiagonal === 0 ? 0 : 1), 4);
}

export function paperToneForPcmmC(row, col, blockSize) {
  const outputBlock = Math.floor(row / blockSize);
  const localDiagonal = mod((row % blockSize) - (col % blockSize), blockSize);
  return mod(2 * outputBlock + (localDiagonal === 0 ? 0 : 1), 4);
}

export function matrixArtifact({ id, label, values, revealAt, badge, meta, toneForCell }) {
  return {
    id,
    type: "matrix",
    label,
    badge,
    meta: meta || `${values.length}x${values[0].length}`,
    revealAt,
    rows: values.map((row, i) => row.map((value, j) => ({
      value,
      label: String(value),
      tone: toneForCell ? toneForCell(i, j, value) : mod(i + j, 4),
      provenance: [`${id}:${i}:${j}`],
    }))),
  };
}

export function vectorArtifact({ id, label, values, revealAt, badge = "CT", meta = "", tones = [], provenance = [], kind = "vector" }) {
  return {
    id,
    type: kind,
    label,
    badge,
    meta,
    revealAt,
    cells: values.map((value, index) => ({
      value,
      label: String(value),
      tone: tones[index] ?? PAPER_TONES.neutral,
      provenance: provenance[index] || [`${id}:${index}`],
    })),
  };
}

export function noteArtifact(id, label, text, revealAt) {
  return { id, type: "note", label, text, revealAt };
}

export function operator(id, symbol, label, revealAt) {
  return { id, type: "operator", symbol, label, revealAt };
}

export function refsForMatrix(id, rows, cols) {
  return Array.from({ length: rows }, (_, i) =>
    Array.from({ length: cols }, (_, j) => `${id}:${i}:${j}`)
  );
}

export function ctState(id, label, values, slots, secret = true, kind = "tensor", sourceRefs = []) {
  const padded = Array.from({ length: slots }, (_, index) => values[index] ?? 0);
  return {
    id,
    label,
    secret,
    illustrative: true,
    layout: `THOR paper layout - ${slots} slots`,
    slots: padded.map((value, slot) => ({
      id: `${id}:slot:${slot}`,
      logicalId: `${id}:${slot}`,
      slot,
      value,
      expression: `${id}[${slot}] = ${value}`,
      kind,
      sourceRefs: sourceRefs[slot] || [],
    })),
  };
}

export function makeSteps(prefix, definitions) {
  let total = emptyMetrics();
  return definitions.map((definition, index) => {
    const delta = { ...emptyMetrics(), ...(definition.delta || {}) };
    total = Object.fromEntries(Object.keys(total).map((key) => [key, total[key] + (delta[key] || 0)]));
    return {
      id: `${prefix}-${index}`,
      index,
      phase: definition.phase,
      title: definition.title,
      description: definition.description,
      formula: { plain: definition.formula || "" },
      notes: definition.notes || [],
      operation: { type: definition.operation || definition.phase },
      tensors: definition.tensors,
      ciphertexts: definition.ciphertexts || [],
      durationMs: definition.durationMs || 1500,
      metricsDelta: delta,
      metricsTotal: { ...total },
    };
  });
}

export function ref(tensorId, row, col, cols) {
  return {
    tensorId,
    coords: [row, col],
    linear: row * cols + col,
    logicalId: `${tensorId}:${row * cols + col}`,
  };
}
