// Input parsing + validation for the custom matrix panel (spec §8.2).
//
// All validation errors are returned keyed by field so the page can render
// them next to the offending control (never console-only). The parser is
// pure and node-testable.

import { isPowerOfTwo, nextPow2 } from "./packing_planner.js";
import { Layout } from "../../layout_engine.js";

export const MAX_DIM = 64;
export const MAX_SLOTS = 512;
// Bounds the trace size so large inputs stay within product limits and the
// page degrades to Tile/Summary display instead of rendering millions of slots.
export const MAX_TOTAL_CELLS = 1200;

/** Small deterministic PRNG (mulberry32) so "Random" is reproducible by seed. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Generate a rows×cols matrix by mode.
 * @param {"sequential"|"random"|"zero-one"} mode
 */
export function generateValues(rows, cols, mode, seed) {
  const rand = mulberry32(seed == null ? 1 : Number(seed));
  const out = [];
  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) {
      switch (mode) {
        case "sequential":
          row.push(i * cols + j + 1);
          break;
        case "zero-one":
          row.push(rand() < 0.5 ? 0 : 1);
          break;
        case "random":
          row.push(Math.floor(rand() * 9) - 4); // -4..4
          break;
        default:
          row.push(0);
      }
    }
    out.push(row);
  }
  return out;
}

/**
 * Parse a matrix from pasted text: either JSON `[[...],[...]]` or whitespace /
 * comma separated rows with rows separated by newlines or semicolons.
 * @returns {number[][]} or throws with a readable message.
 */
export function parseMatrixText(text) {
  const t = String(text == null ? "" : text).trim();
  if (!t) throw new Error("No matrix values provided");
  if (t.startsWith("[") || t.startsWith("(")) {
    let parsed;
    try {
      parsed = JSON.parse(t);
    } catch {
      throw new Error("Could not parse JSON matrix. Expected e.g. [[1,2,3],[4,5,6]]");
    }
    if (!Array.isArray(parsed) || !parsed.length || !Array.isArray(parsed[0])) {
      throw new Error("JSON matrix must be a non-empty 2D array");
    }
    return parsed.map((row) =>
      row.map((v) => {
        const num = Number(v);
        if (!Number.isFinite(num)) throw new Error("All matrix values must be finite numbers");
        return num;
      })
    );
  }
  // Whitespace/comma rows.
  const rows = t
    .split(/[;\n]+/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (!rows.length) throw new Error("No matrix values provided");
  return rows.map((line) => {
    const cells = line
      .split(/[\s,]+/)
      .map((c) => c.trim())
      .filter(Boolean)
      .map((c) => {
        const num = Number(c);
        if (!Number.isFinite(num)) throw new Error(`"${c}" is not a finite number`);
        return num;
      });
    if (!cells.length) throw new Error("Matrix row is empty");
    return cells;
  });
}

function checkDim(value, name, allowZero = false) {
  const n = Number(value);
  if (!Number.isInteger(n) || !(allowZero ? n >= 0 : n > 0)) {
    throw new Error(`${name} must be ${allowZero ? "a non-negative" : "a positive"} integer`);
  }
  if (n > MAX_DIM) throw new Error(`${name} must be ≤ ${MAX_DIM}`);
  return n;
}

/**
 * Validate everything and produce clean inputs, or a field-keyed error map.
 *
 * @param {object} raw form state
 * @param {string} raw.m "3"
 * @param {string} raw.k
 * @param {string} raw.n
 * @param {string} raw.slots "16"
 * @param {string} raw.leftType "ct"|"pt"
 * @param {string} raw.rightType "ct"|"pt"
 * @param {string} raw.strategy "auto"|"row"|"column"|"diagonal"|"custom"
 * @param {string} raw.valueMode "manual"|"sequential"|"random"|"zero-one"
 * @param {string} raw.valuesText pasted/edited matrix text (manual mode)
 * @param {string} raw.seed
 * @param {{left?:string,right?:string,output?:string}} raw.customLayouts
 * @returns {{ok:boolean, errors:object, inputs:object|null}}
 */
export function parseAndValidateInputs(raw) {
  const errors = {};
  let M = 0;
  let K = 0;
  let N = 0;
  let slots = 16;

  try {
    M = checkDim(raw.m, "M");
    K = checkDim(raw.k, "K");
    N = checkDim(raw.n, "N");
    const totalCells = M * K + K * N + M * N;
    if (totalCells > MAX_TOTAL_CELLS) {
      throw new Error(`M*K + K*N + M*N must be ≤ ${MAX_TOTAL_CELLS} (got ${totalCells})`);
    }
  } catch (err) {
    const m = /^(\w+)/.exec(err.message);
    const field = m && m[1] ? m[1].toLowerCase() : "m";
    errors[field] = err.message;
  }

  try {
    slots = Number(raw.slots);
    if (!isPowerOfTwo(slots)) throw new Error("Slots per CT must be a power of 2");
    if (slots < 8 || slots > MAX_SLOTS) {
      throw new Error(`Slots per CT must be between 8 and ${MAX_SLOTS}`);
    }
  } catch (err) {
    errors.slots = err.message;
  }

  const leftType = String(raw.leftType || "ct").toLowerCase();
  const rightType = String(raw.rightType || "pt").toLowerCase();
  const strategy = String(raw.strategy || "auto").toLowerCase();
  const valueMode = String(raw.valueMode || "manual").toLowerCase();

  let A = null;
  let B = null;
  if (!Object.keys(errors).length) {
    try {
      if (valueMode === "manual") {
        A = parseMatrixText(raw.valuesTextA);
        B = parseMatrixText(raw.valuesTextB);
      } else {
        const seed = raw.seed != null && raw.seed !== "" ? Number(raw.seed) : 1;
        if (!Number.isInteger(seed)) throw new Error("Random seed must be an integer");
        A = generateValues(M, K, valueMode === "manual" ? "sequential" : valueMode, seed);
        B = generateValues(K, N, valueMode === "manual" ? "sequential" : valueMode, seed);
      }
    } catch (err) {
      errors.values = err.message;
    }
  }

  if (A && !errors.values) {
    if (A.length !== M || (A.length && A[0].length !== K)) {
      errors.values = `A must be ${M}×${K} (got ${A.length}×${A[0] ? A[0].length : 0})`;
    }
  }
  if (B && !errors.values) {
    if (B.length !== K || (B.length && B[0].length !== N)) {
      errors.values = `B must be ${K}×${N} (got ${B.length}×${B[0] ? B[0].length : 0})`;
    }
  }
  // Finite check on all values.
  if (A && B && !errors.values) {
    const allFinite = [...A.flat(), ...B.flat()].every((v) => Number.isFinite(Number(v)));
    if (!allFinite) errors.values = "All matrix values must be finite numbers";
  }

  // Slots must fit the shared dim (a single contraction row must fit one CT).
  if (!errors.slots && K > 0) {
    try {
      if (nextPow2(K) > slots) {
        errors.slots = `Slots (${slots}) must be ≥ nextPow2(K) (${nextPow2(K)})`;
      }
    } catch {
      /* K invalid, covered above */
    }
  }

  let customLayouts = null;
  if (strategy === "custom") {
    customLayouts = {
      left: String(raw.customLayouts?.left || "").trim(),
      right: String(raw.customLayouts?.right || "").trim(),
      output: String(raw.customLayouts?.output || "").trim() || null,
    };
    if (!customLayouts.left || !customLayouts.right) {
      errors.strategy = "Custom strategy requires layout strings for A and B";
    } else if (!errors.slots) {
      try {
        Layout.fromString(customLayouts.left, slots, leftType === "ct");
        Layout.fromString(customLayouts.right, slots, rightType === "ct");
      } catch (err) {
        errors.strategy = `Custom layout failed to parse: ${err.message}`;
      }
    }
  }

  if (Object.keys(errors).length) {
    return { ok: false, errors, inputs: null };
  }

  return {
    ok: true,
    errors: {},
    inputs: {
      m: M,
      k: K,
      n: N,
      A,
      B,
      leftSecret: leftType === "ct",
      rightSecret: rightType === "ct",
      slotsPerCiphertext: slots,
      strategy,
      customLayouts,
      valueMode,
      thorCcmmMethod: String(raw.thorCcmmMethod || "lower-lower"),
      seed: raw.seed,
    },
  };
}

/** Human label for a strategy (used by the input panel). */
export function strategyLabel(strategy) {
  return String(strategy || "auto").toUpperCase();
}
