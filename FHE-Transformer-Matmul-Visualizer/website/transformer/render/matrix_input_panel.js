// Custom matrix input panel controller (spec §8.2). Reads/writes the form in
// transformer_visualizer.html, auto-fills generated values, toggles the custom
// layout fields, and surfaces field-level errors next to the offending input.

import { generateValues } from "../core/input_parser.js";

export function formatMatrix(matrix) {
  return matrix.map((row) => row.map((v) => String(v)).join(" ")).join("\n");
}

export class MatrixInputPanel {
  /**
   * @param {Element} root form element
   * @param {{onGenerate:(state:object)=>void}} cfg
   */
  constructor(root, { onGenerate }) {
    this.root = root;
    this.onGenerate = onGenerate;
    this.q = (sel) => root.querySelector(sel);
    this._debounceTimer = null;
    this._loading = false;
    this.algorithmMode = "rotom-opp";
    this.defaultConfig = null;

    this.q('[data-field="valueMode"]').addEventListener("change", () => {
      this._syncMode();
      this._scheduleGenerate(0);
    });
    ["m", "k", "n"].forEach((field) => {
      this.q(`[data-field="${field}"]`).addEventListener("input", () => {
        this._syncMode();
        this._scheduleGenerate(350);
      });
    });
    this.q('[data-field="seed"]').addEventListener("input", () => {
      this._syncMode();
      this._scheduleGenerate(350);
    });
    this.q('[data-field="strategy"]').addEventListener("change", () => {
      this._syncCustom();
      this._scheduleGenerate(0);
    });
    ["slots", "leftType", "rightType"].forEach((field) => {
      this.q(`[data-field="${field}"]`).addEventListener("change", () => this._scheduleGenerate(0));
    });
    this.q('[data-field="thorCcmmMethod"]')?.addEventListener("change", () => this._scheduleGenerate(0));
    ["valuesA", "valuesB", "customLeft", "customRight", "customOutput"].forEach((field) => {
      this.q(`[data-field="${field}"]`)?.addEventListener("input", () => this._scheduleGenerate(450));
    });
    this.q('[data-action="generate"]').addEventListener("click", () => this._generate("button"));
    this.q('[data-action="load-defaults"]')?.addEventListener("click", () => {
      this.loadDefault();
      this._syncMode();
      this._syncCustom();
      this._generate("defaults");
    });

    this._syncCustom();
    this._syncMode();
  }

  /** Current form state (raw strings — validation happens in the parser). */
  read() {
    return {
      m: this.q('[data-field="m"]').value,
      k: this.q('[data-field="k"]').value,
      n: this.q('[data-field="n"]').value,
      slots: this.q('[data-field="slots"]').value,
      leftType: this.q('[data-field="leftType"]').value,
      rightType: this.q('[data-field="rightType"]').value,
      strategy: this.q('[data-field="strategy"]').value,
      valueMode: this.q('[data-field="valueMode"]').value,
      thorCcmmMethod: this.q('[data-field="thorCcmmMethod"]')?.value || "lower-lower",
      seed: this.q('[data-field="seed"]').value,
      valuesTextA: this.q('[data-field="valuesA"]').value,
      valuesTextB: this.q('[data-field="valuesB"]').value,
      customLayouts: {
        left: this.q('[data-field="customLeft"]')?.value || "",
        right: this.q('[data-field="customRight"]')?.value || "",
        output: this.q('[data-field="customOutput"]')?.value || "",
      },
    };
  }

  _dims() {
    const m = parseInt(this.q('[data-field="m"]').value, 10);
    const k = parseInt(this.q('[data-field="k"]').value, 10);
    const n = parseInt(this.q('[data-field="n"]').value, 10);
    return { m, k, n };
  }

  _syncMode() {
    const mode = this.q('[data-field="valueMode"]').value;
    const isManual = mode === "manual";
    const ta = [this.q('[data-field="valuesA"]'), this.q('[data-field="valuesB"]')];
    ta.forEach((t) => {
      t.disabled = !isManual;
      t.readOnly = !isManual;
    });
    if (!isManual) this._fillGenerated(mode);
  }

  _fillGenerated(mode) {
    const { m, k, n } = this._dims();
    if (!(m > 0 && k > 0 && n > 0)) return;
    let seed = parseInt(this.q('[data-field="seed"]').value, 10);
    if (!Number.isInteger(seed)) seed = 1;
    try {
      this.q('[data-field="valuesA"]').value = formatMatrix(generateValues(m, k, mode, seed));
      this.q('[data-field="valuesB"]').value = formatMatrix(generateValues(k, n, mode, seed));
    } catch {
      /* dims invalid — leave textareas */
    }
  }

  _syncCustom() {
    const isCustom = this.q('[data-field="strategy"]').value === "custom";
    const group = this.q('[data-field-custom="group"]');
    if (group) group.hidden = !isCustom;
  }

  _scheduleGenerate(delay = 400) {
    if (this._loading) return;
    if (this._debounceTimer) clearTimeout(this._debounceTimer);
    this._setLiveStatus("Editing…", "pending");
    this._debounceTimer = setTimeout(() => {
      this._debounceTimer = null;
      this._generate("live");
    }, Math.max(0, delay));
  }

  _generate(source = "button") {
    if (this._debounceTimer) {
      clearTimeout(this._debounceTimer);
      this._debounceTimer = null;
    }
    this.clearErrors();
    const result = this.onGenerate ? this.onGenerate(this.read(), { source }) : null;
    if (result?.ok === false) {
      this._setLiveStatus(result.message || "Fix the highlighted input to update", "error");
    } else if (result?.ok) {
      this._setLiveStatus(result.message || "Visualization updated", "ok");
    }
    return result;
  }

  _setLiveStatus(message, state = "") {
    const status = this.q("[data-live-status]");
    if (!status) return;
    status.textContent = message || "";
    status.dataset.state = state;
  }

  setStatus(message, state = "") {
    this._setLiveStatus(message, state);
  }

  setAlgorithmMode(mode, { defaults = null, operation = null } = {}) {
    this.algorithmMode = mode || "rotom-opp";
    this.defaultConfig = defaults;
    const isThorPcmm = this.algorithmMode === "thor-pcmm";
    const isThorCcmm = this.algorithmMode === "thor-ccmm";
    const isToyOperation = Boolean(operation);
    const fixedForThor = isThorPcmm || isThorCcmm;
    const fixedForPaper = fixedForThor || isToyOperation;
    this.root.classList.toggle("tv-form--toy", isToyOperation);
    ["m", "k", "n", "slots", "strategy", "leftType", "rightType"].forEach((field) => {
      const input = this.q(`[data-field="${field}"]`);
      if (!input) return;
      const fixed = isToyOperation || (fixedForThor && (field === "leftType" || field === "rightType"));
      input.disabled = fixed;
      input.title = "";
    });
    const strategy = this.q('[data-field="strategy"]');
    if (strategy) {
      [...strategy.options].forEach((option) => {
        const allowed = isToyOperation ? [operation.strategy] : ["auto", "diagonal"];
        option.disabled = fixedForPaper && !allowed.includes(option.value);
      });
      if (isToyOperation) strategy.value = operation.strategy;
      else if (fixedForThor && !["auto", "diagonal"].includes(strategy.value)) strategy.value = "auto";
    }
    const note = this.q("[data-algorithm-note]");
    if (note) {
      note.hidden = !fixedForThor || isToyOperation;
      note.textContent = isThorPcmm
        ? "THOR Figure 3 paper mode: d=4, n=2, s=8. Matrix values are editable; A is PT and B is CT."
        : isThorCcmm
          ? "THOR Figure 4 paper mode: m=2, n=4, s=8. Matrix values and the Upper-Lower / Lower-Lower approach are editable; both operands are CT."
          : "";
    }
    const matrixNames = this.root.querySelectorAll("[data-matrix-name]");
    matrixNames.forEach((node) => {
      const side = node.dataset.matrixName;
      node.textContent = isToyOperation
        ? (side === "left" ? operation.leftLabel : operation.rightLabel)
        : (side === "left" ? "A" : "B");
    });
    const leftShape = this.root.querySelector('[data-matrix-shape="left"]');
    const rightShape = this.root.querySelector('[data-matrix-shape="right"]');
    if (leftShape) leftShape.textContent = isToyOperation ? `${operation.A.length}×${operation.A[0].length}` : "M×K";
    if (rightShape) rightShape.textContent = isToyOperation ? `${operation.B.length}×${operation.B[0].length}` : "K×N";
    const ccmmControls = this.q("[data-thor-ccmm-controls]");
    if (ccmmControls) ccmmControls.hidden = !isThorCcmm;
    const defaultsButton = this.q('[data-action="load-defaults"]');
    if (defaultsButton) defaultsButton.textContent = isToyOperation ? "Reset" : fixedForThor ? "Reset paper example" : "Load 4×4 example";
    this._syncCustom();
  }

  loadDefault() {
    this.load(this.defaultConfig || {
      m: 4,
      k: 4,
      n: 4,
      A: [
        [1, 2, 0, 1],
        [0, 1, 3, 2],
        [2, 0, 1, 1],
        [1, 1, 2, 0],
      ],
      B: [
        [1, 0, 2, 1],
        [0, 1, 1, 2],
        [2, 1, 0, 1],
        [1, 2, 1, 0],
      ],
      leftSecret: true,
      rightSecret: false,
      slotsPerCiphertext: 16,
      strategy: "auto",
    });
  }

  /** Fill the form from a scene descriptor + matrices. */
  load({ m, k, n, A, B, leftSecret, rightSecret, slotsPerCiphertext, strategy, thorCcmmMethod }) {
    this._loading = true;
    this.q('[data-field="m"]').value = String(m);
    this.q('[data-field="k"]').value = String(k);
    this.q('[data-field="n"]').value = String(n);
    this.q('[data-field="slots"]').value = String(slotsPerCiphertext);
    this.q('[data-field="leftType"]').value = leftSecret ? "ct" : "pt";
    this.q('[data-field="rightType"]').value = rightSecret ? "ct" : "pt";
    this.q('[data-field="strategy"]').value = strategy || "auto";
    if (thorCcmmMethod && this.q('[data-field="thorCcmmMethod"]')) {
      this.q('[data-field="thorCcmmMethod"]').value = thorCcmmMethod;
    }
    this.q('[data-field="valueMode"]').value = "manual";
    this.q('[data-field="valuesA"]').value = formatMatrix(A);
    this.q('[data-field="valuesB"]').value = formatMatrix(B);
    this._syncMode();
    this._syncCustom();
    this._loading = false;
  }

  setErrors(errors) {
    this.clearErrors();
    for (const [field, msg] of Object.entries(errors)) {
      const input = this.q(`[data-field="${field}"]`);
      const err = this.q(`[data-error="${field}"]`);
      if (input) input.classList.add("tv-invalid");
      if (field === "values") {
        this.q('[data-field="valuesA"]')?.classList.add("tv-invalid");
        this.q('[data-field="valuesB"]')?.classList.add("tv-invalid");
      }
      if (err) err.textContent = msg;
    }
  }

  clearErrors() {
    this.root.querySelectorAll(".tv-invalid").forEach((el) => el.classList.remove("tv-invalid"));
    this.root.querySelectorAll("[data-error]").forEach((el) => (el.textContent = ""));
  }

  setBusy(busy) {
    const btn = this.q('[data-action="generate"]');
    if (btn) {
      btn.disabled = busy;
      btn.textContent = busy ? "Generating…" : "Generate visualization";
    }
  }
}
