// Application bootstrap for the Transformer matmul visualizer.
//
// Wires the input panel, scene selector, timeline controller, animation
// coordinator and the render views together. The trace is the single source
// of truth; every view is a projection of the current step.

import { SCENES } from "./scenes/defaults.js";
import { buildMatmulScene } from "./core/scene_builder.js";
import { transpose } from "./core/matrix.js";
import { buildThorFigure3PcmmTrace, THOR_FIGURE3_PRESET } from "./algorithms/thor/figure3_pcmm.js";
import { buildThorFigure4CcmmTrace, THOR_FIGURE4_PRESET, THOR_CCMM_METHODS } from "./algorithms/thor/figure4_ccmm.js";
import {
  THOR_OPERATION_IDS,
  buildThorOperationTrace,
  getThorOperation,
  thorOperationDefaults,
  parseThorOperationInputs,
} from "./algorithms/thor/bert_operations.js";
import {
  MOAI_OPERATION_IDS,
  buildMoaiOperationTrace,
  getMoaiOperation,
  moaiOperationDefaults,
  parseMoaiOperationInputs,
} from "./algorithms/moai/bert_operations.js";
import {
  BLB_OPERATION_IDS,
  buildBlbOperationTrace,
  getBlbOperation,
  blbOperationDefaults,
  parseBlbOperationInputs,
} from "./algorithms/blb/bert_operations.js";
import {
  CACHEMIR_OPERATION_IDS,
  buildCachemirOperationTrace,
  getCachemirOperation,
  cachemirOperationDefaults,
  parseCachemirOperationInputs,
} from "./algorithms/cachemir/llama_operations.js";
import {
  EUSTON_OPERATION_IDS,
  buildEustonOperationTrace,
  getEustonOperation,
  eustonOperationDefaults,
  parseEustonOperationInputs,
} from "./algorithms/euston/transformer_operations.js";
import {
  STIP_OPERATION_IDS,
  buildStipOperationTrace,
  getStipOperation,
  stipOperationDefaults,
  parseStipOperationInputs,
} from "./algorithms/stip/transformer_operations.js";
import {
  NEXUS_OPERATION_IDS,
  buildNexusOperationTrace,
  getNexusOperation,
  nexusOperationDefaults,
  parseNexusOperationInputs,
} from "./algorithms/nexus/transformer_operations.js";
import {
  TRICYCLE_OPERATION_IDS,
  buildTricycleOperationTrace,
  getTricycleOperation,
  tricycleOperationDefaults,
  parseTricycleOperationInputs,
} from "./algorithms/tricycle/transformer_operations.js";
import { ALGORITHM_IDS } from "./algorithms/algorithm_registry.js?v=20260813-tricycle";
import { buildTraceFromInputs } from "./algorithms/input_adapter.js?v=20260813-tricycle";
import { parseAndValidateInputs } from "./core/input_parser.js?v=20260813-tricycle";
import { TimelineController } from "./animation/timeline_controller.js";
import { AnimationCoordinator } from "./animation/animation_coordinator.js";
import { prefersReducedMotion, findLogical } from "./animation/flip.js";
import { renderTensors } from "./render/tensor_view.js";
import { renderCiphertexts } from "./render/ciphertext_view.js";
import { renderFormula } from "./render/formula_view.js";
import { renderMetrics } from "./render/metrics_view.js";
import { renderStepChips, syncRange } from "./render/step_navigator.js";
import { renderTiles } from "./render/tile_overview.js";
import { renderConnections } from "./render/connection_overlay.js";
import { MatrixInputPanel } from "./render/matrix_input_panel.js?v=20260812-blb";
import { StoryView } from "./render/story_view.js?v=20260822-step-copy";

const $ = (sel) => document.querySelector(sel);

const METHOD_NAMES = Object.freeze({
  thor: "THOR",
  moai: "MOAI",
  blb: "BLB",
  cachemir: "Cachemir",
  euston: "Euston",
  stip: "STIP",
  nexus: "NEXUS",
  tricycle: "Tricycle",
});

const METHOD_VENUES = Object.freeze({
  thor: "CCS'25",
  moai: "ICLR'26",
  blb: "USENIX Security'25",
  cachemir: "arXiv'26",
  euston: "S&P'26",
  stip: "ePrint'26",
  nexus: "NDSS'25",
  tricycle: "Cryptology ePrint'25",
});

/** Build provenance indexes for the current step (spec §17). */
function buildStepIndex(step, outputId) {
  const sourceToSlot = new Map(); // "X:6" -> Set<logicalId>
  const outputToSlot = new Map(); // "Q:0:0" -> Set<logicalId>
  const slotByLogical = new Map();
  for (const ct of step.ciphertexts || []) {
    for (const s of ct.slots) {
      slotByLogical.set(s.logicalId, s);
      for (const ref of s.sourceRefs || []) {
        if (ref.tensorId != null && ref.linear != null) {
          const key = `${ref.tensorId}:${ref.linear}`;
          if (!sourceToSlot.has(key)) sourceToSlot.set(key, new Set());
          sourceToSlot.get(key).add(s.logicalId);
        }
      }
      if (s.outputCoords && outputId) {
        const key = `${outputId}:${s.outputCoords[0]}:${s.outputCoords[1]}`;
        if (!outputToSlot.has(key)) outputToSlot.set(key, new Set());
        outputToSlot.get(key).add(s.logicalId);
      }
    }
  }
  return { sourceToSlot, outputToSlot, slotByLogical };
}

class App {
  constructor() {
    this.trace = null;
    this.controller = null;
    this.coordinator = null;
    this.index = { sourceToSlot: new Map(), outputToSlot: new Map(), slotByLogical: new Map() };
    this.pinned = { logicalIds: new Set(), cellKeys: new Set() };
    this.currentFocus = null;
    this.activeAlgorithm = THOR_OPERATION_IDS.QUERY;
    this.activeMethod = "thor";
    this.viewMode = "story";
    this._reduced = prefersReducedMotion();

    this._cacheDom();
    this._bindEvents();
    this.inputPanel = new MatrixInputPanel(this.inputForm, {
      onGenerate: (state) => this._onGenerate(state),
    });

    this.story = new StoryView({
      root: this.storyRoot,
      viewport: this.storyViewport,
      canvas: this.storyCanvas,
      title: this.storyTitle,
      subtitle: this.storySubtitle,
      detail: this.storyDetail,
      onStepRequest: (index) => {
        if (this.controller && index !== this.controller.currentStep) this.controller.seek(index);
      },
    });

    this._setView("story");
    this.loadThorOperation(THOR_OPERATION_IDS.QUERY);
  }

  _cacheDom() {
    this.inputForm = $("#input-form");
    this.tensorView = $("#tensor-view");
    this.slotView = $("#ciphertext-view");
    this.formulaView = $("#formula-view");
    this.metricsView = $("#metrics-view");
    this.planView = $("#plan-view");
    this.tileView = $("#tile-view");
    this.planPanel = this.planView?.closest(".tv-panel");
    this.tilePanel = this.tileView?.closest(".tv-panel");
    this.stepChips = $("#step-chips");
    this.stepCount = $("#step-count");
    this.liveRegion = $("#live-region");
    this.gridWrap = $("#grid-wrap");
    this.svg = $("#connections");
    this.stepLabel = $("#step-label");
    this.range = $("#range-step");
    this.btnPlay = $("#btn-play");
    this.speedSelect = $("#speed-select");
    this.storyRoot = $("#story-view");
    this.storyViewport = $("#story-viewport");
    this.storyCanvas = $("#story-canvas");
    this.storyTitle = $("#story-title");
    this.storySubtitle = $("#story-subtitle");
    this.storyDetail = $("#story-detail");
    this.inspectorView = $("#inspector-view");
    this.inspectorButton = $('[data-view="inspector"]');
    this.interleavedLegend = $("#interleaved-legend");
    this.batchSymbols = $("#batch-symbols");
    this.mpcLegend = $("#mpc-legend");
    this.methodContextTitle = $("#method-context-title");
    this.methodContextSubtitle = $("#method-context-subtitle");
    this.contextMatmul = $("#context-matmul");
    this.contextPacking = $("#context-packing");
    this.contextShape = $("#context-shape");
    this.contextSlots = $("#context-slots");
    this.stepFormulaView = $("#step-formula-view");
    this.stepMetricsView = $("#step-metrics-view");
    this.matrixEditorPanel = $("#matrix-editor-panel");
  }

  _bindEvents() {
    // Scene selector.
    document.querySelectorAll("[data-scene]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const desc = SCENES[btn.dataset.scene];
        if (desc) this.loadScene(desc);
      });
    });
    document.querySelectorAll("[data-algorithm]").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (btn.dataset.algorithm === "thor-pcmm") this.loadThorPcmm();
        if (btn.dataset.algorithm === "thor-ccmm") this.loadThorCcmm();
      });
    });
    document.querySelectorAll("[data-thor-operation]").forEach((btn) => {
      btn.addEventListener("click", () => this.loadThorOperation(btn.dataset.thorOperation));
    });
    document.querySelectorAll("[data-moai-operation]").forEach((btn) => {
      btn.addEventListener("click", () => this.loadMoaiOperation(btn.dataset.moaiOperation));
    });
    document.querySelectorAll("[data-blb-operation]").forEach((btn) => {
      btn.addEventListener("click", () => this.loadBlbOperation(btn.dataset.blbOperation));
    });
    document.querySelectorAll("[data-cachemir-operation]").forEach((btn) => {
      btn.addEventListener("click", () => this.loadCachemirOperation(btn.dataset.cachemirOperation));
    });
    document.querySelectorAll("[data-euston-operation]").forEach((btn) => {
      btn.addEventListener("click", () => this.loadEustonOperation(btn.dataset.eustonOperation));
    });
    document.querySelectorAll("[data-stip-operation]").forEach((btn) => {
      btn.addEventListener("click", () => this.loadStipOperation(btn.dataset.stipOperation));
    });
    document.querySelectorAll("[data-nexus-operation]").forEach((btn) => {
      btn.addEventListener("click", () => this.loadNexusOperation(btn.dataset.nexusOperation));
    });
    document.querySelectorAll("[data-tricycle-operation]").forEach((btn) => {
      btn.addEventListener("click", () => this.loadTricycleOperation(btn.dataset.tricycleOperation));
    });
    document.querySelectorAll("[data-method]").forEach((btn) => {
      btn.addEventListener("click", () => this.selectMethod(btn.dataset.method));
    });
    document.querySelectorAll("[data-view]").forEach((btn) => {
      btn.addEventListener("click", () => this._setView(btn.dataset.view));
    });
    $('[data-action="open-matrix-editor"]')?.addEventListener("click", () => {
      if (!this.matrixEditorPanel) return;
      this.matrixEditorPanel.open = true;
      this.matrixEditorPanel.scrollIntoView({
        block: "nearest",
        behavior: this._reduced ? "auto" : "smooth",
      });
    });

    // Timeline buttons.
    $("#btn-first").addEventListener("click", () => this.controller?.seek(0));
    $("#btn-prev").addEventListener("click", () => this.controller?.previous());
    $("#btn-play").addEventListener("click", () => this._togglePlay());
    $("#btn-next").addEventListener("click", () => this.controller?.next());
    $("#btn-last").addEventListener("click", () =>
      this.controller?.seek((this.trace?.steps?.length || 1) - 1)
    );
    $("#btn-replay").addEventListener("click", () => this.controller?.replay());
    this.speedSelect.addEventListener("change", () => {
      this.controller?.setSpeed(Number(this.speedSelect.value));
    });

    // Keyboard shortcuts (ignored while typing in form fields).
    document.addEventListener("keydown", (e) => this._onKey(e));

    // Hover / click provenance.
    this.gridWrap.addEventListener("mouseover", (e) => this._onHover(e));
    this.gridWrap.addEventListener("click", (e) => this._onClick(e));
    this.gridWrap.addEventListener("mouseleave", () => {
      if (!this.pinned.logicalIds.size && !this.pinned.cellKeys.size) this._clearHighlight();
    });
    this.gridWrap.addEventListener("click", (e) => {
      if (e.target === this.gridWrap) this._clearPinned();
    });

    // Keep the SVG overlay aligned as panes resize.
    const ro = new ResizeObserver(() => this._renderConnections());
    ro.observe(this.gridWrap);
    window.addEventListener("resize", () => this._renderConnections());
  }

  _onKey(e) {
    if (this._isTyping(e.target)) return;
    const k = e.key;
    if (k === " ") {
      e.preventDefault();
      this._togglePlay();
    } else if (k === "ArrowLeft") {
      this.controller?.previous();
    } else if (k === "ArrowRight") {
      this.controller?.next();
    } else if (k === "Home") {
      this.controller?.seek(0);
    } else if (k === "End") {
      this.controller?.seek((this.trace?.steps?.length || 1) - 1);
    } else if (k === "r" || k === "R") {
      this.controller?.replay();
    } else if (k === "Escape") {
      this._clearPinned();
    }
  }

  _isTyping(el) {
    return (
      el &&
      (el.tagName === "INPUT" ||
        el.tagName === "TEXTAREA" ||
        el.tagName === "SELECT" ||
        el.isContentEditable)
    );
  }

  loadScene(desc) {
    try {
      this.activeAlgorithm = ALGORITHM_IDS.ROTOM_OPP;
      const formB = desc.transposeRight ? transpose(desc.right.values) : desc.right.values;
      this.inputPanel.setAlgorithmMode(ALGORITHM_IDS.ROTOM_OPP, {
        defaults: {
          m: desc.left.values.length,
          k: desc.left.values[0].length,
          n: formB[0].length,
          A: desc.left.values,
          B: formB,
          leftSecret: desc.left.secret,
          rightSecret: desc.right.secret,
          slotsPerCiphertext: desc.slotsPerCiphertext,
          strategy: desc.strategy || "auto",
        },
      });
      const trace = buildMatmulScene({
        left: desc.left,
        right: desc.right,
        slotsPerCiphertext: desc.slotsPerCiphertext,
        packingStrategy: desc.strategy || "auto",
        options: {
          transposeRight: desc.transposeRight,
          outputId: desc.outputId,
          outputLabel: desc.outputLabel,
        },
      });
      this._setTrace(trace, desc);
      this._updateSceneButtons(desc.id);
    } catch (err) {
      console.error("Failed to build scene:", err);
      this._announce(`Failed to build scene: ${err.message}`);
    }
  }

  loadThorPcmm() {
    try {
      this.activeAlgorithm = ALGORITHM_IDS.THOR_PCMM;
      const formConfig = {
        m: 4,
        k: 4,
        n: 2,
        A: THOR_FIGURE3_PRESET.A,
        B: THOR_FIGURE3_PRESET.B,
        leftSecret: false,
        rightSecret: true,
        slotsPerCiphertext: 8,
        strategy: "diagonal",
      };
      this.inputPanel.setAlgorithmMode(ALGORITHM_IDS.THOR_PCMM, { defaults: formConfig });
      this.inputPanel.load(formConfig);
      const trace = buildThorFigure3PcmmTrace({ slotsPerCiphertext: 8, packingStrategy: "diagonal" });
      this._setTrace(trace, null);
      this._updateSceneButtons("thor-pcmm");
      this._setView("story");
      this.inputPanel.setStatus("THOR Figure 3 values ready - edits preserve paper colors", "ok");
    } catch (err) {
      console.error("Failed to build THOR scene:", err);
      this._announce(`Failed to build THOR scene: ${err.message}`);
    }
  }

  loadThorCcmm() {
    try {
      this.activeAlgorithm = ALGORITHM_IDS.THOR_CCMM;
      const formConfig = {
        m: 2,
        k: 4,
        n: 4,
        A: THOR_FIGURE4_PRESET.A,
        B: THOR_FIGURE4_PRESET.B,
        leftSecret: true,
        rightSecret: true,
        slotsPerCiphertext: 8,
        strategy: "diagonal",
        thorCcmmMethod: THOR_CCMM_METHODS.LOWER_LOWER,
      };
      this.inputPanel.setAlgorithmMode(ALGORITHM_IDS.THOR_CCMM, { defaults: formConfig });
      this.inputPanel.load(formConfig);
      const trace = buildThorFigure4CcmmTrace({
        A: formConfig.A,
        B: formConfig.B,
        slotsPerCiphertext: 8,
        method: formConfig.thorCcmmMethod,
      });
      this._setTrace(trace, null);
      this._updateSceneButtons("thor-ccmm");
      this._setView("story");
      this.inputPanel.setStatus("THOR Figure 4 values ready - choose either CC-MM approach", "ok");
    } catch (err) {
      console.error("Failed to build THOR Figure 4 scene:", err);
      this._announce(`Failed to build THOR Figure 4 scene: ${err.message}`);
    }
  }

  loadThorOperation(operationId) {
    try {
      const descriptor = getThorOperation(operationId);
      const formConfig = thorOperationDefaults(operationId);
      if (!descriptor || !formConfig) throw new Error(`Unknown THOR operation: ${operationId}`);
      this._activateMethodShell("thor");
      this.activeAlgorithm = operationId;
      this.inputPanel.setAlgorithmMode(operationId, { defaults: formConfig, operation: descriptor });
      this.inputPanel.load(formConfig);
      const trace = buildThorOperationTrace(operationId, formConfig);
      this._setTrace(trace, null);
      this._updateSceneButtons(operationId);
      this._setView("story");
      this.inputPanel.setStatus("✓", "ok");
    } catch (err) {
      console.error("Failed to build THOR operation:", err);
      this._announce(`Failed to build THOR operation: ${err.message}`);
    }
  }

  loadMoaiOperation(operationId) {
    try {
      const descriptor = getMoaiOperation(operationId);
      const formConfig = moaiOperationDefaults(operationId);
      if (!descriptor || !formConfig) throw new Error(`Unknown MOAI operation: ${operationId}`);
      this._activateMethodShell("moai");
      this.activeAlgorithm = operationId;
      this.inputPanel.setAlgorithmMode(operationId, { defaults: formConfig, operation: descriptor });
      this.inputPanel.load(formConfig);
      const trace = buildMoaiOperationTrace(operationId, formConfig);
      this._setTrace(trace, null);
      this._updateSceneButtons(operationId);
      this._setView("story");
      this.inputPanel.setStatus("✓", "ok");
    } catch (err) {
      console.error("Failed to build MOAI operation:", err);
      this._announce(`Failed to build MOAI operation: ${err.message}`);
    }
  }

  loadBlbOperation(operationId) {
    try {
      const descriptor = getBlbOperation(operationId);
      const formConfig = blbOperationDefaults(operationId);
      if (!descriptor || !formConfig) throw new Error(`Unknown BLB operation: ${operationId}`);
      this._activateMethodShell("blb");
      this.activeAlgorithm = operationId;
      this.inputPanel.setAlgorithmMode(operationId, { defaults: formConfig, operation: descriptor });
      this.inputPanel.load(formConfig);
      const trace = buildBlbOperationTrace(operationId, formConfig);
      this._setTrace(trace, null);
      this._updateSceneButtons(operationId);
      this._setView("story");
      this.inputPanel.setStatus("✓", "ok");
    } catch (err) {
      console.error("Failed to build BLB operation:", err);
      this._announce(`Failed to build BLB operation: ${err.message}`);
    }
  }

  loadCachemirOperation(operationId) {
    try {
      const descriptor = getCachemirOperation(operationId);
      const formConfig = cachemirOperationDefaults(operationId);
      if (!descriptor || !formConfig) throw new Error(`Unknown Cachemir operation: ${operationId}`);
      this._activateMethodShell("cachemir");
      this.activeAlgorithm = operationId;
      this.inputPanel.setAlgorithmMode(operationId, { defaults: formConfig, operation: descriptor });
      this.inputPanel.load(formConfig);
      const trace = buildCachemirOperationTrace(operationId, formConfig);
      this._setTrace(trace, null);
      this._updateSceneButtons(operationId);
      this._setView("story");
      this.inputPanel.setStatus("✓", "ok");
    } catch (err) {
      console.error("Failed to build Cachemir operation:", err);
      this._announce(`Failed to build Cachemir operation: ${err.message}`);
    }
  }

  loadEustonOperation(operationId) {
    try {
      const descriptor = getEustonOperation(operationId);
      const formConfig = eustonOperationDefaults(operationId);
      if (!descriptor || !formConfig) throw new Error(`Unknown Euston operation: ${operationId}`);
      this._activateMethodShell("euston");
      this.activeAlgorithm = operationId;
      this.inputPanel.setAlgorithmMode(operationId, { defaults: formConfig, operation: descriptor });
      this.inputPanel.load(formConfig);
      const trace = buildEustonOperationTrace(operationId, formConfig);
      this._setTrace(trace, null);
      this._updateSceneButtons(operationId);
      this._setView("story");
      this.inputPanel.setStatus("✓", "ok");
    } catch (err) {
      console.error("Failed to build Euston operation:", err);
      this._announce(`Failed to build Euston operation: ${err.message}`);
    }
  }

  loadStipOperation(operationId) {
    try {
      const descriptor = getStipOperation(operationId);
      const formConfig = stipOperationDefaults(operationId);
      if (!descriptor || !formConfig) throw new Error(`Unknown STIP operation: ${operationId}`);
      this._activateMethodShell("stip");
      this.activeAlgorithm = operationId;
      this.inputPanel.setAlgorithmMode(operationId, { defaults: formConfig, operation: descriptor });
      this.inputPanel.load(formConfig);
      const trace = buildStipOperationTrace(operationId, formConfig);
      this._setTrace(trace, null);
      this._updateSceneButtons(operationId);
      this._setView("story");
      this.inputPanel.setStatus("✓", "ok");
    } catch (err) {
      console.error("Failed to build STIP operation:", err);
      this._announce(`Failed to build STIP operation: ${err.message}`);
    }
  }

  loadNexusOperation(operationId) {
    try {
      const descriptor = getNexusOperation(operationId);
      const formConfig = nexusOperationDefaults(operationId);
      if (!descriptor || !formConfig) throw new Error(`Unknown NEXUS operation: ${operationId}`);
      this._activateMethodShell("nexus");
      this.activeAlgorithm = operationId;
      this.inputPanel.setAlgorithmMode(operationId, { defaults: formConfig, operation: descriptor });
      this.inputPanel.load(formConfig);
      const trace = buildNexusOperationTrace(operationId, formConfig);
      this._setTrace(trace, null);
      this._updateSceneButtons(operationId);
      this._setView("story");
      this.inputPanel.setStatus("✓", "ok");
    } catch (err) {
      console.error("Failed to build NEXUS operation:", err);
      this._announce(`Failed to build NEXUS operation: ${err.message}`);
    }
  }

  loadTricycleOperation(operationId) {
    try {
      const descriptor = getTricycleOperation(operationId);
      const formConfig = tricycleOperationDefaults(operationId);
      if (!descriptor || !formConfig) throw new Error(`Unknown Tricycle operation: ${operationId}`);
      this._activateMethodShell("tricycle");
      this.activeAlgorithm = operationId;
      this.inputPanel.setAlgorithmMode(operationId, { defaults: formConfig, operation: descriptor });
      this.inputPanel.load(formConfig);
      const trace = buildTricycleOperationTrace(operationId, formConfig);
      this._setTrace(trace, null);
      this._updateSceneButtons(operationId);
      this._setView("story");
      this.inputPanel.setStatus("✓", "ok");
    } catch (err) {
      console.error("Failed to build Tricycle operation:", err);
      this._announce(`Failed to build Tricycle operation: ${err.message}`);
    }
  }

  selectMethod(method) {
    if (!["thor", "moai", "blb", "cachemir", "euston", "stip", "nexus", "tricycle"].includes(method)) return;
    this._activateMethodShell(method);
    if (method === "tricycle") this.loadTricycleOperation(TRICYCLE_OPERATION_IDS.QUERY);
    else if (method === "nexus") this.loadNexusOperation(NEXUS_OPERATION_IDS.QUERY);
    else if (method === "stip") this.loadStipOperation(STIP_OPERATION_IDS.QUERY);
    else if (method === "euston") this.loadEustonOperation(EUSTON_OPERATION_IDS.QUERY);
    else if (method === "cachemir") this.loadCachemirOperation(CACHEMIR_OPERATION_IDS.QUERY);
    else if (method === "blb") this.loadBlbOperation(BLB_OPERATION_IDS.QUERY);
    else if (method === "moai") this.loadMoaiOperation(MOAI_OPERATION_IDS.QUERY);
    else this.loadThorOperation(THOR_OPERATION_IDS.QUERY);
  }

  _activateMethodShell(method) {
    this.activeMethod = method;
    document.body.classList.toggle("tv-method-moai", method === "moai");
    document.body.classList.toggle("tv-method-blb", method === "blb");
    document.body.classList.toggle("tv-method-cachemir", method === "cachemir");
    document.body.classList.toggle("tv-method-euston", method === "euston");
    document.body.classList.toggle("tv-method-stip", method === "stip");
    document.body.classList.toggle("tv-method-nexus", method === "nexus");
    document.body.classList.toggle("tv-method-tricycle", method === "tricycle");
    document.querySelectorAll("[data-method-panel]").forEach((panel) => {
      panel.hidden = panel.dataset.methodPanel !== method;
    });
    document.querySelectorAll("[data-method]").forEach((button) => {
      const pressed = button.dataset.method === method;
      button.classList.toggle("tv-method-btn--active", pressed);
      button.setAttribute("aria-pressed", pressed ? "true" : "false");
    });
  }

  _setTrace(trace, desc) {
    if (this.controller) this.controller.destroy();
    this.trace = trace;
    const diagramOnly = trace.plan?.visualStyle === "diagram-only";
    const traceMethod = trace.plan?.method;
    if (["thor", "moai", "blb", "cachemir", "euston", "stip", "nexus", "tricycle"].includes(traceMethod)) this._activateMethodShell(traceMethod);
    if (this.interleavedLegend) this.interleavedLegend.hidden = !(Number(trace.parameters?.batchCount) > 1);
    if (this.batchSymbols) this.batchSymbols.textContent = trace.plan?.batchLabel === "heads" ? "h₀ h₁" : trace.plan?.batchLabel === "tokens" ? "τ₀ τ₁" : "b₀ b₁";
    if (this.mpcLegend) this.mpcLegend.hidden = traceMethod !== "blb";
    this._renderTraceContext(trace);
    document.body.classList.toggle("tv-diagram-mode", diagramOnly);
    this.gridWrap.classList.toggle("tv-diagram-only", diagramOnly);
    if (this.planPanel) this.planPanel.hidden = diagramOnly;
    if (this.tilePanel) this.tilePanel.hidden = diagramOnly;
    if (this.inspectorButton) this.inspectorButton.hidden = diagramOnly;
    if (diagramOnly) this._setView("story");
    this.pinned = { logicalIds: new Set(), cellKeys: new Set() };
    this.story.setTrace(trace);

    this.coordinator = new AnimationCoordinator({
      container: this.gridWrap,
      renderStep: (t, i) => this.renderStep(t, i),
      reducedMotion: this._reduced,
      onSettled: () => this._renderConnections(),
    });

    this.controller = new TimelineController(
      trace,
      { renderStep: (t, i) => this.renderStep(t, i) },
      this.coordinator,
      { reducedMotion: this._reduced, onStateChange: () => this._onStateChange() }
    );
    this._onStateChange(); // paint the initial step label / chips / range

    this._renderPlan();
    // Prefill the custom panel so the current scene is reproducible there too.
    if (desc) {
      const B = desc.transposeRight ? transpose(desc.right.values) : desc.right.values;
      this.inputPanel.load({
        m: desc.left.values.length,
        k: desc.left.values[0].length,
        n: B[0].length,
        A: desc.left.values,
        B,
        leftSecret: desc.left.secret,
        rightSecret: desc.right.secret,
        slotsPerCiphertext: desc.slotsPerCiphertext,
        strategy: desc.strategy || "auto",
      });
    }
  }

  renderStep(trace, index) {
    const step = trace.steps[index];
    if (!step) return;
    this.currentStep = step;
    this.currentFocus = step.focus || null;

    renderTensors(this.tensorView, { tensors: step.tensors, focus: step.focus });
    renderCiphertexts(this.slotView, { ciphertexts: step.ciphertexts });
    renderFormula(this.formulaView, { step });
    renderMetrics(this.metricsView, { step, trace });
    if (this.stepFormulaView) renderFormula(this.stepFormulaView, { step });
    if (this.stepMetricsView) renderMetrics(this.stepMetricsView, { step, trace });
    renderTiles(this.tileView, { plan: trace.plan, currentStep: step });
    this.story.renderStep(trace, index, { scroll: this.viewMode === "story" && index > 0 });

    this.index = buildStepIndex(step, trace.output?.id);
    this._clearHighlight();
    this._renderConnections();
    this._announce(`Step ${index}: ${step.title}`);
  }

  _onStateChange() {
    const c = this.controller;
    if (!c) return;
    const cur = c.currentStep;
    const total = c.totalSteps;
    this.stepLabel.textContent = `${cur + 1} / ${total}`;
    if (this.stepCount) this.stepCount.textContent = `${total} ${total === 1 ? "step" : "steps"}`;
    this.btnPlay.textContent = c.isPlaying ? "Pause" : "Play";
    this.btnPlay.setAttribute("aria-pressed", c.isPlaying ? "true" : "false");
    syncRange(this.range, {
      trace: this.trace,
      current: cur,
      onSeek: (i) => this.controller?.seek(i),
    });
    renderStepChips(this.stepChips, {
      trace: this.trace,
      current: cur,
      onSelect: (i) => this.controller?.seek(i),
    });
  }

  _togglePlay() {
    if (!this.controller) return;
    if (this.controller.isPlaying) this.controller.pause();
    else this.controller.play();
  }

  _updateSceneButtons(activeId) {
    document.querySelectorAll("[data-scene], [data-algorithm], [data-thor-operation], [data-moai-operation], [data-blb-operation], [data-cachemir-operation], [data-euston-operation], [data-stip-operation], [data-nexus-operation], [data-tricycle-operation]").forEach((btn) => {
      const id = btn.dataset.scene || btn.dataset.algorithm || btn.dataset.thorOperation || btn.dataset.moaiOperation || btn.dataset.blbOperation || btn.dataset.cachemirOperation || btn.dataset.eustonOperation || btn.dataset.stipOperation || btn.dataset.nexusOperation || btn.dataset.tricycleOperation;
      const pressed = id === activeId;
      btn.setAttribute("aria-pressed", pressed ? "true" : "false");
      btn.classList.toggle("tv-op--active", pressed);
    });
  }

  _setView(mode) {
    if (mode !== "story" && mode !== "inspector") return;
    this.viewMode = mode;
    if (this.storyRoot) this.storyRoot.hidden = mode !== "story";
    if (this.inspectorView) this.inspectorView.hidden = mode !== "inspector";
    document.querySelectorAll("[data-view]").forEach((btn) => {
      const pressed = btn.dataset.view === mode;
      btn.classList.toggle("tv-view-btn--active", pressed);
      btn.setAttribute("aria-pressed", pressed ? "true" : "false");
    });
    if (mode === "story" && this.trace && this.controller) {
      this.story.renderStep(this.trace, this.controller.currentStep);
    }
    requestAnimationFrame(() => this._renderConnections());
  }

  _renderPlan() {
    const p = this.trace.plan;
    if (p.visualStyle === "diagram-only") {
      this.planView.textContent = "";
      return;
    }
    const lines = [];
    lines.push(`Matmul: ${p.matmulType}`);
    lines.push(`Logical shape: ${p.logicalShape.m}×${p.logicalShape.k}·${p.logicalShape.k}×${p.logicalShape.n}`);
    lines.push(`Padded shape: ${p.paddedShape.m}×${p.paddedShape.k}·${p.paddedShape.k}×${p.paddedShape.n}`);
    lines.push(`Tiles: ${p.tileGrid.m}×${p.tileGrid.n} (tile ${p.tileShape.m}×${p.tileShape.n})`);
    lines.push(`Slots/CT: ${this.trace.parameters.slotsPerCiphertext}`);
    lines.push(`Packing: ${p.packingStrategyRequested} → ${p.packingStrategyResolved}`);
    lines.push(`Planner: ${p.plannerSource}`);
    lines.push(`Reduction: ${p.reductionStrategy}`);
    lines.push(`Display: ${p.displayMode}`);
    if (p.thorRuntime) {
      lines.push("");
      lines.push(`THOR operation: ${p.thorRuntime.operation}`);
      lines.push(`Logical: ${p.thorRuntime.formula}`);
      lines.push(`Executed: ${p.thorRuntime.executedFormula}`);
      lines.push(`BERT-base: ${p.thorRuntime.runtimeShape}`);
      lines.push(`Batch: ${p.thorRuntime.batchSummary}`);
      lines.push(`Code arrays: ${p.thorRuntime.codeFlow}`);
      lines.push(`Source: ${p.thorRuntime.source}`);
    }
    this.planView.textContent = lines.join("\n");
    if (p.warnings && p.warnings.length) {
      const warn = document.createElement("div");
      warn.className = "tv-warning";
      p.warnings.forEach((w) => {
        const li = document.createElement("div");
        li.textContent = `⚠ ${w}`;
        warn.append(li);
      });
      this.planView.append(warn);
    }
  }

  _renderTraceContext(trace) {
    const plan = trace.plan || {};
    const method = plan.method || this.activeMethod;
    const methodName = METHOD_NAMES[method] || method || "Method";
    const operation = trace.graph?.title || trace.title || "Matrix multiplication";
    const left = trace.operands?.left;
    const right = trace.operands?.right;
    const usesMpc = String(plan.matmulType || "").includes("mpc");
    const leftType = usesMpc ? "MPC" : left?.secret ? "CT" : "PT";
    const rightType = usesMpc ? "MPC" : right?.secret ? "CT" : "PT";
    const shape = plan.logicalShape;
    const packing = plan.packingStrategyResolved || plan.reductionStrategy || "Method-defined packing";
    const executedFormula = left?.label && right?.label ? `${left.label} × ${right.label}` : "";
    const venue = METHOD_VENUES[method] || plan.methodLabel || methodName;

    if (this.methodContextTitle) {
      this.methodContextTitle.textContent = `${methodName} · ${operation}`;
      this.methodContextTitle.title = `${methodName} · ${operation}`;
    }
    if (this.methodContextSubtitle) {
      const subtitle = executedFormula ? `${venue} · Executed ${executedFormula}` : venue;
      this.methodContextSubtitle.textContent = subtitle;
      this.methodContextSubtitle.title = subtitle;
    }
    if (this.contextMatmul) this.contextMatmul.textContent = `${leftType} × ${rightType}`;
    if (this.contextPacking) {
      this.contextPacking.textContent = String(packing);
      this.contextPacking.title = String(packing);
    }
    if (this.contextShape) {
      const leftShape = left?.shape?.join("×");
      const rightShape = right?.shape?.join("×");
      this.contextShape.textContent = leftShape && rightShape
        ? `${leftShape} · ${rightShape}`
        : shape
          ? `${shape.m}×${shape.k} · ${shape.k}×${shape.n}`
          : "Shape defined by method";
    }
    if (this.contextSlots) {
      const slots = trace.parameters?.slotsPerCiphertext;
      this.contextSlots.textContent = slots ? `${slots} slots / CT` : "Slot layout by method";
    }
  }

  // ----- Provenance interaction -----

  _onHover(e) {
    if (this.pinned.logicalIds.size || this.pinned.cellKeys.size) return;
    const cell = e.target.closest("[data-tensor-id]");
    const slot = e.target.closest("[data-logical-id][data-slot]");
    if (cell) this._hoverCell(cell.dataset.tensorId, Number(cell.dataset.linear));
    else if (slot) this._hoverSlot(this.index.slotByLogical.get(slot.dataset.logicalId));
  }

  _onClick(e) {
    const cell = e.target.closest("[data-tensor-id]");
    const slot = e.target.closest("[data-logical-id][data-slot]");
    if (cell) this._pinCell(cell.dataset.tensorId, Number(cell.dataset.linear));
    else if (slot) this._pinSlot(this.index.slotByLogical.get(slot.dataset.logicalId));
  }

  _hoverCell(tensorId, linear) {
    const key = `${tensorId}:${linear}`;
    this._applyHighlight(new Set(this.index.sourceToSlot.get(key) || []), new Set([key]));
  }

  _pinCell(tensorId, linear) {
    const key = `${tensorId}:${linear}`;
    const logicalIds = new Set(this.index.sourceToSlot.get(key) || []);
    const cellKeys = new Set([key]);
    this._togglePin(logicalIds, cellKeys);
  }

  _pinSlot(slotState) {
    if (!slotState) return;
    const { logicalIds, cellKeys } = this._traceFromSlot(slotState);
    this._togglePin(logicalIds, cellKeys);
  }

  _hoverSlot(slotState) {
    const { logicalIds, cellKeys } = this._traceFromSlot(slotState);
    this._applyHighlight(logicalIds, cellKeys);
  }

  _traceFromSlot(slotState) {
    const logicalIds = new Set([slotState.logicalId]);
    const cellKeys = new Set();
    const outId = this.trace?.output?.id;
    if (slotState.outputCoords && outId) {
      const [i, j] = slotState.outputCoords;
      const n = this.trace.parameters.matrixShape.n;
      logicalIds.add(`${outId}:${i * n + j}`); // output cell
      const group = this.index.outputToSlot.get(`${outId}:${i}:${j}`) || new Set();
      for (const lid of group) {
        logicalIds.add(lid);
        const s = this.index.slotByLogical.get(lid);
        if (s) {
          for (const ref of s.sourceRefs || []) {
            if (ref.logicalId) logicalIds.add(ref.logicalId);
            if (ref.tensorId != null && ref.linear != null) cellKeys.add(`${ref.tensorId}:${ref.linear}`);
          }
        }
      }
      this.currentFocus = {
        outputCoords: slotState.outputCoords,
        cellRefs: [],
        contractionIndex: slotState.contractionIndex,
        planeIds: [],
      };
    } else {
      const ref = slotState.sourceRefs && slotState.sourceRefs[0];
      if (ref && ref.tensorId != null && ref.linear != null) {
        cellKeys.add(`${ref.tensorId}:${ref.linear}`);
        for (const lid of this.index.sourceToSlot.get(`${ref.tensorId}:${ref.linear}`) || []) {
          logicalIds.add(lid);
        }
      }
    }
    return { logicalIds, cellKeys };
  }

  _togglePin(logicalIds, cellKeys) {
    if (this.pinned.logicalIds.size || this.pinned.cellKeys.size) {
      this._clearPinned();
    } else {
      this.pinned = { logicalIds, cellKeys };
      this._applyHighlight(logicalIds, cellKeys);
    }
  }

  _clearPinned() {
    this.pinned = { logicalIds: new Set(), cellKeys: new Set() };
    this._clearHighlight();
  }

  _applyHighlight(logicalIds, cellKeys) {
    this._clearHighlight();
    for (const id of logicalIds) {
      const el = findLogical(this.gridWrap, id);
      if (el) el.classList.add("tv-hl");
    }
    for (const key of cellKeys) {
      const m = /^([A-Za-z0-9_]+):(\d+)$/.exec(key);
      if (!m) continue;
      const [_, tid, linear] = m;
      this.gridWrap
        .querySelectorAll(`[data-tensor-id="${tid}"][data-linear="${linear}"]`)
        .forEach((el) => el.classList.add("tv-hl"));
    }
    this._renderConnections();
  }

  _clearHighlight() {
    this.gridWrap.querySelectorAll(".tv-hl").forEach((el) => el.classList.remove("tv-hl"));
  }

  _renderConnections() {
    if (!this.svg || !this.currentStep || this.viewMode !== "inspector") return;
    renderConnections(this.svg, {
      step: this.currentStep,
      container: this.gridWrap,
      focus: this.currentFocus || this.currentStep?.focus || null,
    });
  }

  _announce(message) {
    if (this.liveRegion) this.liveRegion.textContent = message;
  }

  // ----- Custom input -----

  _onGenerate(state) {
    const thorOperation = getThorOperation(this.activeAlgorithm);
    const moaiOperation = getMoaiOperation(this.activeAlgorithm);
    const blbOperation = getBlbOperation(this.activeAlgorithm);
    const cachemirOperation = getCachemirOperation(this.activeAlgorithm);
    const eustonOperation = getEustonOperation(this.activeAlgorithm);
    const stipOperation = getStipOperation(this.activeAlgorithm);
    const nexusOperation = getNexusOperation(this.activeAlgorithm);
    const tricycleOperation = getTricycleOperation(this.activeAlgorithm);
    const result = tricycleOperation
      ? parseTricycleOperationInputs(state, tricycleOperation)
      : nexusOperation
      ? parseNexusOperationInputs(state, nexusOperation)
      : stipOperation
      ? parseStipOperationInputs(state, stipOperation)
      : eustonOperation
      ? parseEustonOperationInputs(state, eustonOperation)
      : cachemirOperation
      ? parseCachemirOperationInputs(state, cachemirOperation)
      : blbOperation
      ? parseBlbOperationInputs(state, blbOperation)
      : moaiOperation
        ? parseMoaiOperationInputs(state, moaiOperation)
        : thorOperation
          ? parseThorOperationInputs(state, thorOperation)
          : parseAndValidateInputs(state);
    if (!result.ok) {
      this.inputPanel.setErrors(result.errors);
      return { ok: false, message: "Waiting for valid matrix values" };
    }
    const { inputs } = result;
    this.inputPanel.setErrors({});
    try {
      const trace = buildTraceFromInputs(this.activeAlgorithm, inputs);
      this._setTrace(trace, null);
      this._updateSceneButtons(
        [ALGORITHM_IDS.THOR_PCMM, ALGORITHM_IDS.THOR_CCMM].includes(this.activeAlgorithm) || getThorOperation(this.activeAlgorithm) || getMoaiOperation(this.activeAlgorithm) || getBlbOperation(this.activeAlgorithm) || getCachemirOperation(this.activeAlgorithm) || getEustonOperation(this.activeAlgorithm) || getStipOperation(this.activeAlgorithm) || getNexusOperation(this.activeAlgorithm) || getTricycleOperation(this.activeAlgorithm)
          ? this.activeAlgorithm
          : null
      );
      this._setView("story");
      return {
        ok: true,
        message:
          this.activeAlgorithm === ALGORITHM_IDS.THOR_PCMM
            ? "THOR Figure 3 PC-MM visualization updated"
            : this.activeAlgorithm === ALGORITHM_IDS.THOR_CCMM
              ? "THOR Figure 4 CC-MM visualization updated"
              : getThorOperation(this.activeAlgorithm)
                ? `THOR ${getThorOperation(this.activeAlgorithm).shortLabel} visualization updated`
              : getMoaiOperation(this.activeAlgorithm)
                ? `MOAI ${getMoaiOperation(this.activeAlgorithm).shortLabel} visualization updated`
              : getBlbOperation(this.activeAlgorithm)
                ? `BLB ${getBlbOperation(this.activeAlgorithm).shortLabel} visualization updated`
              : getCachemirOperation(this.activeAlgorithm)
                ? `Cachemir ${getCachemirOperation(this.activeAlgorithm).shortLabel} visualization updated`
              : getEustonOperation(this.activeAlgorithm)
                ? `Euston ${getEustonOperation(this.activeAlgorithm).shortLabel} visualization updated`
              : getStipOperation(this.activeAlgorithm)
                ? `STIP ${getStipOperation(this.activeAlgorithm).shortLabel} visualization updated`
              : getNexusOperation(this.activeAlgorithm)
                ? `NEXUS ${getNexusOperation(this.activeAlgorithm).shortLabel} visualization updated`
              : getTricycleOperation(this.activeAlgorithm)
                ? `Tricycle ${getTricycleOperation(this.activeAlgorithm).shortLabel} visualization updated`
              : "Generic visualization updated",
      };
    } catch (err) {
      this.inputPanel.setErrors({ [err.field || "values"]: err.message });
      console.error(err);
      return { ok: false, message: err.message };
    }
  }
}

new App();
