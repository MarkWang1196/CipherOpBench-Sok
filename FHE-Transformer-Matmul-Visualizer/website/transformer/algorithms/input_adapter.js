// Pure routing layer between the shared input form and paper-specific builders.
// Keeping this outside app.js makes algorithm selection explicit and testable.

import { buildMatmulScene } from "../core/scene_builder.js";
import { ALGORITHM_IDS } from "./algorithm_registry.js?v=20261118-3";
import { buildThorFigure3PcmmTrace } from "./thor/figure3_pcmm.js";
import { buildThorFigure4CcmmTrace } from "./thor/figure4_ccmm.js";
import { buildThorOperationTrace, getThorOperation } from "./thor/bert_operations.js";
import { buildMoaiOperationTrace, getMoaiOperation } from "./moai/bert_operations.js";
import { buildBlbOperationTrace, getBlbOperation } from "./blb/bert_operations.js";
import { buildCachemirOperationTrace, getCachemirOperation } from "./cachemir/llama_operations.js";
import { buildEustonOperationTrace, getEustonOperation } from "./euston/transformer_operations.js?v=20261118-3";
import { buildStipOperationTrace, getStipOperation } from "./stip/transformer_operations.js?v=20261118-3";
import { buildNexusOperationTrace, getNexusOperation } from "./nexus/transformer_operations.js?v=20261118-3";

export class AlgorithmInputError extends Error {
  constructor(message, field = "values") {
    super(message);
    this.name = "AlgorithmInputError";
    this.field = field;
  }
}

export function buildTraceFromInputs(algorithmId, inputs) {
  const nexusOperation = getNexusOperation(algorithmId);
  if (nexusOperation) {
    if (inputs.leftSecret !== nexusOperation.leftSecret || inputs.rightSecret !== nexusOperation.rightSecret) {
      throw new AlgorithmInputError(
        `${nexusOperation.shortLabel} requires ${nexusOperation.leftSecret ? "CT" : "PT"} × ${nexusOperation.rightSecret ? "CT" : "PT"}`,
        "leftType"
      );
    }
    if (inputs.strategy !== "column") {
      throw new AlgorithmInputError("NEXUS uses batched column/diagonal packing", "strategy");
    }
    return buildNexusOperationTrace(algorithmId, {
      A: inputs.A, B: inputs.B, leftSecret: inputs.leftSecret, rightSecret: inputs.rightSecret,
      slotsPerCiphertext: inputs.slotsPerCiphertext,
    });
  }

  const stipOperation = getStipOperation(algorithmId);
  if (stipOperation) {
    if (inputs.leftSecret !== stipOperation.leftSecret || inputs.rightSecret !== stipOperation.rightSecret) {
      throw new AlgorithmInputError(
        `${stipOperation.shortLabel} requires ${stipOperation.leftSecret ? "CT" : "PT"} × ${stipOperation.rightSecret ? "CT" : "PT"}`,
        "leftType"
      );
    }
    if (inputs.strategy !== "column") {
      throw new AlgorithmInputError("STIP uses compact column/diagonal packing", "strategy");
    }
    return buildStipOperationTrace(algorithmId, {
      A: inputs.A, B: inputs.B, leftSecret: inputs.leftSecret, rightSecret: inputs.rightSecret,
      slotsPerCiphertext: inputs.slotsPerCiphertext,
    });
  }

  const eustonOperation = getEustonOperation(algorithmId);
  if (eustonOperation) {
    if (inputs.leftSecret !== eustonOperation.leftSecret || inputs.rightSecret !== eustonOperation.rightSecret) {
      throw new AlgorithmInputError(
        `${eustonOperation.shortLabel} requires ${eustonOperation.leftSecret ? "CT" : "PT"} × ${eustonOperation.rightSecret ? "CT" : "PT"}`,
        "leftType"
      );
    }
    if (inputs.strategy !== "column") {
      throw new AlgorithmInputError("Euston uses batched column/diagonal packing", "strategy");
    }
    return buildEustonOperationTrace(algorithmId, {
      A: inputs.A, B: inputs.B, leftSecret: inputs.leftSecret, rightSecret: inputs.rightSecret,
      slotsPerCiphertext: inputs.slotsPerCiphertext,
    });
  }

  const cachemirOperation = getCachemirOperation(algorithmId);
  if (cachemirOperation) {
    if (inputs.leftSecret !== cachemirOperation.leftSecret || inputs.rightSecret !== cachemirOperation.rightSecret) {
      throw new AlgorithmInputError(
        `${cachemirOperation.shortLabel} requires ${cachemirOperation.leftSecret ? "CT" : "PT"} × ${cachemirOperation.rightSecret ? "CT" : "PT"}`,
        "leftType"
      );
    }
    if (inputs.strategy !== "column") {
      throw new AlgorithmInputError("Cachemir uses interleaved replicated packing", "strategy");
    }
    return buildCachemirOperationTrace(algorithmId, {
      A: inputs.A, B: inputs.B, leftSecret: inputs.leftSecret, rightSecret: inputs.rightSecret,
      slotsPerCiphertext: inputs.slotsPerCiphertext,
    });
  }

  const blbOperation = getBlbOperation(algorithmId);
  if (blbOperation) {
    if (inputs.leftSecret !== blbOperation.leftSecret || inputs.rightSecret !== blbOperation.rightSecret) {
      throw new AlgorithmInputError(
        `${blbOperation.shortLabel} requires ${blbOperation.leftSecret ? "encrypted/shared" : "plaintext"} × ${blbOperation.rightSecret ? "encrypted/shared" : "plaintext"}`,
        "leftType"
      );
    }
    if (inputs.strategy !== "column") {
      throw new AlgorithmInputError("BLB uses spatial-first packing for this toy flow", "strategy");
    }
    return buildBlbOperationTrace(algorithmId, {
      A: inputs.A,
      B: inputs.B,
      leftSecret: inputs.leftSecret,
      rightSecret: inputs.rightSecret,
      slotsPerCiphertext: inputs.slotsPerCiphertext,
    });
  }

  const moaiOperation = getMoaiOperation(algorithmId);
  if (moaiOperation) {
    if (inputs.leftSecret !== moaiOperation.leftSecret || inputs.rightSecret !== moaiOperation.rightSecret) {
      throw new AlgorithmInputError(
        `${moaiOperation.shortLabel} requires ${moaiOperation.leftSecret ? "CT" : "PT"} × ${moaiOperation.rightSecret ? "CT" : "PT"}`,
        "leftType"
      );
    }
    if (inputs.strategy !== "column") {
      throw new AlgorithmInputError("MOAI uses column packing for this toy flow", "strategy");
    }
    return buildMoaiOperationTrace(algorithmId, {
      A: inputs.A,
      B: inputs.B,
      leftSecret: inputs.leftSecret,
      rightSecret: inputs.rightSecret,
      slotsPerCiphertext: inputs.slotsPerCiphertext,
    });
  }

  const thorOperation = getThorOperation(algorithmId);
  if (thorOperation) {
    if (inputs.leftSecret !== thorOperation.leftSecret || inputs.rightSecret !== thorOperation.rightSecret) {
      throw new AlgorithmInputError(
        `${thorOperation.shortLabel} requires ${thorOperation.leftSecret ? "CT" : "PT"} × ${thorOperation.rightSecret ? "CT" : "PT"} teaching operands`,
        "leftType"
      );
    }
    if (inputs.strategy !== "auto" && inputs.strategy !== "diagonal") {
      throw new AlgorithmInputError("THOR toy operations support Auto or Diagonal packing", "strategy");
    }
    return buildThorOperationTrace(algorithmId, {
      A: inputs.A,
      B: inputs.B,
      leftSecret: inputs.leftSecret,
      rightSecret: inputs.rightSecret,
      slotsPerCiphertext: inputs.slotsPerCiphertext,
    });
  }

  if (algorithmId === ALGORITHM_IDS.THOR_PCMM) {
    if (inputs.leftSecret || !inputs.rightSecret) {
      throw new AlgorithmInputError("THOR PC-MM requires plaintext A and ciphertext B", "leftType");
    }
    if (inputs.strategy !== "auto" && inputs.strategy !== "diagonal") {
      throw new AlgorithmInputError("THOR PC-MM supports Auto or Diagonal packing", "strategy");
    }
    try {
      return buildThorFigure3PcmmTrace({
        A: inputs.A,
        B: inputs.B,
        slotsPerCiphertext: inputs.slotsPerCiphertext,
        packingStrategy: inputs.strategy,
      });
    } catch (error) {
      if (/slots/i.test(error.message)) throw new AlgorithmInputError(error.message, "slots");
      throw error;
    }
  }

  if (algorithmId === ALGORITHM_IDS.THOR_CCMM) {
    if (!inputs.leftSecret || !inputs.rightSecret) {
      throw new AlgorithmInputError("THOR CC-MM requires ciphertext A and ciphertext B", "leftType");
    }
    if (inputs.strategy !== "auto" && inputs.strategy !== "diagonal") {
      throw new AlgorithmInputError("THOR CC-MM supports Auto or Diagonal packing", "strategy");
    }
    try {
      return buildThorFigure4CcmmTrace({
        A: inputs.A,
        B: inputs.B,
        slotsPerCiphertext: inputs.slotsPerCiphertext,
        method: inputs.thorCcmmMethod,
      });
    } catch (error) {
      if (/slots/i.test(error.message)) throw new AlgorithmInputError(error.message, "slots");
      throw error;
    }
  }

  return buildMatmulScene({
    left: { id: "A", label: "Operand A", values: inputs.A, secret: inputs.leftSecret },
    right: { id: "B", label: "Operand B", values: inputs.B, secret: inputs.rightSecret },
    slotsPerCiphertext: inputs.slotsPerCiphertext,
    packingStrategy: inputs.strategy,
    customLayouts: inputs.customLayouts,
    options: {
      transposeRight: false,
      outputId: "C",
      outputLabel: "Output C",
    },
  });
}
