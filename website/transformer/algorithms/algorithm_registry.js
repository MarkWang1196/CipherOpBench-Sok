import { buildMatmulScene } from "../core/scene_builder.js";
import { buildThorFigure3PcmmTrace } from "./thor/figure3_pcmm.js";
import { buildThorFigure4CcmmTrace } from "./thor/figure4_ccmm.js";
import {
  THOR_OPERATION_IDS,
  buildThorOperationTrace,
  listThorOperations,
} from "./thor/bert_operations.js";
import {
  MOAI_OPERATION_IDS,
  buildMoaiOperationTrace,
  listMoaiOperations,
} from "./moai/bert_operations.js";
import {
  BLB_OPERATION_IDS,
  buildBlbOperationTrace,
  listBlbOperations,
} from "./blb/bert_operations.js";
import {
  CACHEMIR_OPERATION_IDS,
  buildCachemirOperationTrace,
  listCachemirOperations,
} from "./cachemir/llama_operations.js";
import {
  EUSTON_OPERATION_IDS,
  buildEustonOperationTrace,
  listEustonOperations,
} from "./euston/transformer_operations.js";
import {
  STIP_OPERATION_IDS,
  buildStipOperationTrace,
  listStipOperations,
} from "./stip/transformer_operations.js";
import {
  NEXUS_OPERATION_IDS,
  buildNexusOperationTrace,
  listNexusOperations,
} from "./nexus/transformer_operations.js";
import {
  TRICYCLE_OPERATION_IDS,
  buildTricycleOperationTrace,
  listTricycleOperations,
} from "./tricycle/transformer_operations.js";

export const ALGORITHM_IDS = Object.freeze({
  ROTOM_OPP: "rotom-opp",
  THOR_PCMM: "thor-pcmm",
  THOR_CCMM: "thor-ccmm",
  ...THOR_OPERATION_IDS,
  MOAI_QUERY: MOAI_OPERATION_IDS.QUERY,
  MOAI_KEY: MOAI_OPERATION_IDS.KEY,
  MOAI_VALUE: MOAI_OPERATION_IDS.VALUE,
  MOAI_SCORE: MOAI_OPERATION_IDS.SCORE,
  MOAI_CONTEXT: MOAI_OPERATION_IDS.CONTEXT,
  MOAI_OUTPUT: MOAI_OPERATION_IDS.OUTPUT,
  MOAI_FC1: MOAI_OPERATION_IDS.FC1,
  MOAI_FC2: MOAI_OPERATION_IDS.FC2,
  MOAI_POOLER: MOAI_OPERATION_IDS.POOLER,
  MOAI_CLASSIFIER: MOAI_OPERATION_IDS.CLASSIFIER,
  BLB_QUERY: BLB_OPERATION_IDS.QUERY,
  BLB_KEY: BLB_OPERATION_IDS.KEY,
  BLB_VALUE: BLB_OPERATION_IDS.VALUE,
  BLB_SCORE: BLB_OPERATION_IDS.SCORE,
  BLB_CONTEXT: BLB_OPERATION_IDS.CONTEXT,
  BLB_OUTPUT: BLB_OPERATION_IDS.OUTPUT,
  BLB_FC1: BLB_OPERATION_IDS.FC1,
  BLB_FC2: BLB_OPERATION_IDS.FC2,
  BLB_POOLER: BLB_OPERATION_IDS.POOLER,
  BLB_CLASSIFIER: BLB_OPERATION_IDS.CLASSIFIER,
  CACHEMIR_QUERY: CACHEMIR_OPERATION_IDS.QUERY,
  CACHEMIR_KEY: CACHEMIR_OPERATION_IDS.KEY,
  CACHEMIR_VALUE: CACHEMIR_OPERATION_IDS.VALUE,
  CACHEMIR_SCORE: CACHEMIR_OPERATION_IDS.SCORE,
  CACHEMIR_CONTEXT: CACHEMIR_OPERATION_IDS.CONTEXT,
  CACHEMIR_OUTPUT: CACHEMIR_OPERATION_IDS.OUTPUT,
  CACHEMIR_UP: CACHEMIR_OPERATION_IDS.UP,
  CACHEMIR_GATE: CACHEMIR_OPERATION_IDS.GATE,
  CACHEMIR_DOWN: CACHEMIR_OPERATION_IDS.DOWN,
  EUSTON_PCCDMM: EUSTON_OPERATION_IDS.PCCDMM,
  EUSTON_CCPUMM: EUSTON_OPERATION_IDS.CCPUMM,
  EUSTON_CCCRMM: EUSTON_OPERATION_IDS.CCCRMM,
  EUSTON_CDCCMM: EUSTON_OPERATION_IDS.CDCCMM,
  EUSTON_QUERY: EUSTON_OPERATION_IDS.QUERY,
  EUSTON_KEY: EUSTON_OPERATION_IDS.KEY,
  EUSTON_VALUE: EUSTON_OPERATION_IDS.VALUE,
  EUSTON_SCORE: EUSTON_OPERATION_IDS.SCORE,
  EUSTON_CONTEXT: EUSTON_OPERATION_IDS.CONTEXT,
  EUSTON_OUTPUT: EUSTON_OPERATION_IDS.OUTPUT,
  EUSTON_FC1: EUSTON_OPERATION_IDS.FC1,
  EUSTON_FC2: EUSTON_OPERATION_IDS.FC2,
  STIP_RIHP: STIP_OPERATION_IDS.RIHP,
  STIP_L2: STIP_OPERATION_IDS.L2,
  STIP_DHP: STIP_OPERATION_IDS.DHP,
  STIP_AMCP: STIP_OPERATION_IDS.AMCP,
  STIP_BSGS: STIP_OPERATION_IDS.BSGS,
  STIP_QUERY: STIP_OPERATION_IDS.QUERY,
  STIP_KEY: STIP_OPERATION_IDS.KEY,
  STIP_VALUE: STIP_OPERATION_IDS.VALUE,
  STIP_SCORE: STIP_OPERATION_IDS.SCORE,
  STIP_DISTANCE: STIP_OPERATION_IDS.DISTANCE,
  STIP_CONTEXT: STIP_OPERATION_IDS.CONTEXT,
  STIP_OUTPUT: STIP_OPERATION_IDS.OUTPUT,
  STIP_FC1: STIP_OPERATION_IDS.FC1,
  STIP_FC2: STIP_OPERATION_IDS.FC2,
  NEXUS_DECOMPRESS: NEXUS_OPERATION_IDS.DECOMPRESS,
  NEXUS_OFFLINE_ONLINE: NEXUS_OPERATION_IDS.OFFLINE_ONLINE,
  NEXUS_COLUMN_CCMM: NEXUS_OPERATION_IDS.COLUMN_CCMM,
  NEXUS_QUERY: NEXUS_OPERATION_IDS.QUERY,
  NEXUS_KEY: NEXUS_OPERATION_IDS.KEY,
  NEXUS_VALUE: NEXUS_OPERATION_IDS.VALUE,
  NEXUS_SCORE: NEXUS_OPERATION_IDS.SCORE,
  NEXUS_CONTEXT: NEXUS_OPERATION_IDS.CONTEXT,
  NEXUS_OUTPUT: NEXUS_OPERATION_IDS.OUTPUT,
  NEXUS_FC1: NEXUS_OPERATION_IDS.FC1,
  NEXUS_FC2: NEXUS_OPERATION_IDS.FC2,
  TRICYCLE_BROADCAST: TRICYCLE_OPERATION_IDS.BROADCAST,
  TRICYCLE_BATCH: TRICYCLE_OPERATION_IDS.BATCH,
  TRICYCLE_CONCAT: TRICYCLE_OPERATION_IDS.CONCAT,
  TRICYCLE_BLOCK_BSGS: TRICYCLE_OPERATION_IDS.BLOCK_BSGS,
  TRICYCLE_QUERY: TRICYCLE_OPERATION_IDS.QUERY,
  TRICYCLE_KEY: TRICYCLE_OPERATION_IDS.KEY,
  TRICYCLE_VALUE: TRICYCLE_OPERATION_IDS.VALUE,
  TRICYCLE_KV_RESHAPE: TRICYCLE_OPERATION_IDS.KV_RESHAPE,
  TRICYCLE_SCORE: TRICYCLE_OPERATION_IDS.SCORE,
  TRICYCLE_CONTEXT: TRICYCLE_OPERATION_IDS.CONTEXT,
  TRICYCLE_OUTPUT: TRICYCLE_OPERATION_IDS.OUTPUT,
  TRICYCLE_FC1: TRICYCLE_OPERATION_IDS.FC1,
  TRICYCLE_FC2: TRICYCLE_OPERATION_IDS.FC2,
  TRICYCLE_POOLER: TRICYCLE_OPERATION_IDS.POOLER,
  TRICYCLE_CLASSIFIER: TRICYCLE_OPERATION_IDS.CLASSIFIER,
});

const registry = new Map([
  [ALGORITHM_IDS.ROTOM_OPP, { id: ALGORITHM_IDS.ROTOM_OPP, label: "Generic OPP", build: buildMatmulScene }],
  [ALGORITHM_IDS.THOR_PCMM, { id: ALGORITHM_IDS.THOR_PCMM, label: "THOR Figure 3 PC-MM", build: buildThorFigure3PcmmTrace }],
  [ALGORITHM_IDS.THOR_CCMM, { id: ALGORITHM_IDS.THOR_CCMM, label: "THOR Figure 4 CC-MM", build: buildThorFigure4CcmmTrace }],
  ...listThorOperations().map((operation) => [
    operation.id,
    {
      id: operation.id,
      label: `THOR ${operation.shortLabel}`,
      build: (config = {}) => buildThorOperationTrace(operation.id, config),
    },
  ]),
  ...listMoaiOperations().map((operation) => [
    operation.id,
    {
      id: operation.id,
      label: `MOAI ${operation.shortLabel}`,
      build: (config = {}) => buildMoaiOperationTrace(operation.id, config),
    },
  ]),
  ...listBlbOperations().map((operation) => [
    operation.id,
    {
      id: operation.id,
      label: `BLB ${operation.shortLabel}`,
      build: (config = {}) => buildBlbOperationTrace(operation.id, config),
    },
  ]),
  ...listCachemirOperations().map((operation) => [
    operation.id,
    {
      id: operation.id,
      label: `Cachemir ${operation.shortLabel}`,
      build: (config = {}) => buildCachemirOperationTrace(operation.id, config),
    },
  ]),
  ...listEustonOperations().map((operation) => [
    operation.id,
    {
      id: operation.id,
      label: `Euston ${operation.shortLabel}`,
      build: (config = {}) => buildEustonOperationTrace(operation.id, config),
    },
  ]),
  ...listStipOperations().map((operation) => [
    operation.id,
    {
      id: operation.id,
      label: `STIP ${operation.shortLabel}`,
      build: (config = {}) => buildStipOperationTrace(operation.id, config),
    },
  ]),
  ...listNexusOperations().map((operation) => [
    operation.id,
    {
      id: operation.id,
      label: `NEXUS ${operation.shortLabel}`,
      build: (config = {}) => buildNexusOperationTrace(operation.id, config),
    },
  ]),
  ...listTricycleOperations().map((operation) => [
    operation.id,
    {
      id: operation.id,
      label: `Tricycle ${operation.shortLabel}`,
      build: (config = {}) => buildTricycleOperationTrace(operation.id, config),
    },
  ]),
]);

export function listAlgorithms() {
  return [...registry.values()].map(({ build, ...meta }) => meta);
}

export function getAlgorithm(id) {
  return registry.get(id) || null;
}

export function buildAlgorithmTrace(id, config = {}) {
  const adapter = getAlgorithm(id);
  if (!adapter) throw new Error(`Unknown visualization algorithm: ${id}`);
  return adapter.build(config);
}
