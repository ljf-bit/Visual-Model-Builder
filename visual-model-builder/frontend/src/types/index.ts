/**
 * Core type definitions for Visual Model Builder
 * These types are shared across the frontend application.
 */

// ============================================================
// Parameter Types
// ============================================================

/** Supported parameter types in node configuration */
export type ParamType = 'int' | 'float' | 'bool' | 'select' | 'shape';

/** Definition of a single configurable parameter */
export interface ParamSpec {
  key: string;
  label: string;
  type: ParamType;
  required: boolean;
  defaultValue: unknown;
  helpText?: string;
  options?: string[];
}

// ============================================================
// Node Template (Registry Definition)
// ============================================================

/** Category for grouping nodes in the palette */
export type NodeCategory = 'io' | 'layer' | 'activation' | 'train';

/** Static definition of a node type, used by registry */
export interface NodeTemplate {
  type: string;
  displayName: string;
  category: NodeCategory;
  description?: string;
  inputPorts: number;
  outputPorts: number;
  params: ParamSpec[];
}

/** Behavior configuration for a registered node */
export interface NodeBehavior {
  template: NodeTemplate;
  defaultData: () => Record<string, unknown>;
}

// ============================================================
// Graph Data (Runtime Instances)
// ============================================================

/** Data payload attached to each node instance */
export interface GraphNodeData {
  [key: string]: unknown;
  label: string;
  params: Record<string, unknown>;
  inferredInputShape?: number[] | null;
  inferredOutputShape?: number[] | null;
  errors?: string[];
}

/** A node instance on the canvas */
export interface GraphNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: GraphNodeData;
}

/** An edge connecting two nodes */
export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
}

// ============================================================
// Project Persistence
// ============================================================

/** Project metadata */
export interface ProjectMetadata {
  name: string;
  createdAt: string;
  updatedAt: string;
}

/** Full project graph (serializable) */
export interface ProjectGraph {
  version: string;
  metadata: ProjectMetadata;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

// ============================================================
// API Response Types
// ============================================================

/** Shape inference result for a single node */
export interface NodeShapeResult {
  inputShape: number[] | null;
  outputShape: number[] | null;
  errors: string[];
}

/** Response from /validate-graph */
export interface ValidateGraphResponse {
  ok: boolean;
  globalErrors: string[];
  nodeErrors: Record<string, string[]>;
}

/** Response from /infer-shapes */
export interface InferShapesResponse {
  ok: boolean;
  nodes: Record<string, NodeShapeResult>;
}

/** Response from /generate-code */
export interface GenerateCodeResponse {
  ok: boolean;
  code: string;
  errors: string[];
}

/** Code generation modes shown in the bottom panel */
export type GeneratedCodeMode = 'model' | 'training';

/** Metrics captured for a single training epoch */
export interface TrainingEpochLog {
  epoch: number;
  loss: number;
  accuracy: number | null;
}

/** Saved metadata and artifact locations for one training run */
export interface TrainingRunMetadata {
  projectName: string;
  requestedDatasetName: string;
  datasetUsed: string;
  datasetSize: number;
  imageSize: number;
  numClasses: number;
  trainSplit: boolean;
  batchSize: number;
  shuffle: boolean;
  numWorkers: number;
  epochs: number;
  device: string;
  lossType: string;
  optimizerType: string;
  learningRate: number;
  weightDecay: number;
  momentum: number | null;
  metricType: string | null;
  startedAt: string;
  completedAt: string;
  durationSeconds: number;
  runDirectory: string;
  weightsPath: string;
  logsPath: string;
  summaryPath: string;
}

/** Response from /run-training */
export interface RunTrainingResponse {
  ok: boolean;
  status: string;
  logs: TrainingEpochLog[];
  errors: string[];
  trainingMetadata?: TrainingRunMetadata | null;
}
