/**
 * Core type definitions for Visual Model Builder
 * These types are shared across the frontend application.
 */

export type AppLanguage = 'en' | 'zh';

export interface LocalizedText {
  en: string;
  zh: string;
}

// ============================================================
// Parameter Types
// ============================================================

/** Supported parameter types in node configuration */
export type ParamType = 'int' | 'float' | 'bool' | 'select' | 'shape' | 'text' | 'float_list' | 'string_list';

/** Definition of a single configurable parameter */
export interface ParamSpec {
  key: string;
  label: string;
  type: ParamType;
  required: boolean;
  defaultValue: unknown;
  helpText?: string;
  options?: string[];
  placeholder?: string;
  visible?: (params: Record<string, unknown>) => boolean;
  disabled?: (params: Record<string, unknown>) => boolean;
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
  warnings: string[];
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

/** Response from /inspect-dataset */
export interface InspectDatasetResponse {
  success: boolean;
  datasetMode: string;
  resolvedSplitMode: string;
  taskType: string;
  sampleCount: number;
  numClasses: number;
  classNames: string[];
  splits: Record<string, number>;
  inputShape: number[] | null;
  warnings: string[];
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
  datasetMode: string;
  sampleCount: number;
  imageSize: number;
  numClasses: number;
  classNames: string[];
  splits: Record<string, number>;
  inputShape: number[];
  taskType: string;
  trainSplit: boolean;
  batchSize: number;
  shuffle: boolean;
  numWorkers: number;
  dropLast: boolean;
  pinMemory: boolean;
  persistentWorkers: boolean;
  prefetchFactor: number | null;
  collateFnType: string;
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

/** Structural stats returned by training diagnostics */
export interface TrainingGraphStats {
  totalNodes: number;
  totalEdges: number;
  modelNodeCount: number;
  dataNodeCount: number;
  trainingNodeCount: number;
  learnableLayerCount: number;
  modelDepth: number;
  hasMetric: boolean;
  hasTrainer: boolean;
}

/** Model complexity stats returned by training diagnostics */
export interface TrainingModelStats {
  parameterCount: number;
  trainableParameterCount: number;
  learnableLayerCount: number;
  outputClasses: number;
  flattenFeatures: number | null;
  inputShape: number[];
  complexityLabel: string;
}

/** Training config summary returned by diagnostics */
export interface TrainingConfigStats {
  datasetName: string;
  estimatedDatasetSize: number;
  numClasses: number;
  batchSize: number;
  estimatedBatchesPerEpoch: number;
  epochs: number;
  optimizerType: string;
  learningRate: number;
  weightDecay: number;
  momentum: number | null;
  lossType: string;
  metricType: string | null;
  device: string;
}

/** Training diagnostics returned before a run starts */
export interface TrainingDiagnosticsResponse {
  ok: boolean;
  summary: string;
  warnings: string[];
  errors: string[];
  suggestions: string[];
  graphStats: TrainingGraphStats;
  modelStats: TrainingModelStats;
  trainingStats: TrainingConfigStats;
}

/** Post-run training explanation shown in the analysis view */
export interface TrainingInsightsResponse {
  configurationSummary: string;
  modelSummary: string;
  trendSummary: string;
  qualitySummary: string;
  failureExplanation: string | null;
  possibleCauses: string[];
  suggestedFixes: string[];
}

/** Response from /run-training */
export interface RunTrainingResponse {
  ok: boolean;
  status: string;
  logs: TrainingEpochLog[];
  errors: string[];
  diagnostics?: TrainingDiagnosticsResponse | null;
  insights?: TrainingInsightsResponse | null;
  trainingMetadata?: TrainingRunMetadata | null;
}

/** Lifecycle states for asynchronous training jobs */
export type TrainingJobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';

/** Response from /training-jobs endpoints */
export interface TrainingJobResponse {
  jobId: string;
  ok: boolean;
  status: TrainingJobStatus;
  progress: number;
  cancelRequested: boolean;
  logs: TrainingEpochLog[];
  errors: string[];
  diagnostics?: TrainingDiagnosticsResponse | null;
  insights?: TrainingInsightsResponse | null;
  trainingMetadata?: TrainingRunMetadata | null;
  createdAt: string;
  updatedAt: string;
}
