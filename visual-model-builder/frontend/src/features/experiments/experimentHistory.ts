import type { ExperimentMetricSummary, ExperimentRecord, ProjectGraph, RunTrainingResponse } from '../../types';

export const EXPERIMENT_HISTORY_STORAGE_KEY = 'vmb-experiment-history';
export const EXPERIMENT_HISTORY_EVENT = 'vmb-experiment-history-updated';
const HISTORY_LIMIT = 20;

function safeRead(): ExperimentRecord[] {
  try {
    const raw = window.localStorage.getItem(EXPERIMENT_HISTORY_STORAGE_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((record): record is ExperimentRecord => typeof record?.id === 'string') : [];
  } catch {
    return [];
  }
}

function write(records: ExperimentRecord[]) {
  window.localStorage.setItem(EXPERIMENT_HISTORY_STORAGE_KEY, JSON.stringify(records.slice(0, HISTORY_LIMIT)));
  window.dispatchEvent(new Event(EXPERIMENT_HISTORY_EVENT));
}

function summarizeMetrics(result: RunTrainingResponse): ExperimentMetricSummary {
  const losses = result.logs.map((log) => log.loss).filter(Number.isFinite);
  const valLosses = result.logs
    .map((log) => log.valLoss)
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
  const accuracies = result.logs
    .map((log) => log.accuracy)
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
  const valAccuracies = result.logs
    .map((log) => log.valAccuracy)
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value));

  return {
    firstLoss: losses[0] ?? null,
    finalLoss: losses[losses.length - 1] ?? null,
    bestLoss: losses.length ? Math.min(...losses) : null,
    finalValLoss: valLosses[valLosses.length - 1] ?? null,
    finalAccuracy: accuracies[accuracies.length - 1] ?? null,
    bestAccuracy: accuracies.length ? Math.max(...accuracies) : null,
    finalValAccuracy: valAccuracies[valAccuracies.length - 1] ?? null,
    macroF1: result.evaluation?.macroF1 ?? null,
    weightedF1: result.evaluation?.weightedF1 ?? null,
    epochCount: result.logs.length,
  };
}

function buildGraphSignature(project: ProjectGraph): string {
  return project.nodes
    .map((node) => {
      const params = JSON.stringify(node.data.params ?? {});
      return `${node.id}:${node.type}:${params}`;
    })
    .sort()
    .join('|');
}

function getDatasetName(project: ProjectGraph, result: RunTrainingResponse): string {
  const metadata = result.trainingMetadata;
  if (metadata?.datasetUsed) {
    return metadata.datasetUsed;
  }
  const datasetNode = project.nodes.find((node) => node.type === 'Dataset');
  return String(datasetNode?.data.params.datasetName ?? datasetNode?.data.params.datasetMode ?? 'unknown');
}

function safeNumber(value: unknown): number | null {
  const next = Number(value);
  return Number.isFinite(next) ? next : null;
}

export function loadExperimentHistory(): ExperimentRecord[] {
  return safeRead();
}

export function createExperimentRecord(project: ProjectGraph, result: RunTrainingResponse, jobId: string): ExperimentRecord {
  const metadata = result.trainingMetadata ?? null;
  const diagnostics = result.diagnostics ?? null;
  const trainingStats = diagnostics?.trainingStats ?? null;
  const datasetNode = project.nodes.find((node) => node.type === 'Dataset');
  const optimizerNode = project.nodes.find((node) => node.type === 'Optimizer');
  const dataloaderNode = project.nodes.find((node) => node.type === 'DataLoader');
  const trainerNode = project.nodes.find((node) => node.type === 'Trainer');

  return {
    id: `${jobId}-${Date.now()}`,
    projectName: project.metadata.name || 'Untitled Project',
    createdAt: new Date().toISOString(),
    status: result.status,
    ok: result.ok,
    datasetName: getDatasetName(project, result),
    datasetMode: metadata?.datasetMode ?? String(datasetNode?.data.params.datasetMode ?? 'builtin'),
    optimizerType: metadata?.optimizerType ?? trainingStats?.optimizerType ?? String(optimizerNode?.data.params.optimizerType ?? 'unknown'),
    learningRate: metadata?.learningRate ?? trainingStats?.learningRate ?? safeNumber(optimizerNode?.data.params.lr),
    batchSize: metadata?.batchSize ?? trainingStats?.batchSize ?? safeNumber(dataloaderNode?.data.params.batchSize),
    epochs: metadata?.epochs ?? trainingStats?.epochs ?? safeNumber(trainerNode?.data.params.epochs),
    durationSeconds: metadata?.durationSeconds ?? null,
    graphSignature: buildGraphSignature(project),
    metrics: summarizeMetrics(result),
    logs: result.logs,
    evaluation: result.evaluation ?? null,
    source: 'local',
    diagnostics,
    insights: result.insights ?? null,
    trainingMetadata: metadata,
  };
}

export function saveExperimentRecord(record: ExperimentRecord) {
  const existing = safeRead().filter((item) => item.id !== record.id);
  write([record, ...existing]);
}

export function deleteExperimentRecord(recordId: string) {
  write(safeRead().filter((record) => record.id !== recordId));
}

export function clearExperimentHistory() {
  write([]);
}
