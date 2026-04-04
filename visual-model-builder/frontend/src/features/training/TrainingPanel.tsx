import React, { useMemo, useState } from 'react';

import { hasTrainingNodes } from '../../graph/graphUtils';
import { runTraining } from '../../services';
import { useAppStore } from '../../store';
import type { GraphEdge, GraphNode, TrainingEpochLog } from '../../types';

function buildPolyline(logs: TrainingEpochLog[], selector: (log: TrainingEpochLog) => number | null): string {
  const values = logs.map(selector).filter((value): value is number => value !== null);
  if (values.length < 2) {
    return '';
  }

  const maxValue = Math.max(...values);
  const minValue = Math.min(...values);
  const range = maxValue - minValue || 1;

  return logs
    .map((log, index) => {
      const value = selector(log);
      if (value === null) {
        return null;
      }

      const x = (index / Math.max(logs.length - 1, 1)) * 100;
      const y = 100 - ((value - minValue) / range) * 100;
      return `${x},${y}`;
    })
    .filter((point): point is string => point !== null)
    .join(' ');
}

type ReadinessItem = {
  label: string;
  ok: boolean;
};

function hasNode(nodes: GraphNode[], type: string): GraphNode | undefined {
  return nodes.find((node) => node.type === type);
}

function hasEdge(edges: GraphEdge[], source: string | undefined, target: string | undefined): boolean {
  if (!source || !target) {
    return false;
  }
  return edges.some((edge) => edge.source === source && edge.target === target);
}

const TrainingPanel: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false);
  const project = useAppStore((state) => state.project);
  const globalErrors = useAppStore((state) => state.globalErrors);
  const trainingResult = useAppStore((state) => state.trainingResult);
  const setTrainingResult = useAppStore((state) => state.setTrainingResult);
  const trainingMode = hasTrainingNodes(project.nodes);

  const datasetNode = hasNode(project.nodes, 'Dataset');
  const dataLoaderNode = hasNode(project.nodes, 'DataLoader');
  const inputNode = hasNode(project.nodes, 'Input');
  const outputNode = hasNode(project.nodes, 'Output');
  const lossNode = hasNode(project.nodes, 'Loss');
  const optimizerNode = hasNode(project.nodes, 'Optimizer');
  const trainerNode = hasNode(project.nodes, 'Trainer');
  const metricNode = hasNode(project.nodes, 'Metric');

  const readinessItems = useMemo<ReadinessItem[]>(
    () => [
      { label: 'Dataset node exists', ok: Boolean(datasetNode) },
      { label: 'Dataset → DataLoader connected', ok: hasEdge(project.edges, datasetNode?.id, dataLoaderNode?.id) },
      { label: 'DataLoader → Input connected', ok: hasEdge(project.edges, dataLoaderNode?.id, inputNode?.id) },
      { label: 'Output → Loss connected', ok: hasEdge(project.edges, outputNode?.id, lossNode?.id) },
      { label: 'Optimizer → Trainer connected', ok: hasEdge(project.edges, optimizerNode?.id, trainerNode?.id) },
      { label: 'Loss → Trainer connected', ok: hasEdge(project.edges, lossNode?.id, trainerNode?.id) },
      { label: 'Metric → Trainer connected or omitted', ok: metricNode ? hasEdge(project.edges, metricNode.id, trainerNode?.id) : true },
    ],
    [project.edges, datasetNode, dataLoaderNode, inputNode, outputNode, lossNode, optimizerNode, trainerNode, metricNode],
  );

  const trainingReady = trainingMode && readinessItems.every((item) => item.ok) && globalErrors.length === 0;
  const firstLog = trainingResult?.logs[0] ?? null;
  const lastLog = trainingResult?.logs[trainingResult.logs.length - 1] ?? null;
  const lossImproved = Boolean(firstLog && lastLog && lastLog.loss < firstLog.loss);
  const trainingEvidence = trainingResult?.ok && lastLog
    ? lossImproved
      ? 'Loss decreased across the run, so the configured loop is updating the model.'
      : 'Training ran, but loss did not decrease. The graph is executable, though the current setup may need tuning.'
    : 'No completed training run yet.';

  const lossPolyline = useMemo(
    () => buildPolyline(trainingResult?.logs ?? [], (log) => log.loss),
    [trainingResult],
  );
  const accuracyPolyline = useMemo(
    () => buildPolyline(trainingResult?.logs ?? [], (log) => log.accuracy),
    [trainingResult],
  );

  const handleRunTraining = async () => {
    try {
      setIsRunning(true);
      const result = await runTraining(project);
      setTrainingResult(result);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      setTrainingResult({
        ok: false,
        status: 'request_failed',
        logs: [],
        errors: [message],
      });
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <section className="training-panel">
      <div className="training-panel-header">
        <div>
          <h3 className="training-panel-title">Training Results</h3>
          <p className="training-panel-subtitle">
            {trainingMode
              ? 'Validate the wiring, run the loop, and inspect whether loss and accuracy move in the expected direction.'
              : 'Add Dataset, DataLoader, Loss, Optimizer, Trainer, and Metric nodes to unlock training.'}
          </p>
        </div>
        <button
          className="training-panel-run-btn"
          onClick={handleRunTraining}
          disabled={!trainingReady || isRunning}
        >
          {isRunning ? 'Running...' : 'Run Training'}
        </button>
      </div>

      <div className="training-panel-status">
        <span>Readiness</span>
        <strong>{trainingReady ? 'ready_to_run' : 'incomplete'}</strong>
      </div>

      <div className="training-panel-checklist">
        {readinessItems.map((item) => (
          <div key={item.label} className={`training-panel-check ${item.ok ? 'ok' : 'pending'}`}>
            <span>{item.ok ? 'OK' : 'TODO'}</span>
            <span>{item.label}</span>
          </div>
        ))}
      </div>

      {trainingResult ? (
        <div className="training-panel-status">
          <span>Last Run</span>
          <strong>{trainingResult.status}</strong>
        </div>
      ) : null}

      {trainingResult?.errors.length ? (
        <div className="training-panel-errors">
          {trainingResult.errors.map((error) => (
            <div key={error} className="training-panel-error">
              {error}
            </div>
          ))}
        </div>
      ) : null}

      <div className="training-panel-evidence">
        <div className="training-panel-evidence-title">Why This Run Is Meaningful</div>
        <div className="training-panel-evidence-text">{trainingEvidence}</div>
        {lastLog ? (
          <div className="training-panel-summary-grid">
            <div className="training-panel-summary-card">
              <span>Final Loss</span>
              <strong>{lastLog.loss.toFixed(4)}</strong>
            </div>
            <div className="training-panel-summary-card">
              <span>Loss Trend</span>
              <strong>{firstLog && lastLog ? `${firstLog.loss.toFixed(4)} → ${lastLog.loss.toFixed(4)}` : '--'}</strong>
            </div>
            <div className="training-panel-summary-card">
              <span>Final Accuracy</span>
              <strong>{lastLog.accuracy !== null ? `${(lastLog.accuracy * 100).toFixed(2)}%` : 'Add Metric node'}</strong>
            </div>
          </div>
        ) : null}
      </div>

      <div className="training-panel-chart-grid">
        <div className="training-chart-card">
          <div className="training-chart-title">Loss</div>
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="training-chart">
            {lossPolyline ? <polyline points={lossPolyline} className="training-chart-line loss" /> : null}
          </svg>
        </div>
        <div className="training-chart-card">
          <div className="training-chart-title">Accuracy</div>
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="training-chart">
            {accuracyPolyline ? <polyline points={accuracyPolyline} className="training-chart-line accuracy" /> : null}
          </svg>
        </div>
      </div>

      <div className="training-panel-log">
        {(trainingResult?.logs ?? []).length > 0 ? (
          trainingResult?.logs.map((log) => (
            <div key={log.epoch} className="training-panel-log-row">
              <span>Epoch {log.epoch}</span>
              <span>loss {log.loss.toFixed(4)}</span>
              <span>{log.accuracy !== null ? `accuracy ${(log.accuracy * 100).toFixed(2)}%` : 'accuracy --'}</span>
            </div>
          ))
        ) : (
          <div className="training-panel-empty">No training run yet.</div>
        )}
      </div>
    </section>
  );
};

export default TrainingPanel;
