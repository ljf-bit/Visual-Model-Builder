import React, { useEffect, useMemo, useState } from 'react';

import { hasTrainingNodes } from '../../graph/graphUtils';
import { runTraining } from '../../services';
import { useAppStore } from '../../store';
import type {
  GraphEdge,
  GraphNode,
  RunTrainingResponse,
  TrainingEpochLog,
  TrainingRunMetadata,
} from '../../types';

type ChartPoint = {
  x: number;
  y: number;
};

type SaveFilePickerOptions = {
  suggestedName?: string;
  types?: Array<{
    description: string;
    accept: Record<string, string[]>;
  }>;
};

type SaveFileHandle = {
  createWritable: () => Promise<{
    write: (content: string) => Promise<void>;
    close: () => Promise<void>;
  }>;
};

type SavePickerWindow = Window & {
  showSaveFilePicker?: (options?: SaveFilePickerOptions) => Promise<SaveFileHandle>;
};

function buildChartPoints(logs: TrainingEpochLog[], selector: (log: TrainingEpochLog) => number | null): ChartPoint[] {
  const values = logs.map(selector).filter((value): value is number => value !== null);
  if (values.length === 0) {
    return [];
  }

  if (values.length === 1) {
    return [{ x: 50, y: 50 }];
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

      return {
        x: (index / Math.max(logs.length - 1, 1)) * 100,
        y: 100 - ((value - minValue) / range) * 100,
      };
    })
    .filter((point): point is ChartPoint => point !== null);
}

function buildPolyline(points: ChartPoint[]): string {
  return points.map((point) => `${point.x},${point.y}`).join(' ');
}

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function formatTimestamp(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString('zh-CN', { hour12: false });
}

function formatDuration(seconds: number): string {
  return `${seconds.toFixed(2)} s`;
}

function buildSvgMarkup(title: string, color: string, points: ChartPoint[]): string {
  const polyline = points.length > 1
    ? `<polyline points="${buildPolyline(points)}" fill="none" stroke="${color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />`
    : '';
  const circles = points
    .map((point) => `<circle cx="${point.x}" cy="${point.y}" r="3.5" fill="${color}" />`)
    .join('');

  return `
    <section class="chart-card">
      <h3>${escapeHtml(title)}</h3>
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" class="chart">
        <rect x="0" y="0" width="100" height="100" fill="#0f1117" />
        <g stroke="rgba(255,255,255,0.08)" stroke-width="0.6">
          <line x1="0" y1="25" x2="100" y2="25" />
          <line x1="0" y1="50" x2="100" y2="50" />
          <line x1="0" y1="75" x2="100" y2="75" />
        </g>
        ${polyline}
        ${circles}
      </svg>
    </section>
  `;
}

function buildTrainingReportHtml(
  projectName: string,
  result: RunTrainingResponse,
  trainingMetadata: TrainingRunMetadata,
  lossPoints: ChartPoint[],
  accuracyPoints: ChartPoint[],
): string {
  const logsRows = result.logs
    .map(
      (log) => `
        <tr>
          <td>${log.epoch}</td>
          <td>${log.loss.toFixed(4)}</td>
          <td>${log.accuracy !== null ? `${(log.accuracy * 100).toFixed(2)}%` : '--'}</td>
        </tr>
      `,
    )
    .join('');

  const warningsMarkup = result.errors.length > 0
    ? `
      <section>
        <h2>Warnings</h2>
        <ul>
          ${result.errors.map((error) => `<li>${escapeHtml(error)}</li>`).join('')}
        </ul>
      </section>
    `
    : '';

  return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>${escapeHtml(projectName)} Training Report</title>
  <style>
    body { font-family: "Segoe UI", Arial, sans-serif; background: #0f1117; color: #e8eaed; margin: 0; padding: 32px; }
    h1, h2, h3 { margin: 0 0 12px; }
    section { margin-bottom: 24px; }
    .meta-grid, .chart-grid { display: grid; gap: 12px; }
    .meta-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .chart-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .card, .chart-card { background: #1a1d27; border: 1px solid #363a4e; border-radius: 12px; padding: 16px; }
    .label { color: #9aa0b0; font-size: 12px; margin-bottom: 4px; }
    .value { font-size: 14px; word-break: break-all; }
    .chart { width: 100%; height: 220px; border-radius: 8px; margin-top: 12px; }
    table { width: 100%; border-collapse: collapse; background: #1a1d27; border-radius: 12px; overflow: hidden; }
    th, td { padding: 12px; border-bottom: 1px solid #363a4e; text-align: left; }
    th { color: #9aa0b0; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
    td { font-size: 14px; }
    code { font-family: "Cascadia Code", "JetBrains Mono", monospace; }
  </style>
</head>
<body>
  <section>
    <h1>${escapeHtml(projectName)} Training Report</h1>
    <div class="label">Saved at ${escapeHtml(formatTimestamp(trainingMetadata.completedAt))}</div>
  </section>

  <section>
    <h2>Run Summary</h2>
    <div class="meta-grid">
      <div class="card"><div class="label">Started</div><div class="value">${escapeHtml(formatTimestamp(trainingMetadata.startedAt))}</div></div>
      <div class="card"><div class="label">Completed</div><div class="value">${escapeHtml(formatTimestamp(trainingMetadata.completedAt))}</div></div>
      <div class="card"><div class="label">Duration</div><div class="value">${escapeHtml(formatDuration(trainingMetadata.durationSeconds))}</div></div>
      <div class="card"><div class="label">Dataset Used</div><div class="value">${escapeHtml(trainingMetadata.datasetUsed)}</div></div>
      <div class="card"><div class="label">Requested Dataset</div><div class="value">${escapeHtml(trainingMetadata.requestedDatasetName)}</div></div>
      <div class="card"><div class="label">Model Weights</div><div class="value"><code>${escapeHtml(trainingMetadata.weightsPath)}</code></div></div>
      <div class="card"><div class="label">Logs JSON</div><div class="value"><code>${escapeHtml(trainingMetadata.logsPath)}</code></div></div>
      <div class="card"><div class="label">Summary JSON</div><div class="value"><code>${escapeHtml(trainingMetadata.summaryPath)}</code></div></div>
    </div>
  </section>

  <section>
    <h2>Hyperparameters</h2>
    <div class="meta-grid">
      <div class="card"><div class="label">Epochs</div><div class="value">${trainingMetadata.epochs}</div></div>
      <div class="card"><div class="label">Batch Size</div><div class="value">${trainingMetadata.batchSize}</div></div>
      <div class="card"><div class="label">Optimizer</div><div class="value">${escapeHtml(trainingMetadata.optimizerType)}</div></div>
      <div class="card"><div class="label">Learning Rate</div><div class="value">${trainingMetadata.learningRate}</div></div>
      <div class="card"><div class="label">Weight Decay</div><div class="value">${trainingMetadata.weightDecay}</div></div>
      <div class="card"><div class="label">Loss</div><div class="value">${escapeHtml(trainingMetadata.lossType)}</div></div>
      <div class="card"><div class="label">Metric</div><div class="value">${escapeHtml(trainingMetadata.metricType ?? 'None')}</div></div>
      <div class="card"><div class="label">Device</div><div class="value">${escapeHtml(trainingMetadata.device)}</div></div>
      <div class="card"><div class="label">Image Size</div><div class="value">${trainingMetadata.imageSize}</div></div>
      <div class="card"><div class="label">Classes</div><div class="value">${trainingMetadata.numClasses}</div></div>
      <div class="card"><div class="label">Dataset Size</div><div class="value">${trainingMetadata.datasetSize}</div></div>
      <div class="card"><div class="label">Run Directory</div><div class="value"><code>${escapeHtml(trainingMetadata.runDirectory)}</code></div></div>
    </div>
  </section>

  <section>
    <h2>Curves</h2>
    <div class="chart-grid">
      ${buildSvgMarkup('Loss Curve', '#38bdf8', lossPoints)}
      ${buildSvgMarkup('Accuracy Curve', '#22c55e', accuracyPoints)}
    </div>
  </section>

  <section>
    <h2>Logs</h2>
    <table>
      <thead>
        <tr>
          <th>Epoch</th>
          <th>Loss</th>
          <th>Accuracy</th>
        </tr>
      </thead>
      <tbody>
        ${logsRows}
      </tbody>
    </table>
  </section>

  ${warningsMarkup}
</body>
</html>
  `.trim();
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

const TrainingPanelV2: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [activeView, setActiveView] = useState<'curves' | 'status' | 'logs'>('status');
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
      { label: 'Dataset -> DataLoader connected', ok: hasEdge(project.edges, datasetNode?.id, dataLoaderNode?.id) },
      { label: 'DataLoader -> Input connected', ok: hasEdge(project.edges, dataLoaderNode?.id, inputNode?.id) },
      { label: 'Output -> Loss connected', ok: hasEdge(project.edges, outputNode?.id, lossNode?.id) },
      { label: 'Optimizer -> Trainer connected', ok: hasEdge(project.edges, optimizerNode?.id, trainerNode?.id) },
      { label: 'Loss -> Trainer connected', ok: hasEdge(project.edges, lossNode?.id, trainerNode?.id) },
      { label: 'Metric -> Trainer connected or omitted', ok: metricNode ? hasEdge(project.edges, metricNode.id, trainerNode?.id) : true },
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

  const trainingMetadata = trainingResult?.trainingMetadata ?? null;
  const lossPoints = useMemo(
    () => buildChartPoints(trainingResult?.logs ?? [], (log) => log.loss),
    [trainingResult],
  );
  const accuracyPoints = useMemo(
    () => buildChartPoints(trainingResult?.logs ?? [], (log) => log.accuracy),
    [trainingResult],
  );
  const lossPolyline = useMemo(() => buildPolyline(lossPoints), [lossPoints]);
  const accuracyPolyline = useMemo(() => buildPolyline(accuracyPoints), [accuracyPoints]);
  const canSaveResults = Boolean(trainingResult?.ok && trainingMetadata && trainingResult.logs.length > 0);

  useEffect(() => {
    if ((trainingResult?.logs.length ?? 0) > 0) {
      setActiveView('curves');
      return;
    }
    if (trainingMode) {
      setActiveView('status');
    }
  }, [trainingMode, trainingResult]);

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

  const handleSaveResults = async () => {
    if (!trainingResult || !trainingMetadata) {
      return;
    }

    const reportHtml = buildTrainingReportHtml(
      trainingMetadata.projectName || project.metadata.name,
      trainingResult,
      trainingMetadata,
      lossPoints,
      accuracyPoints,
    );
    const timestamp = trainingMetadata.completedAt.replaceAll(':', '-').replaceAll('.', '-');
    const safeProjectName = (trainingMetadata.projectName || project.metadata.name || 'training-report')
      .replace(/[<>:"/\\|?*]+/g, '-')
      .trim();
    const suggestedName = `${safeProjectName}-${timestamp}.html`;

    try {
      const savePickerWindow = window as SavePickerWindow;
      if (savePickerWindow.showSaveFilePicker) {
        const handle = await savePickerWindow.showSaveFilePicker({
          suggestedName,
          types: [
            {
              description: 'HTML report',
              accept: { 'text/html': ['.html'] },
            },
          ],
        });
        const writable = await handle.createWritable();
        await writable.write(reportHtml);
        await writable.close();
        return;
      }

      const blob = new Blob([reportHtml], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = suggestedName;
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
      URL.revokeObjectURL(url);
    } catch (error) {
      const saveError = error as Error;
      if (saveError.name !== 'AbortError') {
        alert(`Save results failed: ${saveError.message}`);
      }
    }
  };

  return (
    <section className="training-panel">
      <div className="training-panel-header">
        <div>
          <h3 className="training-panel-title">Training Results</h3>
          <p className="training-panel-subtitle">
            {trainingMode
              ? 'Run the loop, then switch between Curves, Status, and Logs without losing the charts below the fold.'
              : 'Add Dataset, DataLoader, Loss, Optimizer, Trainer, and Metric nodes to unlock training.'}
          </p>
        </div>
        <div className="training-panel-actions">
          <button
            className="training-panel-save-btn"
            onClick={handleSaveResults}
            disabled={!canSaveResults || isRunning}
          >
            Save Results
          </button>
          <button
            className="training-panel-run-btn"
            onClick={handleRunTraining}
            disabled={!trainingReady || isRunning}
          >
            {isRunning ? 'Running...' : 'Run Training'}
          </button>
        </div>
      </div>

      <div className="training-panel-toolbar">
        <div className="training-panel-status">
          <span>Readiness</span>
          <strong>{trainingReady ? 'ready_to_run' : 'incomplete'}</strong>
        </div>
        <div className="training-panel-tabs" role="tablist" aria-label="Training panel views">
          <button
            className={`training-panel-tab ${activeView === 'curves' ? 'active' : ''}`}
            onClick={() => setActiveView('curves')}
          >
            Curves
          </button>
          <button
            className={`training-panel-tab ${activeView === 'status' ? 'active' : ''}`}
            onClick={() => setActiveView('status')}
          >
            Status
          </button>
          <button
            className={`training-panel-tab ${activeView === 'logs' ? 'active' : ''}`}
            onClick={() => setActiveView('logs')}
          >
            Logs
          </button>
        </div>
      </div>

      <div className="training-panel-body">
        {activeView === 'curves' ? (
          <div className="training-panel-curves">
            {lastLog ? (
              <div className="training-panel-summary-grid">
                <div className="training-panel-summary-card">
                  <span>Final Loss</span>
                  <strong>{lastLog.loss.toFixed(4)}</strong>
                </div>
                <div className="training-panel-summary-card">
                  <span>Loss Trend</span>
                  <strong>{firstLog && lastLog ? `${firstLog.loss.toFixed(4)} -> ${lastLog.loss.toFixed(4)}` : '--'}</strong>
                </div>
                <div className="training-panel-summary-card">
                  <span>Final Accuracy</span>
                  <strong>{lastLog.accuracy !== null ? `${(lastLog.accuracy * 100).toFixed(2)}%` : 'Add Metric node'}</strong>
                </div>
              </div>
            ) : null}

            <div className="training-panel-chart-grid training-panel-chart-grid-large">
              <div className="training-chart-card">
                <div className="training-chart-title">Loss Curve</div>
                <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="training-chart training-chart-large">
                  {lossPolyline ? <polyline points={lossPolyline} className="training-chart-line loss" /> : null}
                  {lossPoints.map((point, index) => (
                    <circle
                      key={`loss-${index}`}
                      cx={point.x}
                      cy={point.y}
                      r={2.4}
                      className="training-chart-dot loss"
                    />
                  ))}
                </svg>
              </div>
              <div className="training-chart-card">
                <div className="training-chart-title">Accuracy Curve</div>
                <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="training-chart training-chart-large">
                  {accuracyPolyline ? <polyline points={accuracyPolyline} className="training-chart-line accuracy" /> : null}
                  {accuracyPoints.map((point, index) => (
                    <circle
                      key={`accuracy-${index}`}
                      cx={point.x}
                      cy={point.y}
                      r={2.4}
                      className="training-chart-dot accuracy"
                    />
                  ))}
                </svg>
              </div>
            </div>

            {(trainingResult?.logs ?? []).length === 0 ? (
              <div className="training-panel-empty">
                Run training once, then come back to this Curves view to inspect the charts.
              </div>
            ) : null}
          </div>
        ) : null}

        {activeView === 'status' ? (
          <>
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
            </div>

            {trainingMetadata ? (
              <div className="training-panel-evidence">
                <div className="training-panel-evidence-title">Saved Artifacts</div>
                <div className="training-panel-evidence-text">
                  Duration: {formatDuration(trainingMetadata.durationSeconds)}
                </div>
                <div className="training-panel-evidence-text">
                  Dataset used: {trainingMetadata.datasetUsed}
                </div>
                <div className="training-panel-evidence-text">
                  Weights: {trainingMetadata.weightsPath || 'Unavailable'}
                </div>
                <div className="training-panel-evidence-text">
                  Summary: {trainingMetadata.summaryPath || 'Unavailable'}
                </div>
              </div>
            ) : null}
          </>
        ) : null}

        {activeView === 'logs' ? (
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
        ) : null}
      </div>
    </section>
  );
};

export default TrainingPanelV2;
