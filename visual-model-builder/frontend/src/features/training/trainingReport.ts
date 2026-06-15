import type {
  RunTrainingResponse,
  TrainingDiagnosticsResponse,
  TrainingEvaluationSummary,
  TrainingInsightsResponse,
  TrainingRunMetadata,
} from '../../types';
import { buildPolyline, formatDuration, formatTimestamp } from './trainingCharts';
import type { ChartPoint } from './trainingTypes';

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
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

function buildListMarkup(title: string, items: string[]): string {
  if (items.length === 0) {
    return '';
  }

  return `
    <section>
      <h2>${escapeHtml(title)}</h2>
      <ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>
    </section>
  `;
}

export function buildTrainingReportHtml(
  projectName: string,
  result: RunTrainingResponse,
  trainingMetadata: TrainingRunMetadata,
  evaluation: TrainingEvaluationSummary | null,
  lossPoints: ChartPoint[],
  accuracyPoints: ChartPoint[],
  diagnostics: TrainingDiagnosticsResponse | null,
  insights: TrainingInsightsResponse | null,
): string {
  const logsRows = result.logs
    .map(
      (log) => `
        <tr>
          <td>${log.epoch}</td>
          <td>${log.loss.toFixed(4)}</td>
          <td>${log.accuracy !== null ? `${(log.accuracy * 100).toFixed(2)}%` : '--'}</td>
          <td>${log.valLoss !== undefined && log.valLoss !== null ? log.valLoss.toFixed(4) : '--'}</td>
          <td>${log.valAccuracy !== undefined && log.valAccuracy !== null ? `${(log.valAccuracy * 100).toFixed(2)}%` : '--'}</td>
        </tr>
      `,
    )
    .join('');
  const confusionRows = evaluation?.confusionMatrix
    .map((row) => `<tr>${row.map((value) => `<td>${value}</td>`).join('')}</tr>`)
    .join('') ?? '';
  const classMetricRows = evaluation?.classMetrics
    .map(
      (metric) => `
        <tr>
          <td>${escapeHtml(metric.className)}</td>
          <td>${metric.support}</td>
          <td>${(metric.precision * 100).toFixed(2)}%</td>
          <td>${(metric.recall * 100).toFixed(2)}%</td>
          <td>${(metric.f1 * 100).toFixed(2)}%</td>
        </tr>
      `,
    )
    .join('') ?? '';

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
    ul { margin: 0; padding-left: 20px; }
    li { margin-bottom: 8px; }
    .meta-grid, .chart-grid { display: grid; gap: 12px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .card, .chart-card { background: #1a1d27; border: 1px solid #363a4e; border-radius: 12px; padding: 16px; }
    .label { color: #9aa0b0; font-size: 12px; margin-bottom: 4px; }
    .value { font-size: 14px; word-break: break-word; line-height: 1.5; }
    .chart { width: 100%; height: 220px; border-radius: 8px; margin-top: 12px; }
    table { width: 100%; border-collapse: collapse; background: #1a1d27; border-radius: 12px; overflow: hidden; }
    th, td { padding: 12px; border-bottom: 1px solid #363a4e; text-align: left; }
    th { color: #9aa0b0; font-size: 12px; text-transform: uppercase; }
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
      <div class="card"><div class="label">Status</div><div class="value">${escapeHtml(result.status)}</div></div>
      <div class="card"><div class="label">Duration</div><div class="value">${escapeHtml(formatDuration(trainingMetadata.durationSeconds))}</div></div>
      <div class="card"><div class="label">Dataset Used</div><div class="value">${escapeHtml(trainingMetadata.datasetUsed)}</div></div>
      <div class="card"><div class="label">Dataset Mode</div><div class="value">${escapeHtml(trainingMetadata.datasetMode)}</div></div>
      <div class="card"><div class="label">Optimizer</div><div class="value">${escapeHtml(trainingMetadata.optimizerType)}</div></div>
      <div class="card"><div class="label">Learning Rate</div><div class="value">${trainingMetadata.learningRate}</div></div>
      <div class="card"><div class="label">Model Weights</div><div class="value"><code>${escapeHtml(trainingMetadata.weightsPath)}</code></div></div>
      <div class="card"><div class="label">Summary JSON</div><div class="value"><code>${escapeHtml(trainingMetadata.summaryPath)}</code></div></div>
      <div class="card"><div class="label">Run ID</div><div class="value"><code>${escapeHtml(trainingMetadata.runId || '')}</code></div></div>
      <div class="card"><div class="label">Seed</div><div class="value">${evaluation?.seed ?? '--'}</div></div>
    </div>
  </section>

  ${evaluation ? `<section><h2>Evaluation</h2><div class="meta-grid">
    <div class="card"><div class="label">Primary Split</div><div class="value">${escapeHtml(evaluation.primarySplit)}</div></div>
    <div class="card"><div class="label">Macro F1</div><div class="value">${evaluation.macroF1 !== null ? evaluation.macroF1.toFixed(4) : '--'}</div></div>
    <div class="card"><div class="label">Weighted F1</div><div class="value">${evaluation.weightedF1 !== null ? evaluation.weightedF1.toFixed(4) : '--'}</div></div>
    <div class="card"><div class="label">Config Hash</div><div class="value"><code>${escapeHtml(evaluation.configHash)}</code></div></div>
  </div></section>` : ''}

  ${evaluation && confusionRows ? `<section><h2>Confusion Matrix</h2><table><tbody>${confusionRows}</tbody></table></section>` : ''}
  ${evaluation && classMetricRows ? `<section><h2>Per-Class Metrics</h2><table>
    <thead><tr><th>Class</th><th>Support</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
    <tbody>${classMetricRows}</tbody>
  </table></section>` : ''}

  ${diagnostics ? `<section><h2>Training Diagnostics</h2><div class="card">${escapeHtml(diagnostics.summary)}</div></section>` : ''}
  ${insights ? `<section><h2>Training Analysis</h2><div class="meta-grid">
    <div class="card"><div class="label">Configuration</div><div class="value">${escapeHtml(insights.configurationSummary)}</div></div>
    <div class="card"><div class="label">Model</div><div class="value">${escapeHtml(insights.modelSummary)}</div></div>
    <div class="card"><div class="label">Trend</div><div class="value">${escapeHtml(insights.trendSummary)}</div></div>
    <div class="card"><div class="label">Quality</div><div class="value">${escapeHtml(insights.qualitySummary)}</div></div>
  </div></section>` : ''}
  ${buildListMarkup('Runtime Messages', result.errors)}

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
      <thead><tr><th>Epoch</th><th>Loss</th><th>Accuracy</th><th>Val Loss</th><th>Val Accuracy</th></tr></thead>
      <tbody>${logsRows}</tbody>
    </table>
  </section>
</body>
</html>
  `.trim();
}
