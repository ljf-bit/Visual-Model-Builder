import type { RunTrainingResponse, TrainingDiagnosticsResponse, TrainingEvaluationSummary, TrainingRunMetadata } from '../../types';
import type { AutoFixAction, AutoFixSuggestion } from '../../types';
import { AutoFixPanel } from '../autofix';
import { useLanguage } from '../../hooks/useLanguage';
import { translateKnownMessage } from '../../i18n';
import { buildChartPoints, buildPolyline, formatDuration } from './trainingCharts';
import type { ChartPoint, ReadinessItem } from './trainingTypes';

type StatusViewProps = {
  readinessItems: ReadinessItem[];
  currentDiagnostics: TrainingDiagnosticsResponse | null;
  trainingResult: RunTrainingResponse | null;
  trainingMetadata: TrainingRunMetadata | null;
  trainingEvidence: string;
  autoFixSuggestions: AutoFixSuggestion[];
  onApplyAutoFix: (actions: AutoFixAction[]) => void;
};

type CurvesViewProps = {
  trainingResult: RunTrainingResponse | null;
  lossPoints: ChartPoint[];
  accuracyPoints: ChartPoint[];
};

type AnalysisViewProps = {
  analysis: {
    configurationSummary: string;
    modelSummary: string;
    trendSummary: string;
    qualitySummary: string;
    failureExplanation: string | null;
    possibleCauses: string[];
    suggestedFixes: string[];
  };
};

export function TrainingCurvesView({ trainingResult, lossPoints, accuracyPoints }: CurvesViewProps) {
  const { t } = useLanguage();
  const firstLog = trainingResult?.logs[0] ?? null;
  const lastLog = trainingResult?.logs[trainingResult.logs.length - 1] ?? null;
  const evaluation = trainingResult?.evaluation ?? null;
  const valLossPoints = buildChartPoints(trainingResult?.logs ?? [], (log) => log.valLoss ?? null);
  const valAccuracyPoints = buildChartPoints(trainingResult?.logs ?? [], (log) => log.valAccuracy ?? null);
  const lossPolyline = buildPolyline(lossPoints);
  const accuracyPolyline = buildPolyline(accuracyPoints);
  const valLossPolyline = buildPolyline(valLossPoints);
  const valAccuracyPolyline = buildPolyline(valAccuracyPoints);

  return (
    <div className="training-panel-curves">
      {lastLog ? (
        <div className="training-panel-summary-grid">
          <div className="training-panel-summary-card">
            <span>{t('training.finalLoss')}</span>
            <strong>{lastLog.loss.toFixed(4)}</strong>
          </div>
          <div className="training-panel-summary-card">
            <span>{t('training.lossTrend')}</span>
            <strong>{firstLog && lastLog ? `${firstLog.loss.toFixed(4)} -> ${lastLog.loss.toFixed(4)}` : '--'}</strong>
          </div>
          <div className="training-panel-summary-card">
            <span>{t('training.finalAccuracy')}</span>
            <strong>{lastLog.accuracy !== null ? `${(lastLog.accuracy * 100).toFixed(2)}%` : t('training.addMetricNode')}</strong>
          </div>
          <div className="training-panel-summary-card">
            <span>{t('training.validationAccuracy')}</span>
            <strong>{lastLog.valAccuracy !== undefined && lastLog.valAccuracy !== null ? `${(lastLog.valAccuracy * 100).toFixed(2)}%` : '--'}</strong>
          </div>
          <div className="training-panel-summary-card">
            <span>{t('training.macroF1')}</span>
            <strong>{formatScore(evaluation?.macroF1)}</strong>
          </div>
          <div className="training-panel-summary-card">
            <span>{t('training.bestEpoch')}</span>
            <strong>{evaluation?.bestEpoch ?? '--'}</strong>
          </div>
        </div>
      ) : null}

      <div className="training-panel-chart-grid training-panel-chart-grid-large">
        <div className="training-chart-card">
          <div className="training-chart-title">{t('training.lossCurve')}</div>
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="training-chart training-chart-large">
            {lossPolyline ? <polyline points={lossPolyline} className="training-chart-line loss" /> : null}
            {valLossPolyline ? <polyline points={valLossPolyline} className="training-chart-line validation" /> : null}
            {lossPoints.map((point, index) => (
              <circle key={`loss-${index}`} cx={point.x} cy={point.y} r={2.4} className="training-chart-dot loss" />
            ))}
            {valLossPoints.map((point, index) => (
              <circle key={`val-loss-${index}`} cx={point.x} cy={point.y} r={2.4} className="training-chart-dot validation" />
            ))}
          </svg>
          <div className="training-chart-legend"><span className="loss" /> train <span className="validation" /> validation</div>
        </div>
        <div className="training-chart-card">
          <div className="training-chart-title">{t('training.accuracyCurve')}</div>
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="training-chart training-chart-large">
            {accuracyPolyline ? <polyline points={accuracyPolyline} className="training-chart-line accuracy" /> : null}
            {valAccuracyPolyline ? <polyline points={valAccuracyPolyline} className="training-chart-line validation" /> : null}
            {accuracyPoints.map((point, index) => (
              <circle key={`accuracy-${index}`} cx={point.x} cy={point.y} r={2.4} className="training-chart-dot accuracy" />
            ))}
            {valAccuracyPoints.map((point, index) => (
              <circle key={`val-accuracy-${index}`} cx={point.x} cy={point.y} r={2.4} className="training-chart-dot validation" />
            ))}
          </svg>
          <div className="training-chart-legend"><span className="accuracy" /> train <span className="validation" /> validation</div>
        </div>
      </div>

      {evaluation ? <TrainingEvaluationBlock evaluation={evaluation} /> : null}

      {(trainingResult?.logs ?? []).length === 0 ? (
        <div className="training-panel-empty">{t('training.curvesEmpty')}</div>
      ) : null}
    </div>
  );
}

function formatScore(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : '--';
}

function formatPercent(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : '--';
}

function TrainingEvaluationBlock({ evaluation }: { evaluation: TrainingEvaluationSummary }) {
  const { t } = useLanguage();
  const matrixMax = Math.max(1, ...evaluation.confusionMatrix.flat());

  return (
    <div className="training-evaluation">
      <div className="training-panel-analysis-grid">
        <div className="training-panel-analysis-card">
          <div className="training-panel-evidence-title">{t('training.evaluationSplit')}</div>
          <div className="training-panel-evidence-text">{evaluation.primarySplit}</div>
        </div>
        <div className="training-panel-analysis-card">
          <div className="training-panel-evidence-title">{t('training.weightedF1')}</div>
          <div className="training-panel-evidence-text">{formatScore(evaluation.weightedF1)}</div>
        </div>
        <div className="training-panel-analysis-card">
          <div className="training-panel-evidence-title">{t('training.seed')}</div>
          <div className="training-panel-evidence-text">{evaluation.seed}</div>
        </div>
        <div className="training-panel-analysis-card">
          <div className="training-panel-evidence-title">{t('training.configHash')}</div>
          <div className="training-panel-evidence-text">{evaluation.configHash}</div>
        </div>
      </div>

      {evaluation.confusionMatrix.length > 0 ? (
        <div className="training-panel-analysis-card">
          <div className="training-panel-evidence-title">{t('training.confusionMatrix')}</div>
          <div
            className="training-confusion-matrix"
            style={{ gridTemplateColumns: `repeat(${evaluation.confusionMatrix.length}, minmax(28px, 1fr))` }}
          >
            {evaluation.confusionMatrix.flatMap((row, rowIndex) =>
              row.map((value, colIndex) => (
                <div
                  key={`${rowIndex}-${colIndex}`}
                  className="training-confusion-cell"
                  style={{ opacity: 0.25 + (value / matrixMax) * 0.75 }}
                  title={`${evaluation.classNames[rowIndex] ?? rowIndex} -> ${evaluation.classNames[colIndex] ?? colIndex}: ${value}`}
                >
                  {value}
                </div>
              )),
            )}
          </div>
        </div>
      ) : null}

      {evaluation.classMetrics.length > 0 ? (
        <div className="experiment-table-wrap">
          <table className="experiment-table training-class-metrics-table">
            <thead>
              <tr>
                <th>{t('training.class')}</th>
                <th>{t('training.support')}</th>
                <th>{t('training.precision')}</th>
                <th>{t('training.recall')}</th>
                <th>{t('training.f1')}</th>
              </tr>
            </thead>
            <tbody>
              {evaluation.classMetrics.map((metric) => (
                <tr key={metric.classIndex}>
                  <td>{metric.className}</td>
                  <td>{metric.support}</td>
                  <td>{formatPercent(metric.precision)}</td>
                  <td>{formatPercent(metric.recall)}</td>
                  <td>{formatPercent(metric.f1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}

export function TrainingStatusView({
  readinessItems,
  currentDiagnostics,
  trainingResult,
  trainingMetadata,
  trainingEvidence,
  autoFixSuggestions,
  onApplyAutoFix,
}: StatusViewProps) {
  const { language, t } = useLanguage();
  const diagnosticsErrors = currentDiagnostics?.errors ?? [];
  const diagnosticsWarnings = currentDiagnostics?.warnings ?? [];
  const runtimeMessages = trainingResult?.errors ?? [];

  return (
    <>
      <div className="training-panel-checklist">
        {readinessItems.map((item) => (
          <div key={item.label} className={`training-panel-check ${item.ok ? 'ok' : 'pending'}`}>
            <span>{item.ok ? 'OK' : 'TODO'}</span>
            <span>{item.label}</span>
          </div>
        ))}
      </div>

      <AutoFixPanel suggestions={autoFixSuggestions} onApply={onApplyAutoFix} />

      {currentDiagnostics ? (
        <div className="training-panel-summary-grid">
          <div className="training-panel-summary-card">
            <span>{t('training.diagnosticsCard')}</span>
            <strong>{currentDiagnostics.ok ? 'ready' : 'blocked'}</strong>
          </div>
          <div className="training-panel-summary-card">
            <span>{t('training.warningsCard')}</span>
            <strong>{diagnosticsWarnings.length}</strong>
          </div>
          <div className="training-panel-summary-card">
            <span>{t('training.parametersCard')}</span>
            <strong>{currentDiagnostics.modelStats.parameterCount.toLocaleString()}</strong>
          </div>
        </div>
      ) : null}

      {trainingResult || currentDiagnostics ? (
        <div className="training-panel-status">
          <span>{t('training.lastRun')}</span>
          <strong>{trainingResult?.status ?? (currentDiagnostics?.ok ? 'diagnosed' : 'not_started')}</strong>
        </div>
      ) : null}

      {[...diagnosticsErrors, ...diagnosticsWarnings, ...runtimeMessages].length ? (
        <div className="training-panel-errors">
          {diagnosticsErrors.map((error) => <div key={error} className="training-panel-error">{translateKnownMessage(error, language)}</div>)}
          {diagnosticsWarnings.map((warning) => <div key={warning} className="training-panel-warning">{translateKnownMessage(warning, language)}</div>)}
          {runtimeMessages.map((message) => <div key={message} className="training-panel-warning runtime">{translateKnownMessage(message, language)}</div>)}
        </div>
      ) : null}

      <div className="training-panel-evidence">
        <div className="training-panel-evidence-title">{t('training.whyMeaningful')}</div>
        <div className="training-panel-evidence-text">{trainingEvidence}</div>
      </div>

      {trainingMetadata ? (
        <div className="training-panel-evidence">
          <div className="training-panel-evidence-title">{t('training.savedArtifacts')}</div>
          <div className="training-panel-evidence-text">{t('training.duration', { value: formatDuration(trainingMetadata.durationSeconds) })}</div>
          <div className="training-panel-evidence-text">{t('training.datasetUsed', { value: trainingMetadata.datasetUsed })}</div>
          <div className="training-panel-evidence-text">{t('training.datasetMode', { value: trainingMetadata.datasetMode || 'builtin' })}</div>
          <div className="training-panel-evidence-text">{t('training.totalSamples', { value: trainingMetadata.sampleCount || trainingMetadata.datasetSize })}</div>
          <div className="training-panel-evidence-text">
            {t('training.splits', {
              train: trainingMetadata.splits?.train ?? trainingMetadata.datasetSize,
              val: trainingMetadata.splits?.val ?? 0,
              test: trainingMetadata.splits?.test ?? 0,
            })}
          </div>
          <div className="training-panel-evidence-text">{t('training.weights', { value: trainingMetadata.weightsPath || t('training.unavailable') })}</div>
          <div className="training-panel-evidence-text">{t('training.summary', { value: trainingMetadata.summaryPath || t('training.unavailable') })}</div>
          <div className="training-panel-evidence-text">{t('training.runId', { value: trainingMetadata.runId || t('training.unavailable') })}</div>
        </div>
      ) : null}
    </>
  );
}

export function TrainingLogsView({ trainingResult }: { trainingResult: RunTrainingResponse | null }) {
  const { t } = useLanguage();
  return (
    <div className="training-panel-log">
      {(trainingResult?.logs ?? []).length > 0 ? (
        trainingResult?.logs.map((log) => (
          <div key={log.epoch} className="training-panel-log-row">
            <span>{t('training.epoch', { epoch: log.epoch })}</span>
            <span>{t('training.loss', { value: log.loss.toFixed(4) })}</span>
            <span>{t('training.accuracy', { value: log.accuracy !== null ? `${(log.accuracy * 100).toFixed(2)}%` : '--' })}</span>
            <span>{t('training.valLoss', { value: log.valLoss !== undefined && log.valLoss !== null ? log.valLoss.toFixed(4) : '--' })}</span>
            <span>{t('training.valAccuracy', { value: log.valAccuracy !== undefined && log.valAccuracy !== null ? `${(log.valAccuracy * 100).toFixed(2)}%` : '--' })}</span>
          </div>
        ))
      ) : (
        <div className="training-panel-empty">{t('training.noRun')}</div>
      )}
    </div>
  );
}

export function TrainingAnalysisView({ analysis }: AnalysisViewProps) {
  const { t } = useLanguage();
  return (
    <div className="training-panel-curves">
      <div className="training-panel-analysis-grid">
        <div className="training-panel-analysis-card">
          <div className="training-panel-evidence-title">{t('training.configurationSummary')}</div>
          <div className="training-panel-evidence-text">{analysis.configurationSummary}</div>
        </div>
        <div className="training-panel-analysis-card">
          <div className="training-panel-evidence-title">{t('training.modelSummary')}</div>
          <div className="training-panel-evidence-text">{analysis.modelSummary}</div>
        </div>
        <div className="training-panel-analysis-card">
          <div className="training-panel-evidence-title">{t('training.trendSummary')}</div>
          <div className="training-panel-evidence-text">{analysis.trendSummary}</div>
        </div>
        <div className="training-panel-analysis-card">
          <div className="training-panel-evidence-title">{t('training.qualitySummary')}</div>
          <div className="training-panel-evidence-text">{analysis.qualitySummary}</div>
        </div>
      </div>

      {analysis.failureExplanation ? (
        <div className="training-panel-evidence">
          <div className="training-panel-evidence-title">{t('training.failureExplanation')}</div>
          <div className="training-panel-evidence-text">{analysis.failureExplanation}</div>
        </div>
      ) : null}

      {analysis.possibleCauses.length ? (
        <div className="training-panel-analysis-card">
          <div className="training-panel-evidence-title">{t('training.possibleCauses')}</div>
          <div className="training-panel-bullet-list">
            {analysis.possibleCauses.map((item) => <div key={item} className="training-panel-bullet-item">{item}</div>)}
          </div>
        </div>
      ) : null}

      {analysis.suggestedFixes.length ? (
        <div className="training-panel-analysis-card">
          <div className="training-panel-evidence-title">{t('training.suggestedFixes')}</div>
          <div className="training-panel-bullet-list">
            {analysis.suggestedFixes.map((item) => <div key={item} className="training-panel-bullet-item">{item}</div>)}
          </div>
        </div>
      ) : null}

      {!analysis.possibleCauses.length && !analysis.suggestedFixes.length && !analysis.failureExplanation ? (
        <div className="training-panel-empty">{t('training.analysisEmpty')}</div>
      ) : null}
    </div>
  );
}
