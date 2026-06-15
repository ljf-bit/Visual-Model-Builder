import { useEffect, useMemo, useState } from 'react';

import { useLanguage } from '../../hooks/useLanguage';
import { getTrainingRuns } from '../../services';
import type { ExperimentRecord, TrainingEpochLog, TrainingRunRecord } from '../../types';
import { buildChartPoints, buildPolyline, formatDuration } from '../training/trainingCharts';
import {
  clearExperimentHistory,
  deleteExperimentRecord,
  EXPERIMENT_HISTORY_EVENT,
  loadExperimentHistory,
} from './experimentHistory';

function formatMetric(value: number | null, suffix = ''): string {
  if (value === null || !Number.isFinite(value)) {
    return '--';
  }
  return `${value.toFixed(suffix === '%' ? 2 : 4)}${suffix}`;
}

function formatAccuracy(value: number | null): string {
  return value === null ? '--' : `${(value * 100).toFixed(2)}%`;
}

type ExperimentSortKey = 'createdAt' | 'accuracy' | 'macroF1' | 'loss' | 'duration';

function backendRunToExperimentRecord(run: TrainingRunRecord): ExperimentRecord {
  return {
    id: `backend-${run.runId}`,
    projectName: run.projectName || run.runId,
    createdAt: run.createdAt || run.completedAt,
    status: run.status,
    ok: run.ok,
    datasetName: run.datasetUsed || 'unknown',
    datasetMode: run.datasetMode || 'unknown',
    optimizerType: 'from summary',
    learningRate: null,
    batchSize: null,
    epochs: null,
    durationSeconds: run.durationSeconds,
    graphSignature: run.runId,
    metrics: {
      firstLoss: null,
      finalLoss: run.finalLoss,
      bestLoss: run.finalLoss,
      finalValLoss: run.finalLoss,
      finalAccuracy: run.finalAccuracy,
      bestAccuracy: run.finalAccuracy,
      finalValAccuracy: run.finalAccuracy,
      macroF1: run.macroF1,
      weightedF1: run.weightedF1,
      epochCount: 0,
    },
    logs: [],
    evaluation: null,
    source: 'backend',
    trainingMetadata: null,
  };
}

function MiniCurve({ logs, metric }: { logs: TrainingEpochLog[]; metric: 'loss' | 'accuracy' }) {
  const points = buildChartPoints(logs, (log) => (metric === 'loss' ? log.loss : log.accuracy));
  const scaledPoints = points.map((point) => ({ x: point.x, y: point.y * 0.4 }));
  const polyline = buildPolyline(scaledPoints);
  return (
    <svg viewBox="0 0 100 40" preserveAspectRatio="none" className="experiment-mini-chart">
      {polyline ? <polyline points={polyline} className={`training-chart-line ${metric}`} /> : null}
      {scaledPoints.map((point, index) => (
        <circle key={`${metric}-${index}`} cx={point.x} cy={point.y} r={2} className={`training-chart-dot ${metric}`} />
      ))}
    </svg>
  );
}

export function ExperimentCompareView() {
  const { t } = useLanguage();
  const [records, setRecords] = useState<ExperimentRecord[]>(() => loadExperimentHistory());
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [sortKey, setSortKey] = useState<ExperimentSortKey>('createdAt');

  useEffect(() => {
    const refresh = async () => {
      const localRecords = loadExperimentHistory();
      try {
        const backendRuns = await getTrainingRuns(50);
        const backendRecords = backendRuns.runs.map(backendRunToExperimentRecord);
        const localRunIds = new Set(localRecords.map((record) => record.trainingMetadata?.runId).filter(Boolean));
        setRecords([
          ...localRecords,
          ...backendRecords.filter((record) => !localRunIds.has(record.graphSignature)),
        ]);
      } catch {
        setRecords(localRecords);
      }
    };
    void refresh();
    const refreshFromEvent = () => void refresh();
    window.addEventListener(EXPERIMENT_HISTORY_EVENT, refreshFromEvent);
    window.addEventListener('storage', refreshFromEvent);
    return () => {
      window.removeEventListener(EXPERIMENT_HISTORY_EVENT, refreshFromEvent);
      window.removeEventListener('storage', refreshFromEvent);
    };
  }, []);

  const sortedRecords = useMemo(() => {
    const valueFor = (record: ExperimentRecord): number => {
      if (sortKey === 'accuracy') {
        return record.metrics.finalValAccuracy ?? record.metrics.finalAccuracy ?? -1;
      }
      if (sortKey === 'macroF1') {
        return record.metrics.macroF1 ?? -1;
      }
      if (sortKey === 'loss') {
        return -(record.metrics.finalValLoss ?? record.metrics.finalLoss ?? Number.POSITIVE_INFINITY);
      }
      if (sortKey === 'duration') {
        return -(record.durationSeconds ?? Number.POSITIVE_INFINITY);
      }
      return new Date(record.createdAt).valueOf() || 0;
    };
    return [...records].sort((a, b) => valueFor(b) - valueFor(a));
  }, [records, sortKey]);

  const selectedRecords = useMemo(
    () => sortedRecords.filter((record) => selectedIds.includes(record.id)).slice(0, 4),
    [selectedIds, sortedRecords],
  );
  const bestAccuracy = useMemo(
    () => records.reduce<ExperimentRecord | null>((best, record) => {
      const accuracy = record.metrics.finalValAccuracy ?? record.metrics.bestAccuracy ?? -1;
      const bestValue = best ? (best.metrics.finalValAccuracy ?? best.metrics.bestAccuracy ?? -1) : -1;
      return accuracy > bestValue ? record : best;
    }, null),
    [records],
  );
  const bestF1 = useMemo(
    () => records.reduce<ExperimentRecord | null>((best, record) => {
      const f1 = record.metrics.macroF1 ?? -1;
      const bestValue = best?.metrics.macroF1 ?? -1;
      return f1 > bestValue ? record : best;
    }, null),
    [records],
  );
  const bestLoss = useMemo(
    () => records.reduce<ExperimentRecord | null>((best, record) => {
      const loss = record.metrics.bestLoss ?? Number.POSITIVE_INFINITY;
      const bestValue = best?.metrics.bestLoss ?? Number.POSITIVE_INFINITY;
      return loss < bestValue ? record : best;
    }, null),
    [records],
  );

  const toggleSelection = (recordId: string) => {
    setSelectedIds((current) => {
      if (current.includes(recordId)) {
        return current.filter((id) => id !== recordId);
      }
      if (current.length >= 4) {
        return current;
      }
      return [...current, recordId];
    });
  };

  const handleDelete = (recordId: string) => {
    deleteExperimentRecord(recordId);
    setSelectedIds((current) => current.filter((id) => id !== recordId));
  };

  if (records.length === 0) {
    return (
      <div className="experiment-compare-empty">
        <strong>{t('compare.emptyTitle')}</strong>
        <span>{t('compare.empty')}</span>
      </div>
    );
  }

  return (
    <div className="experiment-compare">
      <div className="experiment-compare-copy">
        <strong>{t('compare.title')}</strong>
        <span>{t('compare.description')}</span>
      </div>

      <div className="experiment-compare-header">
        <div className="training-panel-summary-grid">
          <div className="training-panel-summary-card">
            <span>{t('compare.totalRuns')}</span>
            <strong>{records.length}</strong>
          </div>
          <div className="training-panel-summary-card">
            <span>{t('compare.bestAccuracy')}</span>
            <strong>{bestAccuracy ? formatAccuracy(bestAccuracy.metrics.finalValAccuracy ?? bestAccuracy.metrics.bestAccuracy) : '--'}</strong>
          </div>
          <div className="training-panel-summary-card">
            <span>{t('compare.bestLoss')}</span>
            <strong>{bestLoss ? formatMetric(bestLoss.metrics.bestLoss) : '--'}</strong>
          </div>
          <div className="training-panel-summary-card">
            <span>{t('compare.bestF1')}</span>
            <strong>{bestF1 ? formatMetric(bestF1.metrics.macroF1) : '--'}</strong>
          </div>
        </div>
        <div className="experiment-compare-actions">
          <label>
            {t('compare.sortBy')}
            <select value={sortKey} onChange={(event) => setSortKey(event.target.value as ExperimentSortKey)}>
              <option value="createdAt">{t('compare.run')}</option>
              <option value="accuracy">{t('compare.accuracy')}</option>
              <option value="macroF1">{t('compare.macroF1')}</option>
              <option value="loss">{t('compare.loss')}</option>
              <option value="duration">{t('compare.duration')}</option>
            </select>
          </label>
          <button className="training-panel-save-btn danger" onClick={clearExperimentHistory}>
            {t('compare.clear')}
          </button>
        </div>
      </div>

      {selectedRecords.length > 0 ? (
        <div className="experiment-selected-grid">
          {selectedRecords.map((record) => (
            <div key={record.id} className="experiment-selected-card">
              <div className="experiment-selected-title">{record.projectName}</div>
              <div className="experiment-selected-meta">{record.datasetName} / {record.optimizerType} / lr {record.learningRate ?? '--'}</div>
              <MiniCurve logs={record.logs} metric="loss" />
              <div className="experiment-selected-metrics">
                <span>{t('compare.finalLoss')}: {formatMetric(record.metrics.finalLoss)}</span>
                <span>{t('compare.finalAccuracy')}: {formatAccuracy(record.metrics.finalValAccuracy ?? record.metrics.finalAccuracy)}</span>
                <span>{t('compare.macroF1')}: {formatMetric(record.metrics.macroF1)}</span>
                <span>{t('compare.duration')}: {record.durationSeconds === null ? '--' : formatDuration(record.durationSeconds)}</span>
              </div>
            </div>
          ))}
        </div>
      ) : null}

      <div className="experiment-table-wrap">
        <table className="experiment-table">
          <thead>
            <tr>
              <th>{t('compare.select')}</th>
              <th>{t('compare.run')}</th>
              <th>{t('compare.dataset')}</th>
              <th>{t('compare.status')}</th>
              <th>{t('compare.loss')}</th>
              <th>{t('compare.accuracy')}</th>
              <th>{t('compare.macroF1')}</th>
              <th>{t('compare.source')}</th>
              <th>{t('compare.params')}</th>
              <th>{t('compare.actions')}</th>
            </tr>
          </thead>
          <tbody>
            {sortedRecords.map((record) => (
              <tr key={record.id}>
                <td>
                  <input
                    type="checkbox"
                    checked={selectedIds.includes(record.id)}
                    disabled={!selectedIds.includes(record.id) && selectedIds.length >= 4}
                    onChange={() => toggleSelection(record.id)}
                  />
                </td>
                <td>
                  <strong>{new Date(record.createdAt).toLocaleString()}</strong>
                  <span>{record.projectName}</span>
                </td>
                <td>{record.datasetName}<span>{record.datasetMode}</span></td>
                <td><span className={`experiment-status ${record.ok ? 'ok' : 'failed'}`}>{record.status}</span></td>
                <td>{formatMetric(record.metrics.finalValLoss ?? record.metrics.finalLoss)}<MiniCurve logs={record.logs} metric="loss" /></td>
                <td>{formatAccuracy(record.metrics.finalValAccuracy ?? record.metrics.finalAccuracy)}<MiniCurve logs={record.logs} metric="accuracy" /></td>
                <td>{formatMetric(record.metrics.macroF1)}</td>
                <td>{record.source ?? 'local'}</td>
                <td>
                  <span>{record.optimizerType}</span>
                  <span>lr {record.learningRate ?? '--'} / batch {record.batchSize ?? '--'} / epoch {record.epochs ?? '--'}</span>
                </td>
                <td>
                  <button className="training-panel-save-btn" onClick={() => handleDelete(record.id)} disabled={record.source === 'backend'}>
                    {t('compare.delete')}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
