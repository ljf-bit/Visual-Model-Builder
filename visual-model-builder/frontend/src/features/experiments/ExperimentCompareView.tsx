import { useEffect, useMemo, useState } from 'react';

import { useLanguage } from '../../hooks/useLanguage';
import type { ExperimentRecord, TrainingEpochLog } from '../../types';
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

  useEffect(() => {
    const refresh = () => setRecords(loadExperimentHistory());
    window.addEventListener(EXPERIMENT_HISTORY_EVENT, refresh);
    window.addEventListener('storage', refresh);
    return () => {
      window.removeEventListener(EXPERIMENT_HISTORY_EVENT, refresh);
      window.removeEventListener('storage', refresh);
    };
  }, []);

  const selectedRecords = useMemo(
    () => records.filter((record) => selectedIds.includes(record.id)).slice(0, 4),
    [records, selectedIds],
  );
  const bestAccuracy = useMemo(
    () => records.reduce<ExperimentRecord | null>((best, record) => {
      const accuracy = record.metrics.bestAccuracy ?? -1;
      const bestValue = best?.metrics.bestAccuracy ?? -1;
      return accuracy > bestValue ? record : best;
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
            <strong>{bestAccuracy ? formatAccuracy(bestAccuracy.metrics.bestAccuracy) : '--'}</strong>
          </div>
          <div className="training-panel-summary-card">
            <span>{t('compare.bestLoss')}</span>
            <strong>{bestLoss ? formatMetric(bestLoss.metrics.bestLoss) : '--'}</strong>
          </div>
        </div>
        <button className="training-panel-save-btn danger" onClick={clearExperimentHistory}>
          {t('compare.clear')}
        </button>
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
                <span>{t('compare.finalAccuracy')}: {formatAccuracy(record.metrics.finalAccuracy)}</span>
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
              <th>{t('compare.params')}</th>
              <th>{t('compare.actions')}</th>
            </tr>
          </thead>
          <tbody>
            {records.map((record) => (
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
                <td>{formatMetric(record.metrics.finalLoss)}<MiniCurve logs={record.logs} metric="loss" /></td>
                <td>{formatAccuracy(record.metrics.finalAccuracy)}<MiniCurve logs={record.logs} metric="accuracy" /></td>
                <td>
                  <span>{record.optimizerType}</span>
                  <span>lr {record.learningRate ?? '--'} / batch {record.batchSize ?? '--'} / epoch {record.epochs ?? '--'}</span>
                </td>
                <td>
                  <button className="training-panel-save-btn" onClick={() => handleDelete(record.id)}>
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
