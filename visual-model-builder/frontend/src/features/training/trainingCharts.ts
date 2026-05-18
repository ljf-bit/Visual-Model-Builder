import type { TrainingDiagnosticsResponse, TrainingEpochLog } from '../../types';
import { translate } from '../../i18n';
import type { AppLanguage } from '../../types';
import type { ChartPoint } from './trainingTypes';

export function buildChartPoints(logs: TrainingEpochLog[], selector: (log: TrainingEpochLog) => number | null): ChartPoint[] {
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

export function buildPolyline(points: ChartPoint[]): string {
  return points.map((point) => `${point.x},${point.y}`).join(' ');
}

export function formatTimestamp(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString('zh-CN', { hour12: false });
}

export function formatDuration(seconds: number): string {
  return `${seconds.toFixed(2)} s`;
}

export function buildFallbackConfigurationSummary(diagnostics: TrainingDiagnosticsResponse | null, language: AppLanguage = 'en'): string {
  if (!diagnostics) {
    return translate(language, 'training.analysis.noDiagnostics');
  }

  const stats = diagnostics.trainingStats;
  return translate(language, 'training.analysis.configured', {
    datasetName: stats.datasetName,
    optimizerType: stats.optimizerType,
    learningRate: stats.learningRate,
    batchSize: stats.batchSize,
    epochs: stats.epochs,
    device: stats.device,
  });
}

export function buildFallbackModelSummary(diagnostics: TrainingDiagnosticsResponse | null, language: AppLanguage = 'en'): string {
  if (!diagnostics) {
    return translate(language, 'training.analysis.noModelStats');
  }

  const stats = diagnostics.modelStats;
  return translate(language, 'training.analysis.modelStats', {
    parameters: stats.parameterCount.toLocaleString(),
    layers: stats.learnableLayerCount,
    complexity: stats.complexityLabel,
  });
}
