import type {
  AppLanguage,
  GraphEdge,
  GraphNode,
  RunTrainingResponse,
  TrainingDiagnosticsResponse,
  TrainingInsightsResponse,
} from '../../types';
import { translate } from '../../i18n';
import type { ReadinessItem } from './trainingTypes';
import { buildFallbackConfigurationSummary, buildFallbackModelSummary } from './trainingCharts';

function hasNode(nodes: GraphNode[], type: string): GraphNode | undefined {
  return nodes.find((node) => node.type === type);
}

function hasEdge(edges: GraphEdge[], source: string | undefined, target: string | undefined): boolean {
  if (!source || !target) {
    return false;
  }
  return edges.some((edge) => edge.source === source && edge.target === target);
}

export function buildReadinessItems(nodes: GraphNode[], edges: GraphEdge[], language: AppLanguage = 'en'): ReadinessItem[] {
  const datasetNode = hasNode(nodes, 'Dataset');
  const dataLoaderNode = hasNode(nodes, 'DataLoader');
  const inputNode = hasNode(nodes, 'Input');
  const outputNode = hasNode(nodes, 'Output');
  const lossNode = hasNode(nodes, 'Loss');
  const optimizerNode = hasNode(nodes, 'Optimizer');
  const trainerNode = hasNode(nodes, 'Trainer');
  const metricNode = hasNode(nodes, 'Metric');

  return [
    { label: translate(language, 'training.readiness.dataset'), ok: Boolean(datasetNode) },
    { label: translate(language, 'training.readiness.datasetLoader'), ok: hasEdge(edges, datasetNode?.id, dataLoaderNode?.id) },
    { label: translate(language, 'training.readiness.loaderInput'), ok: hasEdge(edges, dataLoaderNode?.id, inputNode?.id) },
    { label: translate(language, 'training.readiness.outputLoss'), ok: hasEdge(edges, outputNode?.id, lossNode?.id) },
    { label: translate(language, 'training.readiness.optimizerTrainer'), ok: hasEdge(edges, optimizerNode?.id, trainerNode?.id) },
    { label: translate(language, 'training.readiness.lossTrainer'), ok: hasEdge(edges, lossNode?.id, trainerNode?.id) },
    { label: translate(language, 'training.readiness.metricTrainer'), ok: metricNode ? hasEdge(edges, metricNode.id, trainerNode?.id) : true },
  ];
}

export function getReadinessLabel(
  trainingMode: boolean,
  trainingReady: boolean,
  currentDiagnostics: TrainingDiagnosticsResponse | null,
  language: AppLanguage = 'en',
): string {
  if (!trainingMode) {
    return translate(language, 'training.ready.trainingNodesMissing');
  }
  if (!trainingReady) {
    return translate(language, 'training.ready.incomplete');
  }
  if (currentDiagnostics && !currentDiagnostics.ok) {
    return translate(language, 'training.ready.blocked');
  }
  if (currentDiagnostics && currentDiagnostics.warnings.length > 0) {
    return translate(language, 'training.ready.warnings');
  }
  return translate(language, 'training.ready.ready');
}

export function buildTrainingEvidence(result: RunTrainingResponse | null, language: AppLanguage = 'en'): string {
  const firstLog = result?.logs[0] ?? null;
  const lastLog = result?.logs[result.logs.length - 1] ?? null;
  const lossImproved = Boolean(firstLog && lastLog && lastLog.loss < firstLog.loss);

  if (result?.ok && lastLog) {
    return lossImproved ? translate(language, 'training.evidence.improved') : translate(language, 'training.evidence.notImproved');
  }
  return translate(language, 'training.evidence.empty');
}

export function buildAnalysisText(
  result: RunTrainingResponse | null,
  diagnostics: TrainingDiagnosticsResponse | null,
  insights: TrainingInsightsResponse | null,
  language: AppLanguage = 'en',
) {
  const runtimeMessages = result?.errors ?? [];
  const evidence = buildTrainingEvidence(result, language);
  const configurationSummary = insights?.configurationSummary ?? buildFallbackConfigurationSummary(diagnostics, language);
  const modelSummary = insights?.modelSummary ?? buildFallbackModelSummary(diagnostics, language);
  const trendSummary = insights?.trendSummary
    ?? (result?.status === 'diagnostics_failed'
      ? translate(language, 'training.analysis.stopped')
      : result && !result.ok
        ? translate(language, 'training.analysis.notComplete')
        : result?.logs.length
          ? evidence
          : translate(language, 'training.analysis.runForCurves'));
  const qualitySummary = insights?.qualitySummary
    ?? ((diagnostics?.warnings.length ?? 0) > 0
      ? translate(language, 'training.analysis.warningQuality')
      : result?.logs.length
        ? evidence
        : translate(language, 'training.analysis.noCompleted'));
  const failureExplanation = insights?.failureExplanation
    ?? (!result?.ok
      ? runtimeMessages[0] ?? (diagnostics?.ok === false ? diagnostics.summary : null)
      : diagnostics?.ok === false
        ? diagnostics.summary
        : null);
  const possibleCauses = Array.from(
    new Set(
      (insights?.possibleCauses?.length
        ? insights.possibleCauses
        : [
          ...(diagnostics?.errors ?? []),
          ...(diagnostics?.warnings ?? []),
          ...(result?.errors ?? []),
        ]).filter((item) => item.trim().length > 0),
    ),
  );
  const suggestedFixes = Array.from(
    new Set(
      (insights?.suggestedFixes?.length ? insights.suggestedFixes : diagnostics?.suggestions ?? []).filter(
        (item) => item.trim().length > 0,
      ),
    ),
  );

  return {
    configurationSummary,
    modelSummary,
    trendSummary,
    qualitySummary,
    failureExplanation,
    possibleCauses,
    suggestedFixes,
  };
}
