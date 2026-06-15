import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { buildAutoFixSuggestions } from '../autofix';
import { ExperimentCompareView, createExperimentRecord, saveExperimentRecord } from '../experiments';
import { hasTrainingNodes } from '../../graph/graphUtils';
import { useLanguage } from '../../hooks/useLanguage';
import { cancelTrainingJob, createTrainingJob, getTrainingJob } from '../../services';
import { useAppStore } from '../../store';
import type { RunTrainingResponse, TrainingJobResponse } from '../../types';
import { buildChartPoints } from './trainingCharts';
import { buildAnalysisText, buildReadinessItems, buildTrainingEvidence, getReadinessLabel } from './trainingReadiness';
import { buildTrainingReportHtml } from './trainingReport';
import type { SavePickerWindow, TrainingView } from './trainingTypes';
import {
  TrainingAnalysisView,
  TrainingCurvesView,
  TrainingLogsView,
  TrainingStatusView,
} from './TrainingViews';

function jobToTrainingResult(job: TrainingJobResponse): RunTrainingResponse {
  return {
    ok: job.ok,
    status: job.status,
    logs: job.logs,
    errors: job.errors,
    diagnostics: job.diagnostics ?? null,
    insights: job.insights ?? null,
    evaluation: job.evaluation ?? null,
    trainingMetadata: job.trainingMetadata ?? null,
  };
}

const terminalJobStatuses = new Set(['completed', 'failed', 'cancelled']);

const TrainingPanelV2: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [activeView, setActiveView] = useState<TrainingView>('status');
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [jobProgress, setJobProgress] = useState(0);
  const [cancelRequested, setCancelRequested] = useState(false);
  const savedJobIdsRef = useRef(new Set<string>());
  const project = useAppStore((state) => state.project);
  const activeJobProjectRef = useRef(project);
  const globalErrors = useAppStore((state) => state.globalErrors);
  const trainingResult = useAppStore((state) => state.trainingResult);
  const trainingDiagnostics = useAppStore((state) => state.trainingDiagnostics);
  const setTrainingResult = useAppStore((state) => state.setTrainingResult);
  const setTrainingDiagnostics = useAppStore((state) => state.setTrainingDiagnostics);
  const applyAutoFixActions = useAppStore((state) => state.applyAutoFixActions);
  const trainingMode = hasTrainingNodes(project.nodes);
  const { language, t } = useLanguage();

  const readinessItems = useMemo(
    () => buildReadinessItems(project.nodes, project.edges, language),
    [language, project.edges, project.nodes],
  );
  const currentDiagnostics = trainingResult?.diagnostics ?? trainingDiagnostics;
  const currentInsights = trainingResult?.insights ?? null;
  const diagnosticsErrors = currentDiagnostics?.errors ?? [];
  const diagnosticsWarnings = currentDiagnostics?.warnings ?? [];
  const diagnosticsSuggestions = currentDiagnostics?.suggestions ?? [];
  const trainingReady = trainingMode && readinessItems.every((item) => item.ok) && globalErrors.length === 0;
  const readinessLabel = getReadinessLabel(trainingMode, trainingReady, currentDiagnostics ?? null, language);
  const trainingEvidence = buildTrainingEvidence(trainingResult, language);
  const trainingMetadata = trainingResult?.trainingMetadata ?? null;
  const lossPoints = useMemo(
    () => buildChartPoints(trainingResult?.logs ?? [], (log) => log.loss),
    [trainingResult],
  );
  const accuracyPoints = useMemo(
    () => buildChartPoints(trainingResult?.logs ?? [], (log) => log.accuracy),
    [trainingResult],
  );
  const analysis = useMemo(
    () => buildAnalysisText(trainingResult, currentDiagnostics ?? null, currentInsights, language),
    [currentDiagnostics, currentInsights, language, trainingResult],
  );
  const autoFixSuggestions = useMemo(
    () => buildAutoFixSuggestions(project, currentDiagnostics ?? null),
    [currentDiagnostics, project],
  );
  const diagnosticsTone = currentDiagnostics
    ? currentDiagnostics.ok
      ? currentDiagnostics.warnings.length > 0 ? 'warning' : 'ok'
      : 'error'
    : trainingReady
      ? 'ok'
      : 'warning';
  const diagnosticsBannerTitle = currentDiagnostics ? t('training.diagnostics') : t('training.preRunDiagnostics');
  const diagnosticsBannerSummary = currentDiagnostics
    ? currentDiagnostics.summary
    : trainingMode
      ? t('training.diagnosticsPending')
      : t('training.diagnosticsMissing');
  const canSaveResults = Boolean(trainingResult?.ok && trainingMetadata && trainingResult.logs.length > 0);

  const persistTerminalJob = useCallback((job: TrainingJobResponse) => {
    const result = jobToTrainingResult(job);
    if (terminalJobStatuses.has(job.status) && !savedJobIdsRef.current.has(job.jobId)) {
      saveExperimentRecord(createExperimentRecord(activeJobProjectRef.current, result, job.jobId));
      savedJobIdsRef.current.add(job.jobId);
    }
    return result;
  }, []);

  useEffect(() => {
    if (!activeJobId || !isRunning) {
      return;
    }

    let disposed = false;
    const poll = async () => {
      try {
        const job = await getTrainingJob(activeJobId);
        if (disposed) {
          return;
        }

        setJobProgress(job.progress);
        setCancelRequested(job.cancelRequested);
        if (job.diagnostics) {
          setTrainingDiagnostics(job.diagnostics);
        }
        setTrainingResult(persistTerminalJob(job));

        if (terminalJobStatuses.has(job.status)) {
          setIsRunning(false);
          setActiveView(job.ok && job.logs.length > 0 ? 'curves' : 'analysis');
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unknown training job polling error';
        setTrainingResult({
          ok: false,
          status: 'request_failed',
          logs: [],
        errors: [message],
        diagnostics: currentDiagnostics ?? null,
        insights: null,
        evaluation: null,
        trainingMetadata: null,
      });
        setIsRunning(false);
        setActiveView('analysis');
      }
    };

    void poll();
    const timer = window.setInterval(poll, 700);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [activeJobId, currentDiagnostics, isRunning, persistTerminalJob, setTrainingDiagnostics, setTrainingResult]);

  const handleRunTraining = async () => {
    try {
      setIsRunning(true);
      setCancelRequested(false);
      setJobProgress(0);
      activeJobProjectRef.current = project;
      const job = await createTrainingJob(project);
      setActiveJobId(job.jobId);
      if (job.diagnostics) {
        setTrainingDiagnostics(job.diagnostics);
      }
      setTrainingResult(persistTerminalJob(job));
      if (terminalJobStatuses.has(job.status)) {
        setIsRunning(false);
        setActiveView('analysis');
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      setTrainingResult({
        ok: false,
        status: 'request_failed',
        logs: [],
        errors: [message],
        diagnostics: null,
        insights: null,
        evaluation: null,
        trainingMetadata: null,
      });
      setIsRunning(false);
      setActiveView('analysis');
    }
  };

  const handleCancelTraining = async () => {
    if (!activeJobId) {
      return;
    }
    const job = await cancelTrainingJob(activeJobId);
    setCancelRequested(job.cancelRequested);
    setTrainingResult(jobToTrainingResult(job));
  };

  const handleSaveResults = async () => {
    if (!trainingResult || !trainingMetadata) {
      return;
    }

    const reportHtml = buildTrainingReportHtml(
      trainingMetadata.projectName || project.metadata.name,
      trainingResult,
      trainingMetadata,
      trainingResult.evaluation ?? null,
      lossPoints,
      accuracyPoints,
      currentDiagnostics ?? null,
      currentInsights,
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
          <h3 className="training-panel-title">{t('training.title')}</h3>
          <p className="training-panel-subtitle">
            {trainingMode
              ? t('training.subtitleReady')
              : t('training.subtitleMissing')}
          </p>
        </div>
        <div className="training-panel-actions">
          <button className="training-panel-save-btn" onClick={handleSaveResults} disabled={!canSaveResults || isRunning}>
            {t('training.saveResults')}
          </button>
          {isRunning ? (
            <button className="training-panel-save-btn danger" onClick={handleCancelTraining} disabled={cancelRequested}>
              {cancelRequested ? t('training.cancelling') : t('training.cancel')}
            </button>
          ) : null}
          <button className="training-panel-run-btn" onClick={handleRunTraining} disabled={!trainingReady || isRunning}>
            {isRunning ? t('training.running') : t('training.run')}
          </button>
        </div>
      </div>

      <div className="training-panel-toolbar">
        <div className="training-panel-status">
          <span>{t('training.readiness')}</span>
          <strong>{readinessLabel}</strong>
        </div>
        <div className="training-panel-progress" aria-label={t('training.progressLabel')}>
          <span style={{ width: `${Math.round(jobProgress * 100)}%` }} />
        </div>
        <div className="training-panel-tabs" role="tablist" aria-label={t('training.viewsLabel')}>
          {(['curves', 'status', 'logs', 'analysis', 'compare'] as const).map((view) => (
            <button
              key={view}
              className={`training-panel-tab ${activeView === view ? 'active' : ''}`}
              onClick={() => setActiveView(view)}
            >
              {t(`training.tab.${view}`)}
            </button>
          ))}
        </div>
      </div>

      <div className="training-panel-body">
        {trainingMode || currentDiagnostics ? (
          <div className={`training-panel-diagnostics-banner ${diagnosticsTone}`}>
            <div className="training-panel-diagnostics-banner-header">
              <span>{diagnosticsBannerTitle}</span>
              {currentDiagnostics ? (
                <strong>
                  {currentDiagnostics.ok
                    ? (diagnosticsWarnings.length > 0 ? t('training.status.warning') : t('training.status.ok'))
                    : t('training.status.blocked')}
                </strong>
              ) : null}
            </div>
            <div className="training-panel-evidence-text">{diagnosticsBannerSummary}</div>
            {currentDiagnostics ? (
              <div className="training-panel-notice-list">
                <span>{t('training.errorsCount', { count: diagnosticsErrors.length })}</span>
                <span>{t('training.warningsCount', { count: diagnosticsWarnings.length })}</span>
                <span>{t('training.suggestionsCount', { count: diagnosticsSuggestions.length })}</span>
              </div>
            ) : null}
          </div>
        ) : null}

        {activeView === 'curves' ? (
          <TrainingCurvesView trainingResult={trainingResult} lossPoints={lossPoints} accuracyPoints={accuracyPoints} />
        ) : null}

        {activeView === 'status' ? (
          <TrainingStatusView
            readinessItems={readinessItems}
            currentDiagnostics={currentDiagnostics ?? null}
            trainingResult={trainingResult}
            trainingMetadata={trainingMetadata}
            trainingEvidence={trainingEvidence}
            autoFixSuggestions={autoFixSuggestions}
            onApplyAutoFix={applyAutoFixActions}
          />
        ) : null}

        {activeView === 'logs' ? <TrainingLogsView trainingResult={trainingResult} /> : null}

        {activeView === 'analysis' ? <TrainingAnalysisView analysis={analysis} /> : null}

        {activeView === 'compare' ? <ExperimentCompareView /> : null}
      </div>
    </section>
  );
};

export default TrainingPanelV2;
