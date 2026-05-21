import React, { startTransition, useEffect, useMemo, useState } from 'react';

import { AutoFixPanel, buildAutoFixSuggestions } from '../autofix';
import { useLanguage } from '../../hooks/useLanguage';
import { getParamHelpText, getParamLabel, translateKnownMessage } from '../../i18n';
import { getNodeBehavior } from '../../registry';
import { inspectDataset } from '../../services';
import { useAppStore } from '../../store';
import type { InspectDatasetResponse, ParamSpec } from '../../types';

function formatListValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value.join(', ');
  }
  return typeof value === 'string' ? value : '';
}

function parseFloatList(value: string): number[] {
  return value
    .split(',')
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isFinite(item));
}

function parseStringList(value: string): string[] {
  return value
    .split(',')
    .map((item) => item.trim())
    .filter((item) => item.length > 0);
}

function formatSplitSummary(splits: Record<string, number>): string {
  return ['train', 'val', 'test']
    .map((key) => `${key}: ${splits[key] ?? 0}`)
    .join(' | ');
}

function InspectorNotice({
  tone,
  label,
  children,
}: {
  tone: 'error' | 'warning';
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className={`inspector-notice inspector-notice--${tone}`}>
      <span className="inspector-notice-label">{label}</span>
      <span>{children}</span>
    </div>
  );
}

function buildParamPatch(
  nodeType: string,
  currentParams: Record<string, unknown>,
  key: string,
  value: unknown,
): Record<string, unknown> {
  const patch: Record<string, unknown> = { [key]: value };

  if (nodeType === 'DataLoader') {
    const nextNumWorkers = key === 'numWorkers' ? Number(value) : Number(currentParams.numWorkers ?? 0);
    if (nextNumWorkers === 0) {
      patch.persistentWorkers = false;
    }
    if (key === 'persistentWorkers' && nextNumWorkers === 0) {
      patch.persistentWorkers = false;
    }
  }

  if (nodeType === 'Dataset' && key === 'datasetMode' && value !== 'csv') {
    patch.taskType = 'classification';
  }

  return patch;
}

function renderParamControl(
  param: ParamSpec,
  params: Record<string, unknown>,
  handleParamChange: (key: string, value: unknown) => void,
) {
  const disabled = param.disabled?.(params) ?? false;

  if (param.type === 'bool') {
    return (
      <input
        type="checkbox"
        checked={Boolean(params[param.key])}
        disabled={disabled}
        onChange={(event) => handleParamChange(param.key, event.target.checked)}
      />
    );
  }

  if (param.type === 'select') {
    return (
      <select
        value={String(params[param.key] ?? param.defaultValue)}
        disabled={disabled}
        onChange={(event) => handleParamChange(param.key, event.target.value)}
      >
        {param.options?.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    );
  }

  if (param.type === 'shape') {
    return (
      <input
        type="text"
        disabled={disabled}
        value={JSON.stringify(params[param.key] ?? param.defaultValue)}
        onChange={(event) => {
          try {
            handleParamChange(param.key, JSON.parse(event.target.value));
          } catch {
            // Ignore invalid shape input until it becomes valid JSON.
          }
        }}
      />
    );
  }

  if (param.type === 'text') {
    return (
      <input
        type="text"
        disabled={disabled}
        placeholder={param.placeholder}
        value={String(params[param.key] ?? param.defaultValue ?? '')}
        onChange={(event) => handleParamChange(param.key, event.target.value)}
      />
    );
  }

  if (param.type === 'float_list') {
    return (
      <input
        type="text"
        disabled={disabled}
        placeholder={param.placeholder ?? '0.5, 0.5, 0.5'}
        value={formatListValue(params[param.key] ?? param.defaultValue)}
        onChange={(event) => handleParamChange(param.key, parseFloatList(event.target.value))}
      />
    );
  }

  if (param.type === 'string_list') {
    return (
      <input
        type="text"
        disabled={disabled}
        placeholder={param.placeholder ?? 'feature_a, feature_b'}
        value={formatListValue(params[param.key] ?? param.defaultValue)}
        onChange={(event) => handleParamChange(param.key, parseStringList(event.target.value))}
      />
    );
  }

  return (
    <input
      type="number"
      disabled={disabled}
      step={param.type === 'float' ? '0.001' : '1'}
      value={Number(params[param.key] ?? param.defaultValue)}
      onChange={(event) => handleParamChange(param.key, Number(event.target.value))}
    />
  );
}

const Inspector: React.FC = () => {
  const selectedNodeId = useAppStore((state) => state.selectedNodeId);
  const project = useAppStore((state) => state.project);
  const nodes = useAppStore((state) => state.project.nodes);
  const trainingDiagnostics = useAppStore((state) => state.trainingDiagnostics);
  const updateNodeParams = useAppStore((state) => state.updateNodeParams);
  const applyAutoFixActions = useAppStore((state) => state.applyAutoFixActions);
  const openDatasetWizard = useAppStore((state) => state.openDatasetWizard);
  const [isInspecting, setIsInspecting] = useState(false);
  const [datasetPreview, setDatasetPreview] = useState<InspectDatasetResponse | null>(null);
  const { language, t } = useLanguage();

  const selectedNode = nodes.find((node) => node.id === selectedNodeId);
  const behavior = selectedNode ? getNodeBehavior(selectedNode.type) : undefined;
  const selectedParams = useMemo(
    () => (selectedNode?.data.params ?? {}) as Record<string, unknown>,
    [selectedNode?.data.params],
  );
  const datasetSignature = useMemo(
    () => (selectedNode?.type === 'Dataset' ? JSON.stringify(selectedNode.data.params ?? {}) : ''),
    [selectedNode],
  );
  const visibleParams = useMemo(
    () => behavior?.template.params.filter((param) => !param.visible || param.visible(selectedParams)) ?? [],
    [behavior, selectedParams],
  );
  const autoFixSuggestions = useMemo(
    () => buildAutoFixSuggestions(project, trainingDiagnostics, datasetPreview),
    [datasetPreview, project, trainingDiagnostics],
  );

  useEffect(() => {
    if (!selectedNode || selectedNode.type !== 'Dataset') {
      setDatasetPreview(null);
      setIsInspecting(false);
      return;
    }

    const timer = window.setTimeout(async () => {
      try {
        setIsInspecting(true);
        const preview = await inspectDataset(selectedParams);
        startTransition(() => {
          setDatasetPreview(preview);
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unknown dataset inspection error';
        startTransition(() => {
          setDatasetPreview({
            success: false,
            datasetMode: String(selectedParams.datasetMode ?? 'builtin'),
            resolvedSplitMode: String(selectedParams.splitMode ?? 'predefined'),
            taskType: String(selectedParams.taskType ?? 'classification'),
            sampleCount: 0,
            numClasses: 0,
            classNames: [],
            splits: { train: 0, val: 0, test: 0 },
            inputShape: null,
            warnings: [],
            errors: [message],
          });
        });
      } finally {
        setIsInspecting(false);
      }
    }, 350);

    return () => window.clearTimeout(timer);
  }, [datasetSignature, selectedNode, selectedParams]);

  if (!selectedNode || !behavior) {
    return (
      <aside className="inspector">
        <h2 className="inspector-title">{t('inspector.title')}</h2>
        <p className="inspector-empty">{t('inspector.empty')}</p>
      </aside>
    );
  }

  const handleParamChange = (key: string, value: unknown) => {
    updateNodeParams(selectedNode.id, buildParamPatch(selectedNode.type, selectedParams, key, value));
  };

  return (
    <aside className="inspector">
      <h2 className="inspector-title">{t('inspector.title')}</h2>
      <div className="inspector-node-header">
        <span className="inspector-node-type">{behavior.template.displayName}</span>
        <span className="inspector-node-id">{selectedNode.id}</span>
      </div>

      <div className="inspector-shapes">
        <div className="inspector-shape">
          <span>{t('inspector.inputShape')}</span>
          <code>{selectedNode.data.inferredInputShape ? JSON.stringify(selectedNode.data.inferredInputShape) : '--'}</code>
        </div>
        <div className="inspector-shape">
          <span>{t('inspector.outputShape')}</span>
          <code>{selectedNode.data.inferredOutputShape ? JSON.stringify(selectedNode.data.inferredOutputShape) : '--'}</code>
        </div>
      </div>

      {selectedNode.data.errors && selectedNode.data.errors.length > 0 ? (
        <div className="inspector-errors">
          {selectedNode.data.errors.map((error, index) => (
            <InspectorNotice key={`${selectedNode.id}-${index}`} tone="error" label={t('inspector.error')}>
              {translateKnownMessage(error, language)}
            </InspectorNotice>
          ))}
        </div>
      ) : null}

      {selectedNode.type === 'Dataset' ? (
        <div className="inspector-errors">
          <div className="inspector-node-header">
            <span className="inspector-node-type">{t('inspector.datasetPreview')}</span>
            <span className="inspector-node-id">
              {isInspecting ? t('inspector.inspecting') : datasetPreview?.success ? t('inspector.ready') : t('inspector.pending')}
            </span>
          </div>
          <button className="inspector-secondary-action" onClick={openDatasetWizard}>
            {t('inspector.openWizard')}
          </button>
          {datasetPreview ? (
            <>
              <div className="inspector-shapes">
                <div className="inspector-shape">
                  <span>{t('inspector.mode')}</span>
                  <code>{datasetPreview.datasetMode}</code>
                </div>
                <div className="inspector-shape">
                  <span>{t('inspector.split')}</span>
                  <code>{datasetPreview.resolvedSplitMode}</code>
                </div>
                <div className="inspector-shape">
                  <span>{t('inspector.samples')}</span>
                  <code>{datasetPreview.sampleCount}</code>
                </div>
                <div className="inspector-shape">
                  <span>{t('inspector.classes')}</span>
                  <code>{datasetPreview.numClasses}</code>
                </div>
                <div className="inspector-shape">
                  <span>{t('inspector.splits')}</span>
                  <code>{formatSplitSummary(datasetPreview.splits)}</code>
                </div>
                <div className="inspector-shape">
                  <span>{t('inspector.input')}</span>
                  <code>{datasetPreview.inputShape ? JSON.stringify(datasetPreview.inputShape) : '--'}</code>
                </div>
              </div>

              {datasetPreview.classNames.length > 0 ? (
                <div className="inspector-shape">
                  <span>{t('inspector.classNames')}</span>
                  <code>{datasetPreview.classNames.join(', ')}</code>
                </div>
              ) : null}

              {datasetPreview.errors.map((error) => (
                <InspectorNotice key={`dataset-preview-error-${error}`} tone="error" label={t('inspector.error')}>
                  {translateKnownMessage(error, language)}
                </InspectorNotice>
              ))}
              {datasetPreview.warnings.map((warning) => (
                <InspectorNotice key={`dataset-preview-warning-${warning}`} tone="warning" label={t('inspector.warning')}>
                  {translateKnownMessage(warning, language)}
                </InspectorNotice>
              ))}
            </>
          ) : (
            <InspectorNotice tone="warning" label={t('inspector.warning')}>
              {t('inspector.configureDataset')}
            </InspectorNotice>
          )}
        </div>
      ) : null}

      {selectedNode.type === 'DataLoader' && Number(selectedParams.numWorkers ?? 0) === 0 ? (
        <div className="inspector-errors">
          <InspectorNotice tone="warning" label={t('inspector.warning')}>
            {t('inspector.dataloaderWorkersNotice')}
          </InspectorNotice>
        </div>
      ) : null}

      {autoFixSuggestions.length > 0 ? (
        <AutoFixPanel suggestions={autoFixSuggestions} onApply={applyAutoFixActions} compact />
      ) : null}

      <div className="inspector-params">
        {visibleParams.map((param) => (
          <div key={param.key} className="inspector-param">
            <label className="inspector-param-label" title={getParamHelpText(selectedNode.type, param, language)}>
              {getParamLabel(selectedNode.type, param, language)}
            </label>
            {renderParamControl(param, selectedParams, handleParamChange)}
          </div>
        ))}
      </div>
    </aside>
  );
};

export default Inspector;
