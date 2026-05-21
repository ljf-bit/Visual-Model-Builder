import { useEffect, useMemo, useState } from 'react';

import { AutoFixPanel, buildAutoFixSuggestions } from '../autofix';
import { useLanguage } from '../../hooks/useLanguage';
import { translateKnownMessage } from '../../i18n';
import { inspectDataset } from '../../services';
import { useAppStore } from '../../store';
import type { DatasetWizardDraft, DatasetWizardStep, InspectDatasetResponse } from '../../types';

const WIZARD_STEPS: DatasetWizardStep[] = ['source', 'preprocess', 'split', 'preview'];

const DEFAULT_DRAFT: DatasetWizardDraft = {
  datasetMode: 'builtin',
  datasetName: 'FakeData',
  trainSplit: true,
  rootPath: '',
  splitMode: 'predefined',
  trainRatio: 0.7,
  valRatio: 0.2,
  testRatio: 0.1,
  shuffleBeforeSplit: true,
  csvPath: '',
  pathColumn: 'image_path',
  labelColumn: 'label',
  featureColumns: [],
  taskType: 'classification',
  imageSize: 28,
  colorMode: 'grayscale',
  normalize: false,
  mean: [0.5],
  std: [0.5],
  augmentationEnabled: false,
  numClasses: 10,
};

function asNumber(value: unknown, fallback: number): number {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

function asBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === 'boolean' ? value : fallback;
}

function asStringArray(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map(String).filter(Boolean);
  }
  if (typeof value === 'string') {
    return value.split(',').map((item) => item.trim()).filter(Boolean);
  }
  return [];
}

function asNumberArray(value: unknown, fallback: number[]): number[] {
  if (Array.isArray(value)) {
    const parsed = value.map(Number).filter(Number.isFinite);
    return parsed.length ? parsed : fallback;
  }
  if (typeof value === 'string') {
    const parsed = value.split(',').map((item) => Number(item.trim())).filter(Number.isFinite);
    return parsed.length ? parsed : fallback;
  }
  return fallback;
}

function listToText(value: string[] | number[]): string {
  return value.join(', ');
}

function parseStringList(value: string): string[] {
  return value.split(',').map((item) => item.trim()).filter(Boolean);
}

function parseNumberList(value: string): number[] {
  return value.split(',').map((item) => Number(item.trim())).filter(Number.isFinite);
}

function buildDraft(params: Record<string, unknown>): DatasetWizardDraft {
  return {
    datasetMode: String(params.datasetMode ?? DEFAULT_DRAFT.datasetMode),
    datasetName: String(params.datasetName ?? DEFAULT_DRAFT.datasetName),
    trainSplit: asBoolean(params.trainSplit, DEFAULT_DRAFT.trainSplit),
    rootPath: String(params.rootPath ?? DEFAULT_DRAFT.rootPath),
    splitMode: String(params.splitMode ?? DEFAULT_DRAFT.splitMode),
    trainRatio: asNumber(params.trainRatio, DEFAULT_DRAFT.trainRatio),
    valRatio: asNumber(params.valRatio, DEFAULT_DRAFT.valRatio),
    testRatio: asNumber(params.testRatio, DEFAULT_DRAFT.testRatio),
    shuffleBeforeSplit: asBoolean(params.shuffleBeforeSplit, DEFAULT_DRAFT.shuffleBeforeSplit),
    csvPath: String(params.csvPath ?? DEFAULT_DRAFT.csvPath),
    pathColumn: String(params.pathColumn ?? DEFAULT_DRAFT.pathColumn),
    labelColumn: String(params.labelColumn ?? DEFAULT_DRAFT.labelColumn),
    featureColumns: asStringArray(params.featureColumns),
    taskType: String(params.taskType ?? DEFAULT_DRAFT.taskType),
    imageSize: asNumber(params.imageSize, DEFAULT_DRAFT.imageSize),
    colorMode: String(params.colorMode ?? DEFAULT_DRAFT.colorMode),
    normalize: asBoolean(params.normalize, DEFAULT_DRAFT.normalize),
    mean: asNumberArray(params.mean, DEFAULT_DRAFT.mean),
    std: asNumberArray(params.std, DEFAULT_DRAFT.std),
    augmentationEnabled: asBoolean(params.augmentationEnabled, DEFAULT_DRAFT.augmentationEnabled),
    numClasses: asNumber(params.numClasses, DEFAULT_DRAFT.numClasses),
  };
}

function formatSplitSummary(splits: Record<string, number>): string {
  return ['train', 'val', 'test'].map((key) => `${key}: ${splits[key] ?? 0}`).join(' | ');
}

export function DatasetWizard() {
  const { language, t } = useLanguage();
  const isOpen = useAppStore((state) => state.isDatasetWizardOpen);
  const closeDatasetWizard = useAppStore((state) => state.closeDatasetWizard);
  const selectedNodeId = useAppStore((state) => state.selectedNodeId);
  const project = useAppStore((state) => state.project);
  const updateNodeParams = useAppStore((state) => state.updateNodeParams);
  const applyAutoFixActions = useAppStore((state) => state.applyAutoFixActions);
  const setSelectedNodeId = useAppStore((state) => state.setSelectedNodeId);
  const [activeStep, setActiveStep] = useState<DatasetWizardStep>('source');
  const [draft, setDraft] = useState<DatasetWizardDraft>(DEFAULT_DRAFT);
  const [preview, setPreview] = useState<InspectDatasetResponse | null>(null);
  const [isInspecting, setIsInspecting] = useState(false);

  const datasetNode = useMemo(() => {
    const selectedNode = project.nodes.find((node) => node.id === selectedNodeId);
    if (selectedNode?.type === 'Dataset') {
      return selectedNode;
    }
    return project.nodes.find((node) => node.type === 'Dataset') ?? null;
  }, [project.nodes, selectedNodeId]);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    setActiveStep('source');
    setPreview(null);
    setDraft(buildDraft(datasetNode?.data.params ?? {}));
  }, [datasetNode?.id, datasetNode?.data.params, isOpen]);

  const autoFixSuggestions = useMemo(
    () => buildAutoFixSuggestions(project, null, preview),
    [preview, project],
  );

  if (!isOpen) {
    return null;
  }

  const updateDraft = (patch: Partial<DatasetWizardDraft>) => {
    setDraft((current) => ({ ...current, ...patch }));
    setPreview(null);
  };

  const handleInspect = async () => {
    setIsInspecting(true);
    try {
      setPreview(await inspectDataset(draft as unknown as Record<string, unknown>));
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown dataset inspection error';
      setPreview({
        success: false,
        datasetMode: draft.datasetMode,
        resolvedSplitMode: draft.splitMode,
        taskType: draft.taskType,
        sampleCount: 0,
        numClasses: 0,
        classNames: [],
        splits: { train: 0, val: 0, test: 0 },
        inputShape: null,
        warnings: [],
        errors: [message],
      });
    } finally {
      setIsInspecting(false);
    }
  };

  const handleApply = () => {
    if (!datasetNode) {
      return;
    }
    updateNodeParams(datasetNode.id, draft as unknown as Record<string, unknown>);
    setSelectedNodeId(datasetNode.id);
    closeDatasetWizard();
  };

  return (
    <div className="dataset-wizard-backdrop" role="presentation">
      <section className="dataset-wizard" role="dialog" aria-modal="true" aria-labelledby="dataset-wizard-title">
        <div className="dataset-wizard-header">
          <div>
            <h2 id="dataset-wizard-title">{t('wizard.title')}</h2>
            <p>{t('wizard.subtitle')}</p>
          </div>
          <button className="dataset-wizard-close" onClick={closeDatasetWizard} aria-label={t('wizard.close')}>
            x
          </button>
        </div>

        {!datasetNode ? (
          <div className="dataset-wizard-empty">
            <strong>{t('wizard.noDataset')}</strong>
            <span>{t('wizard.noDatasetHint')}</span>
          </div>
        ) : (
          <>
            <div className="dataset-wizard-steps">
              {WIZARD_STEPS.map((step) => (
                <button
                  key={step}
                  className={activeStep === step ? 'active' : ''}
                  onClick={() => setActiveStep(step)}
                >
                  {t(`wizard.step.${step}`)}
                </button>
              ))}
            </div>

            <div className="dataset-wizard-body">
              {activeStep === 'source' ? (
                <div className="dataset-wizard-grid">
                  <label>
                    <span>{t('wizard.datasetMode')}</span>
                    <select value={draft.datasetMode} onChange={(event) => updateDraft({ datasetMode: event.target.value })}>
                      <option value="builtin">builtin</option>
                      <option value="image_folder">image_folder</option>
                      <option value="csv">csv</option>
                    </select>
                  </label>
                  {draft.datasetMode === 'builtin' ? (
                    <>
                      <label>
                        <span>{t('wizard.datasetName')}</span>
                        <select value={draft.datasetName} onChange={(event) => updateDraft({ datasetName: event.target.value })}>
                          <option value="FakeData">FakeData</option>
                          <option value="MNIST">MNIST</option>
                        </select>
                      </label>
                      <label className="dataset-wizard-checkbox">
                        <input type="checkbox" checked={draft.trainSplit} onChange={(event) => updateDraft({ trainSplit: event.target.checked })} />
                        <span>{t('wizard.trainSplit')}</span>
                      </label>
                    </>
                  ) : null}
                  {draft.datasetMode === 'image_folder' ? (
                    <label className="dataset-wizard-wide">
                      <span>{t('wizard.rootPath')}</span>
                      <input value={draft.rootPath} onChange={(event) => updateDraft({ rootPath: event.target.value })} placeholder="E:\\datasets\\cats-vs-dogs" />
                    </label>
                  ) : null}
                  {draft.datasetMode === 'csv' ? (
                    <>
                      <label className="dataset-wizard-wide">
                        <span>{t('wizard.csvPath')}</span>
                        <input value={draft.csvPath} onChange={(event) => updateDraft({ csvPath: event.target.value })} placeholder="E:\\datasets\\labels.csv" />
                      </label>
                      <label>
                        <span>{t('wizard.pathColumn')}</span>
                        <input value={draft.pathColumn} onChange={(event) => updateDraft({ pathColumn: event.target.value })} />
                      </label>
                      <label>
                        <span>{t('wizard.labelColumn')}</span>
                        <input value={draft.labelColumn} onChange={(event) => updateDraft({ labelColumn: event.target.value })} />
                      </label>
                      <label className="dataset-wizard-wide">
                        <span>{t('wizard.featureColumns')}</span>
                        <input value={listToText(draft.featureColumns)} onChange={(event) => updateDraft({ featureColumns: parseStringList(event.target.value) })} />
                      </label>
                    </>
                  ) : null}
                </div>
              ) : null}

              {activeStep === 'preprocess' ? (
                <div className="dataset-wizard-grid">
                  <label>
                    <span>{t('wizard.taskType')}</span>
                    <select value={draft.taskType} onChange={(event) => updateDraft({ taskType: event.target.value })}>
                      <option value="classification">classification</option>
                      <option value="regression">regression</option>
                    </select>
                  </label>
                  <label>
                    <span>{t('wizard.imageSize')}</span>
                    <input type="number" value={draft.imageSize} onChange={(event) => updateDraft({ imageSize: Number(event.target.value) })} />
                  </label>
                  <label>
                    <span>{t('wizard.colorMode')}</span>
                    <select value={draft.colorMode} onChange={(event) => updateDraft({ colorMode: event.target.value })}>
                      <option value="grayscale">grayscale</option>
                      <option value="rgb">rgb</option>
                    </select>
                  </label>
                  <label>
                    <span>{t('wizard.numClasses')}</span>
                    <input type="number" value={draft.numClasses} onChange={(event) => updateDraft({ numClasses: Number(event.target.value) })} />
                  </label>
                  <label className="dataset-wizard-checkbox">
                    <input type="checkbox" checked={draft.normalize} onChange={(event) => updateDraft({ normalize: event.target.checked })} />
                    <span>{t('wizard.normalize')}</span>
                  </label>
                  <label className="dataset-wizard-checkbox">
                    <input type="checkbox" checked={draft.augmentationEnabled} onChange={(event) => updateDraft({ augmentationEnabled: event.target.checked })} />
                    <span>{t('wizard.augmentation')}</span>
                  </label>
                  {draft.normalize ? (
                    <>
                      <label>
                        <span>{t('wizard.mean')}</span>
                        <input value={listToText(draft.mean)} onChange={(event) => updateDraft({ mean: parseNumberList(event.target.value) })} />
                      </label>
                      <label>
                        <span>{t('wizard.std')}</span>
                        <input value={listToText(draft.std)} onChange={(event) => updateDraft({ std: parseNumberList(event.target.value) })} />
                      </label>
                    </>
                  ) : null}
                </div>
              ) : null}

              {activeStep === 'split' ? (
                <div className="dataset-wizard-grid">
                  {draft.datasetMode === 'image_folder' ? (
                    <>
                      <label>
                        <span>{t('wizard.splitMode')}</span>
                        <select value={draft.splitMode} onChange={(event) => updateDraft({ splitMode: event.target.value })}>
                          <option value="predefined">predefined</option>
                          <option value="ratio">ratio</option>
                        </select>
                      </label>
                      {draft.splitMode === 'ratio' ? (
                        <>
                          <label>
                            <span>{t('wizard.trainRatio')}</span>
                            <input type="number" step="0.05" value={draft.trainRatio} onChange={(event) => updateDraft({ trainRatio: Number(event.target.value) })} />
                          </label>
                          <label>
                            <span>{t('wizard.valRatio')}</span>
                            <input type="number" step="0.05" value={draft.valRatio} onChange={(event) => updateDraft({ valRatio: Number(event.target.value) })} />
                          </label>
                          <label>
                            <span>{t('wizard.testRatio')}</span>
                            <input type="number" step="0.05" value={draft.testRatio} onChange={(event) => updateDraft({ testRatio: Number(event.target.value) })} />
                          </label>
                          <label className="dataset-wizard-checkbox">
                            <input type="checkbox" checked={draft.shuffleBeforeSplit} onChange={(event) => updateDraft({ shuffleBeforeSplit: event.target.checked })} />
                            <span>{t('wizard.shuffleBeforeSplit')}</span>
                          </label>
                        </>
                      ) : null}
                    </>
                  ) : (
                    <div className="dataset-wizard-note">{t('wizard.splitBuiltinNote')}</div>
                  )}
                </div>
              ) : null}

              {activeStep === 'preview' ? (
                <div className="dataset-wizard-preview">
                  <button className="dataset-wizard-inspect" onClick={handleInspect} disabled={isInspecting}>
                    {isInspecting ? t('wizard.inspecting') : t('wizard.inspect')}
                  </button>
                  {preview ? (
                    <div className="dataset-wizard-preview-grid">
                      <div><span>{t('inspector.mode')}</span><strong>{preview.datasetMode}</strong></div>
                      <div><span>{t('inspector.samples')}</span><strong>{preview.sampleCount}</strong></div>
                      <div><span>{t('inspector.classes')}</span><strong>{preview.numClasses}</strong></div>
                      <div><span>{t('inspector.input')}</span><strong>{preview.inputShape ? JSON.stringify(preview.inputShape) : '--'}</strong></div>
                      <div className="dataset-wizard-wide"><span>{t('inspector.splits')}</span><strong>{formatSplitSummary(preview.splits)}</strong></div>
                    </div>
                  ) : (
                    <div className="dataset-wizard-note">{t('wizard.previewEmpty')}</div>
                  )}
                  {preview?.errors.map((error) => (
                    <div key={error} className="inspector-notice inspector-notice--error">
                      <span className="inspector-notice-label">{t('inspector.error')}</span>
                      <span>{translateKnownMessage(error, language)}</span>
                    </div>
                  ))}
                  {preview?.warnings.map((warning) => (
                    <div key={warning} className="inspector-notice inspector-notice--warning">
                      <span className="inspector-notice-label">{t('inspector.warning')}</span>
                      <span>{translateKnownMessage(warning, language)}</span>
                    </div>
                  ))}
                  {preview?.success ? (
                    <AutoFixPanel suggestions={autoFixSuggestions} onApply={applyAutoFixActions} compact />
                  ) : null}
                </div>
              ) : null}
            </div>

            <div className="dataset-wizard-footer">
              <button onClick={() => setActiveStep(WIZARD_STEPS[Math.max(WIZARD_STEPS.indexOf(activeStep) - 1, 0)])}>
                {t('wizard.back')}
              </button>
              <button onClick={() => setActiveStep(WIZARD_STEPS[Math.min(WIZARD_STEPS.indexOf(activeStep) + 1, WIZARD_STEPS.length - 1)])}>
                {t('wizard.next')}
              </button>
              <button className="dataset-wizard-apply" onClick={handleApply}>
                {t('wizard.apply')}
              </button>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
