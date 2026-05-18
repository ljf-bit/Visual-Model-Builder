import React, { useEffect, useState } from 'react';

import { hasTrainingNodes } from '../../graph/graphUtils';
import { useLanguage } from '../../hooks/useLanguage';
import { generateCode, generateTrainingCode } from '../../services';
import { useAppStore } from '../../store';
import type { GeneratedCodeMode } from '../../types';

const CodePanel: React.FC = () => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [codeMode, setCodeMode] = useState<GeneratedCodeMode>('model');
  const generatedCodeByMode = useAppStore((state) => state.generatedCodeByMode);
  const setGeneratedCode = useAppStore((state) => state.setGeneratedCode);
  const project = useAppStore((state) => state.project);
  const trainingMode = hasTrainingNodes(project.nodes);
  const generatedCode = generatedCodeByMode[codeMode];
  const { t } = useLanguage();

  useEffect(() => {
    if (!trainingMode && codeMode === 'training') {
      setCodeMode('model');
    }
  }, [codeMode, trainingMode]);

  const handleCopy = () => {
    if (generatedCode) {
      navigator.clipboard.writeText(generatedCode);
    }
  };

  const handleGenerate = async () => {
    try {
      setIsGenerating(true);
      const result =
        codeMode === 'training' && trainingMode
          ? await generateTrainingCode(project)
          : await generateCode(project);
      if (result.ok && result.code) {
        setGeneratedCode(codeMode, result.code);
      } else {
        const errors = result.errors.join('\n') || 'Unknown error';
        setGeneratedCode(codeMode, `// Code generation failed:\n${errors}`);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      setGeneratedCode(codeMode, `// Unable to reach backend service\n${message}`);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="code-panel">
      <div className="code-panel-header">
        <div>
          <h3 className="code-panel-title">{t('code.title')}</h3>
          <div className="code-panel-subtitle">
            {codeMode === 'model'
              ? t('code.modelSubtitle')
              : t('code.trainingSubtitle')}
          </div>
        </div>
        <div className="code-panel-actions">
          <button
            className={`code-panel-mode-btn ${codeMode === 'model' ? 'active' : ''}`}
            onClick={() => setCodeMode('model')}
          >
            {t('code.model')}
          </button>
          <button
            className={`code-panel-mode-btn ${codeMode === 'training' ? 'active' : ''}`}
            onClick={() => setCodeMode('training')}
            disabled={!trainingMode}
          >
            {t('code.training')}
          </button>
          <button
            className="code-panel-generate-btn"
            onClick={handleGenerate}
            disabled={isGenerating}
          >
            {isGenerating ? t('code.generating') : codeMode === 'training' ? t('code.generateTraining') : t('code.generateModel')}
          </button>
          <button
            className="code-panel-copy-btn"
            onClick={handleCopy}
            disabled={!generatedCode || generatedCode.startsWith('//')}
          >
            {t('code.copy')}
          </button>
        </div>
      </div>
      <div className="code-panel-content" tabIndex={0}>
        <pre className="code-panel-pre">
          <code>
            {generatedCode ||
              (codeMode === 'training' ? t('code.emptyTraining') : t('code.emptyModel'))}
          </code>
        </pre>
      </div>
    </div>
  );
};

export default CodePanel;
