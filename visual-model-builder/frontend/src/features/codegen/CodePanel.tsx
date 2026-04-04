import React, { useEffect, useState } from 'react';

import { hasTrainingNodes } from '../../graph/graphUtils';
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
          <h3 className="code-panel-title">Generated Code</h3>
          <div className="code-panel-subtitle">
            {codeMode === 'model'
              ? 'Model code ignores training nodes and focuses on the layer graph.'
              : 'Training code includes dataset, optimizer, trainer, and logging.'}
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            className={`code-panel-mode-btn ${codeMode === 'model' ? 'active' : ''}`}
            onClick={() => setCodeMode('model')}
          >
            Model
          </button>
          <button
            className={`code-panel-mode-btn ${codeMode === 'training' ? 'active' : ''}`}
            onClick={() => setCodeMode('training')}
            disabled={!trainingMode}
          >
            Training
          </button>
          <button
            className="code-panel-copy-btn"
            onClick={handleGenerate}
            disabled={isGenerating}
            style={{ backgroundColor: 'var(--color-bg-hover)' }}
          >
            {isGenerating ? 'Generating...' : codeMode === 'training' ? 'Generate Training Code' : 'Generate Model Code'}
          </button>
          <button
            className="code-panel-copy-btn"
            onClick={handleCopy}
            disabled={!generatedCode || generatedCode.startsWith('//')}
          >
            Copy Code
          </button>
        </div>
      </div>
      <div className="code-panel-content" tabIndex={0}>
        <pre className="code-panel-pre">
          <code>
            {generatedCode ||
              `// Build a ${codeMode === 'training' ? 'training graph' : 'model graph'}, then generate code here.`}
          </code>
        </pre>
      </div>
    </div>
  );
};

export default CodePanel;
