import React from 'react';

import { getNodeBehavior } from '../../registry';
import { useAppStore } from '../../store';

const Inspector: React.FC = () => {
  const selectedNodeId = useAppStore((state) => state.selectedNodeId);
  const nodes = useAppStore((state) => state.project.nodes);
  const updateNodeParams = useAppStore((state) => state.updateNodeParams);

  const selectedNode = nodes.find((node) => node.id === selectedNodeId);
  const behavior = selectedNode ? getNodeBehavior(selectedNode.type) : undefined;

  if (!selectedNode || !behavior) {
    return (
      <aside className="inspector">
        <h2 className="inspector-title">Inspector</h2>
        <p className="inspector-empty">Select a node to inspect its parameters.</p>
      </aside>
    );
  }

  const handleParamChange = (key: string, value: unknown) => {
    updateNodeParams(selectedNode.id, { [key]: value });
  };

  return (
    <aside className="inspector">
      <h2 className="inspector-title">Inspector</h2>
      <div className="inspector-node-header">
        <span className="inspector-node-type">{behavior.template.displayName}</span>
        <span className="inspector-node-id">{selectedNode.id}</span>
      </div>

      <div className="inspector-shapes">
        <div className="inspector-shape">
          <span>Input Shape:</span>
          <code>{selectedNode.data.inferredInputShape ? JSON.stringify(selectedNode.data.inferredInputShape) : '--'}</code>
        </div>
        <div className="inspector-shape">
          <span>Output Shape:</span>
          <code>{selectedNode.data.inferredOutputShape ? JSON.stringify(selectedNode.data.inferredOutputShape) : '--'}</code>
        </div>
      </div>

      {selectedNode.data.errors && selectedNode.data.errors.length > 0 ? (
        <div className="inspector-errors">
          {selectedNode.data.errors.map((error, index) => (
            <div key={`${selectedNode.id}-${index}`} className="inspector-error">
              {error}
            </div>
          ))}
        </div>
      ) : null}

      <div className="inspector-params">
        {behavior.template.params.map((param) => (
          <div key={param.key} className="inspector-param">
            <label className="inspector-param-label" title={param.helpText}>
              {param.label}
            </label>
            {param.type === 'bool' ? (
              <input
                type="checkbox"
                checked={Boolean(selectedNode.data.params[param.key])}
                onChange={(event) => handleParamChange(param.key, event.target.checked)}
              />
            ) : param.type === 'select' ? (
              <select
                value={String(selectedNode.data.params[param.key] ?? param.defaultValue)}
                onChange={(event) => handleParamChange(param.key, event.target.value)}
              >
                {param.options?.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            ) : param.type === 'shape' ? (
              <input
                type="text"
                value={JSON.stringify(selectedNode.data.params[param.key] ?? param.defaultValue)}
                onChange={(event) => {
                  try {
                    handleParamChange(param.key, JSON.parse(event.target.value));
                  } catch {
                    // Ignore invalid shape input until it becomes valid JSON.
                  }
                }}
              />
            ) : (
              <input
                type="number"
                step={param.type === 'float' ? '0.001' : '1'}
                value={Number(selectedNode.data.params[param.key] ?? param.defaultValue)}
                onChange={(event) => handleParamChange(param.key, Number(event.target.value))}
              />
            )}
          </div>
        ))}
      </div>
    </aside>
  );
};

export default Inspector;
