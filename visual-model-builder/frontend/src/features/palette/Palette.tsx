import React from 'react';

import { getNodesByCategory } from '../../registry';

const categoryLabels: Record<string, string> = {
  io: 'IO',
  layer: 'Layer',
  activation: 'Activation',
  train: 'Training',
};

const Palette: React.FC = () => {
  const grouped = getNodesByCategory();

  const onDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData('application/vmb-node-type', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <aside className="palette">
      <h2 className="palette-title">Palette</h2>
      {Object.entries(grouped).map(([category, behaviors]) => (
        <div key={category} className="palette-category">
          <h3 className="palette-category-title">{categoryLabels[category] ?? category}</h3>
          {category === 'train' ? (
            <div className="palette-guide">
              <div className="palette-guide-title">Recommended Build Order</div>
              <div className="palette-guide-step">1. `Dataset` → `DataLoader`</div>
              <div className="palette-guide-step">2. `DataLoader` → `Input`</div>
              <div className="palette-guide-step">3. Model `Output` → `Loss`</div>
              <div className="palette-guide-step">4. `Optimizer` + `Loss` + optional `Metric` → `Trainer`</div>
            </div>
          ) : null}
          <div className="palette-nodes">
            {behaviors.map((behavior) => (
              <div
                key={behavior.template.type}
                className="palette-node"
                draggable
                onDragStart={(event) => onDragStart(event, behavior.template.type)}
                title={behavior.template.description}
              >
                {behavior.template.displayName}
              </div>
            ))}
          </div>
        </div>
      ))}
    </aside>
  );
};

export default Palette;
