import React from 'react';

import { useLanguage } from '../../hooks/useLanguage';
import { getNodeDescription } from '../../i18n';
import { getNodesByCategory } from '../../registry';

const Palette: React.FC = () => {
  const grouped = getNodesByCategory();
  const { language, t } = useLanguage();

  const onDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData('application/vmb-node-type', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <aside className="palette">
      <h2 className="palette-title">{t('palette.title')}</h2>
      {Object.entries(grouped).map(([category, behaviors]) => (
        <div key={category} className="palette-category">
          <h3 className="palette-category-title">{t(`palette.category.${category}`)}</h3>
          {category === 'train' ? (
            <div className="palette-guide">
              <div className="palette-guide-title">{t('palette.buildOrder')}</div>
              <div className="palette-guide-step">{t('palette.step.dataset')}</div>
              <div className="palette-guide-step">{t('palette.step.input')}</div>
              <div className="palette-guide-step">{t('palette.step.loss')}</div>
              <div className="palette-guide-step">{t('palette.step.trainer')}</div>
            </div>
          ) : null}
          <div className="palette-nodes">
            {behaviors.map((behavior) => (
              <div
                key={behavior.template.type}
                className={`palette-node palette-node--${category}`}
                draggable
                onDragStart={(event) => onDragStart(event, behavior.template.type)}
                title={getNodeDescription(behavior.template.type, behavior.template.description, language)}
              >
                <span className="palette-node-icon" aria-hidden="true" />
                <span>{behavior.template.displayName}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </aside>
  );
};

export default Palette;
