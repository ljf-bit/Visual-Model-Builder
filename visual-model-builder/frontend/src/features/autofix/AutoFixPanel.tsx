import { useMemo, useState } from 'react';

import { useLanguage } from '../../hooks/useLanguage';
import type { AutoFixAction, AutoFixSuggestion } from '../../types';

type AutoFixPanelProps = {
  suggestions: AutoFixSuggestion[];
  onApply: (actions: AutoFixAction[]) => void;
  compact?: boolean;
};

export function AutoFixPanel({ suggestions, onApply, compact = false }: AutoFixPanelProps) {
  const { t } = useLanguage();
  const [previewId, setPreviewId] = useState<string | null>(null);
  const visibleSuggestions = compact ? suggestions.slice(0, 3) : suggestions;
  const safeActions = useMemo(
    () => suggestions.filter((suggestion) => suggestion.safe).flatMap((suggestion) => suggestion.actions),
    [suggestions],
  );

  if (suggestions.length === 0) {
    return (
      <div className="autofix-panel autofix-panel-empty">
        <div className="autofix-panel-title">{t('autofix.title')}</div>
        <div className="autofix-panel-subtitle">{t('autofix.empty')}</div>
      </div>
    );
  }

  return (
    <div className="autofix-panel">
      <div className="autofix-panel-header">
        <div>
          <div className="autofix-panel-title">{t('autofix.title')}</div>
          <div className="autofix-panel-subtitle">{t('autofix.subtitle', { count: suggestions.length })}</div>
        </div>
        <button
          className="autofix-apply-all"
          onClick={() => onApply(safeActions)}
          disabled={safeActions.length === 0}
        >
          {t('autofix.applyAll')}
        </button>
      </div>

      <div className="autofix-list">
        {visibleSuggestions.map((suggestion) => (
          <div key={suggestion.id} className={`autofix-card autofix-card--${suggestion.category}`}>
            <div className="autofix-card-main">
              <span className="autofix-card-badge">{suggestion.safe ? t('autofix.safe') : t('autofix.review')}</span>
              <div>
                <div className="autofix-card-title">{suggestion.title}</div>
                <div className="autofix-card-description">{suggestion.description}</div>
              </div>
            </div>
            <div className="autofix-card-actions">
              <button onClick={() => setPreviewId(previewId === suggestion.id ? null : suggestion.id)}>
                {previewId === suggestion.id ? t('autofix.hidePreview') : t('autofix.preview')}
              </button>
              <button onClick={() => onApply(suggestion.actions)}>{t('autofix.applyFix')}</button>
            </div>
            {previewId === suggestion.id ? (
              <div className="autofix-preview">
                {suggestion.previewLines.map((line) => (
                  <code key={line}>{line}</code>
                ))}
              </div>
            ) : null}
          </div>
        ))}
      </div>

      {compact && suggestions.length > visibleSuggestions.length ? (
        <div className="autofix-panel-more">
          {t('autofix.more', { count: suggestions.length - visibleSuggestions.length })}
        </div>
      ) : null}
    </div>
  );
}
