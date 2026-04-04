import React, { useCallback, useEffect, useMemo, useState } from 'react';
import type { CSSProperties } from 'react';
import { ReactFlowProvider } from '@xyflow/react';

import { Canvas } from './features/canvas';
import { CodePanel } from './features/codegen';
import { Inspector } from './features/inspector';
import { Palette } from './features/palette';
import { TrainingPanel } from './features/training';
import { hasTrainingNodes, normalizeProjectGraph } from './graph/graphUtils';
import { inferShapes, validateGraph, validateTrainingGraph } from './services';
import { useAppStore } from './store';
import type { ProjectGraph } from './types';

import './App.css';

type SaveFilePickerOptions = {
  suggestedName?: string;
  types?: Array<{
    description: string;
    accept: Record<string, string[]>;
  }>;
};

type SaveFileHandle = {
  createWritable: () => Promise<{
    write: (content: string) => Promise<void>;
    close: () => Promise<void>;
  }>;
};

type SavePickerWindow = Window & {
  showSaveFilePicker?: (options?: SaveFilePickerOptions) => Promise<SaveFileHandle>;
};

function isEditableTarget(target: EventTarget | null): boolean {
  const element = target instanceof HTMLElement ? target : null;
  if (!element) {
    return false;
  }

  if (element.isContentEditable) {
    return true;
  }

  const interactiveAncestor = element.closest('input, textarea, select, [contenteditable="true"]');
  return interactiveAncestor instanceof HTMLElement;
}

function hasTextSelection(): boolean {
  const selection = window.getSelection();
  return Boolean(selection && !selection.isCollapsed && selection.toString().trim().length > 0);
}

function isProjectGraph(value: unknown): value is ProjectGraph {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const candidate = value as Partial<ProjectGraph>;
  return (
    typeof candidate.version === 'string' &&
    Array.isArray(candidate.nodes) &&
    Array.isArray(candidate.edges)
  );
}

const App: React.FC = () => {
  const project = useAppStore((state) => state.project);
  const isDirty = useAppStore((state) => state.isDirty);
  const globalErrors = useAppStore((state) => state.globalErrors);
  const setProject = useAppStore((state) => state.setProject);
  const setRemoteFeedback = useAppStore((state) => state.setRemoteFeedback);
  const markSaved = useAppStore((state) => state.markSaved);
  const [paletteWidth, setPaletteWidth] = useState(240);
  const [inspectorWidth, setInspectorWidth] = useState(300);
  const [bottomPanelHeight, setBottomPanelHeight] = useState(280);
  const [activeResizer, setActiveResizer] = useState<null | {
    type: 'palette' | 'inspector' | 'bottom';
    startX: number;
    startY: number;
    startSize: number;
  }>(null);

  useEffect(() => {
    if (!activeResizer) {
      return;
    }

    const handlePointerMove = (event: PointerEvent) => {
      if (activeResizer.type === 'palette') {
        const delta = event.clientX - activeResizer.startX;
        setPaletteWidth(Math.max(180, Math.min(420, activeResizer.startSize + delta)));
        return;
      }

      if (activeResizer.type === 'inspector') {
        const delta = activeResizer.startX - event.clientX;
        setInspectorWidth(Math.max(220, Math.min(520, activeResizer.startSize + delta)));
        return;
      }

      const delta = activeResizer.startY - event.clientY;
      setBottomPanelHeight(Math.max(180, Math.min(420, activeResizer.startSize + delta)));
    };

    const handlePointerUp = () => {
      setActiveResizer(null);
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);

    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };
  }, [activeResizer]);

  const appLayoutStyle = useMemo<CSSProperties>(
    () =>
      ({
        ['--palette-width' as const]: `${paletteWidth}px`,
        ['--inspector-width' as const]: `${inspectorWidth}px`,
        ['--code-panel-height' as const]: `${bottomPanelHeight}px`,
      }) as CSSProperties,
    [bottomPanelHeight, inspectorWidth, paletteWidth],
  );

  useEffect(() => {
    if (!isDirty) {
      return;
    }

    const timer = window.setTimeout(async () => {
      try {
        const validationRequest = hasTrainingNodes(project.nodes)
          ? validateTrainingGraph(project)
          : validateGraph(project);
        const [validationResult, shapeResult] = await Promise.all([
          validationRequest,
          inferShapes(project),
        ]);
        setRemoteFeedback(validationResult, shapeResult);
      } catch (error) {
        console.error('Failed to sync with backend:', error);
      }
    }, 500);

    return () => window.clearTimeout(timer);
  }, [isDirty, project, setRemoteFeedback]);

  const handleSave = useCallback(async () => {
    try {
      const json = JSON.stringify(project, null, 2);
      const savePickerWindow = window as SavePickerWindow;

      if (savePickerWindow.showSaveFilePicker) {
        const handle = await savePickerWindow.showSaveFilePicker({
          suggestedName: `${project.metadata.name || 'project'}.json`,
          types: [
            {
              description: 'JSON file',
              accept: { 'application/json': ['.json'] },
            },
          ],
        });
        const writable = await handle.createWritable();
        await writable.write(json);
        await writable.close();
      } else {
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = `${project.metadata.name || 'project'}.json`;
        document.body.appendChild(anchor);
        anchor.click();
        document.body.removeChild(anchor);
        URL.revokeObjectURL(url);
      }

      markSaved();
    } catch (error) {
      const saveError = error as Error;
      if (saveError.name !== 'AbortError') {
        alert(`Save failed: ${saveError.message}`);
      }
    }
  }, [markSaved, project]);

  const handleLoad = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (!file) {
        return;
      }

      const reader = new FileReader();
      reader.onload = (loadEvent) => {
        try {
          const result = loadEvent.target?.result;
          const parsed = typeof result === 'string' ? JSON.parse(result) : null;

          if (isProjectGraph(parsed)) {
            const normalizedProject = normalizeProjectGraph(parsed);
            if (normalizedProject) {
              setProject(normalizedProject);
            } else {
              alert('Invalid project file format.');
            }
          } else {
            alert('Invalid project file format.');
          }
        } catch {
          alert('Failed to parse the selected project file.');
        }
      };
      reader.readAsText(file);
    };
    input.click();
  }, [setProject]);

  const handleGlobalShortcuts = useCallback(
    (event: KeyboardEvent) => {
      const isPrimaryModifier = event.ctrlKey || event.metaKey;
      if (!isPrimaryModifier || event.altKey) {
        return;
      }

      const key = event.key.toLowerCase();
      const editableTarget = isEditableTarget(event.target);

      if (key === 's') {
        event.preventDefault();
        void handleSave();
        return;
      }

      if (editableTarget) {
        return;
      }

      const store = useAppStore.getState();

      if (key === 'c') {
        if (hasTextSelection()) {
          return;
        }
        if (store.copySelectedNode()) {
          event.preventDefault();
        }
        return;
      }

      if (key === 'v') {
        if (store.pasteCopiedNode()) {
          event.preventDefault();
        }
        return;
      }

      if (key === 'z' && !event.shiftKey) {
        if (store.undoProjectChange()) {
          event.preventDefault();
        }
      }
    },
    [handleSave],
  );

  useEffect(() => {
    window.addEventListener('keydown', handleGlobalShortcuts);

    return () => {
      window.removeEventListener('keydown', handleGlobalShortcuts);
    };
  }, [handleGlobalShortcuts]);

  return (
    <ReactFlowProvider>
      <div className="app-layout" style={appLayoutStyle}>
        <header className="app-header">
          <div className="app-title">Visual Model Builder</div>
          <div className="app-actions">
            <button className="app-btn" onClick={handleLoad}>
              Open Project
            </button>
            <button className="app-btn" onClick={handleSave}>
              Save Project
            </button>
          </div>
        </header>

        {globalErrors.length > 0 ? (
          <section className="app-global-errors" aria-label="Graph validation errors">
            {globalErrors.map((message) => (
              <div key={message} className="app-global-error">
                {message}
              </div>
            ))}
          </section>
        ) : null}

        <div className="app-body">
          <Palette />
          <div
            className="app-resizer app-resizer-vertical"
            onPointerDown={(event) =>
              setActiveResizer({
                type: 'palette',
                startX: event.clientX,
                startY: event.clientY,
                startSize: paletteWidth,
              })
            }
          />

          <main className="app-main">
            <div className="app-canvas-area">
              <Canvas />
            </div>
            <div
              className="app-resizer app-resizer-horizontal"
              onPointerDown={(event) =>
                setActiveResizer({
                  type: 'bottom',
                  startX: event.clientX,
                  startY: event.clientY,
                  startSize: bottomPanelHeight,
                })
              }
            />
            <div className="app-code-area">
              <CodePanel />
              <TrainingPanel />
            </div>
          </main>

          <div
            className="app-resizer app-resizer-vertical"
            onPointerDown={(event) =>
              setActiveResizer({
                type: 'inspector',
                startX: event.clientX,
                startY: event.clientY,
                startSize: inspectorWidth,
              })
            }
          />
          <Inspector />
        </div>
      </div>
    </ReactFlowProvider>
  );
};

export default App;
