/**
 * Application state store using Zustand.
 *
 * Manages:
 *  - Project graph (nodes, edges, metadata)
 *  - Selection state
 *  - Generated code
 *  - Global errors
 *  - Dirty flag for unsaved changes
 */

import { create } from 'zustand';
import type {
  GeneratedCodeMode,
  GraphEdge,
  GraphNode,
  ProjectGraph,
  RunTrainingResponse,
  TrainingDiagnosticsResponse,
} from '../types';

// ============================================================
// State Shape
// ============================================================

export interface AppState {
  /** Current project data */
  project: ProjectGraph;

  /** Currently selected node ID */
  selectedNodeId: string | null;

  /** Internal clipboard for node copy/paste */
  copiedNode: GraphNode | null;

  /** Number of times the current clipboard payload has been pasted */
  clipboardPasteCount: number;

  /** Generated PyTorch code from backend, separated by mode */
  generatedCodeByMode: Record<GeneratedCodeMode, string>;

  /** Global-level validation errors */
  globalErrors: string[];

  /** Latest training execution result */
  trainingResult: RunTrainingResponse | null;

  /** Latest training diagnostics result */
  trainingDiagnostics: TrainingDiagnosticsResponse | null;

  /** Whether the project has unsaved changes */
  isDirty: boolean;

  /** Undo history for graph edits */
  historyPast: ProjectGraph[];

  /** Future history reserved for follow-up editing flows */
  historyFuture: ProjectGraph[];

  // --- Actions ---

  /** Replace the entire project */
  setProject: (project: ProjectGraph) => void;

  /** Add a node to the graph */
  addNode: (node: GraphNode) => void;

  /** Remove a node (and its connected edges) */
  removeNode: (nodeId: string) => void;

  /** Update a node's params */
  updateNodeParams: (nodeId: string, params: Record<string, unknown>) => void;

  /** Add an edge */
  addEdge: (edge: GraphEdge) => void;

  /** Remove an edge */
  removeEdge: (edgeId: string) => void;

  /** Set nodes (bulk replacement, for React Flow sync) */
  setNodes: (nodes: GraphNode[]) => void;

  /** Set edges (bulk replacement, for React Flow sync) */
  setEdges: (edges: GraphEdge[]) => void;

  /** Sync React Flow internal node state without creating an undo step */
  syncNodesFromCanvas: (nodes: GraphNode[]) => void;

  /** Sync React Flow internal edge state without creating an undo step */
  syncEdgesFromCanvas: (edges: GraphEdge[]) => void;

  /** Select a node */
  setSelectedNodeId: (nodeId: string | null) => void;

  /** Copy the currently selected node into the internal clipboard */
  copySelectedNode: () => boolean;

  /** Paste the copied node with an offset and select the new node */
  pasteCopiedNode: () => boolean;

  /** Undo the last graph edit */
  undoProjectChange: () => boolean;

  /** Set generated code for one mode */
  setGeneratedCode: (mode: GeneratedCodeMode, code: string) => void;

  /** Set global errors */
  setGlobalErrors: (errors: string[]) => void;

  /** Set latest training result */
  setTrainingResult: (result: RunTrainingResponse | null) => void;

  /** Set latest training diagnostics */
  setTrainingDiagnostics: (diagnostics: TrainingDiagnosticsResponse | null) => void;

  /** Sync backend feedback into nodes and reset dirty flag */
  setRemoteFeedback: (valRes: import('../types').ValidateGraphResponse, shapeRes: import('../types').InferShapesResponse) => void;

  /** Mark project as saved */
  markSaved: () => void;

  /** Reset to a new empty project */
  resetProject: (name?: string) => void;
}

// ============================================================
// Default Project Factory
// ============================================================

function createEmptyProject(name = 'Untitled Project'): ProjectGraph {
  const now = new Date().toISOString();
  return {
    version: '2.0.0',
    metadata: {
      name,
      createdAt: now,
      updatedAt: now,
    },
    nodes: [],
    edges: [],
  };
}

function createEmptyGeneratedCode(): Record<GeneratedCodeMode, string> {
  return {
    model: '',
    training: '',
  };
}

const HISTORY_LIMIT = 50;
const PASTE_OFFSET = 40;
const NODE_COLLISION_WIDTH = 220;
const NODE_COLLISION_HEIGHT = 96;
const NODE_COLLISION_GAP = 24;

function cloneProject(project: ProjectGraph): ProjectGraph {
  return JSON.parse(JSON.stringify(project)) as ProjectGraph;
}

function cloneNode(node: GraphNode): GraphNode {
  return JSON.parse(JSON.stringify(node)) as GraphNode;
}

function sanitizeNode(node: GraphNode): GraphNode {
  return {
    id: node.id,
    type: node.type,
    position: {
      x: node.position.x,
      y: node.position.y,
    },
    data: {
      ...node.data,
    },
  };
}

function sanitizeEdge(edge: GraphEdge): GraphEdge {
  return {
    id: edge.id,
    source: edge.source,
    target: edge.target,
    sourceHandle: edge.sourceHandle,
    targetHandle: edge.targetHandle,
  };
}

function sanitizeNodes(nodes: GraphNode[]): GraphNode[] {
  return nodes.map(sanitizeNode);
}

function sanitizeEdges(edges: GraphEdge[]): GraphEdge[] {
  return edges.map(sanitizeEdge);
}

function pushHistory(historyPast: ProjectGraph[], currentProject: ProjectGraph): ProjectGraph[] {
  const nextHistory = [...historyPast, cloneProject(currentProject)];
  return nextHistory.length > HISTORY_LIMIT ? nextHistory.slice(nextHistory.length - HISTORY_LIMIT) : nextHistory;
}

function createProjectChangeState(
  state: AppState,
  nextProject: ProjectGraph,
  overrides: Partial<AppState> = {},
): Partial<AppState> {
  return {
    project: nextProject,
    historyPast: pushHistory(state.historyPast, state.project),
    historyFuture: [],
    generatedCodeByMode: createEmptyGeneratedCode(),
    trainingResult: null,
    trainingDiagnostics: null,
    isDirty: true,
    ...overrides,
  };
}

function generateNodeId(type: string, nodes: GraphNode[]): string {
  const pattern = new RegExp(`^${type}_(\\d+)$`);
  let maxSuffix = 0;

  for (const node of nodes) {
    const match = node.id.match(pattern);
    if (match) {
      maxSuffix = Math.max(maxSuffix, Number(match[1]));
    }
  }

  return `${type}_${maxSuffix + 1}`;
}

function boxesOverlap(a: GraphNode['position'], b: GraphNode['position']): boolean {
  return !(
    a.x + NODE_COLLISION_WIDTH + NODE_COLLISION_GAP <= b.x ||
    b.x + NODE_COLLISION_WIDTH + NODE_COLLISION_GAP <= a.x ||
    a.y + NODE_COLLISION_HEIGHT + NODE_COLLISION_GAP <= b.y ||
    b.y + NODE_COLLISION_HEIGHT + NODE_COLLISION_GAP <= a.y
  );
}

function findNonOverlappingPosition(basePosition: GraphNode['position'], nodes: GraphNode[], pasteIndex: number) {
  let candidate = {
    x: basePosition.x + PASTE_OFFSET * pasteIndex,
    y: basePosition.y + PASTE_OFFSET * pasteIndex,
  };

  for (let step = 0; step < 100; step += 1) {
    const collides = nodes.some((node) => boxesOverlap(candidate, node.position));
    if (!collides) {
      return candidate;
    }

    candidate = {
      x: candidate.x + NODE_COLLISION_WIDTH / 2,
      y: candidate.y + NODE_COLLISION_HEIGHT / 2,
    };
  }

  return candidate;
}

// ============================================================
// Store
// ============================================================

export const useAppStore = create<AppState>((set, get) => ({
  project: createEmptyProject(),
  selectedNodeId: null,
  copiedNode: null,
  clipboardPasteCount: 0,
  generatedCodeByMode: createEmptyGeneratedCode(),
  globalErrors: [],
  trainingResult: null,
  trainingDiagnostics: null,
  isDirty: false,
  historyPast: [],
  historyFuture: [],

  setProject: (project) =>
    set({
      project: {
        ...project,
        nodes: sanitizeNodes(project.nodes),
        edges: sanitizeEdges(project.edges),
      },
      isDirty: false,
      selectedNodeId: null,
      copiedNode: null,
      clipboardPasteCount: 0,
      generatedCodeByMode: createEmptyGeneratedCode(),
      globalErrors: [],
      trainingResult: null,
      trainingDiagnostics: null,
      historyPast: [],
      historyFuture: [],
    }),

  addNode: (node) =>
    set((s) =>
      createProjectChangeState(s, {
        ...s.project,
        nodes: [...s.project.nodes, node],
        metadata: { ...s.project.metadata, updatedAt: new Date().toISOString() },
      }),
    ),

  removeNode: (nodeId) =>
    set((s) =>
      createProjectChangeState(
        s,
        {
          ...s.project,
          nodes: s.project.nodes.filter((n) => n.id !== nodeId),
          edges: s.project.edges.filter((e) => e.source !== nodeId && e.target !== nodeId),
          metadata: { ...s.project.metadata, updatedAt: new Date().toISOString() },
        },
        {
          selectedNodeId: s.selectedNodeId === nodeId ? null : s.selectedNodeId,
        },
      ),
    ),

  updateNodeParams: (nodeId, params) =>
    set((s) =>
      createProjectChangeState(s, {
        ...s.project,
        nodes: s.project.nodes.map((n) =>
          n.id === nodeId ? { ...n, data: { ...n.data, params: { ...n.data.params, ...params } } } : n,
        ),
        metadata: { ...s.project.metadata, updatedAt: new Date().toISOString() },
      }),
    ),

  addEdge: (edge) =>
    set((s) =>
      createProjectChangeState(s, {
        ...s.project,
        edges: [...s.project.edges, edge],
        metadata: { ...s.project.metadata, updatedAt: new Date().toISOString() },
      }),
    ),

  removeEdge: (edgeId) =>
    set((s) =>
      createProjectChangeState(s, {
        ...s.project,
        edges: s.project.edges.filter((e) => e.id !== edgeId),
        metadata: { ...s.project.metadata, updatedAt: new Date().toISOString() },
      }),
    ),

  setNodes: (nodes) =>
    set((s) =>
      createProjectChangeState(s, {
        ...s.project,
        nodes: sanitizeNodes(nodes),
        metadata: { ...s.project.metadata, updatedAt: new Date().toISOString() },
      }),
    ),

  setEdges: (edges) =>
    set((s) =>
      createProjectChangeState(s, {
        ...s.project,
        edges: sanitizeEdges(edges),
        metadata: { ...s.project.metadata, updatedAt: new Date().toISOString() },
      }),
    ),

  syncNodesFromCanvas: (nodes) =>
    set((s) => ({
      project: {
        ...s.project,
        nodes: sanitizeNodes(nodes),
      },
    })),

  syncEdgesFromCanvas: (edges) =>
    set((s) => ({
      project: {
        ...s.project,
        edges: sanitizeEdges(edges),
      },
    })),

  setSelectedNodeId: (nodeId) => set({ selectedNodeId: nodeId }),

  copySelectedNode: () => {
    const { project, selectedNodeId } = get();
    if (!selectedNodeId) {
      return false;
    }

    const selectedNode = project.nodes.find((node) => node.id === selectedNodeId);
    if (!selectedNode) {
      return false;
    }

    set({
      copiedNode: sanitizeNode(cloneNode(selectedNode)),
      clipboardPasteCount: 0,
    });
    return true;
  },

  pasteCopiedNode: () => {
    const state = get();
    if (!state.copiedNode) {
      return false;
    }

    const pasteIndex = state.clipboardPasteCount + 1;
    const nextPosition = findNonOverlappingPosition(state.copiedNode.position, state.project.nodes, pasteIndex);
    const copiedNodeData = cloneNode(state.copiedNode);
    const newNode: GraphNode = {
      ...copiedNodeData,
      id: generateNodeId(state.copiedNode.type, state.project.nodes),
      position: nextPosition,
      data: {
        ...copiedNodeData.data,
        inferredInputShape: null,
        inferredOutputShape: null,
        errors: [],
      },
    };

    set((s) =>
      createProjectChangeState(
        s,
        {
          ...s.project,
          nodes: [...s.project.nodes, newNode],
          metadata: { ...s.project.metadata, updatedAt: new Date().toISOString() },
        },
        {
          clipboardPasteCount: pasteIndex,
          selectedNodeId: newNode.id,
        },
      ),
    );

    return true;
  },

  undoProjectChange: () => {
    const state = get();
    if (state.historyPast.length === 0) {
      return false;
    }

    const previousProject = cloneProject(state.historyPast[state.historyPast.length - 1]);
    const nextFuture = [cloneProject(state.project), ...state.historyFuture].slice(0, HISTORY_LIMIT);

    set({
      project: previousProject,
      selectedNodeId: null,
      generatedCodeByMode: createEmptyGeneratedCode(),
      trainingResult: null,
      globalErrors: [],
      isDirty: true,
      historyPast: state.historyPast.slice(0, -1),
      historyFuture: nextFuture,
    });

    return true;
  },

  setGeneratedCode: (mode, code) =>
    set((s) => ({
      generatedCodeByMode: {
        ...s.generatedCodeByMode,
        [mode]: code,
      },
    })),

  setGlobalErrors: (errors) => set({ globalErrors: errors }),

  setTrainingResult: (result) => set({ trainingResult: result }),

  setTrainingDiagnostics: (diagnostics) => set({ trainingDiagnostics: diagnostics }),

  setRemoteFeedback: (valRes, shapeRes) => set((s) => {
    const updatedNodes = s.project.nodes.map(node => {
        const shapeData = shapeRes.nodes[node.id];
        const valErrors = valRes.nodeErrors[node.id] || [];
        const shapeErrors = shapeData?.errors || [];
        const allErrors = [...valErrors, ...shapeErrors];

        return {
            ...node,
            data: {
                ...node.data,
                inferredInputShape: shapeData?.inputShape ?? null,
                inferredOutputShape: shapeData?.outputShape ?? null,
                errors: allErrors
            }
        };
    });

    return {
        project: {
            ...s.project,
            nodes: updatedNodes
        },
        globalErrors: valRes.globalErrors,
        trainingResult: s.trainingResult,
        trainingDiagnostics: s.trainingDiagnostics,
        isDirty: false
    };
  }),

  markSaved: () => set({ isDirty: false }),

  resetProject: (name) => set({
    project: createEmptyProject(name),
    selectedNodeId: null,
    copiedNode: null,
    clipboardPasteCount: 0,
    generatedCodeByMode: createEmptyGeneratedCode(),
    globalErrors: [],
    trainingResult: null,
    trainingDiagnostics: null,
    isDirty: false,
    historyPast: [],
    historyFuture: [],
  }),
}));
