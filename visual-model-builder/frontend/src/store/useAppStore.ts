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
import { applyAutoFixActionsToProject } from '../graph/autoFixActions';
import { getNextNodeId, normalizeProjectGraph } from '../graph/graphUtils';
import type {
  AutoFixAction,
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

  /** Whether the dataset setup wizard is visible */
  isDatasetWizardOpen: boolean;

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

  /** Open the dataset wizard */
  openDatasetWizard: () => void;

  /** Close the dataset wizard */
  closeDatasetWizard: () => void;

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

  /** Apply one or more deterministic graph repair actions */
  applyAutoFixActions: (actions: AutoFixAction[]) => void;

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

function createStarterProject(): ProjectGraph {
  const now = new Date().toISOString();
  const nodes: GraphNode[] = [
    {
      id: 'Dataset_1',
      type: 'Dataset',
      position: { x: -20, y: 260 },
      data: {
        label: 'Dataset',
        params: {
          datasetMode: 'builtin',
          datasetName: 'FakeData',
          trainSplit: true,
          rootPath: '',
          splitMode: 'predefined',
          trainRatio: 0.7,
          valRatio: 0.2,
          testRatio: 0.1,
          shuffleBeforeSplit: true,
          imageSize: 28,
          colorMode: 'grayscale',
          normalize: false,
          mean: [0.5],
          std: [0.5],
          augmentationEnabled: false,
          csvPath: '',
          labelColumn: 'label',
          pathColumn: 'image_path',
          featureColumns: [],
          taskType: 'classification',
          numClasses: 10,
        },
      },
    },
    {
      id: 'DataLoader_1',
      type: 'DataLoader',
      position: { x: 210, y: 260 },
      data: {
        label: 'DataLoader',
        params: {
          batchSize: 32,
          shuffle: true,
          numWorkers: 0,
          dropLast: false,
          pinMemory: false,
          persistentWorkers: false,
          prefetchFactor: 2,
          collateFnType: 'default',
        },
      },
    },
    {
      id: 'Input_1',
      type: 'Input',
      position: { x: 40, y: 40 },
      data: { label: 'Input', params: { inputShape: [1, 28, 28] } },
    },
    {
      id: 'Conv2d_1',
      type: 'Conv2d',
      position: { x: 270, y: 40 },
      data: {
        label: 'Conv2d',
        params: { in_channels: 1, out_channels: 16, kernel_size: 3, stride: 1, padding: 1 },
      },
    },
    {
      id: 'ReLU_1',
      type: 'ReLU',
      position: { x: 500, y: 40 },
      data: { label: 'ReLU', params: { inplace: false } },
    },
    {
      id: 'MaxPool2d_1',
      type: 'MaxPool2d',
      position: { x: 730, y: 40 },
      data: { label: 'MaxPool2d', params: { kernel_size: 2, stride: 2, padding: 0 } },
    },
    {
      id: 'Flatten_1',
      type: 'Flatten',
      position: { x: 960, y: 40 },
      data: { label: 'Flatten', params: { start_dim: 0, end_dim: -1 } },
    },
    {
      id: 'Linear_1',
      type: 'Linear',
      position: { x: 1190, y: 40 },
      data: { label: 'Linear', params: { in_features: 3136, out_features: 10, bias: true } },
    },
    {
      id: 'Output_1',
      type: 'Output',
      position: { x: 1420, y: 40 },
      data: { label: 'Output', params: {} },
    },
    {
      id: 'Loss_1',
      type: 'Loss',
      position: { x: 1650, y: 40 },
      data: { label: 'Loss', params: { lossType: 'CrossEntropyLoss' } },
    },
    {
      id: 'Optimizer_1',
      type: 'Optimizer',
      position: { x: 1420, y: 260 },
      data: { label: 'Optimizer', params: { optimizerType: 'Adam', lr: 0.001, weightDecay: 0, momentum: 0.9 } },
    },
    {
      id: 'Metric_1',
      type: 'Metric',
      position: { x: 1650, y: 260 },
      data: { label: 'Metric', params: { metricType: 'Accuracy' } },
    },
    {
      id: 'Trainer_1',
      type: 'Trainer',
      position: { x: 1880, y: 150 },
      data: { label: 'Trainer', params: { epochs: 2, device: 'cpu', logInterval: 1, validateEveryEpoch: false } },
    },
  ];

  return {
    version: '2.0.0',
    metadata: {
      name: 'Portfolio CNN Training Demo',
      createdAt: now,
      updatedAt: now,
    },
    nodes,
    edges: [
      { id: 'e1', source: 'Dataset_1', target: 'DataLoader_1' },
      { id: 'e2', source: 'DataLoader_1', target: 'Input_1' },
      { id: 'e3', source: 'Input_1', target: 'Conv2d_1' },
      { id: 'e4', source: 'Conv2d_1', target: 'ReLU_1' },
      { id: 'e5', source: 'ReLU_1', target: 'MaxPool2d_1' },
      { id: 'e6', source: 'MaxPool2d_1', target: 'Flatten_1' },
      { id: 'e7', source: 'Flatten_1', target: 'Linear_1' },
      { id: 'e8', source: 'Linear_1', target: 'Output_1' },
      { id: 'e9', source: 'Output_1', target: 'Loss_1' },
      { id: 'e10', source: 'Loss_1', target: 'Trainer_1' },
      { id: 'e11', source: 'Optimizer_1', target: 'Trainer_1' },
      { id: 'e12', source: 'Metric_1', target: 'Trainer_1' },
    ],
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
  project: createStarterProject(),
  selectedNodeId: null,
  copiedNode: null,
  clipboardPasteCount: 0,
  generatedCodeByMode: createEmptyGeneratedCode(),
  globalErrors: [],
  trainingResult: null,
  trainingDiagnostics: null,
  isDirty: false,
  isDatasetWizardOpen: false,
  historyPast: [],
  historyFuture: [],

  setProject: (project) =>
    set(() => {
      const normalizedProject = normalizeProjectGraph(project) ?? createEmptyProject(project.metadata.name);

      return {
        project: {
          ...normalizedProject,
          nodes: sanitizeNodes(normalizedProject.nodes),
          edges: sanitizeEdges(normalizedProject.edges),
        },
        isDirty: false,
        selectedNodeId: null,
        copiedNode: null,
        clipboardPasteCount: 0,
        generatedCodeByMode: createEmptyGeneratedCode(),
        globalErrors: [],
        trainingResult: null,
        trainingDiagnostics: null,
        isDatasetWizardOpen: false,
        historyPast: [],
        historyFuture: [],
      };
    }),

  addNode: (node) =>
    set((s) => {
      const nodeId = s.project.nodes.some((existingNode) => existingNode.id === node.id)
        ? getNextNodeId(node.type, s.project.nodes)
        : node.id;
      const nextNode = sanitizeNode({
        ...node,
        id: nodeId,
      });

      return createProjectChangeState(s, {
        ...s.project,
        nodes: [...s.project.nodes, nextNode],
        metadata: { ...s.project.metadata, updatedAt: new Date().toISOString() },
      });
    }),

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

  openDatasetWizard: () => set({ isDatasetWizardOpen: true }),

  closeDatasetWizard: () => set({ isDatasetWizardOpen: false }),

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
      id: getNextNodeId(state.copiedNode.type, state.project.nodes),
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

  applyAutoFixActions: (actions) =>
    set((s) => {
      if (actions.length === 0) {
        return {};
      }

      return createProjectChangeState(
        s,
        applyAutoFixActionsToProject(s.project, actions),
        {
          selectedNodeId: s.selectedNodeId,
          globalErrors: [],
        },
      );
    }),

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
    isDatasetWizardOpen: false,
    isDirty: false,
    historyPast: [],
    historyFuture: [],
  }),
}));
