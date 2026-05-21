import { getNextNodeId, topologicalSort } from '../../graph/graphUtils';
import type {
  AutoFixSuggestion,
  GraphEdge,
  GraphNode,
  InspectDatasetResponse,
  ProjectGraph,
  TrainingDiagnosticsResponse,
} from '../../types';

const MODEL_NODE_TYPES = new Set(['Input', 'Conv2d', 'ReLU', 'MaxPool2d', 'Flatten', 'Linear', 'Output']);

function arraysEqual(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function asNumber(value: unknown, fallback: number): number {
  const next = Number(value);
  return Number.isFinite(next) ? next : fallback;
}

function getNodesByType(project: ProjectGraph, type: string): GraphNode[] {
  return project.nodes.filter((node) => node.type === type);
}

function getFirstNode(project: ProjectGraph, type: string): GraphNode | null {
  return getNodesByType(project, type)[0] ?? null;
}

function getNode(project: ProjectGraph, nodeId: string): GraphNode | null {
  return project.nodes.find((node) => node.id === nodeId) ?? null;
}

function modelOrder(project: ProjectGraph): GraphNode[] {
  const orderedIds = topologicalSort(project.nodes, project.edges);
  const order = orderedIds ?? project.nodes.map((node) => node.id);
  const nodeMap = new Map(project.nodes.map((node) => [node.id, node]));
  return order
    .map((nodeId) => nodeMap.get(nodeId))
    .filter((node): node is GraphNode => Boolean(node && MODEL_NODE_TYPES.has(node.type)));
}

function getLastLinear(project: ProjectGraph): GraphNode | null {
  const linearNodes = modelOrder(project).filter((node) => node.type === 'Linear');
  return linearNodes[linearNodes.length - 1] ?? null;
}

function edgeExists(project: ProjectGraph, source: string, target: string): boolean {
  return project.edges.some((edge) => edge.source === source && edge.target === target);
}

function buildEdge(source: string, target: string): GraphEdge {
  return {
    id: `autofix-${source}-${target}`.replace(/[^A-Za-z0-9_-]+/g, '-'),
    source,
    target,
  };
}

function buildDatasetShape(datasetNode: GraphNode | null, datasetPreview?: InspectDatasetResponse | null): number[] | null {
  if (datasetPreview?.success && datasetPreview.inputShape) {
    return datasetPreview.inputShape;
  }
  if (!datasetNode) {
    return null;
  }
  const params = datasetNode.data.params;
  const imageSize = asNumber(params.imageSize, 28);
  if (imageSize <= 0) {
    return null;
  }
  const channels = String(params.colorMode ?? 'grayscale') === 'rgb' ? 3 : 1;
  return [channels, imageSize, imageSize];
}

function buildDatasetClasses(datasetNode: GraphNode | null, datasetPreview?: InspectDatasetResponse | null): number | null {
  if (datasetPreview?.success && datasetPreview.numClasses > 0) {
    return datasetPreview.numClasses;
  }
  if (!datasetNode) {
    return null;
  }
  const params = datasetNode.data.params;
  if (String(params.datasetName ?? '') === 'MNIST') {
    return 10;
  }
  const numClasses = asNumber(params.numClasses, 0);
  return numClasses > 0 ? numClasses : null;
}

function buildDatasetTask(datasetNode: GraphNode | null, datasetPreview?: InspectDatasetResponse | null): string {
  if (datasetPreview?.taskType) {
    return datasetPreview.taskType;
  }
  return String(datasetNode?.data.params.taskType ?? 'classification');
}

function getIncomingModelEdge(project: ProjectGraph, node: GraphNode): GraphEdge | null {
  return project.edges.find((edge) => {
    if (edge.target !== node.id) {
      return false;
    }
    const sourceNode = getNode(project, edge.source);
    return Boolean(sourceNode && MODEL_NODE_TYPES.has(sourceNode.type));
  }) ?? null;
}

function createFlattenBetween(project: ProjectGraph, sourceNode: GraphNode, targetNode: GraphNode): GraphNode {
  return {
    id: getNextNodeId('Flatten', project.nodes),
    type: 'Flatten',
    position: {
      x: (sourceNode.position.x + targetNode.position.x) / 2,
      y: (sourceNode.position.y + targetNode.position.y) / 2 + 84,
    },
    data: {
      label: 'Flatten',
      type: 'Flatten',
      params: {
        start_dim: 0,
        end_dim: -1,
      },
      inferredInputShape: null,
      inferredOutputShape: null,
      errors: [],
    },
  };
}

function addSuggestion(
  suggestions: AutoFixSuggestion[],
  suggestion: AutoFixSuggestion,
) {
  if (!suggestions.some((item) => item.id === suggestion.id)) {
    suggestions.push(suggestion);
  }
}

export function buildAutoFixSuggestions(
  project: ProjectGraph,
  diagnostics?: TrainingDiagnosticsResponse | null,
  datasetPreview?: InspectDatasetResponse | null,
): AutoFixSuggestion[] {
  const suggestions: AutoFixSuggestion[] = [];
  const datasetNode = getFirstNode(project, 'Dataset');
  const inputNode = getFirstNode(project, 'Input');
  const lossNode = getFirstNode(project, 'Loss');
  const outputNode = getFirstNode(project, 'Output');
  const dataloaderNode = getFirstNode(project, 'DataLoader');
  const optimizerNode = getFirstNode(project, 'Optimizer');
  const metricNode = getFirstNode(project, 'Metric');
  const trainerNode = getFirstNode(project, 'Trainer');
  const datasetShape = buildDatasetShape(datasetNode, datasetPreview);
  const datasetClasses = buildDatasetClasses(datasetNode, datasetPreview);
  const datasetTask = buildDatasetTask(datasetNode, datasetPreview);
  const lastLinear = getLastLinear(project);
  const diagnosticsStats = diagnostics?.trainingStats;

  if (datasetNode && inputNode && datasetShape && !arraysEqual(inputNode.data.params.inputShape, datasetShape)) {
    addSuggestion(suggestions, {
      id: `dataset-input-shape-${inputNode.id}`,
      title: 'Sync Input shape with Dataset',
      description: 'Make the model entry match the image shape resolved by Dataset inspection.',
      category: 'dataset',
      safe: true,
      targetNodeId: inputNode.id,
      actions: [{ kind: 'update_node_params', nodeId: inputNode.id, params: { inputShape: datasetShape } }],
      previewLines: [
        `${inputNode.id}.inputShape: ${JSON.stringify(inputNode.data.params.inputShape)} -> ${JSON.stringify(datasetShape)}`,
      ],
    });
  }

  if (lastLinear && datasetClasses && datasetTask === 'classification' && asNumber(lastLinear.data.params.out_features, 0) !== datasetClasses) {
    addSuggestion(suggestions, {
      id: `linear-output-classes-${lastLinear.id}`,
      title: 'Match classifier outputs to Dataset classes',
      description: 'The final Linear layer should produce one logit per class for classification.',
      category: 'shape',
      safe: true,
      targetNodeId: lastLinear.id,
      actions: [{ kind: 'update_node_params', nodeId: lastLinear.id, params: { out_features: datasetClasses } }],
      previewLines: [
        `${lastLinear.id}.out_features: ${String(lastLinear.data.params.out_features)} -> ${datasetClasses}`,
      ],
    });
  }

  for (const linearNode of getNodesByType(project, 'Linear')) {
    const inputShape = linearNode.data.inferredInputShape;
    const expectedFeatures = Array.isArray(inputShape) ? inputShape[inputShape.length - 1] : null;
    if (expectedFeatures && asNumber(linearNode.data.params.in_features, 0) !== expectedFeatures) {
      addSuggestion(suggestions, {
        id: `linear-in-features-${linearNode.id}`,
        title: 'Match Linear in_features to upstream shape',
        description: 'Use the inferred feature size so the Linear layer receives the tensor it expects.',
        category: 'shape',
        safe: true,
        targetNodeId: linearNode.id,
        actions: [{ kind: 'update_node_params', nodeId: linearNode.id, params: { in_features: expectedFeatures } }],
        previewLines: [
          `${linearNode.id}.in_features: ${String(linearNode.data.params.in_features)} -> ${expectedFeatures}`,
        ],
      });
    }

    if (Array.isArray(inputShape) && inputShape.length > 1) {
      const incomingEdge = getIncomingModelEdge(project, linearNode);
      const sourceNode = incomingEdge ? getNode(project, incomingEdge.source) : null;
      if (incomingEdge && sourceNode && sourceNode.type !== 'Flatten') {
        const flattenNode = createFlattenBetween(project, sourceNode, linearNode);
        addSuggestion(suggestions, {
          id: `insert-flatten-${sourceNode.id}-${linearNode.id}`,
          title: 'Insert Flatten before Linear',
          description: 'Linear layers expect vector features. Flatten makes the classifier head explicit.',
          category: 'shape',
          safe: true,
          targetNodeId: linearNode.id,
          actions: [{
            kind: 'insert_node_between',
            sourceId: sourceNode.id,
            targetId: linearNode.id,
            node: flattenNode,
          }],
          previewLines: [
            `Replace ${sourceNode.id} -> ${linearNode.id} with ${sourceNode.id} -> ${flattenNode.id} -> ${linearNode.id}`,
          ],
        });
      }
    }
  }

  const trainingPairs: Array<[GraphNode | null, GraphNode | null, string]> = [
    [datasetNode, dataloaderNode, 'Connect Dataset to DataLoader'],
    [dataloaderNode, inputNode, 'Connect DataLoader to Input'],
    [outputNode, lossNode, 'Connect Output to Loss'],
    [lossNode, trainerNode, 'Connect Loss to Trainer'],
    [optimizerNode, trainerNode, 'Connect Optimizer to Trainer'],
    [metricNode, trainerNode, 'Connect Metric to Trainer'],
  ];

  for (const [source, target, title] of trainingPairs) {
    if (!source || !target || edgeExists(project, source.id, target.id)) {
      continue;
    }
    addSuggestion(suggestions, {
      id: `training-edge-${source.id}-${target.id}`,
      title,
      description: 'Complete the runnable training pipeline expected by the diagnostics.',
      category: 'training',
      safe: true,
      actions: [{ kind: 'add_edge', edge: buildEdge(source.id, target.id) }],
      previewLines: [`Add edge ${source.id} -> ${target.id}`],
    });
  }

  if (lossNode) {
    const desiredLoss = datasetTask === 'regression' ? 'MSELoss' : 'CrossEntropyLoss';
    if (String(lossNode.data.params.lossType ?? '') !== desiredLoss) {
      addSuggestion(suggestions, {
        id: `loss-type-${lossNode.id}`,
        title: `Switch Loss to ${desiredLoss}`,
        description: 'Align the training objective with the Dataset task type.',
        category: 'training',
        safe: true,
        targetNodeId: lossNode.id,
        actions: [{ kind: 'update_node_params', nodeId: lossNode.id, params: { lossType: desiredLoss } }],
        previewLines: [`${lossNode.id}.lossType: ${String(lossNode.data.params.lossType)} -> ${desiredLoss}`],
      });
    }
  }

  if (datasetNode) {
    const params = datasetNode.data.params;
    if (String(params.datasetName ?? '') === 'MNIST') {
      const patch: Record<string, unknown> = {};
      if (asNumber(params.numClasses, 0) !== 10) patch.numClasses = 10;
      if (String(params.colorMode ?? 'grayscale') !== 'grayscale') patch.colorMode = 'grayscale';
      if (asNumber(params.imageSize, 28) !== 28) patch.imageSize = 28;

      if (Object.keys(patch).length > 0) {
        addSuggestion(suggestions, {
          id: `mnist-defaults-${datasetNode.id}`,
          title: 'Restore MNIST teaching defaults',
          description: 'MNIST is a 10-class grayscale 28x28 dataset; these defaults keep the demo predictable.',
          category: 'dataset',
          safe: true,
          targetNodeId: datasetNode.id,
          actions: [{ kind: 'update_node_params', nodeId: datasetNode.id, params: patch }],
          previewLines: Object.entries(patch).map(([key, value]) => `${datasetNode.id}.${key}: ${String(params[key])} -> ${String(value)}`),
        });
      }
    }

    if (String(params.datasetName ?? '') === 'FakeData' && asNumber(params.numClasses, 0) <= 0) {
      const fallbackClasses = diagnosticsStats?.numClasses && diagnosticsStats.numClasses > 0 ? diagnosticsStats.numClasses : 10;
      addSuggestion(suggestions, {
        id: `fakedata-classes-${datasetNode.id}`,
        title: 'Set FakeData class count',
        description: 'FakeData needs a positive class count for classification training.',
        category: 'dataset',
        safe: true,
        targetNodeId: datasetNode.id,
        actions: [{ kind: 'update_node_params', nodeId: datasetNode.id, params: { numClasses: fallbackClasses } }],
        previewLines: [`${datasetNode.id}.numClasses: ${String(params.numClasses)} -> ${fallbackClasses}`],
      });
    }
  }

  return suggestions;
}
