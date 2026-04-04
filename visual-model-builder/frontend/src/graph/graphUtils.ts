/**
 * Graph utility functions.
 *
 * Provides topology sort, cycle detection, and other graph operations.
 * Placeholder for Phase 1 — will be expanded as needed.
 */

import type { GraphEdge, GraphNode, ProjectGraph } from '../types';

function escapeForRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function buildDefaultEdgeId(source: string, target: string): string {
  return `xy-edge__${source}-${target}`;
}

function getNextEdgeId(source: string, target: string, usedEdgeIds: Set<string>): string {
  const baseId = buildDefaultEdgeId(source, target);
  if (!usedEdgeIds.has(baseId)) {
    return baseId;
  }

  let suffix = 2;
  while (usedEdgeIds.has(`${baseId}-${suffix}`)) {
    suffix += 1;
  }

  return `${baseId}-${suffix}`;
}

export function getNextNodeId(type: string, nodes: Pick<GraphNode, 'id' | 'type'>[]): string {
  const pattern = new RegExp(`^${escapeForRegex(type)}_(\\d+)$`);
  let maxSuffix = 0;

  for (const node of nodes) {
    const match = node.id.match(pattern);
    if (match) {
      maxSuffix = Math.max(maxSuffix, Number(match[1]));
    }
  }

  return `${type}_${maxSuffix + 1}`;
}

function normalizeNode(rawNode: unknown, nodes: GraphNode[]): GraphNode | null {
  if (!isRecord(rawNode)) {
    return null;
  }

  const rawData = isRecord(rawNode.data) ? rawNode.data : {};
  const nodeType =
    typeof rawNode.type === 'string'
      ? rawNode.type
      : typeof rawData.type === 'string'
        ? rawData.type
        : typeof rawData.label === 'string'
          ? rawData.label
          : 'Unknown';

  const requestedId =
    typeof rawNode.id === 'string' && rawNode.id.trim().length > 0
      ? rawNode.id
      : getNextNodeId(nodeType, nodes);
  const nodeId = nodes.some((node) => node.id === requestedId) ? getNextNodeId(nodeType, nodes) : requestedId;

  return {
    id: nodeId,
    type: nodeType,
    position: {
      x: isRecord(rawNode.position) && isFiniteNumber(rawNode.position.x) ? rawNode.position.x : 0,
      y: isRecord(rawNode.position) && isFiniteNumber(rawNode.position.y) ? rawNode.position.y : 0,
    },
    data: {
      ...rawData,
      type: typeof rawData.type === 'string' ? rawData.type : nodeType,
      label: typeof rawData.label === 'string' ? rawData.label : nodeType,
      params: isRecord(rawData.params) ? rawData.params : {},
      inferredInputShape: Array.isArray(rawData.inferredInputShape) ? rawData.inferredInputShape : null,
      inferredOutputShape: Array.isArray(rawData.inferredOutputShape) ? rawData.inferredOutputShape : null,
      errors: Array.isArray(rawData.errors) ? rawData.errors.filter((error): error is string => typeof error === 'string') : [],
    },
  };
}

function normalizeEdge(rawEdge: unknown, validNodeIds: Set<string>, usedEdgeIds: Set<string>): GraphEdge | null {
  if (!isRecord(rawEdge) || typeof rawEdge.source !== 'string' || typeof rawEdge.target !== 'string') {
    return null;
  }

  if (rawEdge.source === rawEdge.target) {
    return null;
  }

  if (!validNodeIds.has(rawEdge.source) || !validNodeIds.has(rawEdge.target)) {
    return null;
  }

  const requestedId =
    typeof rawEdge.id === 'string' && rawEdge.id.trim().length > 0
      ? rawEdge.id
      : buildDefaultEdgeId(rawEdge.source, rawEdge.target);
  const edgeId = usedEdgeIds.has(requestedId)
    ? getNextEdgeId(rawEdge.source, rawEdge.target, usedEdgeIds)
    : requestedId;

  usedEdgeIds.add(edgeId);

  return {
    id: edgeId,
    source: rawEdge.source,
    target: rawEdge.target,
    sourceHandle: typeof rawEdge.sourceHandle === 'string' ? rawEdge.sourceHandle : undefined,
    targetHandle: typeof rawEdge.targetHandle === 'string' ? rawEdge.targetHandle : undefined,
  };
}

/**
 * Topological sort using Kahn's algorithm.
 * Returns sorted node IDs or null if cycle detected.
 */
export function topologicalSort(
  nodes: GraphNode[],
  edges: GraphEdge[],
): string[] | null {
  const inDegree = new Map<string, number>();
  const adjacency = new Map<string, string[]>();

  for (const node of nodes) {
    inDegree.set(node.id, 0);
    adjacency.set(node.id, []);
  }

  for (const edge of edges) {
    adjacency.get(edge.source)?.push(edge.target);
    inDegree.set(edge.target, (inDegree.get(edge.target) ?? 0) + 1);
  }

  const queue: string[] = [];
  for (const [id, deg] of inDegree) {
    if (deg === 0) queue.push(id);
  }

  const sorted: string[] = [];
  while (queue.length > 0) {
    const current = queue.shift()!;
    sorted.push(current);
    for (const neighbor of adjacency.get(current) ?? []) {
      const newDeg = (inDegree.get(neighbor) ?? 1) - 1;
      inDegree.set(neighbor, newDeg);
      if (newDeg === 0) queue.push(neighbor);
    }
  }

  return sorted.length === nodes.length ? sorted : null;
}

/**
 * Check if graph contains a cycle.
 */
export function hasCycle(nodes: GraphNode[], edges: GraphEdge[]): boolean {
  return topologicalSort(nodes, edges) === null;
}

const TRAINING_NODE_TYPES = new Set([
  'Dataset',
  'DataLoader',
  'Loss',
  'Optimizer',
  'Trainer',
  'Metric',
]);

/**
 * Whether the current project contains Phase 2 training nodes.
 */
export function hasTrainingNodes(nodes: GraphNode[]): boolean {
  return nodes.some((node) => TRAINING_NODE_TYPES.has(node.type));
}

/**
 * Normalize a loaded project so old Phase 1 files stay compatible.
 */
export function normalizeProjectGraph(project: unknown): ProjectGraph | null {
  if (
    !project ||
    typeof project !== 'object' ||
    !Array.isArray((project as { nodes?: unknown }).nodes) ||
    !Array.isArray((project as { edges?: unknown }).edges)
  ) {
    return null;
  }

  const candidate = project as Partial<ProjectGraph> & { nodes: unknown[]; edges: unknown[] };
  const now = new Date().toISOString();
  const normalizedNodes: GraphNode[] = [];

  for (const rawNode of candidate.nodes) {
    const normalizedNode = normalizeNode(rawNode, normalizedNodes);
    if (normalizedNode) {
      normalizedNodes.push(normalizedNode);
    }
  }

  const validNodeIds = new Set(normalizedNodes.map((node) => node.id));
  const usedEdgeIds = new Set<string>();
  const normalizedEdges = candidate.edges
    .map((rawEdge) => normalizeEdge(rawEdge, validNodeIds, usedEdgeIds))
    .filter((edge): edge is GraphEdge => edge !== null);

  return {
    version: typeof candidate.version === 'string' ? candidate.version : '2.0.0',
    metadata: {
      name: typeof candidate.metadata?.name === 'string' ? candidate.metadata.name : 'Imported Project',
      createdAt: typeof candidate.metadata?.createdAt === 'string' ? candidate.metadata.createdAt : now,
      updatedAt: typeof candidate.metadata?.updatedAt === 'string' ? candidate.metadata.updatedAt : now,
    },
    nodes: normalizedNodes,
    edges: normalizedEdges,
  };
}
