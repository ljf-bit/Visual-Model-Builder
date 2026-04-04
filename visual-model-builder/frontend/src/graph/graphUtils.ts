/**
 * Graph utility functions.
 *
 * Provides topology sort, cycle detection, and other graph operations.
 * Placeholder for Phase 1 — will be expanded as needed.
 */

import type { GraphNode, GraphEdge, ProjectGraph } from '../types';

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

  const candidate = project as ProjectGraph;
  const now = new Date().toISOString();

  return {
    version: candidate.version ?? '2.0.0',
    metadata: {
      name: candidate.metadata?.name ?? 'Imported Project',
      createdAt: candidate.metadata?.createdAt ?? now,
      updatedAt: candidate.metadata?.updatedAt ?? now,
    },
    nodes: candidate.nodes.map((node) => ({
      ...node,
      data: {
        type: (node.data as { type?: string } | undefined)?.type ?? node.type,
        label: node.data?.label ?? node.type,
        params: node.data?.params ?? {},
        inferredInputShape: node.data?.inferredInputShape ?? null,
        inferredOutputShape: node.data?.inferredOutputShape ?? null,
        errors: node.data?.errors ?? [],
      },
    })),
    edges: candidate.edges ?? [],
  };
}
