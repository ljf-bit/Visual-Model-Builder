import type { AutoFixAction, GraphEdge, ProjectGraph } from '../types';

function cloneProject(project: ProjectGraph): ProjectGraph {
  return JSON.parse(JSON.stringify(project)) as ProjectGraph;
}

function sanitizeId(value: string): string {
  return value.replace(/[^A-Za-z0-9_-]+/g, '-');
}

function createEdgeId(source: string, target: string, edges: GraphEdge[]): string {
  const baseId = `autofix-${sanitizeId(source)}-${sanitizeId(target)}`;
  const usedIds = new Set(edges.map((edge) => edge.id));
  if (!usedIds.has(baseId)) {
    return baseId;
  }

  let suffix = 2;
  while (usedIds.has(`${baseId}-${suffix}`)) {
    suffix += 1;
  }
  return `${baseId}-${suffix}`;
}

function hasEdge(edges: GraphEdge[], source: string, target: string): boolean {
  return edges.some((edge) => edge.source === source && edge.target === target);
}

function addEdgeOnce(edges: GraphEdge[], edge: GraphEdge): GraphEdge[] {
  if (hasEdge(edges, edge.source, edge.target)) {
    return edges;
  }
  const id = edge.id || createEdgeId(edge.source, edge.target, edges);
  return [...edges, { ...edge, id }];
}

export function applyAutoFixActionsToProject(project: ProjectGraph, actions: AutoFixAction[]): ProjectGraph {
  const nextProject = cloneProject(project);

  for (const action of actions) {
    if (action.kind === 'update_node_params' && action.nodeId && action.params) {
      nextProject.nodes = nextProject.nodes.map((node) =>
        node.id === action.nodeId
          ? {
              ...node,
              data: {
                ...node.data,
                params: {
                  ...node.data.params,
                  ...action.params,
                },
              },
            }
          : node,
      );
      continue;
    }

    if (action.kind === 'add_edge' && action.edge) {
      nextProject.edges = addEdgeOnce(nextProject.edges, action.edge);
      continue;
    }

    if (action.kind === 'insert_node_between' && action.sourceId && action.targetId && action.node) {
      const nodeExists = nextProject.nodes.some((node) => node.id === action.node?.id);
      if (!nodeExists) {
        nextProject.nodes = [...nextProject.nodes, action.node];
      }
      nextProject.edges = nextProject.edges.filter(
        (edge) => !(edge.source === action.sourceId && edge.target === action.targetId),
      );
      nextProject.edges = addEdgeOnce(nextProject.edges, {
        id: createEdgeId(action.sourceId, action.node.id, nextProject.edges),
        source: action.sourceId,
        target: action.node.id,
      });
      nextProject.edges = addEdgeOnce(nextProject.edges, {
        id: createEdgeId(action.node.id, action.targetId, nextProject.edges),
        source: action.node.id,
        target: action.targetId,
      });
    }
  }

  nextProject.metadata = {
    ...nextProject.metadata,
    updatedAt: new Date().toISOString(),
  };

  return nextProject;
}
