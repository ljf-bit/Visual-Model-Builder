import React, { useCallback, useEffect, useMemo, useRef } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Panel,
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type Edge,
  type EdgeChange,
  Handle,
  type Node,
  type NodeChange,
  type NodeProps,
  Position,
  useReactFlow,
  useNodesInitialized,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { getNextNodeId } from '../../graph/graphUtils';
import { getNodeBehavior } from '../../registry';
import { nodeRegistry } from '../../registry/nodeRegistry';
import { useAppStore } from '../../store';
import type { GraphEdge, GraphNode, GraphNodeData } from '../../types';

const GenericNode: React.FC<NodeProps> = ({ data, selected }) => {
  const behavior = getNodeBehavior(String(data.type ?? data.label));
  const category = behavior?.template.category ?? 'layer';
  const accent =
    category === 'train'
      ? '#22c55e'
      : category === 'io'
        ? '#38bdf8'
        : category === 'activation'
          ? '#f59e0b'
          : 'var(--color-accent)';

  return (
    <div
      className={`react-flow__node-default ${selected ? 'selected' : ''}`}
      style={{
        background: 'var(--color-bg-secondary)',
        color: 'var(--color-text-primary)',
        borderColor: selected ? accent : 'var(--color-border)',
        borderRadius: '8px',
        padding: '10px 15px',
        minWidth: '176px',
      }}
    >
      {behavior?.template.inputPorts ? (
        <Handle type="target" position={Position.Left} style={{ background: '#555' }} />
      ) : null}
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', alignItems: 'center' }}>
        <div style={{ fontSize: '14px', fontWeight: 'bold' }}>{String(data.label)}</div>
        <span
          style={{
            fontSize: '10px',
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
            color: accent,
          }}
        >
          {category}
        </span>
      </div>
      {'errors' in data && Array.isArray(data.errors) && data.errors.length > 0 ? (
        <div style={{ marginTop: '8px', fontSize: '11px', color: '#fca5a5' }}>{data.errors[0]}</div>
      ) : null}
      {behavior?.template.outputPorts ? (
        <Handle type="source" position={Position.Right} style={{ background: '#555' }} />
      ) : null}
    </div>
  );
};

const nodeTypes = Object.fromEntries(Object.keys(nodeRegistry).map((key) => [key, GenericNode]));

function isFinitePosition(position: GraphNode['position']): boolean {
  return Number.isFinite(position.x) && Number.isFinite(position.y);
}

const Canvas: React.FC = () => {
  const { fitView, screenToFlowPosition } = useReactFlow();
  const nodesInitialized = useNodesInitialized();
  const selectedNodeId = useAppStore((state) => state.selectedNodeId);
  const setSelectedNodeId = useAppStore((state) => state.setSelectedNodeId);
  const project = useAppStore((state) => state.project);
  const addNode = useAppStore((state) => state.addNode);
  const setStoreNodes = useAppStore((state) => state.setNodes);
  const setStoreEdges = useAppStore((state) => state.setEdges);
  const syncNodesFromCanvas = useAppStore((state) => state.syncNodesFromCanvas);
  const syncEdgesFromCanvas = useAppStore((state) => state.syncEdgesFromCanvas);
  const previousNodeSignatureRef = useRef('');

  const nodes = project.nodes.map((node) => ({
    ...node,
    selected: node.id === selectedNodeId,
  })) as Node<GraphNodeData>[];
  const edges = project.edges as Edge[];
  const nodeSignature = useMemo(
    () => project.nodes.map((node) => `${node.id}:${node.position.x}:${node.position.y}`).join('|'),
    [project.nodes],
  );

  useEffect(() => {
    if (!nodesInitialized || project.nodes.length === 0) {
      if (project.nodes.length === 0) {
        previousNodeSignatureRef.current = '';
      }
      return;
    }

    if (previousNodeSignatureRef.current === nodeSignature) {
      return;
    }

    previousNodeSignatureRef.current = nodeSignature;

    const frameId = window.requestAnimationFrame(() => {
      void fitView({
        padding: 0.2,
        duration: 180,
        includeHiddenNodes: true,
      });
    });

    return () => window.cancelAnimationFrame(frameId);
  }, [fitView, nodeSignature, nodesInitialized, project.nodes.length]);

  const shouldRecordNodeHistory = (changes: NodeChange[]) =>
    changes.some((change) => {
      if (change.type === 'remove' || change.type === 'replace' || change.type === 'add') {
        return true;
      }
      if (change.type === 'position') {
        return change.dragging === false;
      }
      return false;
    });

  const shouldRecordEdgeHistory = (changes: EdgeChange[]) =>
    changes.some((change) => change.type === 'remove' || change.type === 'replace' || change.type === 'add');

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      const relevantChanges = changes.filter(
        (change) =>
          change.type === 'remove'
          || change.type === 'replace'
          || change.type === 'add'
          || change.type === 'position',
      );

      if (relevantChanges.length === 0) {
        return;
      }

      const latestNodes = useAppStore.getState().project.nodes as unknown as Node[];
      const nextNodes = applyNodeChanges(relevantChanges, latestNodes) as unknown as GraphNode[];
      if (shouldRecordNodeHistory(relevantChanges)) {
        setStoreNodes(nextNodes);
        return;
      }
      syncNodesFromCanvas(nextNodes);
    },
    [setStoreNodes, syncNodesFromCanvas],
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      const relevantChanges = changes.filter(
        (change) => change.type === 'remove' || change.type === 'replace' || change.type === 'add',
      );

      if (relevantChanges.length === 0) {
        return;
      }

      const latestEdges = useAppStore.getState().project.edges as unknown as Edge[];
      const nextEdges = applyEdgeChanges(relevantChanges, latestEdges) as unknown as GraphEdge[];
      if (shouldRecordEdgeHistory(relevantChanges)) {
        setStoreEdges(nextEdges);
        return;
      }
      syncEdgesFromCanvas(nextEdges);
    },
    [setStoreEdges, syncEdgesFromCanvas],
  );

  const onConnect = useCallback(
    (params: Connection) => {
      const latestEdges = useAppStore.getState().project.edges as unknown as Edge[];
      setStoreEdges(addEdge(params, latestEdges) as unknown as GraphEdge[]);
    },
    [setStoreEdges],
  );

  const onNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      setSelectedNodeId(node.id);
    },
    [setSelectedNodeId],
  );

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null);
  }, [setSelectedNodeId]);

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const nodeType = event.dataTransfer.getData('application/vmb-node-type');
      if (!nodeType) {
        return;
      }

      const behavior = getNodeBehavior(nodeType);
      if (!behavior) {
        return;
      }

      const latestNodes = useAppStore.getState().project.nodes;

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });
      const safePosition = isFinitePosition(position)
        ? position
        : {
            x: 80 + latestNodes.length * 40,
            y: 80 + latestNodes.length * 40,
          };

      const newNode: GraphNode = {
        id: getNextNodeId(nodeType, latestNodes),
        type: nodeType,
        position: safePosition,
        data: {
          label: behavior.template.displayName,
          type: behavior.template.type,
          params: behavior.defaultData(),
          inferredInputShape: null,
          inferredOutputShape: null,
          errors: [],
        },
      };

      addNode(newNode);
      setSelectedNodeId(newNode.id);
    },
    [addNode, screenToFlowPosition, setSelectedNodeId],
  );

  return (
    <div className="canvas-wrapper">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        onDragOver={onDragOver}
        onDrop={onDrop}
      >
        <Background />
        <Controls />
        <MiniMap />
        <Panel position="top-right">
          <button
            onClick={() => useAppStore.getState().resetProject()}
            style={{
              background: 'var(--color-bg-hover)',
              color: 'var(--color-text-primary)',
              border: '1px solid var(--color-border)',
              padding: '6px 12px',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            Clear Canvas
          </button>
        </Panel>
      </ReactFlow>
    </div>
  );
};

export default Canvas;
