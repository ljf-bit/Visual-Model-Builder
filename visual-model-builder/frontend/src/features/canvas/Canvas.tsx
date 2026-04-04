import React, { useCallback } from 'react';
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
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { getNodeBehavior } from '../../registry';
import { nodeRegistry } from '../../registry/nodeRegistry';
import { useAppStore } from '../../store';
import type { GraphEdge, GraphNode, GraphNodeData } from '../../types';

let nodeIdCounter = 0;

function generateNodeId(type: string): string {
  nodeIdCounter += 1;
  return `${type}_${nodeIdCounter}`;
}

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

const Canvas: React.FC = () => {
  const { screenToFlowPosition } = useReactFlow();
  const selectedNodeId = useAppStore((state) => state.selectedNodeId);
  const setSelectedNodeId = useAppStore((state) => state.setSelectedNodeId);
  const project = useAppStore((state) => state.project);
  const addNode = useAppStore((state) => state.addNode);
  const setStoreNodes = useAppStore((state) => state.setNodes);
  const setStoreEdges = useAppStore((state) => state.setEdges);
  const syncNodesFromCanvas = useAppStore((state) => state.syncNodesFromCanvas);
  const syncEdgesFromCanvas = useAppStore((state) => state.syncEdgesFromCanvas);

  const nodes = project.nodes.map((node) => ({
    ...node,
    selected: node.id === selectedNodeId,
  })) as Node<GraphNodeData>[];
  const edges = project.edges as Edge[];

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
      const nextNodes = applyNodeChanges(changes, nodes) as unknown as GraphNode[];
      if (shouldRecordNodeHistory(changes)) {
        setStoreNodes(nextNodes);
        return;
      }
      syncNodesFromCanvas(nextNodes);
    },
    [nodes, setStoreNodes, syncNodesFromCanvas],
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      const nextEdges = applyEdgeChanges(changes, edges) as unknown as GraphEdge[];
      if (shouldRecordEdgeHistory(changes)) {
        setStoreEdges(nextEdges);
        return;
      }
      syncEdgesFromCanvas(nextEdges);
    },
    [edges, setStoreEdges, syncEdgesFromCanvas],
  );

  const onConnect = useCallback(
    (params: Connection) => {
      setStoreEdges(addEdge(params, edges) as unknown as GraphEdge[]);
    },
    [edges, setStoreEdges],
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

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const newNode: GraphNode = {
        id: generateNodeId(nodeType),
        type: nodeType,
        position,
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
    },
    [addNode, screenToFlowPosition],
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
        fitView
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
