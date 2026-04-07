/**
 * API service for communicating with the FastAPI backend.
 *
 * All backend calls are centralized here.
 * Base URL is configurable via environment variable VITE_API_BASE_URL.
 */

import type {
  GenerateCodeResponse,
  InferShapesResponse,
  InspectDatasetResponse,
  ProjectGraph,
  TrainingDiagnosticsResponse,
  RunTrainingResponse,
  ValidateGraphResponse,
} from '../types';

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

// ============================================================
// Public API
// ============================================================

/** Health check */
export async function healthCheck(): Promise<{ status: string }> {
  return get('/health');
}

/** Validate the project graph */
export async function validateGraph(project: ProjectGraph): Promise<ValidateGraphResponse> {
  return post('/validate-graph', { project });
}

/** Infer shapes for all nodes */
export async function inferShapes(project: ProjectGraph): Promise<InferShapesResponse> {
  return post('/infer-shapes', { project });
}

/** Generate PyTorch code from the graph */
export async function generateCode(project: ProjectGraph): Promise<GenerateCodeResponse> {
  return post('/generate-code', { project });
}

/** Validate a training graph */
export async function validateTrainingGraph(project: ProjectGraph): Promise<ValidateGraphResponse> {
  return post('/validate-training-graph', { project });
}

/** Generate a training script from the graph */
export async function generateTrainingCode(project: ProjectGraph): Promise<GenerateCodeResponse> {
  return post('/generate-training-code', { project });
}

/** Execute the Phase 2 training loop */
export async function runTraining(project: ProjectGraph): Promise<RunTrainingResponse> {
  return post('/run-training', { project });
}

/** Diagnose a training graph before running it */
export async function diagnoseTrainingGraph(project: ProjectGraph): Promise<TrainingDiagnosticsResponse> {
  return post('/diagnose-training-graph', { project });
}

/** Inspect a dataset config before training */
export async function inspectDataset(config: Record<string, unknown>): Promise<InspectDatasetResponse> {
  return post('/inspect-dataset', { config });
}
