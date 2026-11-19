/**
 * Service layer used by the UI.
 * Connects directly to the SatQuery AI FastAPI backend server.
 * All boilerplate mock responses and artificial data seeds have been removed.
 */
import { API_ROUTES, apiRequest, IS_LIVE_BACKEND } from "./client";
import type {
  AnalysisMode,
  AnalysisRequest,
  AnalysisResult,
  AnalysisSummary,
  DatasetInfo,
  ExecutionTrace,
  ImageRole,
  ModelInfo,
  PipelineStep,
  SystemStatus,
  UploadedImage,
  ValidationResult,
} from "@/types";

export { IS_LIVE_BACKEND };

export async function getSystemStatus(): Promise<SystemStatus> {
  return apiRequest<SystemStatus>(API_ROUTES.systemStatus);
}

export async function getModels(): Promise<ModelInfo[]> {
  return apiRequest<ModelInfo[]>(API_ROUTES.models);
}

export async function getDatasets(): Promise<DatasetInfo[]> {
  return apiRequest<DatasetInfo[]>(API_ROUTES.datasets);
}

export async function uploadImage(file: File, role: ImageRole = "primary"): Promise<UploadedImage> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("role", role);
  return apiRequest<UploadedImage>(API_ROUTES.upload, {
    method: "POST",
    body: formData,
  });
}

export async function validateInputs(
  mode: AnalysisMode,
  images: UploadedImage[],
): Promise<ValidationResult> {
  return apiRequest<ValidationResult>(API_ROUTES.validate, {
    method: "POST",
    body: JSON.stringify({ mode, imageIds: images.map((i) => i.id), images }),
  });
}

export async function runAnalysis(
  request: AnalysisRequest,
  images: UploadedImage[],
  onStep: (steps: PipelineStep[]) => void,
): Promise<AnalysisResult> {
  // Initial in-progress step animation for live 22-agent backend
  const initialSteps: PipelineStep[] = [
    {
      id: "orchestrator",
      agentId: "lead_orchestrator",
      label: "01. Lead Orchestrator — Intent & DAG Compilation",
      status: "running",
      outputType: "rule_based",
      startedAt: new Date().toISOString(),
    },
    {
      id: "validator",
      agentId: "geo_validator",
      label: "02. GeoValidator — CRS & Metadata Extraction",
      status: "waiting",
      outputType: "rule_based",
    },
    {
      id: "specialist",
      agentId: "single_scene_vqa",
      label: "Perception & Reasoning Swarm",
      status: "waiting",
      outputType: "model",
    },
    {
      id: "ledger",
      agentId: "audit_ledger",
      label: "Cryptographic Audit Ledger & Bayesian Calibrator",
      status: "waiting",
      outputType: "rule_based",
    },
  ];
  onStep(initialSteps);

  const result = await apiRequest<AnalysisResult>(API_ROUTES.analyze, {
    method: "POST",
    body: JSON.stringify({
      ...request,
      imageIds: images.map((i) => i.id),
    }),
  });

  if (result.steps && result.steps.length > 0) {
    onStep(result.steps);
  }
  return result;
}

export async function getAnalysis(id: string): Promise<AnalysisResult> {
  return apiRequest<AnalysisResult>(API_ROUTES.analysis(id));
}

export async function getTrace(id: string): Promise<ExecutionTrace> {
  return apiRequest<ExecutionTrace>(API_ROUTES.trace(id));
}

export async function getHistory(): Promise<AnalysisSummary[]> {
  return apiRequest<AnalysisSummary[]>(API_ROUTES.history);
}

export function reportUrl(id: string, format: "pdf" | "json" | "html"): string {
  return API_ROUTES.report(id, format === "html" ? "json" : format);
}

/** Builds the report payload from live analysis result. */
export function buildReportPayload(result: AnalysisResult) {
  return {
    generatedAt: new Date().toISOString(),
    source: "backend",
    analysisId: result.id,
    query: result.query,
    mode: result.mode,
    task: result.intentLabel,
    inputs: result.images.map((i) => ({
      role: i.role,
      ...i.metadata,
    })),
    agentsUsed: result.agentsUsed,
    answer: result.answer,
    confidence: result.confidence,
    evidence: result.evidence,
    executionSummary: result.trace,
    totalExecutionMs: result.totalExecutionMs,
    merkleRoot: (result as Record<string, unknown>).merkleRoot,
    merkleProof: (result as Record<string, unknown>).merkleProof,
  };
}
