/**
 * Core domain types for SatQuery AI.
 * Mirrors the Pydantic v2 contracts from the Python/FastAPI backend.
 * See: ISRO/SAC Problem Statement 26167
 */

/* ── Enums ─────────────────────────────────────────────────── */

export type Modality = "OPTICAL" | "SAR" | "UNKNOWN";

export type AnalysisMode = "single" | "bitemporal" | "optical-sar" | "ask-anything";

export type ImageRole = "primary" | "before" | "after" | "optical" | "sar";

export type ProcessingStatus = "pending" | "uploading" | "validating" | "ready" | "error";

/** Intent classification types from LeadOrchestratorAgent */
export type IntentType =
  "vqa" | "captioning" | "grounding" | "change-detection" | "optical-sar-fusion" | "segmentation";

/** Output type distinguishing real vs mock vs rule-based results (Rule #12) */
export type OutputType = "model" | "rule_based" | "mock" | "unavailable";

/** Agent identifiers matching the 7-agent registry */
export type AgentId =
  | "lead_orchestrator"
  | "geo_validator"
  | "audit_ledger"
  | "single_scene_vqa"
  | "visual_grounding"
  | "bi_temporal_change"
  | "sar_cloud_penetration";

/* ── Image types ───────────────────────────────────────────── */

export interface ImageMetadata {
  filename: string;
  fileType: string;
  fileSizeBytes: number;
  width: number | null;
  height: number | null;
  bands: number | null;
  crs: string | null;
  modality: Modality;
  sensor: string | null;
  acquisitionDate: string | null;
  resolutionMeters: number | null;
  polarizations: string[] | null;
  boundsWgs84: [number, number, number, number] | null;
  cloudFraction: number | null;
}

export interface UploadedImage {
  id: string;
  role: ImageRole;
  previewUrl: string;
  status: ProcessingStatus;
  progress: number;
  metadata: ImageMetadata;
  error?: string | undefined;
}

/* ── Validation ────────────────────────────────────────────── */

export type CheckStatus = "pass" | "fail" | "warn" | "unknown";

export interface ValidationCheck {
  id: string;
  label: string;
  status: CheckStatus;
  detail?: string | undefined;
}

export interface ValidationResult {
  ok: boolean;
  checks: ValidationCheck[];
  blockingMessage?: string | undefined;
}

/* ── Analysis request / response ───────────────────────────── */

export interface AnalysisRequest {
  mode: AnalysisMode;
  query: string;
  imageIds: string[];
  demo?: boolean | undefined;
}

export type PipelineStepStatus = "waiting" | "running" | "completed" | "failed";

/** A single step in the 7-agent pipeline */
export interface PipelineStep {
  id: string;
  agentId: AgentId;
  label: string;
  detail?: string | undefined;
  status: PipelineStepStatus;
  outputType: OutputType;
  startedAt?: string | undefined;
  durationMs?: number | undefined;
}

export interface ExecutionTraceEntry {
  agent: AgentId;
  stage: string;
  outcome: string;
  outputType: OutputType;
  timestamp: string;
  durationMs?: number | undefined;
}

export interface ExecutionTrace {
  analysisId: string;
  entries: ExecutionTraceEntry[];
  totalDurationMs: number;
}

export interface BoundingBox {
  id: string;
  label: string;
  /** Normalized 0..1 coordinates relative to image width/height. */
  x: number;
  y: number;
  w: number;
  h: number;
  score: number | null;
}

export interface LandCoverClass {
  label: string;
  percentage: number;
  colorToken: string;
}

export interface Evidence {
  groundingBoxes: BoundingBox[] | null;
  changeMapUrl: string | null;
  changedAreaPercent: number | null;
  segmentationUrl: string | null;
  landCover: LandCoverClass[] | null;
  notes: string | null;
}

export type AnalysisStatus = "queued" | "running" | "completed" | "failed";

export interface AnalysisResult {
  id: string;
  createdAt: string;
  mode: AnalysisMode;
  intent: IntentType | null;
  intentLabel: string | null;
  query: string;
  status: AnalysisStatus;
  answer: string | null;
  confidence: number | null;
  outputType: OutputType;
  agentsUsed: AgentId[];
  images: UploadedImage[];
  evidence: Evidence | null;
  steps: PipelineStep[];
  trace: ExecutionTrace | null;
  totalExecutionMs: number | null;
  demo: boolean;
  error?: string | undefined;
}

export interface AnalysisSummary {
  id: string;
  createdAt: string;
  mode: AnalysisMode;
  intentLabel: string | null;
  imageCount: number;
  query: string;
  status: AnalysisStatus;
  confidence: number | null;
  outputType: OutputType;
  totalExecutionMs: number | null;
  demo: boolean;
}

/* ── Model registry ────────────────────────────────────────── */

export type ModelStatus = "ready" | "loading" | "unavailable" | "not-configured";

export interface ModelInfo {
  id: string;
  capability: string;
  task: IntentType;
  name: string;
  candidates: string[];
  inputSchema: string;
  outputSchema: string;
  version: string | null;
  lastLoadedAt: string | null;
  status: ModelStatus;
  device: string | null;
}

/* ── System status ─────────────────────────────────────────── */

export interface ServiceHealth {
  id: string;
  label: string;
  status: "online" | "degraded" | "offline" | "unknown";
  detail?: string | undefined;
}

export interface SystemStatus {
  version: string;
  mode: "mock" | "real";
  agentController: "READY" | "BUSY" | "UNAVAILABLE";
  services: ServiceHealth[];
  agents: AgentStatus[];
  compute: {
    device: string;
    utilizationPercent: number | null;
    memoryUsedGb: number | null;
    memoryTotalGb: number | null;
  };
}

export interface AgentStatus {
  id: AgentId;
  name: string;
  status: "ready" | "busy" | "unavailable";
  memPalaceWing: string;
}

/* ── Datasets ──────────────────────────────────────────────── */

export interface DatasetInfo {
  id: string;
  name: string;
  description: string;
  modality: Modality | "MIXED";
  sceneCount: number;
  resolution: string;
  tasks: string[];
}

/* ── Errors ────────────────────────────────────────────────── */

export interface ApiError {
  code: string;
  message: string;
  hint?: string | undefined;
}
