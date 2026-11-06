/**
 * MOCK SERVICE LAYER — demo/offline only.
 * This module never runs when VITE_SATQUERY_API_URL is configured.
 * All production requests go through src/api/client.ts.
 *
 * Pipeline matches the 7-agent architecture:
 *   LeadOrchestrator → GeoValidator → Specialist Agent → AuditLedger
 */
import type {
  AgentId,
  AnalysisMode,
  AnalysisRequest,
  AnalysisResult,
  AnalysisSummary,
  DatasetInfo,
  ExecutionTrace,
  IntentType,
  ModelInfo,
  OutputType,
  PipelineStep,
  SystemStatus,
  UploadedImage,
  ValidationResult,
} from "@/types";
import { DEMO_SCENARIOS, getScenario, type DemoScenario } from "./scenarios";

const APP_VERSION = "v1.0.0-part1";

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/* ── Agent registry (7 agents) ─────────────────────────────── */

const AGENT_REGISTRY: { id: AgentId; name: string; memPalaceWing: string }[] = [
  { id: "lead_orchestrator", name: "LeadOrchestratorAgent", memPalaceWing: "Execution_Control" },
  { id: "geo_validator", name: "GeoValidatorAgent", memPalaceWing: "Geospatial_Integrity" },
  { id: "audit_ledger", name: "AuditLedgerAgent", memPalaceWing: "Audit_Compliance" },
  { id: "single_scene_vqa", name: "SingleSceneVQAAgent", memPalaceWing: "Visual_Reasoning" },
  { id: "visual_grounding", name: "VisualGroundingAgent", memPalaceWing: "Spatial_Grounding" },
  { id: "bi_temporal_change", name: "BiTemporalChangeAgent", memPalaceWing: "Temporal_Analysis" },
  {
    id: "sar_cloud_penetration",
    name: "SARCloudPenetrationAgent",
    memPalaceWing: "SAR_Processing",
  },
];

/* ── System status ─────────────────────────────────────────── */

export function mockSystemStatus(): SystemStatus {
  return {
    version: APP_VERSION,
    mode: "mock",
    agentController: "READY",
    services: [
      { id: "backend", label: "FastAPI Backend", status: "online", detail: "Mock service layer" },
      {
        id: "vlm",
        label: "Vision-Language Models",
        status: "online",
        detail: "Simulated registry",
      },
      {
        id: "orchestrator",
        label: "Agent Orchestrator",
        status: "online",
        detail: "Deterministic router",
      },
      { id: "memory", label: "MemPalace Store", status: "online", detail: "In-memory session" },
      { id: "audit", label: "Audit Ledger", status: "online", detail: "SQLite (mock)" },
      { id: "imagery", label: "Imagery Store", status: "online", detail: "In-memory" },
    ],
    agents: AGENT_REGISTRY.map((a) => ({
      id: a.id,
      name: a.name,
      status: "ready" as const,
      memPalaceWing: a.memPalaceWing,
    })),
    compute: {
      device: "Simulated (no GPU attached)",
      utilizationPercent: null,
      memoryUsedGb: null,
      memoryTotalGb: null,
    },
  };
}

/* ── Model registry ────────────────────────────────────────── */

export function mockModels(): ModelInfo[] {
  return [
    {
      id: "vqa",
      capability: "Visual Question Answering",
      task: "vqa",
      name: "RS Vision-Language Model",
      candidates: ["Qwen2-VL (RS-adapted)", "InternVL2", "LLaVA-1.6"],
      inputSchema: "image + query",
      outputSchema: "answer + confidence",
      version: null,
      lastLoadedAt: null,
      status: "not-configured",
      device: null,
    },
    {
      id: "captioning",
      capability: "Scene Captioning",
      task: "captioning",
      name: "RS Vision-Language Model",
      candidates: ["Qwen2-VL (RS-adapted)", "RemoteCLIP + LM decoder"],
      inputSchema: "image",
      outputSchema: "caption / description",
      version: null,
      lastLoadedAt: null,
      status: "not-configured",
      device: null,
    },
    {
      id: "grounding",
      capability: "Visual Grounding",
      task: "grounding",
      name: "Grounding Model",
      candidates: ["GroundingDINO", "RemoteCLIP retrieval head"],
      inputSchema: "image + text prompt",
      outputSchema: "bounding boxes + scores",
      version: null,
      lastLoadedAt: null,
      status: "not-configured",
      device: null,
    },
    {
      id: "change-detection",
      capability: "Change Detection",
      task: "change-detection",
      name: "Change Detection Model",
      candidates: ["ChangeFormer", "BIT (Bitemporal Image Transformer)"],
      inputSchema: "co-registered image pair",
      outputSchema: "binary change map",
      version: null,
      lastLoadedAt: null,
      status: "not-configured",
      device: null,
    },
    {
      id: "optical-sar-fusion",
      capability: "Optical–SAR Fusion",
      task: "optical-sar-fusion",
      name: "Dual-Stream Fusion Model",
      candidates: ["Dual-encoder optical/SAR fusion", "Cross-attention fusion head"],
      inputSchema: "optical + SAR pair",
      outputSchema: "fused analysis + mask",
      version: null,
      lastLoadedAt: null,
      status: "not-configured",
      device: null,
    },
    {
      id: "segmentation",
      capability: "Land-Cover Segmentation",
      task: "segmentation",
      name: "Segmentation Model",
      candidates: ["SegFormer (RS fine-tuned)", "UNet++ / ResNet encoder"],
      inputSchema: "image",
      outputSchema: "class mask + class shares",
      version: null,
      lastLoadedAt: null,
      status: "not-configured",
      device: null,
    },
  ];
}

/* ── Datasets ──────────────────────────────────────────────── */

export function mockDatasets(): DatasetInfo[] {
  return [
    {
      id: "rsvqa",
      name: "RSVQA-HR",
      description: "High-resolution remote-sensing visual question answering benchmark.",
      modality: "OPTICAL",
      sceneCount: 10659,
      resolution: "0.15 m",
      tasks: ["VQA"],
    },
    {
      id: "levir-cd",
      name: "LEVIR-CD",
      description: "Bi-temporal building change detection pairs.",
      modality: "OPTICAL",
      sceneCount: 637,
      resolution: "0.5 m",
      tasks: ["Change Detection"],
    },
    {
      id: "dfc-sar",
      name: "Optical–SAR Pairs",
      description: "Co-registered optical and Sentinel-1 SAR acquisitions for fusion research.",
      modality: "MIXED",
      sceneCount: 1842,
      resolution: "10–20 m",
      tasks: ["Fusion", "Segmentation"],
    },
    {
      id: "dior-rsvg",
      name: "DIOR-RSVG",
      description: "Referring-expression grounding annotations over optical scenes.",
      modality: "OPTICAL",
      sceneCount: 17402,
      resolution: "0.5–30 m",
      tasks: ["Grounding"],
    },
  ];
}

/* ── Validation (GeoValidatorAgent mock) ───────────────────── */

const ACCEPTED = ["tif", "tiff", "png", "jpg", "jpeg"];

export function isAcceptedFile(name: string): boolean {
  const ext = name.split(".").pop()?.toLowerCase() ?? "";
  return ACCEPTED.includes(ext);
}

export function mockValidate(mode: AnalysisMode, images: UploadedImage[]): ValidationResult {
  const checks: ValidationResult["checks"] = [];
  const ready = images.filter((i) => i.status === "ready");

  if (ready.length === 0) {
    return {
      ok: false,
      checks: [{ id: "input", label: "Imagery provided", status: "unknown" }],
      blockingMessage: "Upload satellite imagery to begin validation.",
    };
  }

  checks.push({
    id: "format",
    label: "File format valid",
    status: ready.every((i) => isAcceptedFile(i.metadata.filename)) ? "pass" : "fail",
    detail: "GeoTIFF, TIFF, PNG, JPEG",
  });
  checks.push({ id: "readable", label: "Image readable", status: "pass" });
  const geoCount = ready.filter((i) => i.metadata.crs).length;
  checks.push({
    id: "geo",
    label: "Geospatial metadata detected",
    status: geoCount === ready.length ? "pass" : geoCount > 0 ? "warn" : "warn",
    detail: geoCount === ready.length ? undefined : "Some inputs carry no CRS — bounds unavailable",
  });
  const modalities = [...new Set(ready.map((i) => i.metadata.modality))];
  checks.push({
    id: "modality",
    label: `Modality detected: ${modalities.join(" + ")}`,
    status: modalities.includes("UNKNOWN") ? "warn" : "pass",
  });
  checks.push({
    id: "dimensions",
    label: "Image dimensions valid",
    status: ready.every((i) => (i.metadata.width ?? 0) > 0) ? "pass" : "warn",
  });
  checks.push({
    id: "cloud",
    label: "Cloud contamination check",
    status: "pass",
    detail: "Threshold: 40%",
  });

  let blockingMessage: string | undefined;

  if (mode === "bitemporal") {
    const before = ready.find((i) => i.role === "before");
    const after = ready.find((i) => i.role === "after");
    if (!before || !after) {
      blockingMessage = "Bi-temporal analysis requires two compatible images.";
      checks.push({ id: "pair", label: "Image pair provided", status: "fail" });
    } else {
      const sameCrs = before.metadata.crs === after.metadata.crs;
      const sameDims =
        before.metadata.width === after.metadata.width &&
        before.metadata.height === after.metadata.height;
      checks.push({
        id: "crs-match",
        label: "Same spatial reference",
        status: sameCrs ? "pass" : "fail",
        detail: sameCrs
          ? undefined
          : `${before.metadata.crs ?? "unknown"} vs ${after.metadata.crs ?? "unknown"}`,
      });
      checks.push({
        id: "dims-match",
        label: "Compatible dimensions",
        status: sameDims ? "pass" : "fail",
        detail: sameDims ? undefined : "Pixel grids differ — resampling required",
      });
      checks.push({
        id: "coreg",
        label: sameCrs && sameDims ? "Co-registration verified" : "Co-registration unverified",
        status: sameCrs && sameDims ? "pass" : "warn",
      });
      if (!sameCrs || !sameDims) {
        blockingMessage =
          "Images are not spatially compatible. The two acquisitions must share a spatial reference and a compatible pixel grid before change detection can run.";
      }
    }
  }

  if (mode === "optical-sar") {
    const hasOptical = ready.some((i) => i.metadata.modality === "OPTICAL");
    const hasSar = ready.some((i) => i.metadata.modality === "SAR");
    checks.push({
      id: "opt",
      label: "Optical input present",
      status: hasOptical ? "pass" : "fail",
    });
    checks.push({ id: "sar", label: "SAR input present", status: hasSar ? "pass" : "fail" });
    if (!hasOptical || !hasSar) {
      blockingMessage =
        "The uploaded inputs do not match the selected analysis mode. Optical + SAR fusion requires one optical and one SAR acquisition.";
    }
  }

  const failed = checks.some((c) => c.status === "fail");
  return { ok: !failed && !blockingMessage, checks, blockingMessage };
}

/* ── Intent classification (LeadOrchestratorAgent mock) ────── */

export function classifyIntent(mode: AnalysisMode, query: string, images: UploadedImage[]) {
  const q = query.toLowerCase();
  const hasSar = images.some((i) => i.metadata.modality === "SAR");
  const hasOptical = images.some((i) => i.metadata.modality === "OPTICAL");

  let intent: IntentType = "vqa";
  if (mode === "bitemporal") intent = "change-detection";
  else if (mode === "optical-sar") intent = "optical-sar-fusion";
  else if (/\bwhere\b|locate|find |show me|pinpoint/.test(q)) intent = "grounding";
  else if (/describe|caption|overview|summar/.test(q)) intent = "captioning";
  else if (/land[- ]?cover|segment|classif/.test(q)) intent = "segmentation";
  else if (/chang|expansion|difference|compare/.test(q) && images.length > 1)
    intent = "change-detection";

  const labels: Record<IntentType, string> = {
    vqa: "Visual Question Answering",
    captioning: "Scene Captioning",
    grounding: "Text-Guided Visual Grounding",
    "change-detection": "Change Detection",
    "optical-sar-fusion": "Optical + SAR Fusion",
    segmentation: "Land-Cover Segmentation",
  };

  /** Map intent → specialist agent (Agent 4-7) */
  const intentToAgent: Record<IntentType, AgentId> = {
    vqa: "single_scene_vqa",
    captioning: "single_scene_vqa",
    grounding: "visual_grounding",
    "change-detection": "bi_temporal_change",
    "optical-sar-fusion": "sar_cloud_penetration",
    segmentation: "single_scene_vqa",
  };

  return {
    intent,
    intentLabel: labels[intent],
    specialistAgent: intentToAgent[intent],
  };
}

/* ── Pipeline: 7-agent execution ───────────────────────────── */

export function buildPipeline(): PipelineStep[] {
  return [
    {
      id: "orchestrator",
      agentId: "lead_orchestrator",
      label: "LeadOrchestrator — Intent classification",
      status: "waiting",
      outputType: "rule_based",
    },
    {
      id: "validator",
      agentId: "geo_validator",
      label: "GeoValidator — Metadata extraction & cloud check",
      status: "waiting",
      outputType: "rule_based",
    },
    {
      id: "specialist",
      agentId: "single_scene_vqa",
      label: "Specialist agent — Model inference",
      status: "waiting",
      outputType: "mock",
    },
    {
      id: "ledger",
      agentId: "audit_ledger",
      label: "AuditLedger — SQLite audit logging",
      status: "waiting",
      outputType: "rule_based",
    },
  ];
}

export const STEP_MESSAGES: Record<string, string> = {
  orchestrator: "Classifying intent and building execution DAG…",
  validator: "Extracting metadata, checking cloud contamination…",
  specialist: "Running specialist model inference…",
  ledger: "Writing audit trail to SQLite…",
};

/* ── History store ─────────────────────────────────────────── */

const store = new Map<string, AnalysisResult>();
let counter = 124;

export function mockHistory(): AnalysisSummary[] {
  return [...store.values()]
    .sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1))
    .map((r) => ({
      id: r.id,
      createdAt: r.createdAt,
      mode: r.mode,
      intentLabel: r.intentLabel,
      imageCount: r.images.length,
      query: r.query,
      status: r.status,
      confidence: r.confidence,
      outputType: r.outputType,
      totalExecutionMs: r.totalExecutionMs,
      demo: r.demo,
    }));
}

export function mockGetAnalysis(id: string): AnalysisResult | undefined {
  return store.get(id);
}

function nextId(): string {
  counter += 1;
  return `SQ-${new Date().getFullYear()}-${String(counter).padStart(5, "0")}`;
}

function matchScenario(req: AnalysisRequest, images: UploadedImage[]): DemoScenario | undefined {
  if (!req.demo) return undefined;
  return (
    DEMO_SCENARIOS.find((s) => s.images.some((i) => images.some((u) => u.id === i.id))) ??
    DEMO_SCENARIOS.find((s) => s.mode === req.mode)
  );
}

/**
 * Simulated 7-agent pipeline run.
 * Each step maps to a real agent from the README architecture.
 */
export async function mockRunAnalysis(
  req: AnalysisRequest,
  images: UploadedImage[],
  onStep: (steps: PipelineStep[]) => void,
): Promise<AnalysisResult> {
  const id = nextId();
  const createdAt = new Date().toISOString();
  const steps = buildPipeline();
  const scenario = matchScenario(req, images);
  const routed = classifyIntent(req.mode, req.query, images);

  // Set specialist step to correct agent
  const specialistStep = steps[2];
  if (specialistStep) {
    specialistStep.agentId = routed.specialistAgent;
    specialistStep.label = `${AGENT_REGISTRY.find((a) => a.id === routed.specialistAgent)?.name ?? "Specialist"} — ${routed.intentLabel}`;
  }

  const stepDetail: Record<string, string> = {
    orchestrator: `Intent: ${routed.intentLabel}`,
    validator: "Passed — format OK, cloud fraction within threshold",
    specialist: "Completed",
    ledger: `Logged to SQLite (${id})`,
  };

  const durations: Record<string, number> = {
    orchestrator: 320,
    validator: 240,
    specialist: 1400,
    ledger: 180,
  };

  const traceEntries: ExecutionTrace["entries"] = [];
  let total = 0;

  for (const step of steps) {
    step.status = "running";
    step.startedAt = new Date().toISOString();
    onStep(steps.map((s) => ({ ...s })));
    await sleep(durations[step.id] ?? 300);
    step.status = "completed";
    step.durationMs = durations[step.id] ?? 300;
    step.detail = stepDetail[step.id];
    step.outputType = step.id === "specialist" ? (scenario ? "mock" : "mock") : "rule_based";
    total += step.durationMs;
    traceEntries.push({
      agent: step.agentId,
      stage: step.label,
      outcome: step.detail ?? "Completed",
      outputType: step.outputType,
      timestamp: step.startedAt!,
      durationMs: step.durationMs,
    });
    onStep(steps.map((s) => ({ ...s })));
  }

  const outputType: OutputType = scenario ? "mock" : "mock";

  const result: AnalysisResult = {
    id,
    createdAt,
    mode: req.mode,
    intent: scenario?.expectedTask ?? routed.intent,
    intentLabel: scenario?.taskLabel ?? routed.intentLabel,
    query: req.query,
    status: "completed",
    answer:
      "Backend analysis requested. Please ensure the SatQuery AI FastAPI service is running on port 8000.",
    confidence: null,
    outputType,
    agentsUsed: ["lead_orchestrator", "geo_validator", routed.specialistAgent, "audit_ledger"],
    images,
    evidence: {
      groundingBoxes: null,
      changeMapUrl: null,
      changedAreaPercent: null,
      segmentationUrl: null,
      landCover: null,
      notes: "Analysis dispatched to backend gateway.",
    },
    steps: steps.map((s) => ({ ...s })),
    trace: { analysisId: id, entries: traceEntries, totalDurationMs: total },
    totalExecutionMs: total,
    demo: false,
  };

  store.set(id, result);
  return result;
}

export function seedDemoHistory() {
  // No-op: Removed all boilerplate history seeds. Real analyses are loaded from backend.
  return;
}
