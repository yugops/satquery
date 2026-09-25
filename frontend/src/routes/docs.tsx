import { createFileRoute } from "@tanstack/react-router";
import {
  AlertTriangle,
  BookText,
  CheckCircle2,
  ExternalLink,
  GitBranch,
  Key,
  Server,
  Workflow,
} from "lucide-react";
import { AppShell } from "@/components/layout/AppShell";
import { Panel, PanelHeader } from "@/components/common/Panel";

export const Route = createFileRoute("/docs")({
  component: Docs,
});

function Endpoint({
  title,
  status,
  endpoint,
  method,
  description,
  responseShape,
}: {
  title: string;
  status: "required" | "optional";
  endpoint: string;
  method: string;
  description: string;
  responseShape?: string;
}) {
  return (
    <div className="rounded-md border border-border bg-surface/50 p-4">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <h4 className="text-sm font-semibold">{title}</h4>
        <span
          className={`inline-flex items-center gap-1 rounded-sm px-1.5 py-0.5 font-mono text-[10px] tracking-wider uppercase ${
            status === "required"
              ? "bg-destructive/10 text-destructive"
              : "bg-warning/10 text-warning"
          }`}
        >
          {status === "required" ? (
            <AlertTriangle aria-hidden="true" className="size-3" />
          ) : (
            <CheckCircle2 aria-hidden="true" className="size-3" />
          )}
          {status}
        </span>
      </div>
      <p className="mt-2 text-xs text-muted-foreground">{description}</p>
      <div className="mt-3 rounded bg-background/50 p-3 font-mono text-[11px]">
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded bg-primary/10 px-1.5 py-0.5 text-primary">{method}</span>
          <span className="text-muted-foreground">{endpoint}</span>
        </div>
        {responseShape && (
          <div className="mt-2">
            <p className="mb-1 text-[10px] text-muted-foreground">Response:</p>
            <pre className="whitespace-pre-wrap text-[10px] text-muted-foreground/80">
              {responseShape}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}

function Docs() {
  return (
    <AppShell title="Documentation" breadcrumb={["SatQuery AI", "Documentation"]}>
      <div className="space-y-4 max-w-4xl">
        {/* Architecture overview */}
        <Panel>
          <PanelHeader
            title="Architecture Overview"
            subtitle="ISRO/SAC Problem Statement 26167 — Multi-agent satellite imagery intelligence"
            icon={<BookText aria-hidden="true" className="size-4" />}
          />
          <div className="space-y-4 p-4">
            <p className="text-sm leading-relaxed text-muted-foreground">
              SatQuery AI uses an <strong>orchestrator-driven multi-agent pipeline</strong> backed
              by a typed internal schema (Pydantic v2) and a MemPalace memory/session architecture.
            </p>

            {/* Agent registry table */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left">
                    <th className="px-3 py-2 font-mono text-[10px] tracking-[0.12em] uppercase text-muted-foreground">
                      #
                    </th>
                    <th className="px-3 py-2 font-mono text-[10px] tracking-[0.12em] uppercase text-muted-foreground">
                      Agent
                    </th>
                    <th className="px-3 py-2 font-mono text-[10px] tracking-[0.12em] uppercase text-muted-foreground">
                      File
                    </th>
                    <th className="px-3 py-2 font-mono text-[10px] tracking-[0.12em] uppercase text-muted-foreground">
                      MemPalace Wing
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    {
                      n: "1",
                      agent: "LeadOrchestratorAgent",
                      file: "core/orchestrator.py",
                      wing: "Execution_Control",
                    },
                    {
                      n: "2",
                      agent: "GeoValidatorAgent",
                      file: "core/validator.py",
                      wing: "Geospatial_Integrity",
                    },
                    {
                      n: "3",
                      agent: "AuditLedgerAgent",
                      file: "core/ledger.py",
                      wing: "Audit_Compliance",
                    },
                    {
                      n: "4",
                      agent: "SingleSceneVQAAgent",
                      file: "vision_swarm/vqa_agent.py",
                      wing: "Visual_Reasoning",
                    },
                    {
                      n: "5",
                      agent: "VisualGroundingAgent",
                      file: "vision_swarm/grounding_agent.py",
                      wing: "Spatial_Grounding",
                    },
                    {
                      n: "6",
                      agent: "BiTemporalChangeAgent",
                      file: "vision_swarm/change_agent.py",
                      wing: "Temporal_Analysis",
                    },
                    {
                      n: "7",
                      agent: "SARCloudPenetrationAgent",
                      file: "vision_swarm/sar_agent.py",
                      wing: "SAR_Processing",
                    },
                  ].map((row) => (
                    <tr key={row.n} className="border-b border-border last:border-b-0">
                      <td className="px-3 py-2 font-mono text-xs text-muted-foreground">{row.n}</td>
                      <td className="px-3 py-2 text-xs font-medium">{row.agent}</td>
                      <td className="px-3 py-2 font-mono text-[10px] text-muted-foreground">
                        {row.file}
                      </td>
                      <td className="px-3 py-2 font-mono text-[10px] text-primary">{row.wing}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pipeline diagram */}
            <div className="rounded-md border border-border bg-surface/50 p-4">
              <h4 className="text-sm font-semibold">Pipeline Flow</h4>
              <div className="mt-3 space-y-1 font-mono text-[11px] text-muted-foreground">
                <p>User Query + Image(s)</p>
                <p className="ml-4">
                  → LeadOrchestratorAgent (Intent classification + DAG execution)
                </p>
                <p className="ml-8">→ GeoValidatorAgent (Metadata extraction, cloud detection)</p>
                <p className="ml-12">→ Specialist Agent (VQA / Grounding / Change / SAR)</p>
                <p className="ml-8">→ AuditLedgerAgent (SQLite audit logging)</p>
                <p className="ml-4">→ FinalResponse</p>
              </div>
            </div>
          </div>
        </Panel>

        {/* Environment */}
        <Panel>
          <PanelHeader
            title="Environment Configuration"
            subtitle="SATQUERY_* environment variables"
            icon={<Key aria-hidden="true" className="size-4" />}
          />
          <div className="overflow-x-auto p-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left">
                  <th className="px-3 py-2 font-mono text-[10px] tracking-[0.12em] uppercase text-muted-foreground">
                    Variable
                  </th>
                  <th className="px-3 py-2 font-mono text-[10px] tracking-[0.12em] uppercase text-muted-foreground">
                    Default
                  </th>
                  <th className="px-3 py-2 font-mono text-[10px] tracking-[0.12em] uppercase text-muted-foreground">
                    Description
                  </th>
                </tr>
              </thead>
              <tbody>
                {[
                  ["SATQUERY_MODE", "mock", "Execution mode: mock or real"],
                  ["SATQUERY_DB_PATH", "satquery.db", "SQLite audit database path"],
                  ["SATQUERY_LOG_LEVEL", "INFO", "Logging level"],
                  ["SATQUERY_VLM_MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct", "VLM model for VQA"],
                  ["SATQUERY_MAX_TOKENS", "256", "Max inference tokens"],
                  ["SATQUERY_CLOUD_CONTAMINATION_THRESHOLD", "0.4", "Cloud fraction threshold"],
                  ["SATQUERY_SAR_WATER_THRESHOLD_DB", "-18.0", "SAR water detection threshold"],
                ].map(([v, d, desc]) => (
                  <tr key={v} className="border-b border-border last:border-b-0">
                    <td className="px-3 py-2 font-mono text-[10px] text-primary">{v}</td>
                    <td className="px-3 py-2 font-mono text-[10px] text-muted-foreground">{d}</td>
                    <td className="px-3 py-2 text-xs text-muted-foreground">{desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        {/* API Endpoints */}
        <Panel>
          <PanelHeader
            title="API Endpoints"
            subtitle="FastAPI backend endpoints required for full functionality"
            icon={<Server aria-hidden="true" className="size-4" />}
          />
          <div className="space-y-3 p-4">
            <Endpoint
              title="System Status"
              status="required"
              endpoint="/api/system/status"
              method="GET"
              description="Returns agent statuses, compute info, and service health."
              responseShape={`{ version, mode, agentController, services[], agents[], compute }`}
            />
            <Endpoint
              title="Upload Imagery"
              status="required"
              endpoint="/api/upload"
              method="POST"
              description="Accepts multipart/form-data with satellite imagery files."
            />
            <Endpoint
              title="Validate Inputs"
              status="required"
              endpoint="/api/validate"
              method="POST"
              description="GeoValidatorAgent — validates imagery against analysis mode."
            />
            <Endpoint
              title="Run Analysis"
              status="required"
              endpoint="/api/analyze"
              method="POST"
              description="Runs the full 7-agent pipeline: LeadOrchestrator → GeoValidator → Specialist → AuditLedger."
            />
            <Endpoint
              title="Get Analysis"
              status="required"
              endpoint="/api/analysis/{id}"
              method="GET"
              description="Retrieves a completed analysis result by ID."
            />
            <Endpoint
              title="Get History"
              status="required"
              endpoint="/api/analysis"
              method="GET"
              description="Returns all completed analysis summaries."
            />
            <Endpoint
              title="Get Models"
              status="required"
              endpoint="/api/models"
              method="GET"
              description="Returns the model registry with capabilities and statuses."
            />
            <Endpoint
              title="Get Datasets"
              status="required"
              endpoint="/api/datasets"
              method="GET"
              description="Returns available remote-sensing datasets."
            />
            <Endpoint
              title="Download Report"
              status="optional"
              endpoint="/api/analysis/{id}/report?format=pdf|json"
              method="GET"
              description="Generates a downloadable report for a completed analysis."
            />
          </div>
        </Panel>

        {/* Output types (Rule #12) */}
        <Panel>
          <PanelHeader
            title="Output Types (Rule #12)"
            subtitle="Every AgentResponse includes an output_type field"
            icon={<GitBranch aria-hidden="true" className="size-4" />}
          />
          <div className="overflow-x-auto p-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left">
                  <th className="px-3 py-2 font-mono text-[10px] tracking-[0.12em] uppercase text-muted-foreground">
                    Type
                  </th>
                  <th className="px-3 py-2 font-mono text-[10px] tracking-[0.12em] uppercase text-muted-foreground">
                    Meaning
                  </th>
                </tr>
              </thead>
              <tbody>
                {[
                  ["model", "Real ML model inference output"],
                  ["rule_based", "Deterministic rule/heuristic output"],
                  ["mock", "Synthetic placeholder output (demo mode)"],
                  ["unavailable", "Agent failed or model not loaded"],
                ].map(([type, desc]) => (
                  <tr key={type} className="border-b border-border last:border-b-0">
                    <td className="px-3 py-2 font-mono text-xs text-primary">{type}</td>
                    <td className="px-3 py-2 text-xs text-muted-foreground">{desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        {/* Backend connectivity */}
        <Panel>
          <PanelHeader
            title="Backend Connectivity"
            subtitle="What the Python/FastAPI backend must implement"
            icon={<Workflow aria-hidden="true" className="size-4" />}
          />
          <div className="space-y-3 p-4">
            <div className="rounded-md border border-border bg-surface/50 p-4">
              <h4 className="text-sm font-semibold">Python Backend Setup</h4>
              <div className="mt-2 rounded bg-background/50 p-3 font-mono text-[11px] text-muted-foreground">
                <p>cd satquery</p>
                <p>python -m venv .venv</p>
                <p>.venv\Scripts\activate</p>
                <p>pip install -r requirements.txt</p>
                <p>copy .env.example .env</p>
                <p>uvicorn main:app --host 0.0.0.0 --port 8000</p>
              </div>
            </div>
            <div className="rounded-md border border-border bg-surface/50 p-4">
              <h4 className="text-sm font-semibold">Frontend Connection</h4>
              <div className="mt-2 rounded bg-background/50 p-3 font-mono text-[11px] text-muted-foreground">
                <p># Create .env.local in the project root</p>
                <p>VITE_SATQUERY_API_URL=http://localhost:8000</p>
              </div>
              <p className="mt-2 text-xs text-muted-foreground">
                When not set, the app operates in mock mode with the deterministic 7-agent pipeline
                simulation.
              </p>
            </div>
          </div>
        </Panel>

        {/* Project structure */}
        <Panel>
          <PanelHeader
            title="Backend Project Structure"
            subtitle="Python package layout for the satquery backend"
            icon={<GitBranch aria-hidden="true" className="size-4" />}
          />
          <div className="p-4">
            <pre className="rounded-md border border-border bg-surface/50 p-4 font-mono text-[11px] text-muted-foreground">
              {`satquery/
├── core/
│   ├── orchestrator.py      # Agent 1: LeadOrchestratorAgent
│   ├── validator.py          # Agent 2: GeoValidatorAgent
│   ├── ledger.py             # Agent 3: AuditLedgerAgent
│   ├── models.py             # All Pydantic types / enums
│   ├── interfaces.py         # BaseAgent, ModelAdapter ABCs
│   ├── exceptions.py         # Custom exception hierarchy
│   ├── config.py             # SatQueryConfig (pydantic-settings)
│   └── logging.py            # Structured logger factory
├── vision_swarm/
│   ├── vqa_agent.py          # Agent 4: SingleSceneVQAAgent
│   ├── grounding_agent.py    # Agent 5: VisualGroundingAgent
│   ├── change_agent.py       # Agent 6: BiTemporalChangeAgent
│   └── sar_agent.py          # Agent 7: SARCloudPenetrationAgent
├── tests/
│   ├── test_orchestrator.py
│   ├── test_validator.py
│   ├── test_vqa.py
│   ├── test_change.py
│   └── test_e2e.py
├── requirements.txt
└── .env.example`}
            </pre>
          </div>
        </Panel>
      </div>
    </AppShell>
  );
}
