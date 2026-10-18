import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { getAnalysis } from "@/api/satquery";
import { AppShell } from "@/components/layout/AppShell";
import { Panel, PanelHeader } from "@/components/common/Panel";
import { ConfidenceMeter } from "@/components/common/ConfidenceMeter";
import { MarkdownContent } from "@/components/common/MarkdownContent";
import { DemoBadge } from "@/components/common/DemoBadge";
import { ErrorState } from "@/components/common/StateBlocks";
import { formatDateTime, formatDuration, MODE_LABELS } from "@/utils/format";
import { ArrowLeft, Clock, Download, FileText, Image, Layers } from "lucide-react";
import { reportUrl, buildReportPayload } from "@/api/satquery";

export const Route = createFileRoute("/analysis/$id")({
  component: AnalysisDetail,
});

function AnalysisDetail() {
  const { id } = Route.useParams();
  const {
    data: result,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["analysis", id],
    queryFn: () => getAnalysis(id),
    staleTime: 60_000,
  });

  if (isLoading) {
    return (
      <AppShell title="Loading…" breadcrumb={["SatQuery AI", "Analysis", id]}>
        <div className="flex items-center justify-center py-20">
          <div className="size-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          <span className="ml-3 text-sm text-muted-foreground">Loading analysis…</span>
        </div>
      </AppShell>
    );
  }

  if (error || !result) {
    return (
      <AppShell title="Not found" breadcrumb={["SatQuery AI", "Analysis", id]}>
        <ErrorState
          title="Analysis not found"
          message={error instanceof Error ? error.message : `No analysis found with ID ${id}.`}
          action={
            <Link
              to="/history"
              className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
            >
              <ArrowLeft className="size-3" /> Back to history
            </Link>
          }
        />
      </AppShell>
    );
  }

  const downloadReport = (format: "pdf" | "json") => {
    const url = reportUrl(id, format);
    if (url) {
      window.open(url, "_blank");
    } else {
      const payload = buildReportPayload(result);
      const blob = new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json",
      });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `${result.id}-report.json`;
      link.click();
      URL.revokeObjectURL(link.href);
    }
  };

  return (
    <AppShell
      title={`Analysis ${result.id}`}
      breadcrumb={["SatQuery AI", "Analysis", result.id]}
      actions={
        <div className="flex items-center gap-2">
          {result.demo && <DemoBadge />}
          <button
            onClick={() => downloadReport("json")}
            className="inline-flex items-center gap-1.5 rounded-md border border-border bg-surface px-2.5 py-1.5 text-[11px] font-medium transition-colors hover:bg-accent"
          >
            <Download className="size-3" />
            JSON
          </button>
        </div>
      }
    >
      <div className="space-y-4">
        {/* Summary header */}
        <Panel>
          <div className="flex flex-wrap items-start gap-4 p-4">
            <div className="min-w-0 flex-1 space-y-2">
              <div className="flex flex-wrap items-center gap-2">
                <span
                  className={`inline-flex items-center gap-1 rounded-sm px-1.5 py-0.5 font-mono text-[10px] tracking-wider uppercase ${
                    result.status === "completed"
                      ? "bg-success/10 text-success"
                      : result.status === "running"
                        ? "bg-primary/10 text-primary"
                        : result.status === "failed"
                          ? "bg-destructive/10 text-destructive"
                          : "bg-muted text-muted-foreground"
                  }`}
                >
                  {result.status}
                </span>
                <span className="text-xs text-muted-foreground">{MODE_LABELS[result.mode]}</span>
                {result.demo && <DemoBadge />}
              </div>
              <p className="text-sm leading-relaxed text-muted-foreground">{result.query}</p>
            </div>
            <div className="shrink-0">
              <ConfidenceMeter value={result.confidence} className="w-48" />
            </div>
          </div>
        </Panel>

        {/* Metadata grid */}
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div className="panel flex items-center gap-3 p-3">
            <Clock aria-hidden="true" className="size-4 text-primary" />
            <div>
              <p className="font-mono text-[10px] tracking-wider uppercase text-muted-foreground">
                Created
              </p>
              <p className="text-xs">{formatDateTime(result.createdAt)}</p>
            </div>
          </div>
          <div className="panel flex items-center gap-3 p-3">
            <Layers aria-hidden="true" className="size-4 text-primary" />
            <div>
              <p className="font-mono text-[10px] tracking-wider uppercase text-muted-foreground">
                Task
              </p>
              <p className="text-xs">{result.intentLabel ?? "—"}</p>
            </div>
          </div>
          <div className="panel flex items-center gap-3 p-3">
            <Image aria-hidden="true" className="size-4 text-primary" />
            <div>
              <p className="font-mono text-[10px] tracking-wider uppercase text-muted-foreground">
                Images
              </p>
              <p className="text-xs">{result.images.length} uploaded</p>
            </div>
          </div>
          <div className="panel flex items-center gap-3 p-3">
            <Clock aria-hidden="true" className="size-4 text-primary" />
            <div>
              <p className="font-mono text-[10px] tracking-wider uppercase text-muted-foreground">
                Duration
              </p>
              <p className="text-xs">{formatDuration(result.totalExecutionMs)}</p>
            </div>
          </div>
        </div>

        {/* Answer */}
        <Panel>
          <PanelHeader
            title="Answer"
            subtitle={result.intentLabel ?? "Analysis complete"}
            icon={<FileText aria-hidden="true" className="size-4" />}
          />
          <div className="p-4">
            <MarkdownContent content={result.answer ?? "No answer available."} />
          </div>
        </Panel>

        {/* Models used */}
        {result.agentsUsed.length > 0 && (
          <Panel>
            <PanelHeader
              title="Agents Used"
              subtitle="Pipeline agent execution trace"
              icon={<Layers aria-hidden="true" className="size-4" />}
            />
            <div className="flex flex-wrap gap-2 p-4">
              {result.agentsUsed.map((m) => (
                <span
                  key={m}
                  className="inline-flex items-center rounded-sm border border-border bg-surface px-2 py-0.5 font-mono text-[10px] tracking-wider text-muted-foreground"
                >
                  {m}
                </span>
              ))}
            </div>
          </Panel>
        )}

        {/* Pipeline trace */}
        {result.steps.length > 0 && (
          <Panel>
            <PanelHeader
              title="Pipeline Trace"
              subtitle="Execution timeline"
              icon={<Clock aria-hidden="true" className="size-4" />}
            />
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left">
                    <th className="px-4 py-2 font-mono text-[10px] tracking-[0.12em] uppercase text-muted-foreground">
                      Step
                    </th>
                    <th className="px-4 py-2 font-mono text-[10px] tracking-[0.12em] uppercase text-muted-foreground">
                      Status
                    </th>
                    <th className="hidden px-4 py-2 font-mono text-[10px] tracking-[0.12em] uppercase text-muted-foreground sm:table-cell">
                      Detail
                    </th>
                    <th className="px-4 py-2 font-mono text-[10px] tracking-[0.12em] uppercase text-muted-foreground">
                      Duration
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {result.steps.map((s) => (
                    <tr key={s.id} className="border-b border-border last:border-b-0">
                      <td className="px-4 py-2 text-xs font-medium">{s.label}</td>
                      <td className="px-4 py-2">
                        <span
                          className={`inline-flex items-center gap-1 rounded-sm px-1.5 py-0.5 font-mono text-[10px] tracking-wider uppercase ${
                            s.status === "completed"
                              ? "bg-success/10 text-success"
                              : s.status === "running"
                                ? "bg-primary/10 text-primary"
                                : s.status === "failed"
                                  ? "bg-destructive/10 text-destructive"
                                  : "bg-muted text-muted-foreground"
                          }`}
                        >
                          {s.status}
                        </span>
                      </td>
                      <td className="hidden px-4 py-2 text-xs text-muted-foreground sm:table-cell">
                        {s.detail ?? "—"}
                      </td>
                      <td className="px-4 py-2 font-mono text-xs tabular-nums">
                        {s.durationMs != null ? `${s.durationMs}ms` : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        )}

        {/* Evidence notes */}
        {result.evidence?.notes && (
          <Panel>
            <PanelHeader title="Evidence Notes" subtitle="Supplementary information" />
            <div className="p-4 text-xs text-muted-foreground">{result.evidence.notes}</div>
          </Panel>
        )}
      </div>
    </AppShell>
  );
}
