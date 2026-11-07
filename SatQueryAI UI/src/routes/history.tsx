import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { getHistory } from "@/api/satquery";
import { AppShell } from "@/components/layout/AppShell";
import { Panel, PanelHeader } from "@/components/common/Panel";
import { DemoBadge } from "@/components/common/DemoBadge";
import { formatDateTime, MODE_LABELS } from "@/utils/format";
import { Clock, History, ScanSearch } from "lucide-react";

export const Route = createFileRoute("/history")({
  component: AnalysisHistory,
});

function AnalysisHistory() {
  const { data: history, isLoading } = useQuery({
    queryKey: ["history"],
    queryFn: getHistory,
    staleTime: 30_000,
  });

  return (
    <AppShell title="Analysis History" breadcrumb={["SatQuery AI", "History"]}>
      <div className="space-y-4">
        <Panel>
          <PanelHeader
            title="Analysis History"
            subtitle="All completed analysis runs"
            icon={<History aria-hidden="true" className="size-4" />}
            actions={
              <Link
                to="/"
                className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground transition-colors hover:bg-primary/90"
              >
                <ScanSearch aria-hidden="true" className="size-3.5" />
                New Analysis
              </Link>
            }
          />

          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <div className="size-5 animate-spin rounded-full border-2 border-primary border-t-transparent" />
              <span className="ml-2 text-sm text-muted-foreground">Loading history…</span>
            </div>
          ) : !history || history.length === 0 ? (
            <div className="flex flex-col items-center gap-3 py-12 text-center">
              <History aria-hidden="true" className="size-8 text-muted-foreground" />
              <p className="text-sm font-medium">No analyses yet</p>
              <p className="text-xs text-muted-foreground">
                Run your first analysis to see results here.
              </p>
              <Link
                to="/"
                className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground transition-colors hover:bg-primary/90"
              >
                <ScanSearch aria-hidden="true" className="size-3.5" />
                Start analysis
              </Link>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left">
                    <th className="px-4 py-2 font-mono text-[10px] tracking-[0.12em] uppercase text-muted-foreground">
                      ID
                    </th>
                    <th className="hidden px-4 py-2 font-mono text-[10px] tracking-[0.12em] uppercase text-muted-foreground sm:table-cell">
                      Mode
                    </th>
                    <th className="px-4 py-2 font-mono text-[10px] tracking-[0.12em] uppercase text-muted-foreground">
                      Task
                    </th>
                    <th className="hidden px-4 py-2 font-mono text-[10px] tracking-[0.12em] uppercase text-muted-foreground md:table-cell">
                      Query
                    </th>
                    <th className="hidden px-4 py-2 font-mono text-[10px] tracking-[0.12em] uppercase text-muted-foreground sm:table-cell">
                      Confidence
                    </th>
                    <th className="hidden px-4 py-2 font-mono text-[10px] tracking-[0.12em] uppercase text-muted-foreground lg:table-cell">
                      Created
                    </th>
                    <th className="px-4 py-2 font-mono text-[10px] tracking-[0.12em] uppercase text-muted-foreground">
                      Status
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((a) => (
                    <tr
                      key={a.id}
                      className="border-b border-border last:border-b-0 hover:bg-surface/50"
                    >
                      <td className="px-4 py-2.5">
                        <div className="flex items-center gap-2">
                          <Link
                            to="/analysis/$id"
                            params={{ id: a.id }}
                            className="font-mono text-xs text-primary hover:underline"
                          >
                            {a.id}
                          </Link>
                          {a.demo && <DemoBadge />}
                        </div>
                      </td>
                      <td className="hidden px-4 py-2.5 text-xs sm:table-cell">
                        {MODE_LABELS[a.mode]}
                      </td>
                      <td className="px-4 py-2.5 text-xs text-muted-foreground">
                        {a.intentLabel ?? "—"}
                      </td>
                      <td className="hidden max-w-[220px] truncate px-4 py-2.5 text-xs text-muted-foreground md:table-cell">
                        {a.query}
                      </td>
                      <td className="hidden px-4 py-2.5 font-mono text-xs tabular-nums sm:table-cell">
                        {a.confidence !== null
                          ? `${a.confidence <= 1.0 ? Math.round(a.confidence * 100) : Math.round(a.confidence)}%`
                          : "—"}
                      </td>
                      <td className="hidden whitespace-nowrap px-4 py-2.5 text-xs text-muted-foreground lg:table-cell">
                        <Clock aria-hidden="true" className="mr-1 inline size-3" />
                        {formatDateTime(a.createdAt)}
                      </td>
                      <td className="px-4 py-2.5">
                        <span
                          className={`inline-flex items-center gap-1 rounded-sm px-1.5 py-0.5 font-mono text-[10px] tracking-wider uppercase ${
                            a.status === "completed"
                              ? "bg-success/10 text-success"
                              : a.status === "running"
                                ? "bg-primary/10 text-primary"
                                : a.status === "failed"
                                  ? "bg-destructive/10 text-destructive"
                                  : "bg-muted text-muted-foreground"
                          }`}
                        >
                          {a.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      </div>
    </AppShell>
  );
}
