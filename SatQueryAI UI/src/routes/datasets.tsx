import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { getDatasets } from "@/api/satquery";
import { AppShell } from "@/components/layout/AppShell";
import { Panel, PanelHeader } from "@/components/common/Panel";
import { Database, Globe, Layers, MapPin } from "lucide-react";

export const Route = createFileRoute("/datasets")({
  component: Datasets,
});

function Datasets() {
  const { data: datasets, isLoading } = useQuery({
    queryKey: ["datasets"],
    queryFn: getDatasets,
    staleTime: 60_000,
  });

  return (
    <AppShell title="Datasets" breadcrumb={["SatQuery AI", "Datasets"]}>
      <div className="space-y-4">
        <Panel>
          <PanelHeader
            title="Dataset Registry"
            subtitle="Available remote-sensing datasets for analysis"
            icon={<Database aria-hidden="true" className="size-4" />}
          />

          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <div className="size-5 animate-spin rounded-full border-2 border-primary border-t-transparent" />
              <span className="ml-2 text-sm text-muted-foreground">Loading datasets…</span>
            </div>
          ) : !datasets || datasets.length === 0 ? (
            <div className="flex flex-col items-center gap-3 py-12 text-center">
              <Database aria-hidden="true" className="size-8 text-muted-foreground" />
              <p className="text-sm font-medium">No datasets available</p>
              <p className="text-xs text-muted-foreground">
                Connect to the backend to access available datasets.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-3 p-3 sm:grid-cols-2">
              {datasets.map((ds) => (
                <div
                  key={ds.id}
                  className="flex flex-col gap-3 rounded-md border border-border bg-surface/50 p-4 transition-colors hover:border-primary/30"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <h3 className="text-sm font-semibold">{ds.name}</h3>
                      <p className="mt-1 text-xs text-muted-foreground">{ds.description}</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-3">
                    <div className="flex items-center gap-1.5">
                      <Globe aria-hidden="true" className="size-3 text-primary" />
                      <span className="text-muted-foreground">Modality:</span>
                      <span className="font-medium">{ds.modality}</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <MapPin aria-hidden="true" className="size-3 text-primary" />
                      <span className="text-muted-foreground">Resolution:</span>
                      <span className="font-medium">{ds.resolution}</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <Layers aria-hidden="true" className="size-3 text-primary" />
                      <span className="text-muted-foreground">Scenes:</span>
                      <span className="font-mono tabular-nums">
                        {ds.sceneCount.toLocaleString()}
                      </span>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-1">
                    {ds.tasks.map((task) => (
                      <span
                        key={task}
                        className="inline-flex items-center rounded-sm border border-border bg-muted px-1.5 py-0.5 font-mono text-[10px] tracking-wider text-muted-foreground"
                      >
                        {task}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </Panel>
      </div>
    </AppShell>
  );
}
