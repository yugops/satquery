import { cn } from "@/lib/utils";

export type DotTone = "online" | "degraded" | "offline" | "unknown" | "running";

const TONE_CLASS: Record<DotTone, string> = {
  online: "bg-success",
  degraded: "bg-warning",
  offline: "bg-destructive",
  unknown: "bg-muted-foreground",
  running: "bg-primary animate-status-pulse",
};

export function StatusDot({ tone, className }: { tone: DotTone; className?: string }) {
  return (
    <span
      aria-hidden="true"
      className={cn("inline-block size-2 shrink-0 rounded-full", TONE_CLASS[tone], className)}
    />
  );
}
