import { cn } from "@/lib/utils";

export function ConfidenceMeter({
  value,
  className,
}: {
  value: number | null;
  className?: string;
}) {
  if (value === null || value === undefined) {
    return (
      <div className={cn("space-y-2", className)}>
        <p className="font-mono text-sm text-muted-foreground">Awaiting backend result</p>
        <div className="h-1.5 w-full rounded-full bg-muted" />
      </div>
    );
  }

  // Backend delivers confidence as a normalized float in [0.0, 1.0] (e.g. 0.639 = 64%, 0.88 = 88%).
  // If value <= 1.0, scale by 100 to get the correct percentage.
  const percentage = value <= 1.0 && value > 0 ? Math.round(value * 100) : Math.round(value);

  const tone =
    percentage >= 75 ? "bg-emerald-400" : percentage >= 50 ? "bg-sky-400" : "bg-amber-400";

  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-baseline gap-2">
        <span className="font-mono text-3xl font-semibold tabular-nums">{percentage}%</span>
        <span className="text-xs text-muted-foreground">model-reported confidence</span>
      </div>
      <div
        role="meter"
        aria-valuenow={percentage}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Answer confidence"
        className="h-1.5 w-full overflow-hidden rounded-full bg-muted"
      >
        <div
          className={cn("h-full rounded-full transition-[width] duration-700", tone)}
          style={{ width: `${Math.min(100, Math.max(0, percentage))}%` }}
        />
      </div>
    </div>
  );
}
