import type { ReactNode } from "react";
import { AlertTriangle, Inbox } from "lucide-react";
import { cn } from "@/lib/utils";

export function EmptyState({
  title,
  description,
  icon,
  action,
  className,
}: {
  title: string;
  description: string;
  icon?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-border px-6 py-14 text-center",
        className,
      )}
    >
      <span className="text-muted-foreground">{icon ?? <Inbox className="size-6" />}</span>
      <h3 className="text-sm font-semibold">{title}</h3>
      <p className="max-w-md text-xs text-muted-foreground">{description}</p>
      {action}
    </div>
  );
}

export function ErrorState({
  title,
  message,
  hint,
  action,
  className,
}: {
  title: string;
  message: string;
  hint?: string;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div
      role="alert"
      className={cn(
        "flex gap-3 rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3",
        className,
      )}
    >
      <AlertTriangle aria-hidden="true" className="mt-0.5 size-4 shrink-0 text-destructive" />
      <div className="space-y-1">
        <p className="text-sm font-semibold text-foreground">{title}</p>
        <p className="text-xs text-foreground/80">{message}</p>
        {hint ? <p className="font-mono text-[11px] text-muted-foreground">{hint}</p> : null}
        {action}
      </div>
    </div>
  );
}
