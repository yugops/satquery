import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export function Panel({
  children,
  className,
  as: As = "section",
}: {
  children: ReactNode;
  className?: string;
  as?: "section" | "div" | "aside";
}) {
  return <As className={cn("panel", className)}>{children}</As>;
}

export function PanelHeader({
  title,
  subtitle,
  actions,
  icon,
  className,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  icon?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-3",
        className,
      )}
    >
      <div className="flex items-start gap-2.5">
        {icon ? <span className="mt-0.5 text-primary">{icon}</span> : null}
        <div>
          <h2 className="text-sm font-semibold tracking-tight">{title}</h2>
          {subtitle ? <p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p> : null}
        </div>
      </div>
      {actions ? <div className="flex items-center gap-2">{actions}</div> : null}
    </div>
  );
}

export function MonoLabel({ children, className }: { children: ReactNode; className?: string }) {
  return <span className={cn("label-mono block", className)}>{children}</span>;
}
