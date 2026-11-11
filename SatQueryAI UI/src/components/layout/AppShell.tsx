import { type ReactNode } from "react";
import { AppSidebar } from "./Sidebar";
import { TopBar } from "./TopBar";
import GlobeStudy from "@/components/ui/globe-study";

export function AppShell({
  title,
  breadcrumb,
  actions,
  children,
}: {
  title: string;
  breadcrumb: string[];
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="relative flex flex-col md:flex-row min-h-screen bg-black text-foreground overflow-x-hidden">
      {/* Background Interactive Globe */}
      <div
        aria-hidden="true"
        className="pointer-events-none fixed inset-0 z-0 flex items-center justify-center overflow-hidden select-none"
      >
        <div className="h-[880px] w-[880px] max-w-[90vw] max-h-[90vh] opacity-40">
          <GlobeStudy mode="dark" scale={1.05} opacity={0.95} />
        </div>
      </div>

      {/* Modern Animated Expandable Sidebar */}
      <AppSidebar />

      <div className="relative z-10 flex min-w-0 flex-1 flex-col">
        <TopBar title={title} breadcrumb={breadcrumb} actions={actions} />
        <main className="flex-1 px-3 py-4 lg:px-6 lg:py-6">{children}</main>
      </div>
    </div>
  );
}
