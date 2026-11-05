import { BookText, Database, History, LayoutDashboard, Layers, Activity } from "lucide-react";
import type { LucideIcon } from "lucide-react";

export interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  exact?: boolean;
}

export const NAV_ITEMS: NavItem[] = [
  { to: "/", label: "Overview", icon: LayoutDashboard, exact: true },
  { to: "/history", label: "Analysis History", icon: History },
  { to: "/datasets", label: "Datasets", icon: Database },
  { to: "/models", label: "Model Registry", icon: Layers },
  { to: "/status", label: "System Status", icon: Activity },
  { to: "/docs", label: "Documentation", icon: BookText },
];
