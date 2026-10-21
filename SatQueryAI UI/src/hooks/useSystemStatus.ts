import { useQuery } from "@tanstack/react-query";
import { getSystemStatus } from "@/api/satquery";

export function useSystemStatus() {
  return useQuery({
    queryKey: ["system-status"],
    queryFn: getSystemStatus,
    staleTime: 30_000,
    refetchInterval: 60_000,
  });
}
