/**
 * Thin HTTP client for the SatQuery AI backend (Python / FastAPI).
 * The base URL is provided by the environment — never hard-coded, and no
 * credentials or model API keys are ever held in the frontend.
 */
import type { ApiError } from "@/types";

export const API_BASE_URL: string = (
  import.meta.env["VITE_SATQUERY_API_URL"] ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

export const IS_LIVE_BACKEND = true;

export const API_ROUTES = {
  upload: "/api/upload",
  validate: "/api/validate",
  analyze: "/api/analyze",
  analysis: (id: string) => `/api/analysis/${id}`,
  trace: (id: string) => `/api/analysis/${id}/trace`,
  report: (id: string, format: "pdf" | "json") => `/api/analysis/${id}/report?format=${format}`,
  models: "/api/models",
  systemStatus: "/api/system/status",
  history: "/api/analysis",
  datasets: "/api/datasets",
} as const;

export class SatQueryApiError extends Error implements ApiError {
  code: string;
  hint?: string | undefined;

  constructor(code: string, message: string, hint?: string) {
    super(message);
    this.name = "SatQueryApiError";
    this.code = code;
    this.hint = hint;
  }
}

export async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  if (!IS_LIVE_BACKEND) {
    throw new SatQueryApiError(
      "backend_unavailable",
      "Agent Controller is currently unavailable. Please check the backend service.",
      "Set VITE_SATQUERY_API_URL to the FastAPI service URL.",
    );
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
        ...(init?.headers ?? {}),
      },
    });
  } catch {
    throw new SatQueryApiError(
      "network_error",
      "Agent Controller is currently unavailable. Please check the backend service.",
    );
  }

  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new SatQueryApiError(
      `http_${response.status}`,
      detail || `Request failed with status ${response.status}.`,
    );
  }

  return (await response.json()) as T;
}
