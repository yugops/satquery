import type { AnalysisMode, Evidence, ImageMetadata, IntentType, UploadedImage } from "@/types";
import { DEMO_IMAGERY } from "./imagery";

export interface DemoScenario {
  id: string;
  title: string;
  description: string;
  mode: AnalysisMode;
  query: string;
  images: UploadedImage[];
  expectedTask: IntentType;
  taskLabel: string;
  modelsUsed: string[];
  answer: string;
  confidence: number;
  evidence: Evidence;
}

function meta(partial: Partial<ImageMetadata> & { filename: string }): ImageMetadata {
  return {
    fileType: "image/tiff",
    fileSizeBytes: 18_432_000,
    width: 1024,
    height: 1024,
    bands: 4,
    crs: "EPSG:4326",
    modality: "OPTICAL",
    sensor: "Sentinel-2 MSI (simulated)",
    acquisitionDate: "2024-03-12T05:41:00Z",
    resolutionMeters: 10,
    polarizations: null,
    boundsWgs84: [77.512, 12.882, 77.638, 13.004],
    cloudFraction: 0.05,
    ...partial,
  };
}

function img(
  id: string,
  role: UploadedImage["role"],
  previewUrl: string,
  metadata: ImageMetadata,
): UploadedImage {
  return { id, role, previewUrl, status: "ready", progress: 100, metadata };
}

export const DEMO_SCENARIOS: DemoScenario[] = [];

export function getScenario(id: string): DemoScenario | undefined {
  return DEMO_SCENARIOS.find((s) => s.id === id);
}
