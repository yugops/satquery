"""
SatQuery AI — Agent 21: EdgeDeployExportAgent Tools
Problem Statement ID: 26167 (ISRO / SAC)
"""

from __future__ import annotations

from typing import Any


def tool_generate_edge_export_manifest(model_type: str = "ConvNeXt-SAR + Qwen2-VL") -> dict[str, Any]:
    """Generates edge deployment packaging manifest for Jetson Orin / Hailo-8 NPU / CPU."""
    raise NotImplementedError(
        f"Edge export for '{model_type}' is not implemented — no ONNX/GGUF "
        f"conversion or benchmarking pipeline exists in this build. No "
        f"latency or memory figures were measured or substituted."
    )
