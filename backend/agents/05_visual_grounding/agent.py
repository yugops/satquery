from __future__ import annotations

import logging
from typing import Any, Optional
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

from core.interfaces import BaseAgent
from core.models import ComputationMode
from .tools import tool_grounding_dino_bbox, tool_mobilesam_segment_polygon


class VisualGroundingAgent(BaseAgent):
    agent_id = "visual_grounding"
    agent_name = "VisualGroundingAgent"
    mempalace_wing = "Wing: Grounding_Registry"

    async def _execute(self, **kwargs: Any) -> tuple[dict[str, Any], ComputationMode, float, Optional[str], Optional[str]]:
        image_path = kwargs.get("image_path", "")
        text_prompt = kwargs.get("query") or kwargs.get("target_prompt", "infrastructure and objects")

        if not image_path:
            return {}, ComputationMode.RULE_BASED, 0.0, "No image provided for grounding.", None

        try:
            image = Image.open(image_path)
        except Exception as e:
            logger.error(f"Cannot load image for grounding: {e}")
            return {}, ComputationMode.RULE_BASED, 0.0, f"Cannot load image for grounding: {e}", None

        bboxes = tool_grounding_dino_bbox(image, text_prompt, self.config.default_box_threshold)
        geojson = tool_mobilesam_segment_polygon(image, bboxes)

        comp_mode = ComputationMode.REAL_INFERENCE if len(bboxes) > 0 else ComputationMode.RULE_BASED
        conf = (sum(b.confidence for b in bboxes) / len(bboxes)) if bboxes else 0.0

        summary = f"Grounded {len(bboxes)} bounding instances for '{text_prompt}' with MobileSAM polygon vectors."
        data = {
            "prompt": text_prompt,
            "bboxes": [b.model_dump() for b in bboxes],
            "geojson": geojson,
            "instance_count": len(bboxes),
        }

        return data, comp_mode, round(conf, 3), summary, None
