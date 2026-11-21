"""
SatQuery AI — SAR ConvNeXt Specialist Model Loader
Problem Statement ID: 26167 (ISRO / SAC)
Dataset: BigEarthNet-S1 (Sentinel-1 SAR VV/VH)
Architecture: ConvNeXt-Tiny (28M params) + 3-Channel Radar Normalizer
"""

from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Any, Optional
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from core.models import ComputationMode

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


CORINE_19_CLASSES = [
    'Agro-forestry areas',
    'Arable land',
    'Beaches, dunes, sands',
    'Broad-leaved forest',
    'Coastal wetlands',
    'Complex cultivation patterns',
    'Coniferous forest',
    'Discontinuous urban fabric',
    'Dump sites',
    'Estuaries',
    'Industrial or commercial units',
    'Inland marshes',
    'Mineral extraction sites',
    'Moors and heathlands',
    'Mixed forest',
    'Natural grassland',
    'Pastures',
    'Peatbogs',
    'Water bodies'
]


class SatQuerySARConvNeXt(nn.Module):
    """ConvNeXt-Tiny Multi-Label Classifier for Sentinel-1 SAR."""
    def __init__(self, num_classes: int = 19, pretrained: bool = False):
        super().__init__()
        if HAS_TIMM:
            self.backbone = timm.create_model(
                "convnext_tiny",
                pretrained=pretrained,
                in_chans=3,
                num_classes=0
            )
            self.embed_dim = self.backbone.num_features
        else:
            from torchvision.models import resnet50
            res = resnet50(weights=None)
            self.embed_dim = res.fc.in_features
            res.fc = nn.Identity()
            self.backbone = res

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.embed_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features


class SARModelEngine:
    """Manages SAR ConvNeXt model loading, weights verification, and inference."""
    _instance: Optional[SARModelEngine] = None

    def __init__(self, checkpoint_path: str = "sar_convnext_shard0.pt"):
        self.checkpoint_path = Path(checkpoint_path)
        self.model: Optional[SatQuerySARConvNeXt] = None
        self.weights_sha256: Optional[str] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.is_file():
            candidates = [
                self.checkpoint_path,
                Path("d:/satquery/sar_convnext_shard0.pt"),
                Path("../sar_convnext_shard0.pt"),
                Path("../../sar_convnext_shard0.pt")
            ]
            for c in candidates:
                if c.is_file():
                    self.checkpoint_path = c
                    break

        if self.checkpoint_path.is_file():
            h = hashlib.sha256()
            with open(self.checkpoint_path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            self.weights_sha256 = h.hexdigest()

            try:
                model = SatQuerySARConvNeXt(num_classes=19, pretrained=False)
                state = torch.load(str(self.checkpoint_path), map_location=self.device)
                
                if isinstance(state, dict) and "model_state_dict" in state:
                    model.load_state_dict(state["model_state_dict"], strict=False)
                elif isinstance(state, dict):
                    model.load_state_dict(state, strict=False)
                
                model.to(self.device)
                model.eval()
                self.model = model
            except Exception:
                self.model = None

    def normalize_radar(self, vv: np.ndarray, vh: np.ndarray) -> torch.Tensor:
        """3-Channel SAR normalizer: [VV_dB, VH_dB, VV/VH ratio] in [0, 1]."""
        # Ensure 2D
        if vv.ndim == 3:
            vv = vv[..., 0]
        if vh.ndim == 3:
            vh = vh[..., 1] if vh.shape[-1] > 1 else vh[..., 0]

        # Align shapes if differing
        if vv.shape != vh.shape:
            min_h = min(vv.shape[0], vh.shape[0])
            min_w = min(vv.shape[1], vh.shape[1])
            vv = vv[:min_h, :min_w]
            vh = vh[:min_h, :min_w]

        eps = 1e-10
        if vv.min() >= 0:
            vv_db = 10.0 * np.log10(np.clip(vv ** 2 if vv.max() > 2.0 else vv, eps, None))
        else:
            vv_db = vv

        if vh.min() >= 0:
            vh_db = 10.0 * np.log10(np.clip(vh ** 2 if vh.max() > 2.0 else vh, eps, None))
        else:
            vh_db = vh

        vv_norm = np.clip((vv_db - (-25.0)) / (0.0 - (-25.0)), 0.0, 1.0)
        vh_norm = np.clip((vh_db - (-32.0)) / (-5.0 - (-32.0)), 0.0, 1.0)
        ratio_norm = np.clip((vv_db - vh_db - 0.0) / (25.0 - 0.0), 0.0, 1.0)

        tensor = torch.from_numpy(np.stack([vv_norm, vh_norm, ratio_norm], axis=0)).float()
        tensor = TF.resize(tensor, [120, 120], antialias=True)
        tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return tensor

    def predict(self, vv_array: np.ndarray, vh_array: np.ndarray) -> dict[str, Any]:
        """Run inference on SAR image pair."""
        if self.model is not None:
            tensor = self.normalize_radar(vv_array, vh_array).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, features = self.model(tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0]

            top_indices = np.argsort(probs)[::-1]
            top_classes = [
                {"class": CORINE_19_CLASSES[i], "probability": round(float(probs[i]), 4)}
                for i in top_indices[:4]
            ]

            return {
                "computation_mode": ComputationMode.REAL_INFERENCE,
                "weights_sha256": self.weights_sha256,
                "corine_predictions": top_classes,
                "feature_dim": int(features.shape[-1]),
                "status": "real_model_evaluated"
            }
        else:
            return {
                "computation_mode": ComputationMode.SYNTHETIC_FALLBACK,
                "weights_sha256": None,
                "corine_predictions": [],
                "note": "SAR ConvNeXt checkpoint unavailable — no classification performed.",
                "confidence": 0.0,
                "status": "synthetic_fallback_no_weights"
            }
