import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from models_registry.sar_convnext import SARModelEngine, CORINE_19_CLASSES

# Instantiate engine with real checkpoint if present
engine = SARModelEngine("sar_convnext_shard0.pt")

print(f"Model loaded: {engine.model is not None}")
print(f"Weights SHA256: {engine.weights_sha256}")

# 1. Water synthetic input (low backscatter)
water_vv = np.full((120, 120), 0.01, dtype=np.float32)
water_vh = np.full((120, 120), 0.002, dtype=np.float32)

# 2. Urban synthetic input (high backscatter)
urban_vv = np.full((120, 120), 0.8, dtype=np.float32)
urban_vh = np.full((120, 120), 0.4, dtype=np.float32)

if engine.model is not None:
    t_water = engine.normalize_radar(water_vv, water_vh).unsqueeze(0).to(engine.device)
    t_urban = engine.normalize_radar(urban_vv, urban_vh).unsqueeze(0).to(engine.device)

    with torch.no_grad():
        water_logits, water_feats = engine.model(t_water)
        urban_logits, urban_feats = engine.model(t_urban)

    w_logits = water_logits.cpu().numpy()[0]
    u_logits = urban_logits.cpu().numpy()[0]

    w_probs = torch.sigmoid(water_logits).cpu().numpy()[0]
    u_probs = torch.sigmoid(urban_logits).cpu().numpy()[0]

    print("\n--- WATER INPUT RAW LOGITS ---")
    print(f"Min logit: {w_logits.min():.6f}, Max logit: {w_logits.max():.6f}, Mean: {w_logits.mean():.6f}, Std: {w_logits.std():.6f}")
    for idx, cls in enumerate(CORINE_19_CLASSES):
        print(f"  {cls:35s}: logit = {w_logits[idx]:8.4f} | prob = {w_probs[idx]:.6f}")

    print("\n--- URBAN INPUT RAW LOGITS ---")
    print(f"Min logit: {u_logits.min():.6f}, Max logit: {u_logits.max():.6f}, Mean: {u_logits.mean():.6f}, Std: {u_logits.std():.6f}")
    for idx, cls in enumerate(CORINE_19_CLASSES):
        print(f"  {cls:35s}: logit = {u_logits[idx]:8.4f} | prob = {u_probs[idx]:.6f}")

    logit_diff = np.abs(w_logits - u_logits)
    print(f"\nMax absolute logit difference between water and urban: {logit_diff.max():.6f}")
    print(f"Mean absolute logit difference: {logit_diff.mean():.6f}")
else:
    print("Engine model is None (checkpoint not found)")
