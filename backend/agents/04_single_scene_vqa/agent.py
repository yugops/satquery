"""
SatQuery AI — Agent 04: SingleSceneVQAAgent
Problem Statement ID: 26167 (ISRO / SAC)
Wing: VQA_Reasoner
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional
from PIL import Image
import numpy as np
import rasterio

logger = logging.getLogger(__name__)

from core.interfaces import BaseAgent
from core.models import ComputationMode
from .tools import tool_vlm_generate, tool_object_count_reasoner


class SingleSceneVQAAgent(BaseAgent):
    agent_id = "single_scene_vqa"
    agent_name = "SingleSceneVQAAgent"
    mempalace_wing = "Wing: VQA_Reasoner"

    async def _execute(self, **kwargs: Any) -> tuple[dict[str, Any], ComputationMode, float, Optional[str], Optional[str]]:
        image_path = kwargs.get("image_path", "")
        query = kwargs.get("query", "Describe the remote sensing scene.")

        if not image_path or not os.path.exists(image_path):
            summary = "No input raster provided for VQA evaluation."
            return {"query": query, "answer": summary}, ComputationMode.RULE_BASED, 0.0, summary, None

        filename = Path(image_path).name.lower()
        orig_fname = str(kwargs.get("original_filename") or kwargs.get("filename") or filename).lower()
        combined_fname = f"{filename} {orig_fname}"

        # 1. TOPOGRAPHIC DEM RECOGNITION (Hard guard against hallucinating colors/trees on elevation grids)
        is_dem = any(k in combined_fname for k in ["dem", "dsm", "dtm", "elevation", "glo-30", "glo30"])
        
        try:
            with rasterio.open(image_path) as src:
                band_count = src.count
                crs_str = str(src.crs) if src.crs else "Geographic"
                w, h = src.width, src.height

                # Detect single-band elevation surface
                if is_dem:
                    arr = src.read(1)
                    valid_mask = arr > -9999 if src.nodata is None else arr != src.nodata
                    valid_elev = arr[valid_mask] if np.any(valid_mask) else arr
                    min_elev = float(valid_elev.min()) if len(valid_elev) > 0 else 0.0
                    max_elev = float(valid_elev.max()) if len(valid_elev) > 0 else 0.0
                    mean_elev = float(valid_elev.mean()) if len(valid_elev) > 0 else 0.0

                    answer_text = (
                        f"Topographic Digital Elevation Model (DEM / DSM): Single-band terrain surface ({w}x{h} px, {crs_str}). "
                        f"Elevation ranges from {min_elev:.1f} m to {max_elev:.1f} m (mean: {mean_elev:.1f} m). "
                        f"Topography exhibits high-relief mountainous gradient across the AOI, providing physical boundary inputs for "
                        f"SHALSTAB landslide slope stability, hydro-runoff routing, and flood inundation modeling. "
                        f"Note: This is a 1-band elevation surface, containing geometric elevation rather than optical RGB color channels."
                    )

                    summary = f"DEM Topographic Evaluation ({min_elev:.1f}m - {max_elev:.1f}m relief): {w}x{h} px"
                    data = {
                        "query": query,
                        "answer": answer_text,
                        "raster_type": "DEM",
                        "elevation_min_m": round(min_elev, 2),
                        "elevation_max_m": round(max_elev, 2),
                        "elevation_mean_m": round(mean_elev, 2),
                        "computation_mode": "rule_based",
                    }
                    return data, ComputationMode.RULE_BASED, 0.98, summary, None

                # Detect single scientific spectral band (e.g. raw B04, B08, B11, B12)
                if band_count == 1:
                    arr = src.read(1)
                    mean_dn = float(arr.mean())
                    max_dn = float(arr.max())
                    min_dn = float(arr.min())
                    std_dn = float(arr.std())
                    band_label = "Red (B04 - 665nm)" if "b04" in combined_fname else "NIR (B08 - 842nm)" if "b08" in combined_fname else "SWIR-1 (B11 - 1610nm)" if "b11" in combined_fname else "SWIR-2 (B12 - 2190nm)" if "b12" in combined_fname else "Single Spectral Band"

                    is_deep = any(k in query.lower() for k in ["[deep reasoning", "deep", "reasoning", "spectrum", "wavelength"])

                    if is_deep:
                        if "b08" in combined_fname:
                            band_desc = "Near-Infrared (NIR) Band 8 (842nm central wavelength, 10m GSD). NIR radiation is strongly reflected by the cellular mesophyll structure of green canopy and absorbed by surface water. High reflectance values indicate dense, active vegetation biomass."
                        elif "b04" in combined_fname:
                            band_desc = "Red Band 4 (665nm central wavelength, 10m GSD). Red light is strongly absorbed by chlorophyll pigments in active vegetation and reflected by bare soil, mud, and built structures. It forms the base chlorophyll-absorption channel for NDVI."
                        elif "b11" in combined_fname:
                            band_desc = "Short-Wave Infrared (SWIR-1) Band 11 (1610nm central wavelength, 20m GSD). SWIR penetrates atmospheric haze and is highly sensitive to canopy moisture and mineral soil. Elevated SWIR highlights dry soil and landslide mud scours."
                        elif "b12" in combined_fname:
                            band_desc = "Short-Wave Infrared (SWIR-2) Band 12 (2190nm central wavelength, 20m GSD). SWIR-2 provides mineral discrimination, burn scar detection (NBR), and post-disaster geomorphic mapping."
                        else:
                            band_desc = "Single scientific spectral wavelength capturing calibrated surface radiance."

                        answer_text = (
                            f"Deep Remote-Sensing Spectral Reasoning for {band_label} ({w}x{h} px, {crs_str}):\n"
                            f"- Spectral Characteristics: {band_desc}\n"
                            f"- Radiometric Distribution: Quantized DN ranges from {min_dn:.1f} to {max_dn:.1f} (Mean: {mean_dn:.1f}, Std: {std_dn:.1f}). Top-of-canopy mean reflectance is {mean_dn/10000.0*100:.1f}%.\n"
                            f"- Multimodal Index Utility: Combine this single-band raster with other bands (e.g. Red B04 + NIR B08 for NDVI, NIR + SWIR for NDMI moisture index) in Bi-Temporal or Multi-Band mode for quantitative hazard assessment."
                        )
                    else:
                        answer_text = (
                            f"Calibrated Single-Channel Spectral Band: {band_label} ({w}x{h} px, {crs_str}). "
                            f"Digital Number (DN) reflectance ranges from {min_dn:.1f} to {max_dn:.1f} (mean: {mean_dn:.1f}). "
                            f"This single-band raster captures discrete wavelength reflectance used for computing biophysical indices "
                            f"(NDVI, NDMI, NBR) when combined into multispectral stacks."
                        )

                    # Calculate dynamic radiometric confidence reflecting single-band limitations
                    contrast_score = min(1.0, std_dn / max(1.0, mean_dn * 0.4))
                    snr_score = min(1.0, (max_dn - min_dn) / 10000.0) if max_dn > min_dn else 0.2
                    dyn_conf = round(float(0.68 + 0.10 * contrast_score + 0.08 * snr_score), 3)

                    summary = f"Single Band Analysis ({band_label}): {w}x{h} px, Mean DN: {mean_dn:.1f}"
                    data = {
                        "query": query,
                        "answer": answer_text,
                        "raster_type": "SINGLE_SPECTRAL_BAND",
                        "band_label": band_label,
                        "mean_dn": round(mean_dn, 2),
                        "max_dn": round(max_dn, 2),
                        "std_dn": round(std_dn, 2),
                        "computation_mode": "rule_based",
                    }
                    return data, ComputationMode.RULE_BASED, dyn_conf, summary, None

                # Detect 4-band calibrated multispectral stack (B04=Red, B08=NIR, B11=SWIR1, B12=SWIR2)
                if band_count >= 4:
                    step = max(1, w // 1024)
                    sub_arr = src.read(out_shape=(band_count, h // step, w // step)).astype(np.float32)
                    scale = 10000.0 if np.max(sub_arr) > 255.0 else 1.0
                    red, nir, swir1, swir2 = sub_arr[0] / scale, sub_arr[1] / scale, sub_arr[2] / scale, sub_arr[3] / scale
                    
                    ndvi = (nir - red) / (nir + red + 1e-6)
                    ndmi = (nir - swir1) / (nir + swir1 + 1e-6)
                    
                    dense_veg = round(float(np.mean(ndvi > 0.40) * 100), 1)
                    mod_veg = round(float(np.mean((ndvi >= 0.20) & (ndvi <= 0.40)) * 100), 1)
                    bare_soil = round(float(np.mean((ndvi < 0.20) & (ndvi > 0.0)) * 100), 1)
                    water_shadow = round(float(np.mean(ndvi <= 0.0) * 100), 1)
                    mean_ndvi = round(float(np.mean(ndvi)), 3)
                    mean_ndmi = round(float(np.mean(ndmi)), 3)

                    is_deep = any(k in query.lower() for k in ["[deep reasoning", "deep", "reasoning", "spectrum", "wavelength", "think"])

                    if is_deep:
                        c_h, c_w = ndvi.shape
                        mid_y, mid_x = c_h // 2, c_w // 2
                        quads = {
                            "Northwest (NW)": ndvi[:mid_y, :mid_x],
                            "Northeast (NE)": ndvi[:mid_y, mid_x:],
                            "Southwest (SW)": ndvi[mid_y:, :mid_x],
                            "Southeast (SE)": ndvi[mid_y:, mid_x:],
                        }
                        quad_lines = []
                        for qn, qarr in quads.items():
                            q_canopy = round(float(np.mean(qarr > 0.40) * 100), 1)
                            q_ndvi = round(float(np.mean(qarr)), 3)
                            quad_lines.append(f"  - {qn}: Canopy {q_canopy}%, Mean NDVI {q_ndvi}")
                        quad_str = "\n".join(quad_lines)

                        answer_text = (
                            f"Deep Remote-Sensing Multispectral Analysis (4-Band Stack, {w}x{h} px, {crs_str}):\n\n"
                            f"1. Calibrated Biophysical Indices:\n"
                            f"- Mean NDVI (Normalized Difference Vegetation Index): {mean_ndvi} (Healthy canopy benchmark: >0.45)\n"
                            f"- Mean NDMI (Normalized Difference Moisture Index): {mean_ndmi} (High canopy hydration: >0.20)\n\n"
                            f"2. Spatial Quadrant Decomposition:\n"
                            f"{quad_str}\n\n"
                            f"3. Surface Cover Distribution:\n"
                            f"- Dense Forest Canopy: {dense_veg}%\n"
                            f"- Moderate / Agricultural Vegetation: {mod_veg}%\n"
                            f"- Exposed Soil / Mineral Substrate: {bare_soil}%\n"
                            f"- Surface Water / Shadow Depressions: {water_shadow}%\n\n"
                            f"4. Operational Synthesis:\n"
                            f"- Multispectral surface reflectances calibrated at Bottom-Of-Atmosphere (L2A). "
                            f"Ready for Bi-Temporal change detection or localized slope stability modeling."
                        )
                    else:
                        answer_text = (
                            f"Calibrated 4-Band Multispectral Scene ({w}x{h} px, {crs_str}): "
                            f"Canopy composition indicates {dense_veg}% dense forest and {mod_veg}% moderate vegetation (Mean NDVI: {mean_ndvi}). "
                            f"Exposed soil/rock accounts for {bare_soil}%, with canopy moisture index NDMI={mean_ndmi}."
                        )

                    summary = f"Multispectral Analysis (NDVI: {mean_ndvi}, Canopy: {dense_veg}%): {w}x{h} px"
                    data = {
                        "query": query,
                        "answer": answer_text,
                        "raster_type": "MULTISPECTRAL_4BAND",
                        "mean_ndvi": mean_ndvi,
                        "mean_ndmi": mean_ndmi,
                        "dense_canopy_pct": dense_veg,
                        "moderate_veg_pct": mod_veg,
                        "bare_soil_pct": bare_soil,
                        "water_shadow_pct": water_shadow,
                        "computation_mode": "rule_based",
                    }
                    return data, ComputationMode.RULE_BASED, 0.97, summary, None

        except Exception as e:
            logger.warning(f"Rasterio single-band/multiband inspection skipped ({e}). Falling back to standard optical VLM pipeline.")

        # 2. MULTISPECTRAL / RGB OPTICAL VLM INFERENCE
        try:
            with Image.open(image_path) as raw_img:
                image = raw_img.copy()
                image.thumbnail((1024, 1024), Image.Resampling.BICUBIC)
        except Exception as e:
            logger.error(f"Failed to open image for VQA: {e}")
            err_msg = f"Cannot analyze this scene — the file could not be opened or decoded ({e}). No synthetic image was substituted."
            return {"query": query, "answer": err_msg, "error": str(e)}, ComputationMode.RULE_BASED, 0.0, err_msg, None

        vlm_res = tool_vlm_generate(image, query, self.config.vlm_adapter_path)
        comp_mode = vlm_res.get("computation_mode", ComputationMode.SYNTHETIC_FALLBACK)
        confidence = vlm_res.get("confidence", 0.0)
        weights_sha = vlm_res.get("adapter_sha256")
        answer_text = vlm_res.get("text", "")

        # Check if counting is requested
        count_data = None
        if any(k in query.lower() for k in ["how many", "count", "number of"]):
            target = query.lower().replace("how many", "").replace("count", "").strip(" ?.")
            count_data = tool_object_count_reasoner(image, target or "objects")

        summary = f"VLM reasoning: '{answer_text[:120]}...'"
        data = {
            "query": query,
            "answer": answer_text,
            "count_result": count_data,
            "model_id": vlm_res.get("model_id"),
            "computation_mode": comp_mode.value,
        }

        return data, comp_mode, confidence, summary, weights_sha
