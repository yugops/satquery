# ==============================================================================
# 🛰️ SatQuery AI — Standalone VLM Inference Engine
# Tests your fine-tuned Qwen2-VL LoRA Adapter on Remote Sensing Images
# ==============================================================================

import os
import torch
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import PeftModel

# Paths
BASE_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_PATH = "./kaggle/working/satquery_final_adapter"

def load_satquery_vlm(adapter_path=ADAPTER_PATH):
    """Loads the base model in 4-bit and attaches the fine-tuned LoRA adapter."""
    print(f"🔧 Loading Base Model: {BASE_MODEL_ID}...")
    
    # 4-bit Quantization for fast lightweight inference (<4 GB VRAM)
    use_cuda = torch.cuda.is_available()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16 if use_cuda else torch.float32
    ) if use_cuda else None

    processor = AutoProcessor.from_pretrained(adapter_path)

    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config if use_cuda else None,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else "cpu"
    )

    print(f"🔗 Attaching SatQuery LoRA Adapter from: {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print("✅ SatQuery Remote Sensing VLM ready for inference!\n")
    return model, processor

def query_satellite_image(model, processor, image_input, prompt):
    """
    Runs multimodal query over an image (PIL Image or file path).
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    # Resize thumbnail to preserve optimal aspect ratio
    image.thumbnail((512, 512), Image.Resampling.BICUBIC)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            top_p=0.9
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

    return response

if __name__ == "__main__":
    print("=" * 65)
    print("🛰️ SatQuery AI — Remote Sensing Specialist VLM Tester")
    print("=" * 65)
    
    # Load model
    model, processor = load_satquery_vlm()

    # Create a synthetic aerial/satellite test image if none provided
    test_image_path = "sample_test_satellite.jpg"
    if not os.path.exists(test_image_path):
        print(f"Creating a sample satellite patch ({test_image_path})...")
        sample_img = Image.new("RGB", (512, 512), color=(34, 139, 34)) # Forest green
        sample_img.save(test_image_path)

    # Test Query
    prompt = "Describe the scene, identify land cover types, and detect any notable spatial features."
    print(f"📸 Test Image: {test_image_path}")
    print(f"💬 Query: {prompt}\n")

    print("🤖 Generating SatQuery Response...")
    answer = query_satellite_image(model, processor, test_image_path, prompt)

    print("=" * 65)
    print("🎯 SATQUERY PREDICTION:")
    print(answer)
    print("=" * 65)
