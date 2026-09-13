# ==============================================================================
# 🛰️ SatQuery AI — Guaranteed Fast Qwen2-VL LoRA Fine-Tuning Script
# Problem Statement ID: 26167 (ISRO / SAC)
# ==============================================================================

import os
import glob
import json
import time
import tarfile
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import hf_hub_download

def main():
    print("=" * 70)
    print("🛰️ SatQuery AI — Calibrated Fast Qwen2-VL LoRA Fine-Tuning")
    print("=" * 70)

    # 1. Environment & GPU Verification
    if not torch.cuda.is_available():
        print("❌ Error: No GPU detected. Make sure GPU accelerator is enabled (T4 x2 or P100).")
        return

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name} | VRAM: {vram_gb:.2f} GB")

    # 2. Data Directories
    DATA_DIR = "/kaggle/working/geozero_data" if os.path.exists("/kaggle") else "./geozero_data"
    IMG_DIR = os.path.join(DATA_DIR, "images")
    OUTPUT_DIR = "/kaggle/working/satquery_vlm_checkpoints" if os.path.exists("/kaggle") else "./satquery_vlm_checkpoints"
    ADAPTER_DIR = "/kaggle/working/satquery_vlm_adapter" if os.path.exists("/kaggle") else "./satquery_vlm_adapter"
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. Targeted Download of Golden Shards
    REPO_ID = "hjvsl/GeoZero_Train_Datasets"
    TARGET_SHARDS = [
        "images/RSVQA-HR-0000.tar",
        "images/RSVQA-LR-0000.tar",
        "images/VRS-0000.tar",
        "images/NASC-TG2-0000.tar"
    ]

    instruct_file = os.path.join(DATA_DIR, "GeoZero-Instruct.json")
    if not os.path.exists(instruct_file):
        print("📥 Downloading GeoZero-Instruct.json (306 MB)...")
        hf_hub_download(repo_id=REPO_ID, filename="GeoZero-Instruct.json", repo_type="dataset", local_dir=DATA_DIR)

    for idx, shard in enumerate(TARGET_SHARDS, start=2):
        shard_name = os.path.basename(shard)
        extracted_flag = os.path.join(IMG_DIR, f".{shard_name}.done")
        if not os.path.exists(extracted_flag):
            print(f"📥 [{idx}/5] Downloading & Extracting {shard_name}...")
            tar_path = hf_hub_download(repo_id=REPO_ID, filename=shard, repo_type="dataset", local_dir=DATA_DIR)
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=IMG_DIR)
            if os.path.exists(tar_path):
                os.remove(tar_path)
            with open(extracted_flag, 'w') as f:
                f.write("done")
        else:
            print(f"⚡ {shard_name} already available locally.")

    # 4. Filter High-Yield Remote Sensing Samples (Calibrated to 1,200 for ~20-25 mins)
    with open(instruct_file, 'r', encoding='utf-8') as f:
        all_instruct = json.load(f)

    filtered_train_samples = []
    MAX_SAMPLES = 1200  # Highly focused instruction set for rapid LoRA adaptation

    for item in all_instruct:
        img_rel = item["images"][0]
        full_img_path = os.path.join(IMG_DIR, img_rel)
        if os.path.exists(full_img_path):
            user_msg = item["messages"][0]["content"]
            # Exclude long polygon coordinates
            if "outlines of buildings" in user_msg or "vectorized key points" in item["messages"][1]["content"]:
                continue
            
            filtered_train_samples.append({
                "image_path": full_img_path,
                "messages": item["messages"]
            })
            if len(filtered_train_samples) >= MAX_SAMPLES:
                break

    print(f"🎯 Selected {len(filtered_train_samples)} remote sensing training samples.")

    # 5. Load Model & Hard-Cap Vision Token Dimensions
    MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
    print(f"🔧 Loading base model {MODEL_ID} in 4-bit NF4...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )

    # ⚡ THE HARD TOKEN CAP: Restricts resolution to ~384x384 max (576 tokens vs 2800 tokens)
    min_pixels = 256 * 28 * 28
    max_pixels = 384 * 28 * 28
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=min_pixels, max_pixels=max_pixels)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. PyTorch Dataset & Fast Batch Collator
    class GeoZeroFastDataset(Dataset):
        def __init__(self, samples, processor):
            self.samples = samples
            self.processor = processor

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            item = self.samples[idx]
            image = Image.open(item["image_path"]).convert("RGB")
            # Downscale large raw satellite images if oversized
            image.thumbnail((448, 448), Image.Resampling.BICUBIC)

            user_text = item["messages"][0]["content"].replace("<image>\n", "").replace("<image>", "").strip()
            assistant_text = item["messages"][1]["content"].strip()

            formatted_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_text}
                    ]
                },
                {"role": "assistant", "content": assistant_text}
            ]

            text = self.processor.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=False)
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            )

            input_ids = inputs["input_ids"].squeeze(0)
            labels = input_ids.clone()

            return {
                "input_ids": input_ids,
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "pixel_values": inputs["pixel_values"].squeeze(0),
                "image_grid_thw": inputs["image_grid_thw"].squeeze(0),
                "labels": labels
            }

    def qwen_collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
        image_grid_thw = torch.cat([item["image_grid_thw"] for item in batch], dim=0)

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=151643)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels
        }

    train_dataset = GeoZeroFastDataset(filtered_train_samples, processor)

    # 7. Training Configuration (Total Steps = 1200 / 16 = 75 steps per epoch, 2 epochs = 150 steps total)
    BATCH_SIZE = 4
    GRAD_ACCUM = 4
    EPOCHS = 2
    TOTAL_STEPS = (len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM)) * EPOCHS
    
    print("\n" + "=" * 50)
    print(f"📊 Training Plan:")
    print(f"   • Samples: {len(train_dataset)}")
    print(f"   • Effective Batch Size: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"   • Total Epochs: {EPOCHS}")
    print(f"   • Total Optimization Steps: {TOTAL_STEPS} steps")
    print(f"   • Projected Time: ~18 to 25 minutes")
    print("=" * 50 + "\n")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="steps",       # ⚡ Saves every 25 steps to disk
        save_steps=25,
        save_total_limit=3,
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=qwen_collate_fn
    )

    # 8. Check for Checkpoints to Auto-Resume
    last_checkpoint = None
    existing_ckpts = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
    if existing_ckpts:
        last_checkpoint = max(existing_ckpts, key=os.path.getctime)
        print(f"🔄 Checkpoint found! Resuming from: {last_checkpoint}")
    else:
        print("🚀 Starting training...")

    t0 = time.time()
    trainer.train(resume_from_checkpoint=last_checkpoint)
    elapsed_mins = (time.time() - t0) / 60
    print(f"\n🎉 Training Finished in {elapsed_mins:.2f} minutes!")

    # 9. Save Final LoRA Weights
    print(f"💾 Saving adapter to {ADAPTER_DIR}...")
    trainer.model.save_pretrained(ADAPTER_DIR)
    processor.save_pretrained(ADAPTER_DIR)

    zip_path = "/kaggle/working/satquery_vlm_adapter.zip" if os.path.exists("/kaggle") else "./satquery_vlm_adapter.zip"
    os.system(f"zip -r {zip_path} {ADAPTER_DIR}")
    print(f"🎁 satquery_vlm_adapter.zip ready for download: {zip_path}")

if __name__ == "__main__":
    main()
