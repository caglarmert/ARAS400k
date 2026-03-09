import os
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from huggingface_hub import login
login("HF_LOGIN_TOKEN")  # Replace with your actual token or use environment variable for security

# -------------------------------------------------
# ---------------- CONFIGURATION ------------------
# -------------------------------------------------

model_id = "Qwen/Qwen3-VL-8B-Instruct" # Select your model here
image_folder = "ARAS400k/synth/images"  # Path to your .png folder
output_csv = "ARAS400k_synth_vision_language_qwen3_vl_8b.csv" 

percentage_csv = "ARAS400k/class_percentages/synth.csv" # CSV with land cover percentages for images
batch_size = 32
num_workers = 16
enable_gradient_checkpointing = True

base_prompt = (
    "Describe this remote sensing image in 1 or 2 concise sentences, "
    "focusing on land use and key geographical features."
)
# Force CUDA device (or use "cpu" if you want to run on CPU, but performance will be much slower)
device = "cuda"

# -------------------------------------------------
# -------- LOAD CLASS PERCENTAGE DATA ------------
# -------------------------------------------------

print("Loading land cover percentage data...")
df_pct = pd.read_csv(percentage_csv)

# Ensure filename column is string
df_pct['filename'] = df_pct['filename'].astype(str)

df_pct.set_index('filename', inplace=True)
percentage_map = df_pct.to_dict(orient='index')

print(f"Loaded land cover data for {len(percentage_map)} images.")

# -------------------------------------------------
# --------------- MODEL SETUP ---------------------
# -------------------------------------------------
# Check if flash_attn is available for optimized attention
try:
    import flash_attn
    use_flash_attention = True
except ImportError:
    use_flash_attention = False

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    use_fast=True
)

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2"
    if use_flash_attention and device == "cuda"
    else "eager"
).to(device)

if enable_gradient_checkpointing and device == "cuda":
    model.gradient_checkpointing_enable()

model.eval()

if device == "cuda":
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total VRAM available: {total_vram:.2f} GB")

# -------------------------------------------------
# ---------------- RESUME LOGIC -------------------
# -------------------------------------------------
# Check for existing output CSV to avoid reprocessing
processed_files = set()
if os.path.exists(output_csv):
    try:
        df_existing = pd.read_csv(output_csv)
        processed_files = set(df_existing['filename'].astype(str).tolist())
    except Exception:
        pass

all_images = [f for f in os.listdir(image_folder) if f.lower().endswith('.png')]
images_to_process = [f for f in all_images if f not in processed_files]

print(f"Total images to process: {len(images_to_process)}")

# -------------------------------------------------
# ----------------- UTILITIES ---------------------
# -------------------------------------------------

def load_image(image_path):
    return Image.open(image_path).convert("RGB")


def build_prompt(filename):
    """
    Create a dynamic prompt based on class percentages.
    """

    if filename not in percentage_map:
        return base_prompt

    data = percentage_map[filename]

    # Keep only present classes
    present_classes = [
        f"{k} ({round(v, 2)}%)"
        for k, v in data.items()
        if v > 0
    ]

    if len(present_classes) == 0:
        return base_prompt

    class_info = ", ".join(present_classes)
    # Construct a detailed prompt that includes class distribution
    dynamic_prompt = (
        f"This remote sensing image contains the following land cover distribution: "
        f"{class_info}. "
        f"Based on this information, describe the scene in 1 or 2 concise sentences "
        f"focusing on spatial layout and dominant land use types."
    )

    return dynamic_prompt


# -------------------------------------------------
# ----------- BATCH CAPTION GENERATION -----------
# -------------------------------------------------

def generate_captions_batch(batch_filenames, image_folder):

    # Parallel image loading
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        image_paths = [os.path.join(image_folder, fn) for fn in batch_filenames]
        images = list(executor.map(load_image, image_paths))

    # Build dynamic prompts per image
    batch_messages = []

    for filename, image in zip(batch_filenames, images):

        prompt_text = build_prompt(filename)

        batch_messages.append(
            [{
                "role": "user",
                "content": [
                    dict(type="text", text=prompt_text),
                    dict(type="image", image=image)
                ]
            }]
        )

    # Tokenize entire batch
    inputs = processor.apply_chat_template(
        batch_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
        padding_side = "left"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            use_cache=True,
            do_sample=False
        )

    captions = []

    for i, filename in enumerate(batch_filenames):
        generated_tokens = generated_ids[i, inputs['input_ids'][i].size(0):]
        caption = processor.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        captions.append({
            "filename": filename,
            "caption": caption
        })

    torch.cuda.empty_cache()

    return captions


# -------------------------------------------------
# ------------------- MAIN LOOP -------------------
# -------------------------------------------------

num_batches = (len(images_to_process) + batch_size - 1) // batch_size

for batch_idx in tqdm(range(num_batches), desc="Processing batches"):

    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(images_to_process))
    batch_filenames = images_to_process[start_idx:end_idx]

    try:
        batch_results = generate_captions_batch(batch_filenames, image_folder)

        pd.DataFrame(batch_results).to_csv(
            output_csv,
            mode='a',
            index=False,
            header=not os.path.exists(output_csv) or start_idx == 0
        )

        processed_count = min(end_idx, len(images_to_process))
        tqdm.write(f"Processed {processed_count}/{len(images_to_process)} images")

    except Exception as e:
        print(f"\nError processing batch {batch_idx}: {e}")
        import traceback
        traceback.print_exc()

print("Done!")
