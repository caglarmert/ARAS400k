import os
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm

from huggingface_hub import login
login("HF_LOGIN_TOKEN")  # Replace with your actual token or use environment variable for security


# --- Configuration ---
model_id = "google/gemma-3-4b-it"
image_folder = "ARAS400k/train/images" # Path to your .png folder
output_csv = "ARAS400k_train_vision_gemma3_4b.csv" 
batch_size = 10
prompt_text = "Describe this remote sensing image in 1 or 2 concise sentences, focusing on land use and key geographical features."
# Determine device (Force CUDA if available)
device = "cuda" # if torch.cuda.is_available() else "cpu"

# --- Initialization ---
# Note: Removed device_map="auto" to prevent meta-tensor issues
processor = AutoProcessor.from_pretrained(
    model_id, 
    trust_remote_code=True
)

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True
).to(device) # Move entire model to device immediately

# --- Resume Logic ---
processed_files = set()
if os.path.exists(output_csv):
    try:
        df_existing = pd.read_csv(output_csv)
        processed_files = set(df_existing['filename'].astype(str).tolist())
    except Exception: pass

all_images = [f for f in os.listdir(image_folder) if f.lower().endswith('.png')]
images_to_process = [f for f in all_images if f not in processed_files]

def generate_caption(image_path):
    # Remote sensing images can have 4 channels (RGBA) or 1 (L). 
    # Molmo REQUIRES exactly 3 (RGB).
    image = Image.open(image_path).convert("RGB")
    
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": [
            dict(type="text", text=prompt_text),
            dict(type="image", image=image)
        ]}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    # Move inputs to the SAME device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=100,
            use_cache=True
        )
    
    generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
    return processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

# --- Loop ---
current_batch = []
for i, filename in enumerate(tqdm(images_to_process, desc="Captioning")):
    try:
        caption = generate_caption(os.path.join(image_folder, filename))
        current_batch.append({"filename": filename, "caption": caption})
        
        if (len(current_batch) >= batch_size) or (i == len(images_to_process) - 1):
            pd.DataFrame(current_batch).to_csv(
                output_csv, mode='a', index=False, header=not os.path.exists(output_csv)
            )
            current_batch = []
    except Exception as e:
        print(f"\nError on {filename}: {e}")

print("Done!")