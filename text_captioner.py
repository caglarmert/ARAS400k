import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from huggingface_hub import login
login("HF_LOGIN_TOKEN")  # Replace with your actual token or use environment variable for security
model_name = "Qwen/Qwen3-4B-Instruct-2507"
percentage_csv = "ARAS400k/class_percentages/synth.csv" # CSV with land cover percentages for images
output_csv = "ARAS400k_synth_language_qwen3_4b.csv" 
batch_size = 1 
device = "cuda"

print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()

# -------------------------------------------------
# -------- DATA LOADING & RESUME LOGIC ------------
# -------------------------------------------------

df_pct = pd.read_csv(percentage_csv)
df_pct['filename'] = df_pct['filename'].astype(str)

processed_files = set()
if os.path.exists(output_csv):
    try:
        processed_files = set(pd.read_csv(output_csv)['filename'].astype(str).tolist())
    except Exception: pass

df_to_process = df_pct[~df_pct['filename'].isin(processed_files)].copy()
print(f"Total rows to process: {len(df_to_process)}")

# -------------------------------------------------
# ----------------- UTILITIES ---------------------
# -------------------------------------------------

def build_prompt(row):
    """Formats the class percentages into a text-only prompt."""
    data = row.drop(labels=['filename']).to_dict()
    present_classes = [f"{k} ({round(v, 2)}%)" for k, v in data.items() if v > 0]
    
    class_info = ", ".join(present_classes) if present_classes else "unidentified land cover"
    
    # Using the Chat Template format required by Qwen3
    messages = [
        {"role": "system", "content": "You are a remote sensing expert."},
        {"role": "user", "content": (
            f"An image shows: {class_info}. "
            "Describe the scene in 1 or 2 concise sentences focusing on dominant land use."
        )}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# -------------------------------------------------
# ----------- BATCH GENERATION LOOP --------------
# -------------------------------------------------

rows = df_to_process.to_dict(orient='records')

for i in tqdm(range(0, len(rows), batch_size), desc="Processing Batches"):
    batch = rows[i : i + batch_size]
    batch_filenames = [r['filename'] for r in batch]
    
    # 1. Prepare Inputs
    prompts = [build_prompt(pd.Series(r)) for r in batch]
    model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    # 2. Generate (Applying the official snippet logic)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=100,
            do_sample=True,      # Recommended for Instruct models
            temperature=0.7,   
            top_p=0.8,
            pad_token_id=tokenizer.pad_token_id
        )

    # 3. Decode & Extract (Slice out the input tokens)
    input_len = model_inputs.input_ids.shape[1]
    batch_outputs = tokenizer.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)

    # 4. Save Results
    results = [{"filename": fn, "caption": cap.strip()} for fn, cap in zip(batch_filenames, batch_outputs)]
    pd.DataFrame(results).to_csv(output_csv, mode='a', index=False, header=not os.path.exists(output_csv))

print("Done!")