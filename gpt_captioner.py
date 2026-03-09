import time
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import os

# ---------------- CONFIG ----------------
INPUT_CSV = "ARAS400k_train_stats.csv"
OUTPUT_CSV = "ARAS400k_train_cgc.csv"
BATCH_FILE = "batch_requests.jsonl"
BATCH_STATUS_FILE = "batch_status.txt"

MODEL = "gpt-4-mini"  # Changed to a model that supports batches

SAVE_EVERY = 10                 # save every N rows
BATCH_SIZE = 100000              # OpenAI batch size limit
POLLING_INTERVAL = 10            # seconds between status checks
# ----------------------------------------

client = OpenAI(api_key="API_KEY")

LANDCOVER_CLASSES = [
    "Tree", "Shrub", "Grass", "Crop",
    "Built-up", "Barren", "Water"
]

# 🔹 All rules live here (sent once per request, but cheaper & stable)
SYSTEM_MESSAGE = (
    "You generate natural, fluent, human-like satellite image captions (1-2 sentences)."
    "Do not list percentages, statistics, segmentation, or datasets."
)


def build_user_message(row):
    """
    Minimal per-row message: just the land-cover composition.
    """
    return ", ".join(
        f"{cls} {int(row[cls])}%"
        for cls in LANDCOVER_CLASSES
        if row[cls] > 0
    )


def create_batch_request(idx, user_message, custom_id):
    """Create a single batch request in OpenAI format."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            "temperature": 1,  # Default temperature
        }
    }


def create_batch_file(df, indices):
    """Create a JSONL batch file from pending rows."""
    with open(BATCH_FILE, 'w') as f:
        for idx in indices:
            row = df.loc[idx]
            user_msg = build_user_message(row)
            request = create_batch_request(idx, user_msg, f"row_{idx}")
            f.write(json.dumps(request) + '\n')
    print(f"Created batch file with {len(indices)} requests: {BATCH_FILE}")
    return len(indices)


def submit_batch(file_path):
    """Submit batch file to OpenAI."""
    with open(file_path, 'rb') as f:
        batch_response = client.beta.batches.create(
            input_file=f,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
    batch_id = batch_response.id
    print(f"Batch submitted successfully. Batch ID: {batch_id}")
    
    # Save batch ID for reference
    with open(BATCH_STATUS_FILE, 'w') as f:
        f.write(batch_id)
    
    return batch_id


def poll_batch_status(batch_id, max_wait_seconds=None):
    """Poll batch status until completion."""
    start_time = time.time()
    
    while True:
        batch_status = client.beta.batches.retrieve(batch_id)
        
        print(f"Batch {batch_id} status: {batch_status.status}")
        print(f"   - Completed: {batch_status.request_counts.completed}/{batch_status.request_counts.total}")
        
        if batch_status.status == "completed":
            print(f"Batch completed!")
            return batch_status
        elif batch_status.status in ["failed", "expired"]:
            raise Exception(f"Batch failed with status: {batch_status.status}")
        
        if max_wait_seconds:
            elapsed = time.time() - start_time
            if elapsed > max_wait_seconds:
                print(f"Batch still processing. Elapsed: {elapsed:.0f}s. You can check status later with batch ID: {batch_id}")
                return None
        
        print(f"Waiting {POLLING_INTERVAL}s before next check...")
        time.sleep(POLLING_INTERVAL)


def process_batch_results(df, batch_id):
    """Retrieve and process batch results."""
    batch_status = client.beta.batches.retrieve(batch_id)
    
    if batch_status.status != "completed":
        print(f"Batch not completed. Status: {batch_status.status}")
        return False
    
    # Get the results
    result_file = client.beta.batches.results(batch_id)
    
    updated_count = 0
    error_count = 0
    
    for result in result_file:
        custom_id = result.custom_id
        idx = int(custom_id.split('_')[1])
        
        if result.result.error:
            print(f"Error for row {idx}: {result.result.error}")
            error_count += 1
        else:
            # Extract caption from response
            try:
                caption = result.result.message.content[0].text.strip()
                df.at[idx, "caption"] = caption
                updated_count += 1
            except (AttributeError, IndexError) as e:
                print(f"Failed to parse response for row {idx}: {e}")
                error_count += 1
    
    print(f"\nUpdated {updated_count} captions")
    if error_count > 0:
        print(f"{error_count} rows had errors")
    
    return True


def main():
    df = pd.read_csv(INPUT_CSV)

    if "caption" not in df.columns:
        df["caption"] = ""

    pending_indices = df[df["caption"] == ""].index.tolist()
    print(f"Remaining captions: {len(pending_indices)}")
    
    if len(pending_indices) == 0:
        print("All captions already generated!")
        return

    # Step 1: Create and submit batch
    print("\nCreating batch file...")
    create_batch_file(df, pending_indices)
    
    print("\nSubmitting batch to OpenAI...")
    batch_id = submit_batch(BATCH_FILE)
    
    # Step 2: Poll for completion (with optional timeout)
    print("\nPolling for batch completion (this may take a while)...")
    max_wait = 3600  # Wait up to 1 hour initially
    result = poll_batch_status(batch_id, max_wait_seconds=max_wait)
    
    if result is None:
        print(f"\nBatch is still processing. To check status and retrieve results later, run:")
        print(f"   python retrieve_batch_results.py {batch_id}")
        return
    
    # Step 3: Process results
    print("\n📥 Processing batch results...")
    process_batch_results(df, batch_id)
    
    # Save final results
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCaption generation completed. Results saved to {OUTPUT_CSV}")


def retrieve_results_by_id(batch_id):
    """Helper function to retrieve results for a previously submitted batch."""
    print(f"📥 Retrieving results for batch {batch_id}...")
    df = pd.read_csv(INPUT_CSV)

    if "caption" not in df.columns:
        df["caption"] = ""
    
    process_batch_results(df, batch_id)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    import sys
    
    # Allow retrieving results from a previous batch
    if len(sys.argv) > 1:
        batch_id = sys.argv[1]
        retrieve_results_by_id(batch_id)
    else:
        main()
