import os
import csv
import time
import argparse
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO
import requests
import json


def image_to_base64(image_path):
    """Convert image to base64 string."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (handles RGBA, P mode, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize large images to reduce token usage (optional)
            max_size = 1024
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return img_str
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def generate_caption_ollama(image_path, model="moondream", prompt="Describe this image in detail."):
    """
    Generate caption using Ollama's local multimodal model (non-streaming).

    Parameters:
    -----------
    image_path : str
        Path to the image file
    model : str
        Ollama model name (default: "moondream")
    prompt : str
        Prompt to send with the image

    Returns:
    --------
    str or None
        Generated caption or None if failed
    """
    try:
        # Convert image to base64
        image_b64 = image_to_base64(image_path)
        if image_b64 is None:
            return None

        # Prepare the request for Ollama's generate endpoint (non-streaming)
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,  # Non-streaming mode
            "images": [image_b64]
        }

        # Make request to Ollama (uses local model)
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()
        return result.get('response', '').strip()

    except Exception as e:
        print(f"Error generating caption for {image_path}: {e}")
        return None


def generate_description_ollama(prompt, model="gemma3:4b-it-qat"):
    """
    Generate a description using Ollama's local language model (non-streaming).

    Parameters:
    -----------
    prompt : str
        Prompt for the description
    model : str
        Ollama model name (default: "gemma3:4b-it-qat")

    Returns:
    --------
    str or None
        Generated description or None if failed
    """
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }

        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()
        return result.get('response', '').strip()

    except Exception as e:
        print(f"Error generating description: {e}")
        return None


def process_images_with_ollama(
        input_path,
        output_csv,
        model="gemma3:4b-it-qat",
        prompt="You are an expert in remote sensing and satellite imagery analysis. Based only on the visual content of the provided 256×256 satellite image (with ~10 meter spatial resolution), generate a concise, factual, and objective caption describing the scene. Focus only on clearly observable land cover, land use, or geographic features (e.g., urban areas, agricultural fields, forests, water bodies, roads). Do not speculate about specific locations, names, activities, or objects that cannot be confidently identified at this resolution. Avoid making assumptions or adding details not visible in the image. ",
        batch_size=100,
        start_from=0,
        supported_formats=None
):
    """
    Processes either an image folder or a CSV file to generate descriptions using Ollama.

    Parameters:
    -----------
    input_path : str
        Path to either an image folder or a CSV file.
    output_csv : str
        Path to output CSV file.
    model : str
        Ollama model to use.
    prompt : str
        Prompt for image description (used for image folder).
    batch_size : int
        Save to CSV after processing this many items.
    start_from : int
        Skip first N items (useful for resuming).
    supported_formats : list
        List of supported image formats.
    """

    input_path = Path(input_path)

    if input_path.is_dir():
        # Process images in a folder
        if supported_formats is None:
            supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        image_files = []
        for ext in supported_formats:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        image_files = sorted(set(image_files))

        if not image_files:
            print(f"No images found in {input_path} with supported formats.")
            return

        print(f"Found {len(image_files)} images to process")

        if start_from > 0:
            image_files = image_files[start_from:]
            print(f"Starting from image {start_from + 1}, {len(image_files)} remaining")

        processed_filenames = set()
        if os.path.exists(output_csv):
            with open(output_csv, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    processed_filenames.add(row['filename'])

        image_files = [f for f in image_files if f.name not in processed_filenames]
        print(f"Images remaining to process: {len(image_files)}")

        if not image_files:
            print("No new images to process!")
            return

        batch = []
        total_processed = 0

        for i, image_path in enumerate(image_files):
            print(f"Processing {i + 1}/{len(image_files)}: {image_path.name}")
            caption = generate_caption_ollama(str(image_path), model, prompt)
            batch.append({
                'filename': image_path.name,
                'caption': caption if caption else 'ERROR: Failed to generate caption'
            })
            total_processed += 1
            if (total_processed % batch_size == 0) or (i == len(image_files) - 1):
                save_to_csv(output_csv, batch, ['filename', 'caption'])
                batch = []
                time.sleep(0.1)

        print(f"Image processing complete! Total images processed: {total_processed}")

    elif input_path.is_file() and input_path.suffix.lower() == '.csv':
        # Process a CSV file
        print(f"Processing CSV file: {input_path}")

        try:
            with open(input_path, 'r', newline='', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                rows = list(reader)
        except Exception as e:
            print(f"Error reading CSV file {input_path}: {e}")
            return

        print(f"Found {len(rows)} rows to process")

        processed_filenames = set()
        if os.path.exists(output_csv):
            with open(output_csv, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    processed_filenames.add(row['filename'])

        new_rows = [row for row in rows if row['filename'] not in processed_filenames]
        print(f"Rows remaining to process: {len(new_rows)}")

        if not new_rows:
            print("No new rows to process!")
            return

        batch = []
        total_processed = 0

        for i, row in enumerate(new_rows):
            print(f"Processing {i + 1}/{len(new_rows)}: {row['filename']}")

            # Construct a prompt based on the row's data
            description_prompt = create_csv_prompt(row)

            # Use a language model for the description
            description = generate_description_ollama(description_prompt, model)

            # Add the original row data and the new description
            updated_row = row.copy()
            updated_row['description'] = description if description else 'ERROR: Failed to generate description'
            batch.append(updated_row)
            total_processed += 1

            if (total_processed % batch_size == 0) or (i == len(new_rows) - 1):
                # Get the full list of fieldnames, including the new 'description' column
                fieldnames = list(rows[0].keys()) + ['description']
                save_to_csv(output_csv, batch, fieldnames)
                batch = []
                time.sleep(0.1)

        print(f"CSV processing complete! Total rows processed: {total_processed}")

    else:
        print("Invalid input path. Please provide a path to an image folder or a CSV file.")


def create_csv_prompt(row):
    """
    Creates a detailed prompt for the LLM based on the CSV row data.
    """
    # Exclude filename from the prompt
    data = {key: val for key, val in row.items() if key.lower() != 'filename'}

    # Sort data by value in descending order and filter out zero percentages
    sorted_data = sorted(data.items(), key=lambda item: int(item[1]) if item[1].isdigit() else 0, reverse=True)

    filtered_data = {key: int(val) for key, val in sorted_data if val != '0'}

    if not filtered_data:
        return "The provided data indicates no significant land cover percentages. Please describe the scene as a blank or non-existent landscape."

    prompt_parts = [
        "Based on the following percentages of land cover classes, generate a concise, factual, and objective description of the scene.",
        "The scene is primarily composed of:",
    ]

    for i, (key, value) in enumerate(filtered_data.items()):
        if i == 0:
            prompt_parts.append(f"a large amount of {key.lower()} ({value}%),")
        elif i == len(filtered_data) - 1:
            prompt_parts.append(f"and some {key.lower()} ({value}%).")
        else:
            prompt_parts.append(f"{key.lower()} ({value}%),")

    prompt_parts.append("Do not add details that are not provided in the percentages.")

    return " ".join(prompt_parts)


def save_to_csv(output_csv, new_data, fieldnames):
    """
    Saves new data to a CSV file, appending to existing data if the file exists.
    """
    all_data = []
    if os.path.exists(output_csv):
        with open(output_csv, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            all_data = list(reader)

    all_data.extend(new_data)

    # Remove duplicates, keeping the latest entry
    seen = set()
    unique_data = []
    for row in reversed(all_data):
        if row['filename'] not in seen:
            unique_data.append(row)
            seen.add(row['filename'])
    unique_data.reverse()

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_data)

    print(f"Saved batch to {output_csv} (total processed: {len(unique_data)})")


def main():
    parser = argparse.ArgumentParser(description='Generate captions or descriptions using Ollama')
    parser.add_argument('input_path', help='Path to an image folder or a CSV file')
    parser.add_argument('--output_csv', required=True, help='Output CSV file path')
    parser.add_argument('--model', default='moondream', help='Ollama model to use')
    parser.add_argument('--prompt', default='Describe this image in detail.',
                        help='Prompt for image description (used for image folders)')
    parser.add_argument('--batch_size', type=int, default=100, help='Save to CSV every N items')
    parser.add_argument('--start_from', type=int, default=0, help='Skip first N items (for resuming)')

    args = parser.parse_args()

    # Verify Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("Warning: Could not connect to Ollama. Make sure it's running!")
    except requests.exceptions.RequestException:
        print("Warning: Could not connect to Ollama. Make sure it's running on localhost:11434!")

    process_images_with_ollama(
        input_path=args.input_path,
        output_csv=args.output_csv,
        model=args.model,
        prompt=args.prompt,
        batch_size=args.batch_size,
        start_from=args.start_from
    )


# Example usage:
# process_images_with_ollama(input_path="sample_data/train/images", output_csv="captions_qwen2.csv")
# process_images_with_ollama(input_path="../dataset/train/composition_train.csv", output_csv="descriptions_train.csv",
#                            model="gemma3:4b-it-qat",batch_size=10, start_from = 108270)

process_images_with_ollama(input_path="ARAS400k/train/images", output_csv="ARAS400k_train_vision_gemma3_4b.csv",
                           model="moondream:latest",batch_size=1, start_from = 0)

# if __name__ == "__main__":
#     main()