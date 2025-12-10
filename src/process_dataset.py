import json
import torch
import random
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

INPUT_FILE = "acl_dataset/all_papers.json"
OUTPUT_FILE = "acl_dataset/acl_ready_for_training.json"
CHECKPOINT_FILE = "acl_dataset/temp_checkpoint.json"
MODEL_NAME = "JustinDu/BARTxiv"

# Project constraints
MIN_INPUT_TOKENS = 3000
MAX_INPUT_TOKENS = 10000
MIN_ABSTRACT_LEN = 100

def load_resources():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    return tokenizer, model, device

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def process_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_papers = json.load(f)

    tokenizer, model, device = load_resources()
    
    processed_data = []
    stats = {"total": len(raw_papers), "accepted": 0, "rejected_short": 0, "rejected_long": 0, "rejected_empty": 0}

    for paper in tqdm(raw_papers, desc="Processing"):
        content = paper.get("content")
        abstract = paper.get("abstract")

        if not content or not abstract:
            stats["rejected_empty"] += 1
            continue

        # Approximate token count for speed
        content_tokens = len(content.split()) * 1.3
        
        if content_tokens < MIN_INPUT_TOKENS:
            stats["rejected_short"] += 1
            continue
        if content_tokens > MAX_INPUT_TOKENS:
            stats["rejected_long"] += 1
            continue

        # Generate Teacher Label
        inputs = tokenizer(content, max_length=1024, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"], 
                max_length=300, 
                min_length=100, 
                length_penalty=2.0, 
                num_beams=4, 
                early_stopping=True
            )
        
        teacher_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        paper["teacher_summary"] = teacher_summary
        paper["token_count_est"] = int(content_tokens)
        processed_data.append(paper)

        if len(processed_data) % 50 == 0:
            save_json(processed_data, CHECKPOINT_FILE)

    # Create Splits (80/10/10)
    random.shuffle(processed_data)
    n = len(processed_data)
    train_split = int(n * 0.8)
    val_split = int(n * 0.9)

    final_dataset = {
        "metadata": stats,
        "train": processed_data[:train_split],
        "validation": processed_data[train_split:val_split],
        "test": processed_data[val_split:]
    }
    
    stats["accepted"] = n
    save_json(final_dataset, OUTPUT_FILE)
    
    print("\nProcessing Complete.")
    print(f"Stats: {stats}")
    print(f"Train: {len(final_dataset['train'])}, Val: {len(final_dataset['validation'])}, Test: {len(final_dataset['test'])}")

if __name__ == "__main__":
    process_dataset()