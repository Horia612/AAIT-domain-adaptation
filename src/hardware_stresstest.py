import torch
import psutil
import os
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_memory_usage():
    """Returns current RAM usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def get_gpu_memory():
    """Returns GPU VRAM usage in GB if available"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)
    return 0.0

def run_benchmark():
    print(f"System Check Initiated..")
    
    # 1. Check Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   - Device Detected: {device.upper()}")
    if device == "cuda":
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   - Total VRAM: {vram_total:.2f} GB")
    else:
        print("No GPU detected. Model will run on CPU (Very Slow).")

    print("-" * 40)
    
    # 2. Load Model (The Memory Spike Test)
    model_id = "JustinDu/BARTxiv"  # The specific model from your project
    print(f"Loading Model: {model_id} ..")
    
    start_ram = get_memory_usage()
    start_vram = get_gpu_memory()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
        print("Model Loaded Successfully")
    except Exception as e:
        print(f"FAILED to load model. Error: {e}")
        return

    # 3. Run Inference (The Compute Test)
    print("Running Test Inference (Summarization)..")
    
    # Dummy text simulating a paper abstract (technical content)
    sample_text = """
    We present a novel approach to domain adaptation for neural summarization models. 
    While Large Language Models (LLMs) have shown impressive few-shot capabilities, 
    they remain computationally expensive. In this work, we explore lightweight 
    continual pre-training strategies using the BART architecture. Our experiments 
    on the ACL Anthology dataset demonstrate that targeted domain adaptation can 
    bridge the gap between scientific disciplines without catastrophic forgetting.
    """ * 10  # Repeat to make it long enough to stress the encoder

    inputs = tokenizer(sample_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    end_time = time.time()
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # 4. Report Stats
    peak_ram = get_memory_usage()
    peak_vram = get_gpu_memory()
    
    print("-" * 40)
    print("BENCHMARK RESULTS")
    print(f"   - Time to Summarize: {end_time - start_time:.2f} seconds")
    print(f"   - RAM Added: {peak_ram - start_ram:.2f} GB")
    if device == "cuda":
        print(f"   - VRAM Used: {peak_vram - start_vram:.2f} GB")
    
    print("\nGenerated Summary Preview:")
    print(f"   \"{summary[:100]}...\"")
    
    # 5. The Verdict
    print("-" * 40)
    print("VERDICT:")
    if device == "cpu":
        print("CRITICAL: You are on CPU. Teacher Labeling 5,000 papers will take days.")
        print("Recommendation: Use Google Colab (Free Tier) or rent a GPU.")
    elif (peak_vram - start_vram) > 6:
        print(" WARNING: High Memory Usage. You might need 'Gradient Accumulation' for fine-tuning.")
    else:
        print("SUCCESS: Your hardware is capable of local development.")

if __name__ == "__main__":
    run_benchmark()