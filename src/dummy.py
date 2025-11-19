import sys
import json
import importlib

print(f"--- Environment Check Initiated ---")
print(f"Python Version: {sys.version.split()[0]}\n")

# A list of required libraries to test by attempting to import them
libraries_to_check = [
    "torch",
    "transformers",
    "datasets",
    "sentencepiece"
]

all_passed = True

# 1. Core Library Import Check
print("Checking Core Library Imports..")
for lib_name in libraries_to_check:
    try:
        importlib.import_module(lib_name)
        print(f"PASSED: Module '{lib_name}' imported successfully.")
    except ImportError:
        print(f"FAILED: Module '{lib_name}' not found. Please install (pip install {lib_name})")
        all_passed = False

# Exit if core libraries are missing
if not all_passed:
    print("\nOne or more libraries are missing. Please install them to proceed.")
    sys.exit()

# 2. PyTorch (torch) Functionality
print("\nTesting PyTorch (torch) Functionality..")
try:
    import torch
    
    # Test tensor creation
    tensor = torch.rand(2, 2)
    print(f"PASSED: torch.tensor created (shape: {tensor.shape}).")
    
    # Test for CUDA (GPU) availability
    if torch.cuda.is_available():
        print(f"INFO: GPU (CUDA) is available.")
        print(f"INFO: Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("INFO: GPU (CUDA) not found. Operations will run on CPU.")
except Exception as e:
    print(f"FAILED: PyTorch functionality test error: {e}")
    all_passed = False

# 3. JSONL Format Test
print("\nTesting JSON Functionality..")
try:
    test_data = {"text_ro": "Test text.", "summary_ro": "Test summary."}
    json_line = json.dumps(test_data, ensure_ascii=False)
    
    print(f"PASSED: JSON serialization successful.")
    print(f"INFO: Example JSONL output: {json_line}")
except Exception as e:
    print(f"FAILED: JSON serialization error: {e}")
    all_passed = False

# Final Summary
print("\nCheck Complete!")
if all_passed:
    print("SUCCESS: All essential libraries and functionalities are working.")
else:
    print("WARNING: One or more checks failed. Please review the output above.")