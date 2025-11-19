import importlib
import sys

def check_library(lib_name, install_name=None):
    if install_name is None:
        install_name = lib_name
    try:
        importlib.import_module(lib_name)
        print(f"{lib_name:<15} found")
        return True
    except ImportError:
        print(f"{lib_name:<15} NOT found. Run: pip install {install_name}")
        return False

required_libraries = [
    ("torch", "torch"),                 # The Deep Learning engine
    ("transformers", "transformers"),   # HuggingFace models (BART)
    ("datasets", "datasets"),           # Handling large text data
    ("pandas", "pandas"),               # Data manipulation (like SQL/Excel)
    ("numpy", "numpy"),                 # Math operations
    ("PyPDF2", "PyPDF2"),               # PDF parsing
    ("tqdm", "tqdm"),                   # Progress bars
    ("sklearn", "scikit-learn"),        # Basic metrics
    ("rouge_score", "rouge_score")      # Summarization metric
]

print("Checking Environment for AAIT Project..")
all_good = True
for lib, install in required_libraries:
    if not check_library(lib, install):
        all_good = False

print("-" * 40)
if all_good:
    print("Environment is ready!")
else:
    print("Please install the missing libraries above.")