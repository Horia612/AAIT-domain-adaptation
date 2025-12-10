import os
import re
import json
import requests
import fitz  # PyMuPDF
import xml.etree.ElementTree as ET
from datetime import datetime
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor

# -------------------------
# SETTINGS
# -------------------------
ANTHOLOGY_PATH = "acl-anthology/data/xml"
OUTPUT_DIR = "acl_dataset"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "all_papers.json")
PDF_DIR = os.path.join(OUTPUT_DIR, "pdfs")
ACL_BASE_URL = "https://aclanthology.org/"
MAX_PAPERS = 10000  # Threshold: stop after this many papers
NUM_THREADS = 10   # Parallel threads

# Create output folders if not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

current_year = datetime.now().year
start_year = current_year - 1

papers = []

# -------------------------
# PDF Download Function
# -------------------------
def download_pdf(url, target):
    for _ in range(3):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                with open(target, "wb") as f:
                    f.write(r.content)
                return True
        except Exception:
            time.sleep(1)
    return False

# -------------------------
# PDF Text Extraction
# -------------------------
def extract_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return preprocess_pdf_text(text)
    except Exception:
        return None

# -------------------------
# PDF Preprocessing
# -------------------------
HEADER_FOOTER_PATTERNS = [
    r"^\d+$",               # page numbers
    r"^Proceedings of .*",  # conference headers
    r"^ACL \d{4}$",         # ACL year headers
    r"^\[.*\]$"             # reference markers like [1], [2]
]

def preprocess_pdf_text(text):
    text = remove_headers_footers(text)
    text = remove_references(text)
    text = clean_math(text)
    text = normalize_whitespace(text)
    return text

def remove_headers_footers(text):
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(re.match(p, line) for p in HEADER_FOOTER_PATTERNS):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

def remove_references(text):
    ref_start = re.search(r"^references", text, re.I | re.M)
    if ref_start:
        text = text[:ref_start.start()]
    return text.strip()

def clean_math(text):
    text = re.sub(r"\$.*?\$", "", text)       # remove inline math
    text = text.replace("\\(", "").replace("\\)", "")
    return text

def normalize_whitespace(text):
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)         # collapse multiple spaces
    text = re.sub(r"\s*\n\s*", "\n", text)   # normalize newlines
    return text.strip()

def get_full_text(el):
    """Recursively get full text from an XML element, handling line breaks and children."""
    if el is None:
        return None
    texts = [el.text or ""]
    for child in el:
        texts.append(child.text or "")
        texts.append(child.tail or "")
    # Join, normalize spaces, strip
    full_text = " ".join(t.strip() for t in texts if t.strip())
    return full_text

# -------------------------
# XML SCANNING
# -------------------------
print("Scanning XML files...")

for filename in os.listdir(ANTHOLOGY_PATH):
    if not filename.endswith(".xml"):
        continue

    match = re.match(r"(\d{4})", filename)
    if not match:
        continue

    year = int(match.group(1))
    if year < start_year:
        continue

    file_path = os.path.join(ANTHOLOGY_PATH, filename)
    tree = ET.parse(file_path)
    root = tree.getroot()

    for paper in root.findall(".//paper"):
        if len(papers) >= MAX_PAPERS:
            break

        pid = paper.attrib.get("id", "")
        title_el = paper.find("title")
        abstract_el = paper.find("abstract")
        url_el = paper.find("url")
        pdf_url = url_el.text if url_el is not None else None

        title = get_full_text(title_el)
        abstract = get_full_text(abstract_el)

        # Fix URLs: prepend ACL base URL if needed
        if pdf_url:
            pdf_url = pdf_url.strip()
            if not pdf_url.startswith("http"):
                pdf_url = ACL_BASE_URL + pdf_url

        papers.append({
            "id": pid,
            "year": year,
            "title": title,
            "abstract": abstract,
            "pdf_url": pdf_url,
            "content": None
        })

    if len(papers) >= MAX_PAPERS:
        break

print(f"Collected {len(papers)} papers from last 1 year.")
print("Downloading & extracting PDFs in parallel...\n")

# -------------------------
# PAPER PROCESSING FUNCTION
# -------------------------
def process_paper(paper):
    url = paper["pdf_url"]
    if not url:
        return paper

    pdf_filename = paper["id"].replace("/", "_") + ".pdf"
    pdf_path = os.path.join(PDF_DIR, pdf_filename)

    if not os.path.exists(pdf_path):
        ok = download_pdf(url, pdf_path)
        if not ok:
            paper["content"] = None
            return paper

    paper["content"] = extract_text(pdf_path)
    return paper

# -------------------------
# PARALLEL EXECUTION
# -------------------------
with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    papers = list(tqdm(executor.map(process_paper, papers), total=len(papers)))

# -------------------------
# FINAL SAVE
# -------------------------
print("\nSaving final JSON...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(papers, f, ensure_ascii=False, indent=2)

print(f"Done. Saved dataset in: {OUTPUT_FILE}")
