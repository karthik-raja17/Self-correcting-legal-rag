import os
import pdfplumber
from docling.document_converter import DocumentConverter
from llama_index.core import Document # For wrapping Docling output for your utils

# Import the 'Brain' tools we built
from utils.storage_utils import (
    calculate_file_hash, 
    save_to_cache, 
    register_in_db, 
    init_tracker_db,
    PROJECT_ROOT
)

def local_layout_parser(pdf_path):
    """pdfplumber baseline for Pages 6 & 7 (0-indexed)."""
    extracted_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        target_pages = pdf.pages[5:7] 
        for i, page in enumerate(target_pages):
            text = page.extract_text(layout=True)
            extracted_pages.append(f"## [pdfplumber] Page {i+6}\n{text}")
    return "\n\n".join(extracted_pages)

def main():
    # 1. Setup the 'Brain'
    init_tracker_db()
    
    filename = "EN OSC V2 - PPA Final.pdf"
    sample_pdf = os.path.join(PROJECT_ROOT, "Dataset", filename)

    if not os.path.exists(sample_pdf):
        print(f"❌ File not found at {sample_pdf}")
        return

    # 2. Fingerprint the file
    file_hash = calculate_file_hash(sample_pdf)
    print(f"📂 Processing: {filename}\n   Hash: {file_hash[:16]}...")

    # 3. Local Baseline (Always run for comparison)
    print("\n🚀 RUNNING LOCAL BASELINE (pdfplumber)...")
    local_out = local_layout_parser(sample_pdf)

    # 4. Smart AI Parsing with Docling
    print("\n🐘 RUNNING DOCLING AI (Local)...")
    converter = DocumentConverter()

    try:
        # Pass the 1-indexed page range [6, 7]
        result = converter.convert(sample_pdf, page_range=[6, 7])
        markdown_text = result.document.export_to_markdown()

        if markdown_text:
            # Wrap in LlamaIndex Document object so your save_to_cache works
            # We add metadata so the DB knows exactly what this is
            doc_obj = Document(
                text=markdown_text, 
                metadata={"source": filename, "page_range": "8-11", "parser": "docling"}
            )
            
            # --- THE INGESTION LOGIC ---
            # Save to JSON cache
            cache_path = save_to_cache(file_hash, [doc_obj]) 
            # Register in SQLite tracker
            register_in_db(file_hash, filename, cache_path, params="docling_p6-7")
            
            docling_out = markdown_text
            print(f"✅ Success! Data cached and registered at: {cache_path}")
        else:
            docling_out = "⚠️ Docling returned empty results."

    except Exception as e:
        print(f"❌ Docling Failed: {e}")
        docling_out = "ERROR: Processing failed."

    # 2. Print EVERYTHING (No slicing)
    print("\n" + "="*40 + " FULL DOCLING OUTPUT " + "="*40)
    print(markdown_text) 
    print("="*94)
    print(f"✅ Full data cached at: {cache_path}")

if __name__ == "__main__":
    main()