import os
import re
import sys
import logging
from docling.document_converter import DocumentConverter
from llama_index.core import Document
from dotenv import load_dotenv

# Path correction to find 'utils' from the 'pipeline' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.storage_utils import (
    calculate_file_hash, 
    save_to_cache, 
    register_in_db, 
    init_tracker_db,
    is_file_processed,
    PROJECT_ROOT
)

logging.basicConfig(
    filename='ingestion.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()

def clean_markdown(text):
    """Bilingual cleaner for Solar PPA contracts."""
    patterns = [
        r"Open Solar Contracts v2\.0\..*?opensolarcontracts\.org/",
        r"User Note: To be determined on a jurisdiction specific basis\.",
        r"Contrats Solaires Ouverts v2\.0\..*?opensolarcontracts\.org/",
        r"Note de l'utilisateur\s*:\s*À déterminer.*?\.",
        r"https?://opensolarcontracts\.org/\S*",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\n\s*\d+\s*\n', '\n\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def main():
    init_tracker_db()
    dataset_path = os.path.join(PROJECT_ROOT, "Dataset")
    
    print("🐘 Initializing Local AI Ingestor...")
    converter = DocumentConverter()

    all_files = [f for f in os.listdir(dataset_path) if f.lower().endswith('.pdf')]
    print(f"🔍 Found {len(all_files)} files. Synchronizing...")

    for filename in all_files:
        full_path = os.path.join(dataset_path, filename)
        file_hash = calculate_file_hash(full_path)

        # 1. Check if we can skip the work
        if is_file_processed(file_hash): 
            print(f"⏭️  Skipping {filename} (Already in Database).")
            continue

        print(f"📂 Parsing: {filename}...")
        try:
            # 2. Only parse ONCE inside the try block
            result = converter.convert(full_path)
            raw_markdown = result.document.export_to_markdown()

            if raw_markdown:
                cleaned_markdown = clean_markdown(raw_markdown)
                total_pages = len(result.document.pages) if hasattr(result.document, 'pages') else 0
                
                doc_obj = Document(
                    text=cleaned_markdown, 
                    metadata={
                        "source": filename, 
                        "total_pages": total_pages,
                        "parser": "docling_production_v1",
                        "file_hash": file_hash
                    }
                )
                
                cache_path = save_to_cache(file_hash, [doc_obj]) 
                register_in_db(file_hash, filename, cache_path, params="bilingual_cleaned")
                
                logging.info(f"SUCCESS: {filename} | Hash: {file_hash[:8]}")
                print(f"   ✅ Processed & Cached.")
            
        except Exception as e:
            logging.error(f"CRASH: {filename} - Error: {str(e)}")
            print(f"   ❌ Failed: Check ingestion.log")

    print("\n🏁 Pipeline complete.")

if __name__ == "__main__":
    main()