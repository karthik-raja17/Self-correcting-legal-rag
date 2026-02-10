import os
import re
from docling.document_converter import DocumentConverter
from llama_index.core import Document
from dotenv import load_dotenv

# Import the 'Brain' tools we've built
from utils.storage_utils import (
    calculate_file_hash, 
    save_to_cache, 
    register_in_db, 
    init_tracker_db,
    PROJECT_ROOT
)

load_dotenv()

def clean_markdown(text):
    """
    Advanced cleaning to remove multi-line legal footers, 
    standalone page numbers, and repetitive boilerplate.
    """
    # 1. Remove the multi-line IRENA/Terms of Use disclaimer
    footer_pattern = r"Open Solar Contracts v2\.0\..*?from time to time\."
    text = re.sub(footer_pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    
    # 2. Remove standalone page numbers (e.g., \n\n6\n\n) 
    # This keeps the flow clean for the LLM while we store page info in metadata
    text = re.sub(r'\n\s*\d+\s*\n', '\n\n', text)
    
    # 3. Remove standalone opensolarcontracts.org URLs
    text = re.sub(r"https?://opensolarcontracts\.org/\S*", "", text)
    
    # 4. Remove repetitive instructional 'User Notes'
    text = re.sub(r"User Note: To be determined on a jurisdiction specific basis\.", "", text)
    
    # 5. Collapse excessive whitespace left by removals
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def main():
    # 1. Initialize the tracker database
    init_tracker_db()
    
    # 2. File Setup
    filename = "EN OSC V2 - PPA Final.pdf"
    sample_pdf = os.path.join(PROJECT_ROOT, "Dataset", filename)

    if not os.path.exists(sample_pdf):
        print(f"❌ File not found at {sample_pdf}")
        return

    # 3. Fingerprint (Deduplication)
    file_hash = calculate_file_hash(sample_pdf)
    print(f"📂 Processing: {filename}")
    print(f"   Hash: {file_hash[:16]}...")

    # 4. Local AI Parsing with Docling
    print("\n🐘 RUNNING DOCLING AI (Full Local Parse)...")
    print("   (Processing tables and structure on your i7...)")
    converter = DocumentConverter()

    try:
        # Parse the entire document
        result = converter.convert(sample_pdf)
        raw_markdown = result.document.export_to_markdown()

        if raw_markdown:
            # 5. Cleaning Phase
            print("🧹 Scrubbing footers and page artifacts...")
            cleaned_markdown = clean_markdown(raw_markdown)

            # 6. Metadata Creation for Citations
            # We store the page count and source so RAG knows where to look
            total_pages = len(result.document.pages) if hasattr(result.document, 'pages') else "Unknown"
            
            doc_obj = Document(
                text=cleaned_markdown, 
                metadata={
                    "source": filename, 
                    "total_pages": total_pages,
                    "parser": "docling_v2_cleaned",
                    "file_hash": file_hash
                }
            )
            
            # 7. Stateful Ingestion (Cache & DB)
            cache_path = save_to_cache(file_hash, [doc_obj]) 
            register_in_db(file_hash, filename, cache_path, params="docling_full_cleaned")
            
            # 8. Success Reporting
            print(f"✅ Success! Document cached at: {cache_path}")
            print(f"📈 Stats: {total_pages} pages | {len(cleaned_markdown):,} characters.")
            
            # Final verification preview
            print("\n" + "="*20 + " END OF DOCUMENT PREVIEW " + "="*20)
            print(cleaned_markdown[-1000:]) 
            print("="*66)
            
        else:
            print("⚠️ Docling returned empty results.")

    except Exception as e:
        print(f"❌ Ingestion Failed: {e}")

if __name__ == "__main__":
    main()