import os
import shutil
import sqlite3
from utils.storage_utils import CACHE_DIR, PROJECT_ROOT

def reset_system():
    print("🧹 Starting full RAG system reset...")

    # 1. Clear the JSON Cache
    if os.path.exists(CACHE_DIR):
        print(f"  -> Clearing Cache: {CACHE_DIR}")
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR) # Recreate empty folder

    # 2. Clear the ChromaDB (Vector Store)
    chroma_path = os.path.join(PROJECT_ROOT, "chroma_db")
    if os.path.exists(chroma_path):
        print(f"  -> Deleting Vector DB: {chroma_path}")
        shutil.rmtree(chroma_path)

    # 3. Clear the SQLite Ingestion Tracker
    db_path = os.path.join(PROJECT_ROOT, "ingestion_tracker.db")
    if os.path.exists(db_path):
        print(f"  -> Resetting Ingestion Tracker: {db_path}")
        try:
            # We connect and drop the table rather than deleting the file
            # to avoid permission issues in some environments.
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS processed_files")
            conn.commit()
            conn.close()
            os.remove(db_path) # Now safe to remove
        except Exception as e:
            print(f"     ⚠️ Note: Could not delete DB file (might be open): {e}")

    # 4. Clear Logs
    log_file = os.path.join(PROJECT_ROOT, "ingestion.log")
    if os.path.exists(log_file):
        os.remove(log_file)

    print("\n✨ System is fresh. You can now run your pipeline starting with 01_ingest.py.")

if __name__ == "__main__":
    confirm = input("⚠️ This will delete ALL cached data and embeddings. Type 'yes' to proceed: ")
    if confirm.lower() == 'yes':
        reset_system()
    else:
        print("❌ Reset cancelled.")