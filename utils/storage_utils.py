import hashlib
import sqlite3
import json
import os

# --- PATH CONFIGURATION ---
PROJECT_ROOT = os.getcwd()

# Define Absolute Paths
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
DB_PATH = os.path.join(PROJECT_ROOT, "ingestion_tracker.db")

def ensure_environment():
    """Ensures that the cache directory exists."""
    os.makedirs(CACHE_DIR, exist_ok=True)

def calculate_file_hash(file_path, chunk_size=65536):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def init_tracker_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(''' 
            CREATE TABLE IF NOT EXISTS parsed_files (
            file_hash TEXT PRIMARY KEY,
            file_name TEXT,
            json_path TEXT,
            parsing_parameters TEXT, 
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
    conn.commit()
    conn.close()

def save_to_cache(file_hash, llama_documents):
    """
    Save llamaindex documents to local JSON file
    """
    ensure_environment()

    cache_file_name = f"{file_hash}.json"
    full_cache_path = os.path.join(CACHE_DIR,cache_file_name)

    data_to_save = [doc.dict() for doc in llama_documents]

    with open(full_cache_path, "w") as f:
        json.dump(data_to_save, f, indent=4)

    return full_cache_path

def register_in_db(file_hash, file_name, cache_path, params="markdown"):
    """Records the parse in SQLite using the absolute DB_PATH."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
            INSERT OR REPLACE INTO parsed_files (file_hash, file_name, json_path, parsing_parameters)
            VALUES (?, ?, ?, ?)
            """, (file_hash,file_name,cache_path,params))
    conn.commit()
    conn.close()
