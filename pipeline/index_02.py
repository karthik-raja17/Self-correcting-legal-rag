#!/usr/bin/env python3
import os
os.environ['LLAMA_INDEX_LOGGING_LEVEL'] = 'WARNING'

import logging
logging.basicConfig(level=logging.WARNING)

import json
import chromadb
from dotenv import load_dotenv

from llama_index.core import Settings
Settings.llm = None

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Absolute paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")

os.makedirs(CACHE_DIR, exist_ok=True)

def main():
    load_dotenv()
    
    print(f"📁 Cache dir: {CACHE_DIR}")
    print(f"📁 Chroma DB: {CHROMA_DB_PATH}")
    
    # 🚀 EXPLICIT MODEL
    print("Loading BGE-M3 (1024-dim)...")
    try:
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    except Exception as e:
        print(f"❌ Failed to load BGE-M3: {e}")
        raise

    # 📁 Connect to ChromaDB – DO NOT DELETE, just get or create
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    collection_name = "solar_ppa_collection"
    # Get existing collection or create new one
    try:
        chroma_collection = db.get_collection(collection_name)
        print(f"📚 Using existing collection '{collection_name}'")
    except:
        chroma_collection = db.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ Created new collection '{collection_name}'")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 📦 Process ONLY NEW cache files
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        print(f"📁 Created cache directory: {CACHE_DIR}")
    
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]
    if not cache_files:
        print("⏭️ No new cache files to index.")
        return

    all_nodes = []
    parser = MarkdownNodeParser()
    for cf in cache_files:
        cache_path = os.path.join(CACHE_DIR, cf)
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
            doc = Document(text=cache_data[0]['text'], metadata=cache_data[0]['metadata'])
            nodes = parser.get_nodes_from_documents([doc])
            all_nodes.extend(nodes)

    # 🔒 INDEX NEW NODES
    print(f"📦 Indexing {len(all_nodes)} nodes with BGE-M3 (1024-dim)...")
    index = VectorStoreIndex(
        all_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
        insert_batch_size=20  # Prevents memory issues
    )
    
    # ✅ SUCCESS – remove the processed JSON files
    for cf in cache_files:
        cache_path = os.path.join(CACHE_DIR, cf)
        os.remove(cache_path)
        print(f"🗑️ Removed processed cache: {cf}")
    
    print("✅ Vector DB updated incrementally.")

if __name__ == "__main__":
    main()