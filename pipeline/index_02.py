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

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.storage_utils import CACHE_DIR, PROJECT_ROOT



def main():
    load_dotenv()
    
    # 🚀 EXPLICIT MODEL – crash loudly if it fails
    print("Loading BGE-M3 (1024-dim)...")
    try:
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    except Exception as e:
        print(f"❌ Failed to load BGE-M3: {e}")
        raise

    # 💣 FRESH CHROMA COLLECTION
    db_path = os.path.join(PROJECT_ROOT, "chroma_db")
    db = chromadb.PersistentClient(path=db_path)
    
    collection_name = "solar_ppa_collection"
    try:
        db.delete_collection(collection_name)
        print(f"🔥 Deleted old '{collection_name}'")
    except:
        pass
    
    chroma_collection = db.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"✅ Created new collection (dimension will be 1024 on first insert)")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 📦 Process cache (unchanged)
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]
    if not cache_files:
        print("⏭️ No new cache files to index.")
        return

    all_nodes = []
    parser = MarkdownNodeParser()
    for cf in cache_files:
        with open(os.path.join(CACHE_DIR, cf), 'r') as f:
            cache_data = json.load(f)
            doc = Document(text=cache_data[0]['text'], metadata=cache_data[0]['metadata'])
            nodes = parser.get_nodes_from_documents([doc])
            all_nodes.extend(nodes)

    # 🔒 EXPLICIT EMBED MODEL PASSED
    print(f"📦 Indexing {len(all_nodes)} nodes with BGE-M3 (1024-dim)...")
    index = VectorStoreIndex(
        all_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    print("✅ Vector DB updated.")

if __name__ == "__main__":
    main()