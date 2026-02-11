import os
import sys
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. FORCE GLOBAL BILINGUAL SETTINGS
# We do this at the very top to override any LlamaIndex defaults
print("🌍 FORCING BGE-M3 Multilingual Embeddings (1024-dim)...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

# 2. Now import your modules
from pipeline import ingest_01, index_02

def main():
    print("🔋 --- SOLAR RAG PIPELINE --- 🔋")
    
    # Phase 1: Ingest (Skips if already done)
    ingest_01.main() 
    
    # Phase 2: Index (Now guaranteed to use BGE-M3)
    index_02.main()
    
    # Phase 3: Launch Chat
    print("\n🧠 Launching local LLM (Llama 3.1)...")
    os.system("python pipeline/chat_03.py")

if __name__ == "__main__":
    main()