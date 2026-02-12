#!/usr/bin/env python3
import os
import sys
import subprocess
import fcntl
import atexit

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ---------- SINGLETON LOCK ----------
PID_FILE = "/tmp/solar_rag_pipeline.pid"

def enforce_singleton():
    """Exit if another instance is already running."""
    try:
        pid_fd = open(PID_FILE, 'w')
        fcntl.flock(pid_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        print(f"❌ Another instance is already running (PID: {open(PID_FILE).read().strip()})")
        sys.exit(1)
    
    pid_fd.write(str(os.getpid()))
    pid_fd.flush()
    
    def cleanup():
        fcntl.flock(pid_fd, fcntl.LOCK_UN)
        pid_fd.close()
        try:
            os.remove(PID_FILE)
        except:
            pass
    atexit.register(cleanup)

enforce_singleton()
# ------------------------------------

# ---------- GLOBAL SETTINGS ----------
print("🌍 FORCING BGE-M3 Multilingual Embeddings (1024-dim)...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
# --------------------------------------

def main():
    print("🔋 --- SOLAR RAG PIPELINE --- 🔋")
    
    # Absolute paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    print(f"📁 Project root: {project_root}")
    print(f"📁 Cache dir: {os.path.join(project_root, 'cache')}")
    print(f"📁 Chroma DB: {os.path.join(project_root, 'chroma_db')}")
    
    # Phase 1: Ingest
    print("\n🚀 Phase 1: Ingestion")
    subprocess.run([sys.executable, os.path.join(project_root, "pipeline", "ingest_01.py")])
    
    # Phase 2: Index
    print("\n🚀 Phase 2: Indexing")
    subprocess.run([sys.executable, os.path.join(project_root, "pipeline", "index_02.py")])
    
    # Phase 3: Chat
    print("\n🚀 Phase 3: Chat")
    subprocess.run([sys.executable, os.path.join(project_root, "pipeline", "chat_03.py")])

if __name__ == "__main__":
    main()