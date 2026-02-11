import os
import chromadb
import sys
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 👇 New import
from llama_index.llms.groq import Groq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.storage_utils import PROJECT_ROOT

def main():
    load_dotenv()
    
    # Your existing multilingual embedder – perfect for French/English
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    
    # Connect to ChromaDB (already populated with 1024‑dim vectors)
    db_path = os.path.join(PROJECT_ROOT, "chroma_db")
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_collection("solar_ppa_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Build index
    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        embed_model=embed_model
    )
    
    # 👇 GROQ LLM – free, fast, multilingual
    llm = Groq(
        model="llama-3.1-8b-instant",           # or "llama3-70b-8192" (slower, better French)
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
        request_timeout=60.0
    )
    
    # System prompt – keep it bilingual / French‑ready
    system_prompt = (
        "You are an expert Solar Energy Legal Consultant. "
        "Cite the source and respond in the language of the user. "
        "If the user asks in French, answer in fluent French. "
        "If they ask in English, answer in English."
    )
    
    query_engine = index.as_query_engine(
        similarity_top_k=5, 
        llm=llm, 
        system_prompt=system_prompt
    )
    
    print("\n☀️ SOLAR CHAT IS ONLINE (Groq Llama 3 – bilingual)")
    while True:
        query = input("\n💬 Ask a question (exit to quit): ")
        if query.lower() in ['exit', 'quit']:
            break
        response = query_engine.query(query)
        print(f"\n📢 ANSWER:\n{response}")

if __name__ == "__main__":
    main()